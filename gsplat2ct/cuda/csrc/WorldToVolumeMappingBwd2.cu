/**
 * @file WorldToVolumeMappingBwd2.cu
 * @brief Second-order backward CUDA kernel: world-to-volume mapping
 *
 * Computes first-order gradients AND second-order Hessians through
 * the world-to-volume mapping.
 *
 * =========================================================================
 * Mathematical Derivation
 * =========================================================================
 *
 * The mapping has two paths:
 *   Path A: means → means3d_v    (affine, linear)
 *   Path B: covars → Sigma_v → conics   (diagonal scaling + matrix inverse)
 *
 * =========================================================================
 * Path A: means → means3d_v  (affine, purely linear)
 * =========================================================================
 *
 * means3d_v[k] = r[k] * means[k] + const
 * Jacobian: J_A[k][l] = r[k] * delta_{kl}    (diagonal)
 *
 * First-order:
 *   v_means[k] = r[k] * v_means3d_v[k]
 *
 * Second-order (self-Hessian is ZERO since J_A is constant):
 *   h_means[i,j] = r[i] * r[j] * h_means3d_v[i,j]
 *
 * Cross with covars (no coupling through the Jacobian):
 *   h_means_covars[i,j] = r_mean[i] * h_means3d_v_conics[i,j_conic]
 *     where j_conic needs to be remapped through the conics←covars chain
 *
 * =========================================================================
 * Path B: covars → conics  (diagonal scale + blur + matrix inverse)
 * =========================================================================
 *
 * Step 1: Sigma_v[i,j] = r[i] * r[j] * covars[i,j] + eps * delta_{ij}
 *   Jacobian: d(Sigma_v_{ij}) / d(covars_{kl}) = r[i]*r[j] * delta_{ik}*delta_{jl}
 *   This is a diagonal-like scaling (element-wise).
 *
 * Step 2: conics = Sigma_v^{-1}
 *   First-order VJP:
 *     v_Sigma_v = - P @ v_conics_sym @ P     where P = conics
 *   Second-order:
 *     The Hessian of f(Σ) = Σ^{-1} at element level:
 *     ∂²(P_{ab}) / ∂Σ_{cd} ∂Σ_{ef} = P_{ac}*P_{db}*P_{fe} + P_{ae}*P_{fb}*P_{dc}
 *                                     (symmetrized for symmetric Σ)
 *
 * Combined second-order through chains:
 *   h_covars = J_B^T @ h_conics @ J_B + sum_k v_conics_k * Hess_k(covars)
 *
 * Since J_B is element-wise scaling (r[i]*r[j]):
 *   h_covars[i,j; k,l] = r[i]*r[j] * r[k]*r[l] * h_Sv[i,j; k,l]
 *
 * And h_Sv is the Hessian through matrix inverse:
 *   h_Sv = h_conics_mapped + curvature_terms
 *
 * =========================================================================
 * Cross-Hessians
 * =========================================================================
 *
 * Cross means-covars:
 *   Since means and covars go through independent paths (A and B),
 *   the cross-Hessian comes only from downstream h_means3d_v_conics:
 *   h_means_covars = J_A^T @ h_means3d_v_conics @ J_B
 *
 * Cross with opacity (pass-through):
 *   h_means_opacity[k] = r[k] * h_means3d_v_opacity[k]
 *   h_covars_opacity[ij] = r[i]*r[j] * v_Sv_opacity[ij]
 *     where v_Sv_opacity comes from conics_opacity mapped through inverse VJP
 *
 * Author: Dy164
 * Date: 2026-02-21
 */

#include "WorldToVolumeMapping.h"
#include "../include/common.h"
#include <cuda_runtime.h>

namespace gsplat2ct {

// ============================================================================
// Helper: Invert 3x3 symmetric matrix
// ============================================================================
template <typename S>
__device__ __forceinline__ bool invert_sym3x3(
    S a00, S a01, S a02, S a11, S a12, S a22,
    S& i00, S& i01, S& i02, S& i11, S& i12, S& i22
) {
    const S c00 = a11 * a22 - a12 * a12;
    const S c01 = a02 * a12 - a01 * a22;
    const S c02 = a01 * a12 - a02 * a11;
    const S c11 = a00 * a22 - a02 * a02;
    const S c12 = a01 * a02 - a00 * a12;
    const S c22 = a00 * a11 - a01 * a01;

    const S det = a00 * c00 + a01 * c01 + a02 * c02;
    if (abs(det) < static_cast<S>(1e-12)) {
        i00 = i01 = i02 = i11 = i12 = i22 = static_cast<S>(0);
        return false;
    }
    const S inv_det = static_cast<S>(1) / det;
    i00 = c00 * inv_det;
    i01 = c01 * inv_det;
    i02 = c02 * inv_det;
    i11 = c11 * inv_det;
    i12 = c12 * inv_det;
    i22 = c22 * inv_det;
    return true;
}

// ============================================================================
// Helper: Upper-tri index for symmetric 6x6 matrix
// For a symmetric n×n matrix stored as upper triangle,
// index(i,j) with i<=j: i*(2n-1-i)/2 + j
// For n=6: index(i,j) = i*(11-i)/2 + j
// ============================================================================
__device__ __forceinline__ int utri6_idx(int i, int j) {
    if (i > j) { int t = i; i = j; j = t; }
    return i * (11 - i) / 2 + j;
}

// Upper-tri index for symmetric 3x3 matrix (n=3)
// index(i,j) = i*(5-i)/2 + j
__device__ __forceinline__ int utri3_idx(int i, int j) {
    if (i > j) { int t = i; i = j; j = t; }
    return i * (5 - i) / 2 + j;
}

// ============================================================================
// Helper: Map upper-tri indices to (row, col) for 3x3
// ============================================================================
__device__ __forceinline__ void utri3_to_rc(int flat, int& r, int& c) {
    // flat: 0→(0,0), 1→(0,1), 2→(0,2), 3→(1,1), 4→(1,2), 5→(2,2)
    if (flat < 3) { r = 0; c = flat; }
    else if (flat < 5) { r = 1; c = flat - 2; }
    else { r = 2; c = 2; }
}

// ============================================================================
// Second-Order Backward Kernel
// ============================================================================

template <typename S>
__global__ void world_to_volume_mapping_bwd2_kernel(
    const S* __restrict__ means,       // [N, 3]
    const S* __restrict__ covars,      // [N, 6]
    const S* __restrict__ ranges,      // [3]
    const S* __restrict__ centers,     // [3]
    uint32_t dim2,
    uint32_t dim1,
    uint32_t dim0,
    S eps3d,
    // Downstream first-order gradients
    const S* __restrict__ v_means3d_v, // [N, 3]
    const S* __restrict__ v_conics_in, // [N, 6]
    // Downstream phi-phi Hessians
    const S* __restrict__ h_means3d_v_in,       // [N, 6]  upper-tri 3x3
    const S* __restrict__ h_conics_in,           // [N, 21] upper-tri 6x6
    const S* __restrict__ h_means3d_v_conics_in, // [N, 18] 3x6 row-major
    // Downstream phi-psi cross Hessians
    const S* __restrict__ h_means3d_v_opacity_in, // [N, 3]
    const S* __restrict__ h_conics_opacity_in,    // [N, 6]
    // Outputs: first-order gradients
    S* __restrict__ v_means_out,       // [N, 3]
    S* __restrict__ v_covars_out,      // [N, 6]
    // Outputs: theta-theta Hessians
    S* __restrict__ h_means_out,       // [N, 6]  upper-tri 3x3
    S* __restrict__ h_covars_out,      // [N, 21] upper-tri 6x6
    S* __restrict__ h_means_covars_out, // [N, 18] 3x6 row-major
    // Outputs: theta-psi cross Hessians
    S* __restrict__ h_means_opacity_out, // [N, 3]
    S* __restrict__ h_covars_opacity_out, // [N, 6]
    uint32_t N
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // ---- Compute scaling factors ----
    const S rx = ranges[0], ry = ranges[1], rz = ranges[2];
    const S D2 = static_cast<S>(dim2);
    const S D1 = static_cast<S>(dim1);
    const S D0 = static_cast<S>(dim0);
    const S r[3] = { D2 / rx, D1 / ry, D0 / rz };

    // ---- Recompute Sigma_v and P = conics = Sigma_v^{-1} ----
    const S* sv = covars + idx * 6;
    S Sv[6]; // upper-tri
    Sv[0] = r[0]*r[0] * sv[0] + eps3d;
    Sv[1] = r[0]*r[1] * sv[1];
    Sv[2] = r[0]*r[2] * sv[2];
    Sv[3] = r[1]*r[1] * sv[3] + eps3d;
    Sv[4] = r[1]*r[2] * sv[4];
    Sv[5] = r[2]*r[2] * sv[5] + eps3d;

    S P[6]; // P = Sv^{-1}
    bool ok = invert_sym3x3(Sv[0], Sv[1], Sv[2], Sv[3], Sv[4], Sv[5],
                            P[0], P[1], P[2], P[3], P[4], P[5]);

    if (!ok) {
        // Zero all outputs
        for (int i = 0; i < 3; ++i) v_means_out[idx*3+i] = static_cast<S>(0);
        for (int i = 0; i < 6; ++i) v_covars_out[idx*6+i] = static_cast<S>(0);
        for (int i = 0; i < 6; ++i) h_means_out[idx*6+i] = static_cast<S>(0);
        for (int i = 0; i < 21; ++i) h_covars_out[idx*21+i] = static_cast<S>(0);
        for (int i = 0; i < 18; ++i) h_means_covars_out[idx*18+i] = static_cast<S>(0);
        for (int i = 0; i < 3; ++i) h_means_opacity_out[idx*3+i] = static_cast<S>(0);
        for (int i = 0; i < 6; ++i) h_covars_opacity_out[idx*6+i] = static_cast<S>(0);
        return;
    }

    // Expand P to full 3x3 for easier indexing
    S Pf[3][3];
    Pf[0][0] = P[0]; Pf[0][1] = P[1]; Pf[0][2] = P[2];
    Pf[1][0] = P[1]; Pf[1][1] = P[3]; Pf[1][2] = P[4];
    Pf[2][0] = P[2]; Pf[2][1] = P[4]; Pf[2][2] = P[5];

    // ================================================================
    // FIRST-ORDER GRADIENTS
    // ================================================================

    // Path A: v_means[k] = r[k] * v_means3d_v[k]
    for (int k = 0; k < 3; ++k) {
        v_means_out[idx*3+k] = r[k] * v_means3d_v[idx*3+k];
    }

    // Path B: conics → Sigma_v → covars
    //
    // Key insight: v_conics is dL/d(conics_utri), where conics_utri stores
    // ONLY the upper triangle of P = Sv^{-1}.
    // The effective dL/dP_full is UPPER TRIANGULAR (lower = 0).
    //
    // V_P_upper @ P  (upper-tri @ symmetric → general)
    // Then v_Sv_full = -P @ (V_P_upper @ P)  (symmetric @ general → general)
    // v_covars: diag → v_Sv[i][i]*r[i]^2; off-diag → (v_Sv[i][j]+v_Sv[j][i])*r[i]*r[j]

    const S* vc = v_conics_in + idx * 6;

    // V_P_upper: upper-triangular matrix
    //   [ vc[0]  vc[1]  vc[2] ]
    //   [  0     vc[3]  vc[4] ]
    //   [  0      0     vc[5] ]

    // Compute T = V_P_upper @ P  (upper-tri @ symmetric → general 3x3)
    S T[3][3];
    // Row 0: (vc[0], vc[1], vc[2]) @ P
    T[0][0] = vc[0]*Pf[0][0] + vc[1]*Pf[1][0] + vc[2]*Pf[2][0];
    T[0][1] = vc[0]*Pf[0][1] + vc[1]*Pf[1][1] + vc[2]*Pf[2][1];
    T[0][2] = vc[0]*Pf[0][2] + vc[1]*Pf[1][2] + vc[2]*Pf[2][2];
    // Row 1: (0, vc[3], vc[4]) @ P
    T[1][0] = vc[3]*Pf[1][0] + vc[4]*Pf[2][0];
    T[1][1] = vc[3]*Pf[1][1] + vc[4]*Pf[2][1];
    T[1][2] = vc[3]*Pf[1][2] + vc[4]*Pf[2][2];
    // Row 2: (0, 0, vc[5]) @ P
    T[2][0] = vc[5]*Pf[2][0];
    T[2][1] = vc[5]*Pf[2][1];
    T[2][2] = vc[5]*Pf[2][2];

    // Compute vSv_full = -P @ T  (full 3x3, NOT symmetric)
    S vSv_full[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            S val = static_cast<S>(0);
            for (int k = 0; k < 3; ++k)
                val += Pf[i][k] * T[k][j];
            vSv_full[i][j] = -val;
        }

    // v_covars: diag → vSv[i][i]*r[i]^2; off-diag → (vSv[i][j]+vSv[j][i])*r[i]*r[j]
    // Flat 0: (0,0) diag
    v_covars_out[idx*6+0] = vSv_full[0][0] * r[0]*r[0];
    // Flat 1: (0,1) off-diag
    v_covars_out[idx*6+1] = (vSv_full[0][1] + vSv_full[1][0]) * r[0]*r[1];
    // Flat 2: (0,2) off-diag
    v_covars_out[idx*6+2] = (vSv_full[0][2] + vSv_full[2][0]) * r[0]*r[2];
    // Flat 3: (1,1) diag
    v_covars_out[idx*6+3] = vSv_full[1][1] * r[1]*r[1];
    // Flat 4: (1,2) off-diag
    v_covars_out[idx*6+4] = (vSv_full[1][2] + vSv_full[2][1]) * r[1]*r[2];
    // Flat 5: (2,2) diag
    v_covars_out[idx*6+5] = vSv_full[2][2] * r[2]*r[2];

    // ================================================================
    // SECOND-ORDER: MEANS HESSIAN
    // ================================================================
    // h_means[i,j] = r[i] * r[j] * h_means3d_v[i,j]
    // (affine transform has zero curvature, only GN term)
    const S* hm_in = h_means3d_v_in + idx * 6;
    for (int flat = 0; flat < 6; ++flat) {
        int i_r, j_c;
        utri3_to_rc(flat, i_r, j_c);
        h_means_out[idx*6+flat] = r[i_r] * r[j_c] * hm_in[flat];
    }

    // ================================================================
    // SECOND-ORDER: COVARS HESSIAN
    // ================================================================
    // We need: h_covars = J_B^T @ h_conics @ J_B + curvature_terms
    //
    // J_B maps covars[6] → conics[6] with intermediate Sigma_v
    // J_B is the composition: covars →(diag scale) Sigma_v →(inverse) conics
    //
    // The Jacobian of f: covars_ab → conics_cd is:
    //   df_{cd}/d(covars_{ab}) = sum_{ij} (d(conics_{cd})/d(Sv_{ij})) * (d(Sv_{ij})/d(covars_{ab}))
    //
    // d(Sv_{ij})/d(covars_{ab}) = r[i]*r[j] * delta_{ia}*delta_{jb}
    // d(conics_{cd})/d(Sv_{ij}) = -(P_{ci}*P_{jd} + P_{cj}*P_{id})/2 for i!=j
    //                             = -P_{ci}*P_{id} for i==j
    //
    // Since the scaling is element-wise, we can work directly in the
    // Sigma_v space for the Hessian of inverse, then scale by r[i]*r[j].
    //
    // h_Sv[ij,kl] = h_conics propagated through inverse:
    //   For the GN part:  J_inv^T @ h_conics @ J_inv
    //   For curvature:    sum_{cd} v_conics_{cd} * (d^2 conics_{cd})/(dSv_{ij} dSv_{kl})

    // Build J_inv: Jacobian of conics w.r.t. Sigma_v (6x6 matrix)
    // J_inv[cd_flat, ij_flat] = d(conics_{cd})/d(Sv_{ij})
    // For symmetric matrices:
    //   d(P_{ab})/d(Sv_{ij}) = -P_{ai}*P_{jb} when i==j
    //   d(P_{ab})/d(Sv_{ij}) = -(P_{ai}*P_{jb} + P_{aj}*P_{ib}) when i!=j
    //     (because Sv_{ij} appears in both (i,j) and (j,i) positions)
    // But since we store upper-tri and the gradient is accumulated for
    // the unique element, the off-diagonal Sv_{ij} (i<j) has factor 1 not 2.
    // The formula with upper-tri indexing is:
    //   d(P_{ab})/d(Sv_flat) where Sv_flat = utri3_idx(i,j)
    //     = -P_{ai}*P_{jb} - P_{aj}*P_{ib}  if i<j (off-diagonal)
    //     = -P_{ai}*P_{ib}                   if i==j (diagonal)
    // But careful: in upper-tri convention for symmetric derivative,
    // the off-diagonal v_Sv[ij] with i<j represents the element 
    // that appears at both (i,j) and (j,i) in the full matrix.
    // For the inverse derivative of a symmetric matrix stored upper-tri:
    //   d(P_{flat_a})/d(Sv_{flat_b})  requires careful accounting.
    //
    // Let's use the full matrix approach for clarity:

    // For each output pair (cd_flat, ij_flat), compute J_inv
    S J_inv[6][6]; // J_inv[output_conics_flat][input_Sv_flat]
    for (int out_flat = 0; out_flat < 6; ++out_flat) {
        int a, b;
        utri3_to_rc(out_flat, a, b); // conics element (a,b), a<=b
        for (int in_flat = 0; in_flat < 6; ++in_flat) {
            int i_s, j_s;
            utri3_to_rc(in_flat, i_s, j_s); // Sigma_v element (i,j), i<=j
            // d(P_{ab}) / d(Sv_{ij})
            // Full matrix: d(P_{ab})/d(A_{ij}) = -P_{ai}*P_{jb}
            // For symmetric A stored upper-tri, element Sv_flat corresponds
            // to A_{ij} and A_{ji} (when i!=j).
            // Chain rule gives: -P_{ai}*P_{jb} - P_{aj}*P_{ib} for i!=j
            //                   -P_{ai}*P_{ib}                 for i==j
            S val = -Pf[a][i_s] * Pf[j_s][b];
            if (i_s != j_s) {
                val += -Pf[a][j_s] * Pf[i_s][b];
            }
            J_inv[out_flat][in_flat] = val;
        }
    }

    // GN term: h_Sv_GN = J_inv^T @ h_conics_in @ J_inv
    const S* hc_in = h_conics_in + idx * 21;
    // First compute J_inv^T @ h_conics (6x6 symmetric) = temp[6][6]
    // h_conics is stored as 21-element upper-tri of 6x6
    S temp[6][6];
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            S val = static_cast<S>(0);
            for (int k = 0; k < 6; ++k) {
                // h_conics[k,j] with k<=j or j<=k
                int hc_idx = utri6_idx(k, j);
                val += J_inv[k][i] * hc_in[hc_idx]; // J_inv^T[i][k] = J_inv[k][i]
            }
            temp[i][j] = val;
        }
    }
    // h_Sv_GN = temp @ J_inv
    S h_Sv[6][6]; // full 6x6
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            S val = static_cast<S>(0);
            for (int k = 0; k < 6; ++k) {
                val += temp[i][k] * J_inv[k][j];
            }
            h_Sv[i][j] = val;
        }
    }

    // Curvature term: sum_{cd} v_conics_{cd} * d^2(P_{cd})/(dSv_{ij} dSv_{kl})
    // d^2(P_{ab})/(dA_{ij} dA_{kl}) = P_{ai}*P_{jk}*P_{lb} + P_{ak}*P_{li}*P_{jb}
    //                                 (for full non-symmetric A, then symmetrize for symmetric A)
    //
    // For symmetric A with upper-tri storage, the curvature contribution
    // to h_Sv[ij_flat, kl_flat] is:
    //   sum_{ab_flat} v_P_{ab_flat} * d^2(P_{ab}) / d(Sv_{ij_flat}) d(Sv_{kl_flat})
    //
    // This is complex but tractable for 3x3. Let's compute directly:
    for (int in1 = 0; in1 < 6; ++in1) {
        int i1, j1;
        utri3_to_rc(in1, i1, j1);
        for (int in2 = in1; in2 < 6; ++in2) {
            int i2, j2;
            utri3_to_rc(in2, i2, j2);

            S curv = static_cast<S>(0);
            // Sum over all output conics elements
            for (int out_flat = 0; out_flat < 6; ++out_flat) {
                int a, b;
                utri3_to_rc(out_flat, a, b);
                // Account for the weight: v_conics for off-diagonal is the
                // actual upstream gradient for that unique element
                S v_ab = (a == b) ? vc[out_flat] : vc[out_flat];

                // d^2(P_{ab}) / d(Sv_{i1,j1}) d(Sv_{i2,j2})
                // For symmetric Sv with upper-tri storage:
                // We need to sum over all full-matrix index combinations
                // that correspond to the upper-tri element.

                // Contribution from d/dA_{i1,j1} d/dA_{i2,j2}:
                //   P_{a,i1}*P_{j1,i2}*P_{j2,b} + P_{a,i2}*P_{j2,i1}*P_{j1,b}
                S term = Pf[a][i1]*Pf[j1][i2]*Pf[j2][b] + Pf[a][i2]*Pf[j2][i1]*Pf[j1][b];

                // If i1!=j1: also d/dA_{j1,i1} d/dA_{i2,j2}
                if (i1 != j1) {
                    term += Pf[a][j1]*Pf[i1][i2]*Pf[j2][b] + Pf[a][i2]*Pf[j2][j1]*Pf[i1][b];
                }
                // If i2!=j2: also d/dA_{i1,j1} d/dA_{j2,i2}
                if (i2 != j2) {
                    term += Pf[a][i1]*Pf[j1][j2]*Pf[i2][b] + Pf[a][j2]*Pf[i2][i1]*Pf[j1][b];
                }
                // If both off-diagonal: also d/dA_{j1,i1} d/dA_{j2,i2}
                if (i1 != j1 && i2 != j2) {
                    term += Pf[a][j1]*Pf[i1][j2]*Pf[i2][b] + Pf[a][j2]*Pf[i2][j1]*Pf[i1][b];
                }

                curv += v_ab * term;
            }

            h_Sv[in1][in2] += curv;
            if (in1 != in2) {
                h_Sv[in2][in1] += curv;
            }
        }
    }

    // Now scale h_Sv → h_covars: h_covars[ij,kl] = r[i]*r[j] * r[k]*r[l] * h_Sv[ij,kl]
    // Store as 21-element upper-tri of 6x6
    S r_flat[6]; // r factor for each upper-tri element
    for (int flat = 0; flat < 6; ++flat) {
        int i_r, j_c;
        utri3_to_rc(flat, i_r, j_c);
        r_flat[flat] = r[i_r] * r[j_c];
    }

    for (int i = 0; i < 6; ++i) {
        for (int j = i; j < 6; ++j) {
            int out_idx = utri6_idx(i, j);
            h_covars_out[idx*21+out_idx] = r_flat[i] * r_flat[j] * h_Sv[i][j];
        }
    }

    // ================================================================
    // CROSS-HESSIAN: means-covars
    // ================================================================
    // h_means_covars[mean_k, covars_flat] = 
    //   r[k] * sum_{conics_flat} h_means3d_v_conics[k, conics_flat] * J_inv_to_covars[conics_flat, covars_flat]
    //
    // J_inv_to_covars[cf, cov_f] = J_inv[cf, sv_f] * r_flat[sv_f]
    //   but sv_f == cov_f in indexing (same upper-tri element, just scaled)
    // So: J_total[cf, cov_f] = J_inv[cf, cov_f] * r_flat[cov_f]

    const S* hmc_in = h_means3d_v_conics_in + idx * 18; // [3, 6] row-major
    for (int k = 0; k < 3; ++k) {
        for (int cov_f = 0; cov_f < 6; ++cov_f) {
            S val = static_cast<S>(0);
            for (int cf = 0; cf < 6; ++cf) {
                // J_total[cf, cov_f] = J_inv[cf, cov_f] * r_flat[cov_f]
                // (Sv_flat and covars_flat use same indexing)
                val += hmc_in[k*6+cf] * J_inv[cf][cov_f] * r_flat[cov_f];
            }
            h_means_covars_out[idx*18+k*6+cov_f] = r[k] * val;
        }
    }

    // ================================================================
    // CROSS-HESSIAN: means-opacity
    // ================================================================
    // h_means_opacity[k] = r[k] * h_means3d_v_opacity[k]
    const S* hmo_in = h_means3d_v_opacity_in + idx * 3;
    for (int k = 0; k < 3; ++k) {
        h_means_opacity_out[idx*3+k] = r[k] * hmo_in[k];
    }

    // ================================================================
    // CROSS-HESSIAN: covars-opacity
    // ================================================================
    // h_covars_opacity[cov_f] = r_flat[cov_f] * sum_{cf} J_inv^T[cov_f][cf] * h_conics_opacity[cf]
    //                         = r_flat[cov_f] * sum_{cf} J_inv[cf][cov_f] * h_conics_opacity[cf]
    const S* hco_in = h_conics_opacity_in + idx * 6;
    for (int cov_f = 0; cov_f < 6; ++cov_f) {
        S val = static_cast<S>(0);
        for (int cf = 0; cf < 6; ++cf) {
            val += J_inv[cf][cov_f] * hco_in[cf];
        }
        h_covars_opacity_out[idx*6+cov_f] = r_flat[cov_f] * val;
    }
}

// ============================================================================
// Launch Function Implementation
// ============================================================================

template <typename S>
void launch_world_to_volume_mapping_bwd2(
    const S* __restrict__ means,
    const S* __restrict__ covars,
    const S* __restrict__ ranges,
    const S* __restrict__ centers,
    uint32_t dim2,
    uint32_t dim1,
    uint32_t dim0,
    S eps3d,
    const S* __restrict__ v_means3d_v,
    const S* __restrict__ v_conics,
    const S* __restrict__ h_means3d_v,
    const S* __restrict__ h_conics,
    const S* __restrict__ h_means3d_v_conics,
    const S* __restrict__ h_means3d_v_opacity,
    const S* __restrict__ h_conics_opacity,
    S* __restrict__ v_means,
    S* __restrict__ v_covars,
    S* __restrict__ h_means,
    S* __restrict__ h_covars,
    S* __restrict__ h_means_covars,
    S* __restrict__ h_means_opacity,
    S* __restrict__ h_covars_opacity,
    uint32_t N,
    cudaStream_t stream
) {
    if (N == 0) return;
    const int threads = GSPLAT2CT_N_THREADS;
    const int blocks = (N + threads - 1) / threads;

    world_to_volume_mapping_bwd2_kernel<S><<<blocks, threads, 0, stream>>>(
        means, covars, ranges, centers,
        dim2, dim1, dim0, eps3d,
        v_means3d_v, v_conics,
        h_means3d_v, h_conics, h_means3d_v_conics,
        h_means3d_v_opacity, h_conics_opacity,
        v_means, v_covars,
        h_means, h_covars, h_means_covars,
        h_means_opacity, h_covars_opacity,
        N
    );
}

// Explicit template instantiations
template void launch_world_to_volume_mapping_bwd2<float>(
    const float*, const float*, const float*, const float*,
    uint32_t, uint32_t, uint32_t, float,
    const float*, const float*,
    const float*, const float*, const float*,
    const float*, const float*,
    float*, float*,
    float*, float*, float*,
    float*, float*,
    uint32_t, cudaStream_t
);
template void launch_world_to_volume_mapping_bwd2<double>(
    const double*, const double*, const double*, const double*,
    uint32_t, uint32_t, uint32_t, double,
    const double*, const double*,
    const double*, const double*, const double*,
    const double*, const double*,
    double*, double*,
    double*, double*, double*,
    double*, double*,
    uint32_t, cudaStream_t
);

} // namespace gsplat2ct
