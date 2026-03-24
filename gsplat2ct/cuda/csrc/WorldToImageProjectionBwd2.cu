/**
 * @file WorldToImageProjectionBwd2.cu
 * @brief Second-order backward CUDA kernel for world-to-image projection
 *
 * Given downstream first-order gradients and second-order Hessians w.r.t.
 * (means2d, cov2d_3x3), computes upstream gradients and Hessians w.r.t.
 * (means3d, covars), plus cross-Hessians with external parameter (opacity).
 *
 * =========================================================================
 * Notation
 * =========================================================================
 *
 * Input parameters (theta):
 *   p = means3d = (p0, p1, p2)     -- 3 elements
 *   s = covars = (s0..s5)          -- 6 elements (upper-tri of Sigma)
 *   Total: 9 upstream parameters
 *
 * Output parameters (phi):
 *   m = means2d = (m0, m1)         -- 2 elements
 *   c = cov2d_3x3 = (c0..c5)      -- 6 elements (upper-tri of C2d)
 *   Total: 8 downstream parameters
 *
 * External parameters (psi):
 *   opacity                         -- 1 element
 *
 * =========================================================================
 * Core Second-Order Formulas
 * =========================================================================
 *
 * 1. First-order gradient (same as bwd1):
 *    v_theta = J_full^T @ v_phi
 *
 * 2. Hessian (theta-theta):
 *    H_{theta,theta} = J_full^T @ H_{phi,phi} @ J_full
 *                     + sum_k v_phi_k * Hess_k(theta)
 *    where J_full is the (8 x 9) Jacobian d(phi)/d(theta)
 *    and Hess_k is the (9 x 9) Hessian of phi_k w.r.t. theta.
 *
 * 3. Cross-Hessian (theta-psi):
 *    H_{theta,psi} = J_full^T @ H_{phi,psi}
 *
 * =========================================================================
 * Jacobian Structure Analysis
 * =========================================================================
 *
 * The full Jacobian J (8 x 9) where rows=(m0,m1,c0..c5), cols=(p0,p1,p2,s0..s5):
 *
 * Block structure:
 *   dm/dp (2x3):  F * R  where F = [[fx*rz, 0, -fx*x*rz^2], [0, fy*rz, -fy*y*rz^2]]
 *   dm/ds (2x6):  0      (means2d doesn't depend on covars)
 *   dc/dp (6x3):  complex (through J Jacobian, depends on p_c)
 *   dc/ds (6x6):  T matrix (linear map through P=JR)
 *
 * Key simplification for Hessians:
 *
 * For the covars (s) block:
 *   cov2d_3x3 = P * Sigma * P^T where P = J * R
 *   This is LINEAR in Sigma, so d^2(c)/d(s_i)d(s_j) = 0
 *   => The s-s block of Hess_k is ZERO for all k.
 *   => H_{s,s} = T^T @ H_{c,c} @ T  (pure Gauss-Newton, no second-order term!)
 *
 * For the means3d (p) block:
 *   Both means2d and cov2d_3x3 depend on p nonlinearly (through p_c, J, l).
 *   => H_{p,p} = J_p^T @ H_{phi,phi}_pp @ J_p + sum_k v_k * Hess_k(p)
 *   where J_p is the (8 x 3) Jacobian of phi w.r.t. p.
 *
 * For the cross block (p, s):
 *   means2d is independent of s, but cov2d_3x3 depends on both p (through J)
 *   and s (through Sigma). The cross-derivative d^2(c)/d(p)d(s) is nonzero.
 *   => H_{p,s} has both GN term and second-order term.
 *
 * =========================================================================
 * Exact Hessian Strategy (Term 1 + Term 2)
 * =========================================================================
 *
 * The full Hessian has two terms:
 *   H_{theta,theta} = J^T @ H_{phi,phi} @ J        (Term 1, Gauss-Newton)
 *                   + sum_k v_k * Hess_k(theta)    (Term 2, curvature)
 *
 * For each block:
 *   H_{s,s} = T^T @ H_{c,c} @ T     (exact, Term 2 = 0 since c is linear in s)
 *   H_{p,p} = J_p^T @ H @ J_p       (Term 1)
 *           + sum_k v_k * Hess_k(p)  (Term 2: from means2d and cov2d_3x3)
 *   H_{p,s} = J_p^T @ H @ J_s       (Term 1)
 *           + sum_k v_{c_k} * d^2(c_k)/dp/ds  (Term 2: from cov2d_3x3 only)
 *
 * Term 2 sources:
 *   - means2d: d^2(m)/dp^2 involves only 1/z nonlinearity (simple).
 *   - cov2d_3x3: d^2(c)/dp^2 involves d^2(J)/dp_c^2 (Jacobian second
 *     derivatives). These have contributions from 1/z^n terms (rows 0-1 of J)
 *     and 1/l^n terms (row 2 of J, the normalization row).
 *   - For H_{p,s}: d^2(c)/dp/ds = dT/dp, computed via dP/dp_c (=K_m*R).
 *
 * FOV clipping: tx, ty are piecewise linear → d^2(tx) = d^2(ty) = 0.
 * All second-order structure comes from the 1/z and 1/l nonlinearities.
 *
 * =========================================================================
 * T matrix: Linear map Sigma → cov2d_3x3
 * =========================================================================
 *
 * Let P = J * R (3x3). Then cov2d_3x3 = P * Sigma * P^T.
 * The upper-tri elements of cov2d_3x3 as a function of the upper-tri
 * elements of Sigma follow a 6x6 linear map T:
 *
 *   c_k = sum_l T_{kl} * s_l
 *
 * where (using row-major indices for the symmetric matrices):
 *   For diagonal output c_{(a,a)} and diagonal input s_{(i,i)}:
 *     T_{(a,a),(i,i)} = P_{ai}^2
 *   For diagonal output c_{(a,a)} and off-diagonal input s_{(i,j)}, i<j:
 *     T_{(a,a),(i,j)} = 2 * P_{ai} * P_{aj}
 *   For off-diagonal output c_{(a,b)}, a<b, and diagonal input s_{(i,i)}:
 *     T_{(a,b),(i,i)} = P_{ai} * P_{bi}
 *   For off-diagonal output c_{(a,b)}, a<b, and off-diagonal input s_{(i,j)}, i<j:
 *     T_{(a,b),(i,j)} = P_{ai} * P_{bj} + P_{aj} * P_{bi}
 *
 * Author: Dy164
 * Date: 2026-02-19
 */

#include "WorldToImageProjection.h"
#include "../include/common.h"
#include <cuda_runtime.h>

namespace gsplat2ct {

// ============================================================================
// Helper: Compute T matrix (6x6) for P * Sigma * P^T parameterization
// ============================================================================

/**
 * Compute the 6x6 linear map T such that:
 *   vec_upper_tri(P * Sigma * P^T) = T * vec_upper_tri(Sigma)
 *
 * P is 3x3 (row-major: P[row][col] = P_{row,col}).
 * T is 6x6 stored as flat array T[row*6 + col].
 *
 * Row/col indexing for upper-tri 3x3:
 *   0 = (0,0), 1 = (0,1), 2 = (0,2), 3 = (1,1), 4 = (1,2), 5 = (2,2)
 */
template <typename S>
__device__ __forceinline__ void compute_T_matrix(
    const S P[3][3],
    S T[36]
) {
    // Row-col pairs for upper-triangular indexing
    // idx 0: (0,0), 1: (0,1), 2: (0,2), 3: (1,1), 4: (1,2), 5: (2,2)
    const int row_idx[6] = {0, 0, 0, 1, 1, 2};
    const int col_idx[6] = {0, 1, 2, 1, 2, 2};

    #pragma unroll
    for (int out = 0; out < 6; ++out) {
        const int a = row_idx[out];
        const int b = col_idx[out];
        #pragma unroll
        for (int inp = 0; inp < 6; ++inp) {
            const int i = row_idx[inp];
            const int j = col_idx[inp];

            S val;
            if (a == b && i == j) {
                // diag output, diag input: P_{ai}^2
                val = P[a][i] * P[a][i];
            } else if (a == b && i != j) {
                // diag output, off-diag input: 2 * P_{ai} * P_{aj}
                val = static_cast<S>(2) * P[a][i] * P[a][j];
            } else if (a != b && i == j) {
                // off-diag output, diag input: P_{ai} * P_{bi}
                val = P[a][i] * P[b][i];
            } else {
                // off-diag output, off-diag input: P_{ai}*P_{bj} + P_{aj}*P_{bi}
                val = P[a][i] * P[b][j] + P[a][j] * P[b][i];
            }
            T[out * 6 + inp] = val;
        }
    }
}

// ============================================================================
// Second-Order Backward Kernel
// ============================================================================

template <typename S>
__global__ void world_to_image_projection_bwd2_kernel(
    // Forward inputs (for recomputation)
    const S* __restrict__ means,       // [N, 3]
    const S* __restrict__ covars,      // [N, 6]
    const S* __restrict__ viewmats,    // [C, 4, 4]
    const S* __restrict__ Ks,          // [C, 3, 3]
    uint32_t dim1,
    uint32_t dim0,
    // Downstream first-order gradients
    const S* __restrict__ v_means2d,   // [C*N, 2]
    const S* __restrict__ v_cov2d_3x3, // [C*N, 6]
    // Downstream phi-phi Hessians
    const S* __restrict__ h_means2d,            // [C*N, 3]  upper-tri 2x2
    const S* __restrict__ h_cov2d_3x3,          // [C*N, 21] upper-tri 6x6
    const S* __restrict__ h_means2d_cov2d_3x3,  // [C*N, 12] 2x6 row-major
    // Downstream phi-psi cross Hessians (psi = opacity, Dpsi=1)
    const S* __restrict__ h_means2d_opacity,    // [C*N, 2]
    const S* __restrict__ h_cov2d_3x3_opacity,  // [C*N, 6]
    // Outputs: first-order gradients (accumulated into [N,...] via atomicAdd)
    S* __restrict__ v_means_out,       // [N, 3]
    S* __restrict__ v_covars_out,      // [N, 6]
    // Outputs: theta-theta Hessians (accumulated into [N,...] via atomicAdd)
    S* __restrict__ h_means_out,       // [N, 6]  upper-tri 3x3
    S* __restrict__ h_covars_out,      // [N, 21] upper-tri 6x6
    S* __restrict__ h_means_covars_out,// [N, 18] 3x6 row-major
    // Outputs: theta-psi cross Hessians (per-camera, opacity is per-camera)
    S* __restrict__ h_means_opacity,   // [C*N, 3]
    S* __restrict__ h_covars_opacity,  // [C*N, 6]
    // Dimensions
    uint32_t C,
    uint32_t N
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = C * N;
    if (idx >= total) return;

    const uint32_t cam_id = idx / N;
    const uint32_t gs_id = idx % N;

    // Per-camera output pointers (for h_means_opacity, h_covars_opacity)
    S* h_po_ptr = h_means_opacity + idx * 3;
    S* h_so_ptr = h_covars_opacity + idx * 6;

    // ================================================================
    // Read camera parameters
    // ================================================================
    const S* vm = viewmats + cam_id * 16;
    const S R00 = vm[0], R01 = vm[1], R02 = vm[2];
    const S R10 = vm[4], R11 = vm[5], R12 = vm[6];
    const S R20 = vm[8], R21 = vm[9], R22 = vm[10];
    const S t0 = vm[3], t1 = vm[7], t2 = vm[11];

    const S* K = Ks + cam_id * 9;
    const S fx = K[0], fy = K[4];
    const S cx = K[2], cy = K[5];

    // Read world-space Gaussian (shared across cameras, indexed by gs_id)
    const S* p = means + gs_id * 3;
    const S px = p[0], py = p[1], pz = p[2];

    const S* cov_ptr = covars + gs_id * 6;
    const S s00 = cov_ptr[0], s01 = cov_ptr[1], s02 = cov_ptr[2];
    const S s11 = cov_ptr[3], s12 = cov_ptr[4], s22 = cov_ptr[5];

    // ================================================================
    // Recompute forward quantities
    // ================================================================
    const S x = R00 * px + R01 * py + R02 * pz + t0;
    const S y = R10 * px + R11 * py + R12 * pz + t1;
    const S z = R20 * px + R21 * py + R22 * pz + t2;

    // Handle behind-camera case
    if (z <= static_cast<S>(0.01)) {
        // Per-camera outputs still need to be zeroed
        for (int i = 0; i < 3; ++i) h_po_ptr[i] = static_cast<S>(0);
        for (int i = 0; i < 6; ++i) h_so_ptr[i] = static_cast<S>(0);
        // Shared outputs (v_means, v_covars, h_means, h_covars, h_means_covars)
        // are pre-zeroed and atomicAdd won't touch them.
        return;
    }

    const S rz = static_cast<S>(1) / z;
    const S rz2 = rz * rz;

    // FOV clipping
    const S W = static_cast<S>(dim1);
    const S H = static_cast<S>(dim0);
    const S tan_fovx = static_cast<S>(0.5) * W / fx;
    const S tan_fovy = static_cast<S>(0.5) * H / fy;
    const S lim_x_pos = (W - cx) / fx + static_cast<S>(0.3) * tan_fovx;
    const S lim_x_neg = cx / fx + static_cast<S>(0.3) * tan_fovx;
    const S lim_y_pos = (H - cy) / fy + static_cast<S>(0.3) * tan_fovy;
    const S lim_y_neg = cy / fy + static_cast<S>(0.3) * tan_fovy;

    const S xrz = x * rz;
    const S yrz = y * rz;
    const S tx = z * min(lim_x_pos, max(-lim_x_neg, xrz));
    const S ty = z * min(lim_y_pos, max(-lim_y_neg, yrz));

    // Build J
    const S l = sqrt(tx * tx + ty * ty + z * z);
    const S rl = static_cast<S>(1) / l;

    const S J00 = fx * rz;
    const S J02 = -fx * tx * rz2;
    const S J11 = fy * rz;
    const S J12 = -fy * ty * rz2;
    const S J20 = tx * rl;
    const S J21 = ty * rl;
    const S J22 = z * rl;

    // ================================================================
    // Compute P = J * R (3x3, row-major)
    // ================================================================
    // P[i][j] = sum_k J[i][k] * R[k][j]
    // J sparse: J01=J10=0
    S P[3][3];
    P[0][0] = J00*R00 +          J02*R20;
    P[0][1] = J00*R01 +          J02*R21;
    P[0][2] = J00*R02 +          J02*R22;
    P[1][0] =          J11*R10 + J12*R20;
    P[1][1] =          J11*R11 + J12*R21;
    P[1][2] =          J11*R12 + J12*R22;
    P[2][0] = J20*R00 + J21*R10 + J22*R20;
    P[2][1] = J20*R01 + J21*R11 + J22*R21;
    P[2][2] = J20*R02 + J21*R12 + J22*R22;

    // ================================================================
    // Compute T matrix (6x6): maps covars → cov2d_3x3
    // ================================================================
    S T[36];
    compute_T_matrix(P, T);

    // ================================================================
    // Compute F*R = dm/dp Jacobian (2x3)
    // ================================================================
    // F = [[fx*rz, 0, -fx*x*rz2], [0, fy*rz, -fy*y*rz2]]
    // FR[i][j] = sum_k F[i][k] * R[k][j]
    const S F00 = fx * rz, F02 = -fx * x * rz2;
    const S F11 = fy * rz, F12 = -fy * y * rz2;

    S FR[2][3];
    FR[0][0] = F00*R00 +          F02*R20;
    FR[0][1] = F00*R01 +          F02*R21;
    FR[0][2] = F00*R02 +          F02*R22;
    FR[1][0] =          F11*R10 + F12*R20;
    FR[1][1] =          F11*R11 + F12*R21;
    FR[1][2] =          F11*R12 + F12*R22;

    // ================================================================
    // Read downstream first-order gradients
    // ================================================================
    const S vm0 = v_means2d[idx * 2 + 0];
    const S vm1 = v_means2d[idx * 2 + 1];

    const S* vc_ptr = v_cov2d_3x3 + idx * 6;
    S vc[6];
    for (int i = 0; i < 6; ++i) vc[i] = vc_ptr[i];

    // ================================================================
    // Read downstream Hessians
    // ================================================================
    // H_{means2d, means2d}: upper-tri 2x2 = [H00, H01, H11]
    const S Hmm00 = h_means2d[idx * 3 + 0];
    const S Hmm01 = h_means2d[idx * 3 + 1];
    const S Hmm11 = h_means2d[idx * 3 + 2];

    // H_{cov2d_3x3, cov2d_3x3}: upper-tri 6x6 = 21 elements
    const S* Hcc = h_cov2d_3x3 + idx * 21;

    // H_{means2d, cov2d_3x3}: 2x6 row-major = 12 elements
    const S* Hmc = h_means2d_cov2d_3x3 + idx * 12;

    // H_{means2d, opacity}: 2 elements
    const S Hmo0 = h_means2d_opacity[idx * 2 + 0];
    const S Hmo1 = h_means2d_opacity[idx * 2 + 1];

    // H_{cov2d_3x3, opacity}: 6 elements
    const S* Hco = h_cov2d_3x3_opacity + idx * 6;

    // ================================================================
    // STEP 1: First-order gradients (same as bwd1)
    // ================================================================

    // v_covars = T^T @ v_cov2d_3x3
    // v_s[j] = sum_i T[i][j] * vc[i]
    S vs[6];
    #pragma unroll
    for (int j = 0; j < 6; ++j) {
        S sum = static_cast<S>(0);
        #pragma unroll
        for (int i = 0; i < 6; ++i) {
            sum += T[i * 6 + j] * vc[i];
        }
        vs[j] = sum;
    }
    for (int i = 0; i < 6; ++i) atomicAdd(v_covars_out + gs_id * 6 + i, vs[i]);

    // v_means3d: contributions from both means2d and cov2d_3x3
    // From means2d: v_p[j] = sum_i FR[i][j] * vm_i
    S vp[3];
    for (int j = 0; j < 3; ++j) {
        vp[j] = FR[0][j] * vm0 + FR[1][j] * vm1;
    }

    // From cov2d_3x3 through J → p_c → p:
    // This is more complex (same as bwd1). For the GN approximation in bwd2,
    // we need the Jacobian dc/dp. We'll compute it as dc/dp_c * R.
    //
    // dc/dp_c is a (6 x 3) matrix. Under the GN approximation (dropping
    // second-order terms of J w.r.t. p_c), we compute the Jacobian numerically
    // equivalent to the bwd1 v_J → v_pc propagation, but in Jacobian form.
    //
    // Actually, for the bwd2, we need the full Jacobian J_full (8x9) to
    // compute J_full^T @ H @ J_full. Let's compute the blocks:
    //   J_full = [[FR,    0   ],    (2x3, 2x6)
    //             [Jcp,   T   ]]   (6x3, 6x6)
    // where Jcp = dc/dp (6x3).
    //
    // For the GN approximation:
    //   H_{theta,theta} = J_full^T @ H_{phi,phi} @ J_full
    //
    // This decomposes as:
    //   H_{p,p} = FR^T @ Hmm @ FR + FR^T @ Hmc @ Jcp + Jcp^T @ Hmc^T @ FR + Jcp^T @ Hcc @ Jcp
    //   H_{s,s} = T^T @ Hcc @ T
    //   H_{p,s} = FR^T @ Hmc @ T + Jcp^T @ Hcc @ T
    //
    // We need Jcp = dc/dp (6x3).
    // dc/dp = dc/dp_c * R where dc/dp_c = d(vec_upper_tri(J * Sigma_c * J^T)) / dp_c

    // dc/dp_c is obtained from the derivative of cov2d_3x3 w.r.t. p_c.
    // cov2d_3x3 = J(p_c) * Sigma_c * J(p_c)^T
    // d(cov2d_3x3)/d(p_c_m) = dJ/d(p_c_m) * Sigma_c * J^T + J * Sigma_c * dJ^T/d(p_c_m)
    //
    // Under GN approximation, we ignore the second derivatives of J w.r.t. p_c
    // in the Hessian, but we still need the first derivative (Jacobian).
    //
    // dJ/dp_c is a (3x3x3) tensor. Let's compute it.

    // Camera covariance (recompute)
    const S M00 = s00*R00 + s01*R01 + s02*R02;
    const S M01 = s00*R10 + s01*R11 + s02*R12;
    const S M02 = s00*R20 + s01*R21 + s02*R22;
    const S M10 = s01*R00 + s11*R01 + s12*R02;
    const S M11 = s01*R10 + s11*R11 + s12*R12;
    const S M12 = s01*R20 + s11*R21 + s12*R22;
    const S M20 = s02*R00 + s12*R01 + s22*R02;
    const S M21 = s02*R10 + s12*R11 + s22*R12;
    const S M22 = s02*R20 + s12*R21 + s22*R22;

    const S Sc00 = R00*M00 + R01*M10 + R02*M20;
    const S Sc01 = R00*M01 + R01*M11 + R02*M21;
    const S Sc02 = R00*M02 + R01*M12 + R02*M22;
    const S Sc10 = Sc01;  // symmetric
    const S Sc11 = R10*M01 + R11*M11 + R12*M21;
    const S Sc12 = R10*M02 + R11*M12 + R12*M22;
    const S Sc20 = Sc02;
    const S Sc21 = Sc12;
    const S Sc22 = R20*M02 + R21*M12 + R22*M22;

    // We compute Jcp (6x3) = dc/dp_c * R = d(upper_tri(J Sc J^T))/dp_c * R
    // This requires dJ/dp_c.
    //
    // For the GN approximation, let's compute the v_p contribution from
    // cov2d_3x3 using the same approach as bwd1 (which gives the first-order
    // gradient correctly), and also build the Jacobian Jcp for the Hessian.

    // Build v_C2d: the gradient from the 6-element upper-tri parameterization
    // flows as G_upper (lower triangle = 0). For the v_J formula we need
    // (G + G^T), not 2*G_sym. See detailed derivation in Bwd.cu.
    const S GGt00 = static_cast<S>(2) * vc[0];
    const S GGt01 = vc[1];
    const S GGt02 = vc[2];
    const S GGt10 = vc[1];
    const S GGt11 = static_cast<S>(2) * vc[3];
    const S GGt12 = vc[4];
    const S GGt20 = vc[2];
    const S GGt21 = vc[4];
    const S GGt22 = static_cast<S>(2) * vc[5];

    // v_J = (G + G^T) * J * Sigma_c (corrected from bwd1)
    // U = J * Sigma_c
    const S U00 = J00*Sc00 +          J02*Sc20;
    const S U01 = J00*Sc01 +          J02*Sc21;
    const S U02 = J00*Sc02 +          J02*Sc22;
    const S U10 =          J11*Sc10 + J12*Sc20;
    const S U11 =          J11*Sc11 + J12*Sc21;
    const S U12 =          J11*Sc12 + J12*Sc22;
    const S U20 = J20*Sc00 + J21*Sc10 + J22*Sc20;
    const S U21 = J20*Sc01 + J21*Sc11 + J22*Sc21;
    const S U22 = J20*Sc02 + J21*Sc12 + J22*Sc22;

    // v_J = (G + G^T) * U
    const S vJ00 = GGt00*U00 + GGt01*U10 + GGt02*U20;
    const S vJ01 = GGt00*U01 + GGt01*U11 + GGt02*U21;
    const S vJ02 = GGt00*U02 + GGt01*U12 + GGt02*U22;
    const S vJ10 = GGt10*U00 + GGt11*U10 + GGt12*U20;
    const S vJ11_val = GGt10*U01 + GGt11*U11 + GGt12*U21;
    const S vJ12 = GGt10*U02 + GGt11*U12 + GGt12*U22;
    const S vJ20 = GGt20*U00 + GGt21*U10 + GGt22*U20;
    const S vJ21 = GGt20*U01 + GGt21*U11 + GGt22*U21;
    const S vJ22 = GGt20*U02 + GGt21*U12 + GGt22*U22;

    // Propagate v_J → v_pc (same as bwd1)
    const S rz3 = rz2 * rz;
    const S rl2 = rl * rl;
    const bool x_clipped = (xrz > lim_x_pos) || (xrz < -lim_x_neg);
    const bool y_clipped = (yrz > lim_y_pos) || (yrz < -lim_y_neg);

    S v_pc_x = fx * rz * vm0;  // from means2d
    S v_pc_y = fy * rz * vm1;
    S v_pc_z = -(fx * x * rz2 * vm0 + fy * y * rz2 * vm1);

    // From v_J contributions (same logic as bwd1)
    v_pc_z += -fx * rz2 * vJ00;
    v_pc_z += -fy * rz2 * vJ11_val;
    v_pc_z += rl * vJ22;

    if (!x_clipped) {
        v_pc_x += -fx * rz2 * vJ02;
        v_pc_x += rl * vJ20;
    }
    v_pc_z += static_cast<S>(2) * fx * tx * rz3 * vJ02;

    if (!y_clipped) {
        v_pc_y += -fy * rz2 * vJ12;
        v_pc_y += rl * vJ21;
    }
    v_pc_z += static_cast<S>(2) * fy * ty * rz3 * vJ12;

    S v_l = -tx * rl2 * vJ20 - ty * rl2 * vJ21 - z * rl2 * vJ22;

    if (!x_clipped) {
        v_pc_x += v_l * tx * rl;
    }
    if (!y_clipped) {
        v_pc_y += v_l * ty * rl;
    }
    v_pc_z += v_l * z * rl;

    if (x_clipped) {
        v_pc_z += v_l * tx * rl * (tx / z);
        v_pc_z += (tx / z) * rl * vJ20;
        v_pc_z += -fx * (tx / z) * rz2 * vJ02;
    }
    if (y_clipped) {
        v_pc_z += v_l * ty * rl * (ty / z);
        v_pc_z += (ty / z) * rl * vJ21;
        v_pc_z += -fy * (ty / z) * rz2 * vJ12;
    }

    // v_p = R^T * v_pc
    vp[0] += R00 * v_pc_x + R10 * v_pc_y + R20 * v_pc_z;
    vp[1] += R01 * v_pc_x + R11 * v_pc_y + R21 * v_pc_z;
    vp[2] += R02 * v_pc_x + R12 * v_pc_y + R22 * v_pc_z;

    // Wait: v_p from means2d is already FR * vm, but we need to add the
    // contribution from cov2d_3x3 through J. The means2d contribution
    // was already added above (vp[j] = FR * vm). The cov contribution
    // through v_pc should NOT include the means2d part again.
    // Let me fix: separate the two contributions.

    // Actually, let me reconsider. vp was initialized with FR * vm.
    // Then we computed v_pc including BOTH means2d and cov contributions.
    // The v_p = R^T * v_pc line adds the full v_pc.
    // But FR * vm was already the means2d part of R^T * v_pc_from_means2d.
    // So there's double-counting!

    // Let me fix: v_pc_cov = v_pc - v_pc_from_means2d
    // v_pc_from_means2d = (fx*rz*vm0, fy*rz*vm1, -(fx*x*rz2*vm0+fy*y*rz2*vm1))
    // and the rest is v_pc_cov.
    // Then vp = FR * vm + R^T * v_pc_cov
    // But FR * vm = R^T * F^T * vm = R^T * v_pc_from_means2d. So:
    // vp = R^T * v_pc_from_means2d + R^T * v_pc_cov = R^T * v_pc_total.

    // So actually, we should NOT initialize vp with FR*vm and then add R^T*v_pc.
    // We should just do vp = R^T * v_pc_total.

    // Let me redo:
    vp[0] = R00 * v_pc_x + R10 * v_pc_y + R20 * v_pc_z;
    vp[1] = R01 * v_pc_x + R11 * v_pc_y + R21 * v_pc_z;
    vp[2] = R02 * v_pc_x + R12 * v_pc_y + R22 * v_pc_z;

    for (int i = 0; i < 3; ++i) atomicAdd(v_means_out + gs_id * 3 + i, vp[i]);

    // ================================================================
    // STEP 2: Build Jacobian J_p (8 x 3) = d(phi)/d(p)
    //         where phi = (m0, m1, c0..c5)
    // ================================================================
    // We need this for the Gauss-Newton Hessian computation.
    //
    // Row 0-1: dm/dp = FR (2x3) - already computed
    // Row 2-7: dc/dp (6x3) - need to compute
    //
    // dc/dp = dc/dp_c * R
    //
    // dc/dp_c: For each cov2d_3x3 element c_k (corresponding to upper-tri
    // index (a,b) of C2d), we need d(c_k)/d(p_c_m) for m=0,1,2.
    //
    // C2d = J * Sigma_c * J^T
    // dC2d/dp_cm = (dJ/dp_cm) * Sigma_c * J^T + J * Sigma_c * (dJ/dp_cm)^T
    //
    // Under GN, we only need this Jacobian (not its derivative).
    // But computing dJ/dp_cm for each of 3 components and then forming
    // the 6 elements of the resulting symmetric matrix is complex.
    //
    // For GN approximation, we can compute dc/dp_c column by column.
    // For p_c component m:
    //   dC2d/dp_cm = Km * Sigma_c * J^T + J * Sigma_c * Km^T
    //   where Km = dJ/dp_cm is a 3x3 matrix.
    //
    // Let's compute K0 = dJ/dx, K1 = dJ/dy, K2 = dJ/dz:
    //
    // J = [[fx*rz,   0,       -fx*tx*rz2],
    //      [0,       fy*rz,   -fy*ty*rz2],
    //      [tx/l,    ty/l,     z/l       ]]
    //
    // dJ/dx (when not x-clipped):
    //   dJ00/dx = 0
    //   dJ02/dx = -fx * rz2 * (dtx/dx) = -fx * rz2  (dtx/dx = 1 when not clipped)
    //   dJ20/dx = (dtx/dx)/l - tx*(tx/l)*(dtx/dx)/l^2 = (1/l)(1 - tx^2/l^2) = (l^2-tx^2)/l^3
    //   dJ21/dx = -ty*tx/(l^3) (via dl/dtx * dtx/dx)
    //   dJ22/dx = -z*tx/(l^3)
    //   All others = 0
    //
    // When x-clipped:
    //   dtx/dx = 0, all dJ/dx = 0

    // For the GN approximation, we'll precompute the full 8x3 Jacobian Jp
    // (in camera space, then multiply by R to get world space).

    // First, compute dJ/dp_c for each component.
    // We'll store Jp_cam[8][3] = d(phi)/d(p_c) and then Jp = Jp_cam * R

    // But it's simpler to directly compute Jp = d(phi)/dp = d(phi)/dp_c * R.
    // We already know FR (the means2d part).

    // For cov2d_3x3 part: we need dc/dp_c then multiply by R.
    // dc/dp_c is 6x3.

    // Let's compute dC2d/dp_cm for m = 0, 1, 2.
    // dC2d/dp_cm = Km * Sigma_c * J^T + (Km * Sigma_c * J^T)^T
    //            = A_m + A_m^T  where A_m = Km * Sigma_c * J^T

    // Rather than computing all this explicitly (which is very long),
    // let's use a simpler approach for the GN Hessian.
    //
    // For the GN Hessian, we need:
    //   H_{p,p} ≈ Jp^T @ H_{phi,phi} @ Jp
    //   H_{p,s} ≈ Jp^T @ H_{phi,phi} @ [[0], [T]]  (the part of J_full for s)
    //
    // But computing Jp (8x3) explicitly would require 24 dJ/dp_c computations.
    // Instead, let's use a more efficient approach:
    //
    // Note that H_{phi,phi} is 8x8 block matrix:
    //   H = [[Hmm, Hmc], [Hmc^T, Hcc]]
    //
    // And J_full = [[FR, 0], [Jcp, T]]
    //
    // Then:
    //   J_full^T H J_full =
    //     [[FR^T Hmm FR + FR^T Hmc Jcp + Jcp^T Hmc^T FR + Jcp^T Hcc Jcp,
    //       FR^T Hmc T + Jcp^T Hcc T],
    //      [T^T Hmc^T FR + T^T Hcc Jcp,
    //       T^T Hcc T]]
    //
    // For H_{s,s} = T^T Hcc T, this is straightforward.
    // For H_{p,p} and H_{p,s}, we need Jcp.
    //
    // Given the complexity, let's compute Jcp numerically-efficiently.
    // We compute Jcp^T * v_c for arbitrary v_c (as in bwd1), and
    // also Jcp^T * Hcc and Jcp^T * Hmc^T.

    // ================================================================
    // STEP 2a: Compute H_{s,s} = T^T @ Hcc @ T (exact, no Hess term)
    // ================================================================

    // First compute W = Hcc @ T (21-elem symmetric 6x6 times dense 6x6 → dense 6x6)
    // Hcc is stored upper-tri 6x6 with 21 elements:
    //   idx = i*(11-i)/2 + j for i <= j
    // We need to expand to full symmetric access.

    // Helper to access Hcc[i][j] from 21-element upper-tri storage
    // Index formula: for i <= j: flat = i*(11-i)/2 + j
    // For i > j: swap

    S Wmat[6][6];
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        #pragma unroll
        for (int j = 0; j < 6; ++j) {
            S sum = static_cast<S>(0);
            #pragma unroll
            for (int k = 0; k < 6; ++k) {
                // Access Hcc[i][k] from upper-tri storage
                int ii = i, kk = k;
                if (ii > kk) { int tmp = ii; ii = kk; kk = tmp; }
                int flat_idx = ii * (11 - ii) / 2 + kk;
                sum += Hcc[flat_idx] * T[k * 6 + j];
            }
            Wmat[i][j] = sum;
        }
    }

    // H_{s,s} = T^T @ W  (6x6 times 6x6 → 6x6, store upper-tri 21 elements)
    {
        int flat = 0;
        #pragma unroll
        for (int i = 0; i < 6; ++i) {
            #pragma unroll
            for (int j = i; j < 6; ++j) {
                S sum = static_cast<S>(0);
                #pragma unroll
                for (int k = 0; k < 6; ++k) {
                    sum += T[k * 6 + i] * Wmat[k][j];
                }
                atomicAdd(h_covars_out + gs_id * 21 + flat, sum);
                ++flat;
            }
        }
    }

    // ================================================================
    // STEP 2b: Compute H_{p,p} (Term 1: GN + Term 2: curvature)
    // ================================================================
    // For the Gauss-Newton approximation, we use the Jacobian Jp_c (8x3)
    // in camera coordinates, then rotate: Jp = Jp_c (with R applied).
    //
    // But computing the full Jp_c is expensive. Instead, let's use
    // the fact that:
    //   H_{p,p} = R^T @ H_{pc,pc} @ R
    // where H_{pc,pc} is the Hessian in camera coordinates.
    //
    // H_{pc,pc} = Jp_c^T @ H_{phi,phi} @ Jp_c
    //
    // We have Jp_c = [[F], [dc/dp_c]]  (8x3)
    // where F = [[fx*rz, 0, -fx*x*rz2], [0, fy*rz, -fy*y*rz2]]
    //
    // Let's compute Jp_c^T @ H @ Jp_c step by step.
    // First, compute the column vectors of Jp_c.
    // Then compute H @ Jp_c = (8x8) @ (8x3) → (8x3).
    // Then Jp_c^T @ (result) → (3x3).
    //
    // Actually, let's take a different approach to avoid the dJ/dp_c computation.
    // We'll compute H_{pc,pc} via a finite-difference-like approach using
    // the already-computed first-order backward.
    //
    // No, let's just compute the GN Hessian properly. We need Jp_c (8x3):
    //   Rows 0-1: F (2x3)
    //   Rows 2-7: dc/dp_c (6x3)
    //
    // dc/dp_c requires dJ/dp_c, which is a 3x3x3 tensor.
    // Let's compute it explicitly.

    // dJ/dp_c: J is 3x3, p_c = (x, y, z)
    // We need dJ_{ij}/dp_c_m for all i,j,m.
    //
    // Key derivatives (using tx, ty, l, rz):
    // dtx/dx = 1 (not clipped), 0 (clipped)
    // dtx/dy = 0
    // dtx/dz = 0 (not clipped), tx/z (clipped, since tx = z*lim)
    // Similarly for ty.
    // dl/dtx = tx/l, dl/dty = ty/l, dl/dz = z/l
    // drz/dz = -rz^2

    // Precompute some values
    const S l3 = l * l * l;

    // Effective derivatives of tx, ty:
    const S dtx_dx = x_clipped ? static_cast<S>(0) : static_cast<S>(1);
    const S dtx_dz = x_clipped ? (tx / z) : static_cast<S>(0);
    const S dty_dy = y_clipped ? static_cast<S>(0) : static_cast<S>(1);
    const S dty_dz = y_clipped ? (ty / z) : static_cast<S>(0);

    // dl/dx = (tx/l)*dtx_dx, dl/dy = (ty/l)*dty_dy
    // dl/dz = (tx/l)*dtx_dz + (ty/l)*dty_dz + z/l
    const S dl_dx = (tx * rl) * dtx_dx;
    const S dl_dy = (ty * rl) * dty_dy;
    const S dl_dz = (tx * rl) * dtx_dz + (ty * rl) * dty_dz + z * rl;

    // Now compute dJ/dp_c as K[m][i][j] = dJ[i][j]/dp_c[m]

    // dJ/dx (m=0):
    // dJ[0][0]/dx = 0
    // dJ[0][2]/dx = -fx * rz2 * dtx_dx
    // dJ[1][1]/dx = 0
    // dJ[1][2]/dx = 0
    // dJ[2][0]/dx = dtx_dx/l - tx*dl_dx/l^2 = (dtx_dx*l - tx*dl_dx)/l^2
    // dJ[2][1]/dx = -ty*dl_dx/l^2
    // dJ[2][2]/dx = -z*dl_dx/l^2
    const S dJ02_dx = -fx * rz2 * dtx_dx;
    const S dJ20_dx = (dtx_dx - tx * dl_dx * rl) * rl;
    const S dJ21_dx = -ty * dl_dx * rl2;
    const S dJ22_dx = -z * dl_dx * rl2;

    // dJ/dy (m=1):
    const S dJ12_dy = -fy * rz2 * dty_dy;
    const S dJ20_dy = -tx * dl_dy * rl2;
    const S dJ21_dy = (dty_dy - ty * dl_dy * rl) * rl;
    const S dJ22_dy = -z * dl_dy * rl2;

    // dJ/dz (m=2):
    // dJ[0][0]/dz = -fx * rz^2
    // dJ[0][2]/dz = -fx * (dtx_dz * rz^2 + tx * (-2*rz^3)) = -fx*(dtx_dz*rz^2 - 2*tx*rz^3)
    // dJ[1][1]/dz = -fy * rz^2
    // dJ[1][2]/dz = -fy * (dty_dz * rz^2 - 2*ty*rz^3)
    // dJ[2][0]/dz = (dtx_dz - tx*dl_dz*rl) * rl
    // dJ[2][1]/dz = (dty_dz - ty*dl_dz*rl) * rl
    // dJ[2][2]/dz = (1 - z*dl_dz*rl) * rl
    const S dJ00_dz = -fx * rz2;
    const S dJ02_dz = -fx * (dtx_dz * rz2 - static_cast<S>(2) * tx * rz3);
    const S dJ11_dz = -fy * rz2;
    const S dJ12_dz = -fy * (dty_dz * rz2 - static_cast<S>(2) * ty * rz3);
    const S dJ20_dz = (dtx_dz - tx * dl_dz * rl) * rl;
    const S dJ21_dz = (dty_dz - ty * dl_dz * rl) * rl;
    const S dJ22_dz = (static_cast<S>(1) - z * dl_dz * rl) * rl;

    // Store dJ/dp_c matrices in array form: dJdp[m][i][j] = dJ[i][j]/dp_c[m]
    // dJdp[0] = dJ/dx, dJdp[1] = dJ/dy, dJdp[2] = dJ/dz
    S dJdp[3][3][3];
    // dJ/dx
    dJdp[0][0][0] = static_cast<S>(0);  dJdp[0][0][1] = static_cast<S>(0);  dJdp[0][0][2] = dJ02_dx;
    dJdp[0][1][0] = static_cast<S>(0);  dJdp[0][1][1] = static_cast<S>(0);  dJdp[0][1][2] = static_cast<S>(0);
    dJdp[0][2][0] = dJ20_dx;            dJdp[0][2][1] = dJ21_dx;            dJdp[0][2][2] = dJ22_dx;
    // dJ/dy
    dJdp[1][0][0] = static_cast<S>(0);  dJdp[1][0][1] = static_cast<S>(0);  dJdp[1][0][2] = static_cast<S>(0);
    dJdp[1][1][0] = static_cast<S>(0);  dJdp[1][1][1] = static_cast<S>(0);  dJdp[1][1][2] = dJ12_dy;
    dJdp[1][2][0] = dJ20_dy;            dJdp[1][2][1] = dJ21_dy;            dJdp[1][2][2] = dJ22_dy;
    // dJ/dz
    dJdp[2][0][0] = dJ00_dz;            dJdp[2][0][1] = static_cast<S>(0);  dJdp[2][0][2] = dJ02_dz;
    dJdp[2][1][0] = static_cast<S>(0);  dJdp[2][1][1] = dJ11_dz;            dJdp[2][1][2] = dJ12_dz;
    dJdp[2][2][0] = dJ20_dz;            dJdp[2][2][1] = dJ21_dz;            dJdp[2][2][2] = dJ22_dz;

    // Now compute dc/dp_c (6x3):
    // d(C2d)/dp_cm = Km * Sigma_c * J^T + J * Sigma_c * Km^T
    //              = Am + Am^T  where Am = Km * Sc * J^T
    //
    // For the upper-tri elements of this symmetric matrix:
    //   dc_k/dp_cm = (Am + Am^T)[a][b]  where k indexes (a,b) with a<=b
    //   For a==b: = 2*Am[a][a]
    //   For a<b:  = Am[a][b] + Am[b][a]

    // K0 = dJ/dx:
    // K0 is mostly zeros except:
    //   K0[0][2] = dJ02_dx, K0[2][0] = dJ20_dx, K0[2][1] = dJ21_dx, K0[2][2] = dJ22_dx
    // A0 = K0 * Sc * J^T (3x3)
    // K0 * Sc:
    //   row 0: K0[0][2] * Sc[2][j] = dJ02_dx * (Sc20, Sc21, Sc22)
    //   row 1: all zeros
    //   row 2: K0[2][0]*Sc[0][j] + K0[2][1]*Sc[1][j] + K0[2][2]*Sc[2][j]
    S K0Sc[3][3];
    K0Sc[0][0] = dJ02_dx * Sc20;
    K0Sc[0][1] = dJ02_dx * Sc21;
    K0Sc[0][2] = dJ02_dx * Sc22;
    K0Sc[1][0] = static_cast<S>(0);
    K0Sc[1][1] = static_cast<S>(0);
    K0Sc[1][2] = static_cast<S>(0);
    K0Sc[2][0] = dJ20_dx*Sc00 + dJ21_dx*Sc10 + dJ22_dx*Sc20;
    K0Sc[2][1] = dJ20_dx*Sc01 + dJ21_dx*Sc11 + dJ22_dx*Sc21;
    K0Sc[2][2] = dJ20_dx*Sc02 + dJ21_dx*Sc12 + dJ22_dx*Sc22;

    // A0 = K0Sc * J^T (row i of A0 = K0Sc[i] dotted with J rows)
    S A0[3][3];
    // A0[i][j] = K0Sc[i][0]*J[j][0] + K0Sc[i][1]*J[j][1] + K0Sc[i][2]*J[j][2]
    //          = K0Sc[i][0]*J_row_j[0] + ...
    for (int i = 0; i < 3; ++i) {
        A0[i][0] = K0Sc[i][0]*J00 +                K0Sc[i][2]*J02;
        A0[i][1] =                K0Sc[i][1]*J11 + K0Sc[i][2]*J12;
        A0[i][2] = K0Sc[i][0]*J20 + K0Sc[i][1]*J21 + K0Sc[i][2]*J22;
    }

    // K1 = dJ/dy:
    // K1[1][2] = dJ12_dy, K1[2][0] = dJ20_dy, K1[2][1] = dJ21_dy, K1[2][2] = dJ22_dy
    S K1Sc[3][3];
    K1Sc[0][0] = static_cast<S>(0);
    K1Sc[0][1] = static_cast<S>(0);
    K1Sc[0][2] = static_cast<S>(0);
    K1Sc[1][0] = dJ12_dy * Sc20;
    K1Sc[1][1] = dJ12_dy * Sc21;
    K1Sc[1][2] = dJ12_dy * Sc22;
    K1Sc[2][0] = dJ20_dy*Sc00 + dJ21_dy*Sc10 + dJ22_dy*Sc20;
    K1Sc[2][1] = dJ20_dy*Sc01 + dJ21_dy*Sc11 + dJ22_dy*Sc21;
    K1Sc[2][2] = dJ20_dy*Sc02 + dJ21_dy*Sc12 + dJ22_dy*Sc22;

    S A1[3][3];
    for (int i = 0; i < 3; ++i) {
        A1[i][0] = K1Sc[i][0]*J00 +                K1Sc[i][2]*J02;
        A1[i][1] =                K1Sc[i][1]*J11 + K1Sc[i][2]*J12;
        A1[i][2] = K1Sc[i][0]*J20 + K1Sc[i][1]*J21 + K1Sc[i][2]*J22;
    }

    // K2 = dJ/dz:
    // K2[0][0] = dJ00_dz, K2[0][2] = dJ02_dz
    // K2[1][1] = dJ11_dz, K2[1][2] = dJ12_dz
    // K2[2][0] = dJ20_dz, K2[2][1] = dJ21_dz, K2[2][2] = dJ22_dz
    S K2Sc[3][3];
    K2Sc[0][0] = dJ00_dz*Sc00 +               dJ02_dz*Sc20;
    K2Sc[0][1] = dJ00_dz*Sc01 +               dJ02_dz*Sc21;
    K2Sc[0][2] = dJ00_dz*Sc02 +               dJ02_dz*Sc22;
    K2Sc[1][0] =               dJ11_dz*Sc10 + dJ12_dz*Sc20;
    K2Sc[1][1] =               dJ11_dz*Sc11 + dJ12_dz*Sc21;
    K2Sc[1][2] =               dJ11_dz*Sc12 + dJ12_dz*Sc22;
    K2Sc[2][0] = dJ20_dz*Sc00 + dJ21_dz*Sc10 + dJ22_dz*Sc20;
    K2Sc[2][1] = dJ20_dz*Sc01 + dJ21_dz*Sc11 + dJ22_dz*Sc21;
    K2Sc[2][2] = dJ20_dz*Sc02 + dJ21_dz*Sc12 + dJ22_dz*Sc22;

    // Store KSc in indexed array form: KSc[m] = K_m * Sc
    // (alias pointers to avoid copying)
    S* const KSc_arr[3] = { &K0Sc[0][0], &K1Sc[0][0], &K2Sc[0][0] };
    // Helper to access KSc[m][i][j]:
    #define KSc_val(m, i, j) (KSc_arr[m][(i)*3 + (j)])

    S A2[3][3];
    for (int i = 0; i < 3; ++i) {
        A2[i][0] = K2Sc[i][0]*J00 +                K2Sc[i][2]*J02;
        A2[i][1] =                K2Sc[i][1]*J11 + K2Sc[i][2]*J12;
        A2[i][2] = K2Sc[i][0]*J20 + K2Sc[i][1]*J21 + K2Sc[i][2]*J22;
    }

    // Extract dc/dp_c (6x3) from A0, A1, A2:
    // dc_k/dp_cm = (Am + Am^T)[a][b]
    // Upper-tri indexing: k: 0=(0,0),1=(0,1),2=(0,2),3=(1,1),4=(1,2),5=(2,2)
    S dc_dpc[6][3];

    // m=0 (x):
    dc_dpc[0][0] = static_cast<S>(2) * A0[0][0];              // (0,0), diag
    dc_dpc[1][0] = A0[0][1] + A0[1][0];                        // (0,1), off-diag
    dc_dpc[2][0] = A0[0][2] + A0[2][0];                        // (0,2), off-diag
    dc_dpc[3][0] = static_cast<S>(2) * A0[1][1];              // (1,1), diag
    dc_dpc[4][0] = A0[1][2] + A0[2][1];                        // (1,2), off-diag
    dc_dpc[5][0] = static_cast<S>(2) * A0[2][2];              // (2,2), diag

    // m=1 (y):
    dc_dpc[0][1] = static_cast<S>(2) * A1[0][0];
    dc_dpc[1][1] = A1[0][1] + A1[1][0];
    dc_dpc[2][1] = A1[0][2] + A1[2][0];
    dc_dpc[3][1] = static_cast<S>(2) * A1[1][1];
    dc_dpc[4][1] = A1[1][2] + A1[2][1];
    dc_dpc[5][1] = static_cast<S>(2) * A1[2][2];

    // m=2 (z):
    dc_dpc[0][2] = static_cast<S>(2) * A2[0][0];
    dc_dpc[1][2] = A2[0][1] + A2[1][0];
    dc_dpc[2][2] = A2[0][2] + A2[2][0];
    dc_dpc[3][2] = static_cast<S>(2) * A2[1][1];
    dc_dpc[4][2] = A2[1][2] + A2[2][1];
    dc_dpc[5][2] = static_cast<S>(2) * A2[2][2];

    // Jcp_cam = dc/dp_c (6x3, in camera coordinates)
    // Jcp = dc/dp = dc/dp_c * R (6x3)
    // d(c)/d(p_j) = sum_m d(c)/d(pc_m) * R[m][j]
    S Jcp[6][3];
    for (int k = 0; k < 6; ++k) {
        Jcp[k][0] = dc_dpc[k][0] * R00 + dc_dpc[k][1] * R10 + dc_dpc[k][2] * R20;
        Jcp[k][1] = dc_dpc[k][0] * R01 + dc_dpc[k][1] * R11 + dc_dpc[k][2] * R21;
        Jcp[k][2] = dc_dpc[k][0] * R02 + dc_dpc[k][1] * R12 + dc_dpc[k][2] * R22;
    }

    // ================================================================
    // Now we have the full 8x3 Jacobian Jp for position:
    //   Jp[0..1][0..2] = FR[0..1][0..2]  (means2d part)
    //   Jp[2..7][0..2] = Jcp[0..5][0..2] (cov part)
    //
    // Full 8x6 Jacobian Js for covars:
    //   Js[0..1][0..5] = 0  (means2d doesn't depend on covars)
    //   Js[2..7][0..5] = T[0..5][0..5]
    //
    // H_{p,p} = Jp^T @ H_{phi,phi} @ Jp  (GN)
    // ================================================================

    // H_{phi,phi} is 8x8:
    //   [[Hmm(2x2), Hmc(2x6)],
    //    [Hmc^T(6x2), Hcc(6x6)]]
    //
    // H @ Jp is 8x3. Let's compute it block-wise:
    //   (H @ Jp)[i] = Hmm[i]*FR[j] + Hmc[i]*Jcp[k]   for i=0..1 (means2d rows)
    //   (H @ Jp)[i] = Hmc^T[i]*FR[j] + Hcc[i]*Jcp[k]   for i=2..7 (cov rows)

    // Let's define HJp[8][3] = H_{phi,phi} @ Jp:

    // Means2d rows (0-1) of HJp:
    // HJp[0][j] = Hmm[0][0]*FR[0][j] + Hmm[0][1]*FR[1][j]
    //           + sum_k Hmc[0][k]*Jcp[k][j]
    // HJp[1][j] = Hmm[1][0]*FR[0][j] + Hmm[1][1]*FR[1][j]
    //           + sum_k Hmc[1][k]*Jcp[k][j]
    // Note: Hmm is 2x2 symmetric: Hmm[0][0]=Hmm00, Hmm[0][1]=Hmm[1][0]=Hmm01, Hmm[1][1]=Hmm11
    // Hmc is 2x6: Hmc[i][k] stored as h_means2d_cov2d_3x3[idx*12 + i*6 + k]

    S HJp[8][3];
    for (int j = 0; j < 3; ++j) {
        HJp[0][j] = Hmm00 * FR[0][j] + Hmm01 * FR[1][j];
        HJp[1][j] = Hmm01 * FR[0][j] + Hmm11 * FR[1][j];
        #pragma unroll
        for (int k = 0; k < 6; ++k) {
            HJp[0][j] += Hmc[0 * 6 + k] * Jcp[k][j];
            HJp[1][j] += Hmc[1 * 6 + k] * Jcp[k][j];
        }
    }

    // Cov rows (2-7) of HJp:
    // HJp[2+k][j] = Hmc^T[k][0]*FR[0][j] + Hmc^T[k][1]*FR[1][j]
    //             + sum_l Hcc[k][l]*Jcp[l][j]
    // Hmc^T[k][i] = Hmc[i][k]
    for (int k = 0; k < 6; ++k) {
        for (int j = 0; j < 3; ++j) {
            S val = Hmc[0 * 6 + k] * FR[0][j] + Hmc[1 * 6 + k] * FR[1][j];
            #pragma unroll
            for (int l = 0; l < 6; ++l) {
                // Hcc[k][l] from upper-tri storage
                int kk = k, ll = l;
                if (kk > ll) { int tmp = kk; kk = ll; ll = tmp; }
                int fi = kk * (11 - kk) / 2 + ll;
                val += Hcc[fi] * Jcp[l][j];
            }
            HJp[2 + k][j] = val;
        }
    }

    // H_{p,p} = Jp^T @ HJp  (3x8 @ 8x3 → 3x3, store upper-tri 6 elements)
    // This is the GN (Term 1) part. Term 2 is added below.
    S h_pp_gn[6];
    {
        int flat = 0;
        for (int i = 0; i < 3; ++i) {
            for (int j = i; j < 3; ++j) {
                S sum = static_cast<S>(0);
                // Jp^T row i times HJp col j
                // Jp^T[i][row] = Jp[row][i]
                // rows 0-1: FR, rows 2-7: Jcp
                sum += FR[0][i] * HJp[0][j] + FR[1][i] * HJp[1][j];
                #pragma unroll
                for (int k = 0; k < 6; ++k) {
                    sum += Jcp[k][i] * HJp[2 + k][j];
                }
                h_pp_gn[flat] = sum;
                ++flat;
            }
        }
    }

    // ================================================================
    // STEP 2b-extra: Compute Term 2 for H_{p,p}
    // ================================================================
    // Term 2 = sum_k v_phi_k * nabla^2_p phi_k
    //        = sum_i v_{m_i} * R^T * Hess_{p_c}(m_i) * R
    //        + sum_k v_{c_k} * R^T * Hess_{p_c}(c_k) * R
    //
    // where Hess_{p_c}(phi_k) is the 3x3 Hessian of output phi_k w.r.t.
    // camera coordinates p_c = (x, y, z).
    //
    // We first compute the Term 2 contribution in camera coordinates
    // (a 3x3 matrix term2_pc), then rotate: Term2 = R^T * term2_pc * R.

    // ---------------------------------------------------------------
    // Pre-compute second-order auxiliary quantities for J derivatives
    // ---------------------------------------------------------------
    // We need d^2(J_ij)/d(p_c_m)d(p_c_n) for all non-trivial J entries.
    //
    // Key insight: tx = clamp(x/z)*z and ty = clamp(y/z)*z are piecewise
    // linear in (x,y,z), so d^2(tx)=0 and d^2(ty)=0.
    // All second-order terms come from 1/z and 1/l nonlinearities.
    //
    // Precomputed first-order quantities (already available):
    //   rz, rz2, rz3, rl, rl2
    //   dtx_dx, dtx_dz, dty_dy, dty_dz
    //   dl_dx, dl_dy, dl_dz
    //   x_clipped, y_clipped

    const S rz4 = rz3 * rz;
    const S rl3 = rl2 * rl;

    // dS_mn table: partial derivative of S_m w.r.t. p_n
    // S_x = tx * dtx_dx,  S_y = ty * dty_dy,  S_z = tx*dtx_dz + ty*dty_dz + z
    // dS[m][n] is stored as a 3x3 symmetric table.
    // Only 6 independent values (symmetric in m,n? No, S is not symmetric in general)
    // Actually dS_mn is NOT symmetric: dS_x/dz = dtx_dz * dtx_dx, dS_z/dx = dtx_dx * dtx_dz
    // These are equal! So yes, dS_mn IS symmetric: dS_mn = dS_nm.
    const S dSmn_xx = dtx_dx * dtx_dx;  // = dtx_dx^2
    const S dSmn_xy = static_cast<S>(0);
    const S dSmn_xz = dtx_dz * dtx_dx;
    const S dSmn_yy = dty_dy * dty_dy;  // = dty_dy^2
    const S dSmn_yz = dty_dz * dty_dy;
    const S dSmn_zz = dtx_dz * dtx_dz + dty_dz * dty_dz + static_cast<S>(1);

    // ---------------------------------------------------------------
    // Part A of Term2: contribution from means2d
    // ---------------------------------------------------------------
    // m0 = fx * x / z + cx  =>  Hess_{p_c}(m0) has only 2 nonzero entries:
    //   d^2(m0)/dxdz = -fx * rz^2
    //   d^2(m0)/dz^2 = 2 * fx * x * rz^3
    //
    // m1 = fy * y / z + cy  =>  Hess_{p_c}(m1):
    //   d^2(m1)/dydz = -fy * rz^2
    //   d^2(m1)/dz^2 = 2 * fy * y * rz^3
    //
    // term2_means_pc[m][n] = vm0 * d^2(m0)/dp_m dp_n + vm1 * d^2(m1)/dp_m dp_n

    S term2_pc[3][3];  // Term 2 accumulator in camera coords (symmetric)
    // Initialize to zero
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            term2_pc[i][j] = static_cast<S>(0);

    // From m0: only (x,z) and (z,z) entries
    const S d2m0_xz = -fx * rz2;
    const S d2m0_zz = static_cast<S>(2) * fx * x * rz3;

    term2_pc[0][2] += vm0 * d2m0_xz;
    term2_pc[2][0] += vm0 * d2m0_xz;
    term2_pc[2][2] += vm0 * d2m0_zz;

    // From m1: only (y,z) and (z,z) entries
    const S d2m1_yz = -fy * rz2;
    const S d2m1_zz = static_cast<S>(2) * fy * y * rz3;

    term2_pc[1][2] += vm1 * d2m1_yz;
    term2_pc[2][1] += vm1 * d2m1_yz;
    term2_pc[2][2] += vm1 * d2m1_zz;

    // ---------------------------------------------------------------
    // Part B of Term2: contribution from cov2d_3x3
    // ---------------------------------------------------------------
    // c_k = upper_tri(C2d)_k = upper_tri(J * Sigma_c * J^T)_k
    //
    // Hess_{p_c}(c_k)[m][n] = d^2(c_k)/dp_c_m dp_c_n
    //
    // Since C2d_ab = sum_{ij} J_ai * Sc_ij * J_bj, we have:
    //   d^2(C2d_ab)/dp_m dp_n =
    //     sum_{ij} [d^2(J_ai)/dp_m dp_n * Sc_ij * J_bj
    //             + J_ai * Sc_ij * d^2(J_bj)/dp_m dp_n
    //             + dJ_ai/dp_m * Sc_ij * dJ_bj/dp_n
    //             + dJ_ai/dp_n * Sc_ij * dJ_bj/dp_m]
    //
    // All four terms are needed for Term 2. The dJ·dJ cross terms are NOT
    // captured by the GN term (Term 1 = J_p^T H_phiphi J_p), because Term 1
    // uses the second-order loss Hessian H_phiphi, while Term 2 uses the
    // first-order gradient v. These are independent contributions.
    //
    // Let H2J_mn = d^2(J)/dp_m dp_n, K_m = dJ/dp_m (already computed).
    // For each (m,n) pair, the full Hessian of C2d decomposes as:
    //   d^2(C2d)/dp_m dp_n = (B_mn + B_mn^T) + (C_mn + C_mn^T)
    // where:
    //   B_mn = H2J_mn * Sc * J^T    (d²J contribution)
    //   C_mn = K_m * Sc * K_n^T     (dJ·dJ cross contribution)
    //
    // For contraction with vc (upper-tri packing of v_cov2d_3x3):
    //   term2_cov_pc[m][n] = sum_{a<=b} vc_{(a,b)} * X_mn[a][b]
    //   where X_mn = (B_mn + B_mn^T) + (C_mn + C_mn^T) (symmetric 3x3)
    //
    // Since (m,n) is symmetric (Hessian), we have 6 pairs: xx, xy, xz, yy, yz, zz.
    //
    // For C_mn, we already have KmSc = K_m * Sc (computed as K0Sc, K1Sc, K2Sc).
    // C_mn = KmSc * Kn^T where Kn is sparse (dJ/dp_n).

    // ---------------------------------------------------------------
    // Compute d^2(J_ij)/dp_m dp_n for each unique (m,n) pair
    // ---------------------------------------------------------------
    // We compute H2J as a 3x3 matrix for each (m,n), then form B = H2J * Sc * J^T.
    //
    // Structure of d^2(J)/dp^2:
    //   J01 = J10 = 0 (constant) → d^2 = 0 always
    //   J00 = fx*rz:         only (z,z) nonzero: 2*fx*rz^3
    //   J02 = -fx*tx*rz^2:   (x,z): 2*fx*dtx_dx*rz^3, (z,z): fx*(4*dtx_dz*rz^3 - 6*tx*rz^4)
    //   J11 = fy*rz:         only (z,z) nonzero: 2*fy*rz^3
    //   J12 = -fy*ty*rz^2:   (y,z): 2*fy*dty_dy*rz^3, (z,z): fy*(4*dty_dz*rz^3 - 6*ty*rz^4)
    //   J20 = tx*rl:         all 6 pairs nonzero (via rl and dl)
    //   J21 = ty*rl:         all 6 pairs nonzero
    //   J22 = z*rl:          all 6 pairs nonzero
    //
    // For J2k = v_k * rl where v = (tx, ty, z):
    //   d^2(J2k)/dp_m dp_n = -(dv_k/dp_m * dl/dp_n + dv_k/dp_n * dl/dp_m) * rl^2
    //                       + v_k * (3 * dl/dp_m * dl/dp_n - dSmn) * rl^3
    //
    // Derivatives of v = (tx, ty, z) w.r.t. (x, y, z):
    //   dtx: (dtx_dx, 0, dtx_dz)
    //   dty: (0, dty_dy, dty_dz)
    //   dz:  (0, 0, 1)

    // Pre-compute dl array for indexing
    const S dl[3] = {dl_dx, dl_dy, dl_dz};

    // dv[k][m] = d(v_k)/d(p_c_m), v = (tx, ty, z)
    const S dv[3][3] = {
        {dtx_dx,             static_cast<S>(0), dtx_dz},
        {static_cast<S>(0),  dty_dy,            dty_dz},
        {static_cast<S>(0),  static_cast<S>(0), static_cast<S>(1)}
    };
    const S v_row3[3] = {tx, ty, z};  // v_k values for J's third row

    // dSmn[m][n] (symmetric 3x3)
    const S dSmn[3][3] = {
        {dSmn_xx, dSmn_xy, dSmn_xz},
        {dSmn_xy, dSmn_yy, dSmn_yz},
        {dSmn_xz, dSmn_yz, dSmn_zz}
    };

    // Pre-compute d^2(J00)/dz^2 and d^2(J11)/dz^2
    const S d2J00_zz = static_cast<S>(2) * fx * rz3;
    const S d2J11_zz = static_cast<S>(2) * fy * rz3;

    // d^2(J02) entries
    const S d2J02_xz = static_cast<S>(2) * fx * dtx_dx * rz3;
    const S d2J02_zz = fx * (static_cast<S>(4) * dtx_dz * rz3
                             - static_cast<S>(6) * tx * rz4);

    // d^2(J12) entries
    const S d2J12_yz = static_cast<S>(2) * fy * dty_dy * rz3;
    const S d2J12_zz = fy * (static_cast<S>(4) * dty_dz * rz3
                             - static_cast<S>(6) * ty * rz4);

    // For each symmetric pair (m,n) with m <= n, we build the 3x3 matrix
    // H2J_mn[i][j] = d^2(J_ij)/dp_m dp_n, then compute:
    //   B_mn = H2J_mn * Sc * J^T  (3x3)
    //   X_mn = B_mn + B_mn^T      (symmetric 3x3)
    //   term2_cov_pc[m][n] += sum_{a<=b} vc_{(a,b)} * X_mn[a][b]
    //
    // We also accumulate the means2d term2 that was computed above.

    // We'll iterate over all 6 unique (m,n) pairs and accumulate into term2_pc.
    // Mapping: flat index 0=(0,0), 1=(0,1), 2=(0,2), 3=(1,1), 4=(1,2), 5=(2,2)
    // to (m,n) pairs.

    const int mn_m[6] = {0, 0, 0, 1, 1, 2};
    const int mn_n[6] = {0, 1, 2, 1, 2, 2};

    for (int mn_idx = 0; mn_idx < 6; ++mn_idx) {
        const int m = mn_m[mn_idx];
        const int n = mn_n[mn_idx];

        // Build H2J_mn (3x3 matrix of d^2(J_ij)/dp_m dp_n)
        S H2J[3][3];

        // Row 0: J00, 0, J02
        // J00: only (2,2) nonzero
        H2J[0][0] = (m == 2 && n == 2) ? d2J00_zz : static_cast<S>(0);
        H2J[0][1] = static_cast<S>(0);  // J01 = 0 always
        // J02: (0,2) and (2,2) nonzero
        if (m == 0 && n == 2) H2J[0][2] = d2J02_xz;
        else if (m == 2 && n == 2) H2J[0][2] = d2J02_zz;
        else H2J[0][2] = static_cast<S>(0);

        // Row 1: 0, J11, J12
        H2J[1][0] = static_cast<S>(0);  // J10 = 0 always
        // J11: only (2,2) nonzero
        H2J[1][1] = (m == 2 && n == 2) ? d2J11_zz : static_cast<S>(0);
        // J12: (1,2) and (2,2) nonzero
        if (m == 1 && n == 2) H2J[1][2] = d2J12_yz;
        else if (m == 2 && n == 2) H2J[1][2] = d2J12_zz;
        else H2J[1][2] = static_cast<S>(0);

        // Row 2: J20, J21, J22 — all use the unified formula:
        //   d^2(J2k)/dp_m dp_n = -(dv_k_m * dl_n + dv_k_n * dl_m) * rl^2
        //                       + v_k * (3*dl_m*dl_n - dSmn[m][n]) * rl^3
        const S common_rl3 = (static_cast<S>(3) * dl[m] * dl[n] - dSmn[m][n]) * rl3;
        for (int k = 0; k < 3; ++k) {
            H2J[2][k] = -(dv[k][m] * dl[n] + dv[k][n] * dl[m]) * rl2
                        + v_row3[k] * common_rl3;
        }

        // Compute B_mn = H2J * Sc * J^T
        // First: H2J_Sc = H2J * Sc (3x3 * 3x3 → 3x3)
        // Sc is symmetric: Sc00, Sc01, Sc02, Sc01(=Sc10), Sc11, Sc12, Sc02(=Sc20), Sc12(=Sc21), Sc22
        S H2J_Sc[3][3];
        H2J_Sc[0][0] = H2J[0][0]*Sc00 + H2J[0][1]*Sc01 + H2J[0][2]*Sc02;
        H2J_Sc[0][1] = H2J[0][0]*Sc01 + H2J[0][1]*Sc11 + H2J[0][2]*Sc12;
        H2J_Sc[0][2] = H2J[0][0]*Sc02 + H2J[0][1]*Sc12 + H2J[0][2]*Sc22;
        H2J_Sc[1][0] = H2J[1][0]*Sc00 + H2J[1][1]*Sc01 + H2J[1][2]*Sc02;
        H2J_Sc[1][1] = H2J[1][0]*Sc01 + H2J[1][1]*Sc11 + H2J[1][2]*Sc12;
        H2J_Sc[1][2] = H2J[1][0]*Sc02 + H2J[1][1]*Sc12 + H2J[1][2]*Sc22;
        H2J_Sc[2][0] = H2J[2][0]*Sc00 + H2J[2][1]*Sc01 + H2J[2][2]*Sc02;
        H2J_Sc[2][1] = H2J[2][0]*Sc01 + H2J[2][1]*Sc11 + H2J[2][2]*Sc12;
        H2J_Sc[2][2] = H2J[2][0]*Sc02 + H2J[2][1]*Sc12 + H2J[2][2]*Sc22;

        // B_mn = H2J_Sc * J^T (3x3)
        // J^T sparse: J^T[0][0]=J00, J^T[2][0]=J02, J^T[1][1]=J11, J^T[2][1]=J12,
        //             J^T[0][2]=J20, J^T[1][2]=J21, J^T[2][2]=J22
        S B[3][3];
        B[0][0] = H2J_Sc[0][0]*J00 +                H2J_Sc[0][2]*J02;
        B[0][1] =                H2J_Sc[0][1]*J11 + H2J_Sc[0][2]*J12;
        B[0][2] = H2J_Sc[0][0]*J20 + H2J_Sc[0][1]*J21 + H2J_Sc[0][2]*J22;
        B[1][0] = H2J_Sc[1][0]*J00 +                H2J_Sc[1][2]*J02;
        B[1][1] =                H2J_Sc[1][1]*J11 + H2J_Sc[1][2]*J12;
        B[1][2] = H2J_Sc[1][0]*J20 + H2J_Sc[1][1]*J21 + H2J_Sc[1][2]*J22;
        B[2][0] = H2J_Sc[2][0]*J00 +                H2J_Sc[2][2]*J02;
        B[2][1] =                H2J_Sc[2][1]*J11 + H2J_Sc[2][2]*J12;
        B[2][2] = H2J_Sc[2][0]*J20 + H2J_Sc[2][1]*J21 + H2J_Sc[2][2]*J22;

        // ------ Part (ii): C_mn = KmSc * Kn^T (dJ cross terms) ------
        // C_mn[a][b] = sum_c KmSc[a][c] * dJdp[n][b][c]
        // (using K_n^T[c][b] = K_n[b][c] = dJdp[n][b][c])
        S C[3][3];
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                C[a][b] = static_cast<S>(0);
                for (int c = 0; c < 3; ++c) {
                    C[a][b] += KSc_val(m, a, c) * dJdp[n][b][c];
                }
            }
        }

        // X_mn = (B + B^T) + (C + C^T)  (symmetric 3x3)
        // Contract with vc (upper-tri packing):
        //   result = sum_{a<=b} vc_{(a,b)} * X[a][b]
        //          = vc[0]*X[0][0] + vc[1]*X[0][1] + vc[2]*X[0][2]
        //          + vc[3]*X[1][1] + vc[4]*X[1][2] + vc[5]*X[2][2]
        S X00 = static_cast<S>(2) * (B[0][0] + C[0][0]);
        S X01 = (B[0][1] + B[1][0]) + (C[0][1] + C[1][0]);
        S X02 = (B[0][2] + B[2][0]) + (C[0][2] + C[2][0]);
        S X11 = static_cast<S>(2) * (B[1][1] + C[1][1]);
        S X12 = (B[1][2] + B[2][1]) + (C[1][2] + C[2][1]);
        S X22 = static_cast<S>(2) * (B[2][2] + C[2][2]);

        S t2_cov_mn = vc[0]*X00 + vc[1]*X01 + vc[2]*X02
                    + vc[3]*X11 + vc[4]*X12 + vc[5]*X22;

        // Accumulate into term2_pc
        term2_pc[m][n] += t2_cov_mn;
        if (m != n) {
            term2_pc[n][m] += t2_cov_mn;  // symmetric
        }
    }
    #undef KSc_val

    // ---------------------------------------------------------------
    // Rotate Term 2 from camera coords to world coords:
    //   Term2_world = R^T * term2_pc * R
    // Then add to h_pp_gn.
    // ---------------------------------------------------------------
    // Q2 = term2_pc * R  (3x3)
    S Q2[3][3];
    Q2[0][0] = term2_pc[0][0]*R00 + term2_pc[0][1]*R10 + term2_pc[0][2]*R20;
    Q2[0][1] = term2_pc[0][0]*R01 + term2_pc[0][1]*R11 + term2_pc[0][2]*R21;
    Q2[0][2] = term2_pc[0][0]*R02 + term2_pc[0][1]*R12 + term2_pc[0][2]*R22;
    Q2[1][0] = term2_pc[1][0]*R00 + term2_pc[1][1]*R10 + term2_pc[1][2]*R20;
    Q2[1][1] = term2_pc[1][0]*R01 + term2_pc[1][1]*R11 + term2_pc[1][2]*R21;
    Q2[1][2] = term2_pc[1][0]*R02 + term2_pc[1][1]*R12 + term2_pc[1][2]*R22;
    Q2[2][0] = term2_pc[2][0]*R00 + term2_pc[2][1]*R10 + term2_pc[2][2]*R20;
    Q2[2][1] = term2_pc[2][0]*R01 + term2_pc[2][1]*R11 + term2_pc[2][2]*R21;
    Q2[2][2] = term2_pc[2][0]*R02 + term2_pc[2][1]*R12 + term2_pc[2][2]*R22;

    // term2_world = R^T * Q2  (3x3), store upper-tri
    {
        int flat = 0;
        for (int i = 0; i < 3; ++i) {
            for (int j = i; j < 3; ++j) {
                // R^T row i = R col i = (R[0][i], R[1][i], R[2][i])
                S val = static_cast<S>(0);
                // R^T[i][0] = R00_col_i, etc.
                // val = R[0][i]*Q2[0][j] + R[1][i]*Q2[1][j] + R[2][i]*Q2[2][j]
                // Using row-major R: R[row][col]
                // R^T[i][k] = R[k][i]
                // First column of R^T is R00, R10, R20 at row i → R[0][i], R[1][i], R[2][i]
                // But R is stored as R00=vm[0]..., so R[0][i] = {R00,R01,R02}[i]
                // R^T[i][0] = R_{0,i}, R^T[i][1] = R_{1,i}, R^T[i][2] = R_{2,i}
                // We need: sum_k R^T[i][k] * Q2[k][j] = sum_k R[k][i] * Q2[k][j]
                // Cannot easily index R[k][i] without arrays. Use explicit:
                if (i == 0) {
                    val = R00*Q2[0][j] + R10*Q2[1][j] + R20*Q2[2][j];
                } else if (i == 1) {
                    val = R01*Q2[0][j] + R11*Q2[1][j] + R21*Q2[2][j];
                } else {
                    val = R02*Q2[0][j] + R12*Q2[1][j] + R22*Q2[2][j];
                }
                atomicAdd(h_means_out + gs_id * 6 + flat, h_pp_gn[flat] + val);
                ++flat;
            }
        }
    }

    // ================================================================
    // STEP 2c: Compute H_{p,s} (GN Term 1 + Term 2)
    // ================================================================
    // GN (Term 1): H_{p,s} = Jp^T @ H @ Js (same as before)
    //
    // Term 2 for H_{p,s}:
    //   sum_k v_{c_k} * d^2(c_k) / d(p_i) d(s_j)
    //
    // Since means2d doesn't depend on s, only cov2d_3x3 contributes.
    // c_k = upper_tri(J * Sigma_c * J^T)_k = upper_tri(P * Sigma * P^T)_k
    //     = sum_l T_{kl}(P) * s_l
    // where T_{kl} depends on P = J * R, and J depends on p_c.
    //
    // So d(c_k)/d(s_l) = T_{kl}(P), and
    //   d^2(c_k)/d(p_i) d(s_l) = d(T_{kl})/d(p_i) = sum_m d(T_{kl})/d(p_c_m) * R_{mi}
    //
    // term2_ps[i][l] = sum_k vc[k] * sum_m d(T_{kl})/d(p_c_m) * R_{mi}
    //
    // We need d(T_{kl})/d(p_c_m). T depends on P = J*R, so:
    //   d(T_{kl})/d(p_c_m) = sum over P entries: dT_{kl}/dP_{ab} * dP_{ab}/dp_c_m
    //   dP_{ab}/dp_c_m = sum_c dJ_{ac}/dp_c_m * R_{cb} = K_m[a][c] * R_{cb}
    //                  = (K_m * R)[a][b] = dP_m[a][b]
    //
    // T_{kl} where k=(a,b) upper-tri, l=(i,j) upper-tri:
    //   If a==b and i==j: T = P_{ai}^2     → dT/dP_{ab} = 2*P_{ai} * delta(a=a,b→col=i)
    //   If a==b and i<j:  T = 2*P_{ai}*P_{aj} → dT/dP = 2*(P_{aj}*delta_{col=i} + P_{ai}*delta_{col=j})
    //   If a<b and i==j:  T = P_{ai}*P_{bi}   → dT/dP = P_{bi}*delta_{row=a,col=i} + P_{ai}*delta_{row=b,col=i}
    //   If a<b and i<j:   T = P_{ai}*P_{bj} + P_{aj}*P_{bi} → 4 terms
    //
    // This is complex to implement via dT/dP. Instead, use a direct approach:
    //
    // d(T_{kl})/d(p_c_m) can be computed by noting that T maps s → c via P.
    // c_k = upper_tri(P * Sigma * P^T)_k
    // dc_k/dp_c_m = upper_tri(dP_m * Sigma * P^T + P * Sigma * dP_m^T)_k
    //             = upper_tri(dP_m * Sigma * P^T)_k + upper_tri(P * Sigma * dP_m^T)_k
    //
    // Since c_k = sum_l T_{kl} * s_l, we have:
    //   dc_k/dp_c_m = sum_l [d(T_{kl})/dp_c_m] * s_l
    //
    // But we can also express:
    //   dc_k/dp_c_m for specific s_l = e_l (basis vector):
    //     d(T_{kl})/dp_c_m = upper_tri(dP_m * Sigma_l * P^T + P * Sigma_l * dP_m^T)_k
    //   where Sigma_l is the symmetric matrix with 1 in positions corresponding to s_l.
    //
    // This means d(T)/dp_c_m has the same structure as d(c)/dp_c_m but with
    // Sigma replaced by the basis Sigma_l.
    //
    // The Term 2 contraction is:
    //   term2_ps_cam[m][l] = sum_k vc_k * d(T_{kl})/dp_c_m
    //
    // For a fixed m, this is:
    //   term2_ps_cam[m][l] = sum_k vc_k * upper_tri(dP_m * Sigma_l * P^T + P * Sigma_l * dP_m^T)_k
    //
    // Using the upper-tri contraction with vc (same as in the cov2d backward),
    // this equals:
    //   = tr(GGt * dP_m * Sigma_l * P^T)  (by the same argument as before)
    //
    // where GGt = G_upper + G_upper^T.
    //
    // So: term2_ps_cam[m][l] = tr(GGt * dP_m * Sigma_l * P^T)
    //
    // Let Q_m = GGt * dP_m  (3x3), then:
    //   term2_ps_cam[m][l] = tr(Q_m * Sigma_l * P^T)
    //                      = sum_{ab} Q_m[a][b] * (Sigma_l * P^T)[b][a]
    //                      = sum_{ab} Q_m[a][b] * sum_c Sigma_l[b][c] * P[a][c]
    //
    // For the upper-tri s_l corresponding to (i,j) with i<=j:
    //   Sigma_l[b][c] = delta_{bi}*delta_{cj} + delta_{bj}*delta_{ci}  (if i<j, factor 1 each)
    //                 = delta_{bi}*delta_{ci}  (if i==j)
    //
    // Case i==j (diagonal input):
    //   term2_ps_cam[m][l] = sum_a Q_m[a][i] * P[a][i]
    //
    // Case i<j (off-diagonal input):
    //   term2_ps_cam[m][l] = sum_a (Q_m[a][i]*P[a][j] + Q_m[a][j]*P[a][i])
    //
    // This is exactly the T-matrix structure but applied to Q_m instead of P!
    // In other words, term2_ps_cam[m][:] is the "T-like" map applied to Q_m
    // with P as the second factor.
    //
    // We need dP_m = K_m * R for m = 0, 1, 2 (using the already-computed K matrices).
    // Actually, dP_m[a][b] = sum_c K_m[a][c] * R[c][b]. But we already have K_m * Sc.
    // We need K_m * R directly.

    // Compute GGt matrix (already have the elements from the v_J computation)
    // GGt[i][j] = (G_upper + G_upper^T)[i][j]
    // GGt00 = 2*vc[0], GGt01 = vc[1], GGt02 = vc[2]
    // GGt10 = vc[1],   GGt11 = 2*vc[3], GGt12 = vc[4]
    // GGt20 = vc[2],   GGt21 = vc[4],   GGt22 = 2*vc[5]
    // These were already computed above as GGt00..GGt22.

    // Compute dP_m = K_m * R for m = 0, 1, 2
    // K0 (dJ/dx): nonzero entries K0[0][2]=dJ02_dx, K0[2][0]=dJ20_dx, K0[2][1]=dJ21_dx, K0[2][2]=dJ22_dx
    // K1 (dJ/dy): nonzero entries K1[1][2]=dJ12_dy, K1[2][0]=dJ20_dy, K1[2][1]=dJ21_dy, K1[2][2]=dJ22_dy
    // K2 (dJ/dz): nonzero entries K2[0][0]=dJ00_dz, K2[0][2]=dJ02_dz,
    //             K2[1][1]=dJ11_dz, K2[1][2]=dJ12_dz,
    //             K2[2][0]=dJ20_dz, K2[2][1]=dJ21_dz, K2[2][2]=dJ22_dz
    S dP[3][3][3];  // dP[m][a][b] = dP_m[a][b] = sum_c K_m[a][c] * R[c][b]

    // dP[0] = K0 * R:
    //   row 0: K0[0][2]*R[2][j]
    //   row 1: all zero
    //   row 2: K0[2][0]*R[0][j] + K0[2][1]*R[1][j] + K0[2][2]*R[2][j]
    dP[0][0][0] = dJ02_dx * R20;
    dP[0][0][1] = dJ02_dx * R21;
    dP[0][0][2] = dJ02_dx * R22;
    dP[0][1][0] = static_cast<S>(0);
    dP[0][1][1] = static_cast<S>(0);
    dP[0][1][2] = static_cast<S>(0);
    dP[0][2][0] = dJ20_dx*R00 + dJ21_dx*R10 + dJ22_dx*R20;
    dP[0][2][1] = dJ20_dx*R01 + dJ21_dx*R11 + dJ22_dx*R21;
    dP[0][2][2] = dJ20_dx*R02 + dJ21_dx*R12 + dJ22_dx*R22;

    // dP[1] = K1 * R:
    //   row 0: all zero
    //   row 1: K1[1][2]*R[2][j]
    //   row 2: K1[2][0]*R[0][j] + K1[2][1]*R[1][j] + K1[2][2]*R[2][j]
    dP[1][0][0] = static_cast<S>(0);
    dP[1][0][1] = static_cast<S>(0);
    dP[1][0][2] = static_cast<S>(0);
    dP[1][1][0] = dJ12_dy * R20;
    dP[1][1][1] = dJ12_dy * R21;
    dP[1][1][2] = dJ12_dy * R22;
    dP[1][2][0] = dJ20_dy*R00 + dJ21_dy*R10 + dJ22_dy*R20;
    dP[1][2][1] = dJ20_dy*R01 + dJ21_dy*R11 + dJ22_dy*R21;
    dP[1][2][2] = dJ20_dy*R02 + dJ21_dy*R12 + dJ22_dy*R22;

    // dP[2] = K2 * R:
    //   row 0: K2[0][0]*R[0][j] + K2[0][2]*R[2][j]
    //   row 1: K2[1][1]*R[1][j] + K2[1][2]*R[2][j]
    //   row 2: K2[2][0]*R[0][j] + K2[2][1]*R[1][j] + K2[2][2]*R[2][j]
    dP[2][0][0] = dJ00_dz*R00 + dJ02_dz*R20;
    dP[2][0][1] = dJ00_dz*R01 + dJ02_dz*R21;
    dP[2][0][2] = dJ00_dz*R02 + dJ02_dz*R22;
    dP[2][1][0] = dJ11_dz*R10 + dJ12_dz*R20;
    dP[2][1][1] = dJ11_dz*R11 + dJ12_dz*R21;
    dP[2][1][2] = dJ11_dz*R12 + dJ12_dz*R22;
    dP[2][2][0] = dJ20_dz*R00 + dJ21_dz*R10 + dJ22_dz*R20;
    dP[2][2][1] = dJ20_dz*R01 + dJ21_dz*R11 + dJ22_dz*R21;
    dP[2][2][2] = dJ20_dz*R02 + dJ21_dz*R12 + dJ22_dz*R22;

    // For each camera-space direction m (0=x, 1=y, 2=z):
    //   Q_m = GGt * dP_m  (3x3)
    //   Then term2_ps_cam[m][l] = "T-like contraction" of Q_m and P for input l.
    //
    // term2_ps_world[i][l] = sum_m term2_ps_cam[m][l] * R[m][i]
    //                      = sum_m R[m][i] * term2_ps_cam[m][l]
    //
    // (This is the rotation: d/dp_i = sum_m R[m][i] * d/dp_c_m)

    // Upper-tri index pairs for 3x3 symmetric
    const int row_idx6[6] = {0, 0, 0, 1, 1, 2};
    const int col_idx6[6] = {0, 1, 2, 1, 2, 2};

    S term2_ps_world[3][6];  // [world_p_idx][covar_idx]
    for (int i = 0; i < 3; ++i) {
        for (int l = 0; l < 6; ++l) {
            term2_ps_world[i][l] = static_cast<S>(0);
        }
    }

    for (int m = 0; m < 3; ++m) {
        // Compute Q_m = GGt * dP[m]  (3x3 * 3x3 → 3x3)
        S Qm[3][3];
        Qm[0][0] = GGt00*dP[m][0][0] + GGt01*dP[m][1][0] + GGt02*dP[m][2][0];
        Qm[0][1] = GGt00*dP[m][0][1] + GGt01*dP[m][1][1] + GGt02*dP[m][2][1];
        Qm[0][2] = GGt00*dP[m][0][2] + GGt01*dP[m][1][2] + GGt02*dP[m][2][2];
        Qm[1][0] = GGt10*dP[m][0][0] + GGt11*dP[m][1][0] + GGt12*dP[m][2][0];
        Qm[1][1] = GGt10*dP[m][0][1] + GGt11*dP[m][1][1] + GGt12*dP[m][2][1];
        Qm[1][2] = GGt10*dP[m][0][2] + GGt11*dP[m][1][2] + GGt12*dP[m][2][2];
        Qm[2][0] = GGt20*dP[m][0][0] + GGt21*dP[m][1][0] + GGt22*dP[m][2][0];
        Qm[2][1] = GGt20*dP[m][0][1] + GGt21*dP[m][1][1] + GGt22*dP[m][2][1];
        Qm[2][2] = GGt20*dP[m][0][2] + GGt21*dP[m][1][2] + GGt22*dP[m][2][2];

        // For each covar index l (upper-tri index (ii,jj)):
        // Compute the T-like contraction:
        //   If ii == jj: val = sum_a Qm[a][ii] * P[a][ii]
        //   If ii < jj:  val = sum_a (Qm[a][ii]*P[a][jj] + Qm[a][jj]*P[a][ii])
        S t2_cam[6];
        for (int l = 0; l < 6; ++l) {
            const int ii = row_idx6[l];
            const int jj = col_idx6[l];
            S val = static_cast<S>(0);
            if (ii == jj) {
                for (int a = 0; a < 3; ++a) {
                    val += Qm[a][ii] * P[a][ii];
                }
            } else {
                for (int a = 0; a < 3; ++a) {
                    val += Qm[a][ii] * P[a][jj] + Qm[a][jj] * P[a][ii];
                }
            }
            t2_cam[l] = val;
        }

        // Accumulate into world coordinates:
        // term2_ps_world[i][l] += R[m][i] * t2_cam[l]
        // R[m][i]: R[0][0]=R00, R[0][1]=R01, R[0][2]=R02, R[1][0]=R10, etc.
        S Rm[3];
        if (m == 0) { Rm[0] = R00; Rm[1] = R01; Rm[2] = R02; }
        else if (m == 1) { Rm[0] = R10; Rm[1] = R11; Rm[2] = R12; }
        else { Rm[0] = R20; Rm[1] = R21; Rm[2] = R22; }

        for (int i = 0; i < 3; ++i) {
            for (int l = 0; l < 6; ++l) {
                term2_ps_world[i][l] += Rm[i] * t2_cam[l];
            }
        }
    }

    // Now compute GN Term 1 for H_{p,s} and add Term 2:
    // GN: H_{p,s} = Jp^T @ H @ Js
    //   = Jp^T @ [[Hmc @ T], [Hcc @ T]]
    //
    // We already have Wmat = Hcc @ T (6x6).
    // Need also Hmc @ T (2x6).
    S HmcT[2][6];
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 6; ++j) {
            S sum = static_cast<S>(0);
            #pragma unroll
            for (int k = 0; k < 6; ++k) {
                sum += Hmc[i * 6 + k] * T[k * 6 + j];
            }
            HmcT[i][j] = sum;
        }
    }

    // H_{p,s} = Jp^T @ HJs + term2_ps_world
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 6; ++j) {
            S sum = static_cast<S>(0);
            // rows 0-1: FR
            sum += FR[0][i] * HmcT[0][j] + FR[1][i] * HmcT[1][j];
            // rows 2-7: Jcp
            #pragma unroll
            for (int k = 0; k < 6; ++k) {
                sum += Jcp[k][i] * Wmat[k][j];
            }
            atomicAdd(h_means_covars_out + gs_id * 18 + i * 6 + j, sum + term2_ps_world[i][j]);
        }
    }

    // ================================================================
    // STEP 3: Cross-Hessians with opacity (psi)
    // ================================================================
    // H_{theta,psi} = J_full^T @ H_{phi,psi}
    //
    // H_{phi,psi} is 8x1: [(Hmo0, Hmo1), (Hco[0..5])]
    //
    // H_{p, opacity} = Jp^T @ H_{phi,opacity}  (3x1)
    for (int i = 0; i < 3; ++i) {
        S sum = FR[0][i] * Hmo0 + FR[1][i] * Hmo1;
        #pragma unroll
        for (int k = 0; k < 6; ++k) {
            sum += Jcp[k][i] * Hco[k];
        }
        h_po_ptr[i] = sum;
    }

    // H_{s, opacity} = T^T @ H_{cov,opacity}  (6x1)
    for (int j = 0; j < 6; ++j) {
        S sum = static_cast<S>(0);
        #pragma unroll
        for (int k = 0; k < 6; ++k) {
            sum += T[k * 6 + j] * Hco[k];
        }
        h_so_ptr[j] = sum;
    }
}

// ============================================================================
// Launch Function Implementation
// ============================================================================

template <typename S>
void launch_world_to_image_projection_bwd2(
    const S* __restrict__ means,       // [N, 3]
    const S* __restrict__ covars,      // [N, 6]
    const S* __restrict__ viewmats,    // [C, 4, 4]
    const S* __restrict__ Ks,          // [C, 3, 3]
    uint32_t dim1,
    uint32_t dim0,
    const S* __restrict__ v_means2d,
    const S* __restrict__ v_cov2d_3x3,
    const S* __restrict__ h_means2d,
    const S* __restrict__ h_cov2d_3x3,
    const S* __restrict__ h_means2d_cov2d_3x3,
    const S* __restrict__ h_means2d_opacity,
    const S* __restrict__ h_cov2d_3x3_opacity,
    S* __restrict__ v_means_out,       // [N, 3]
    S* __restrict__ v_covars_out,      // [N, 6]
    S* __restrict__ h_means_out,       // [N, 6]
    S* __restrict__ h_covars_out,      // [N, 21]
    S* __restrict__ h_means_covars_out,// [N, 18]
    S* __restrict__ h_means_opacity,   // [C*N, 3]
    S* __restrict__ h_covars_opacity,  // [C*N, 6]
    uint32_t C,
    uint32_t N,
    cudaStream_t stream
) {
    const uint32_t total = C * N;
    if (total == 0) return;
    const int threads = GSPLAT2CT_N_THREADS;
    const int blocks = (total + threads - 1) / threads;

    // Zero-initialize outputs that use atomicAdd accumulation
    cudaMemsetAsync(v_means_out,       0, N * 3  * sizeof(S), stream);
    cudaMemsetAsync(v_covars_out,      0, N * 6  * sizeof(S), stream);
    cudaMemsetAsync(h_means_out,       0, N * 6  * sizeof(S), stream);
    cudaMemsetAsync(h_covars_out,      0, N * 21 * sizeof(S), stream);
    cudaMemsetAsync(h_means_covars_out,0, N * 18 * sizeof(S), stream);

    world_to_image_projection_bwd2_kernel<S><<<blocks, threads, 0, stream>>>(
        means, covars, viewmats, Ks,
        dim1, dim0,
        v_means2d, v_cov2d_3x3,
        h_means2d, h_cov2d_3x3, h_means2d_cov2d_3x3,
        h_means2d_opacity, h_cov2d_3x3_opacity,
        v_means_out, v_covars_out,
        h_means_out, h_covars_out, h_means_covars_out,
        h_means_opacity, h_covars_opacity,
        C, N
    );
}

// Explicit template instantiations
template void launch_world_to_image_projection_bwd2<float>(
    const float*, const float*, const float*, const float*,
    uint32_t, uint32_t,
    const float*, const float*,
    const float*, const float*, const float*,
    const float*, const float*,
    float*, float*,
    float*, float*, float*,
    float*, float*,
    uint32_t, uint32_t, cudaStream_t
);
template void launch_world_to_image_projection_bwd2<double>(
    const double*, const double*, const double*, const double*,
    uint32_t, uint32_t,
    const double*, const double*,
    const double*, const double*, const double*,
    const double*, const double*,
    double*, double*,
    double*, double*, double*,
    double*, double*,
    uint32_t, uint32_t, cudaStream_t
);

} // namespace gsplat2ct
