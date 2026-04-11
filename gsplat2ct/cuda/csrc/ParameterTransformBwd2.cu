/**
 * @file ParameterTransformBwd2.cu
 * @brief Second-order backward CUDA kernel for ParameterTransform module.
 *
 * Given downstream first-order gradients and second-order Hessians w.r.t.
 * covars (6 upper-tri elements), plus cross-Hessians with mean and opacity,
 * computes upstream gradients and Hessians w.r.t. (quats, scale_ori), as
 * well as cross-Hessians with mean and opacity.
 *
 * =========================================================================
 * Notation
 * =========================================================================
 *
 * Input parameters (theta):
 *   q = (w, x, y, z) [4]       -- unit quaternion
 *   so = (so0, so1, so2) [3]   -- raw scale (before sigmoid)
 *
 * Intermediate:
 *   sig_i = sigmoid(so_i)
 *   s_i = sig_i * range + scale_min   (activated scale)
 *
 * Output parameters (phi):
 *   covars = (C00, C01, C02, C11, C12, C22) [6]  -- upper-tri covariance
 *
 * External parameters (psi):
 *   mean = (mx, my, mz) [3]
 *   opacity [1]
 *
 * =========================================================================
 * Core Formulas
 * =========================================================================
 *
 * 1. Jacobian J (6 x 7) structure:
 *    J = d(covars_6) / d(q_4, so_3)
 *    Columns 0-3: d(covars)/d(q_a)
 *    Columns 4-6: d(covars)/d(so_j)
 *
 * 2. First-order gradient:
 *    v_theta = J^T @ v_covars   [7]
 *
 * 3. Second-order theta-theta Hessian:
 *    H_{theta,theta} = J^T @ H_{covar,covar} @ J + sum_k v_k * Hess_k
 *    where Hess_k = d^2(covar_k) / d(theta)^2   [7x7]
 *
 * 4. Cross-Hessian theta-psi:
 *    H_{theta,psi} = J^T @ H_{covar,psi}
 *
 * =========================================================================
 * Jacobian Structure
 * =========================================================================
 *
 * d(C_ij)/d(q_a):
 *   C_ij = sum_k R_ik * R_jk * s_k^2
 *   dC_ij/dq_a = sum_k s_k^2 * (dR_ik/dq_a * R_jk + R_ik * dR_jk/dq_a)
 *
 * d(C_ij)/d(so_m):
 *   dC_ij/ds_m = 2 * s_m * R_im * R_jm
 *   ds_m/d(so_m) = sig_m * (1 - sig_m) * range
 *   dC_ij/d(so_m) = 2 * s_m * R_im * R_jm * sig_m * (1-sig_m) * range
 *   (and 0 for m' != m)
 *
 * =========================================================================
 * Hessian Components
 * =========================================================================
 *
 * d^2(C_ij)/d(q_a)d(q_b):
 *   sum_k s_k^2 * (d2R_ik/dqa.dqb * R_jk + dR_ik/dqa * dR_jk/dqb
 *                 + dR_ik/dqb * dR_jk/dqa + R_ik * d2R_jk/dqa.dqb)
 *   Note: d2R/dq^2 are constants for unit quaternion (R is quadratic in q).
 *
 * d^2(C_ij)/d(so_m)^2:
 *   2 * R_im * R_jm * ((s_m')^2 + s_m * s_m'')
 *   where s_m' = sig_m*(1-sig_m)*range
 *         s_m'' = sig_m*(1-sig_m)*(1-2*sig_m)*range
 *
 * d^2(C_ij)/d(q_a)d(so_m):
 *   2 * s_m * s_m' * (dR_im/dqa * R_jm + R_im * dR_jm/dqa)
 *
 * d^2(C_ij)/d(so_m)d(so_n) = 0  for m != n  (curvature term)
 *   Note: full H_{s,s} can still have non-zero off-diagonal terms from
 *   the Gauss-Newton term J^T H J.
 *
 * =========================================================================
 * Derivatives of R w.r.t. q
 * =========================================================================
 *
 * For unit quaternion q = (w, x, y, z):
 *
 * dR/dw:
 *   [[0,     -2z,    2y],
 *    [2z,     0,    -2x],
 *    [-2y,    2x,    0 ]]
 *
 * dR/dx:
 *   [[0,      2y,    2z],
 *    [2y,    -4x,   -2w],
 *    [2z,     2w,   -4x]]
 *
 * dR/dy:
 *   [[-4y,    2x,    2w],
 *    [2x,     0,     2z],
 *    [-2w,    2z,   -4y]]
 *
 * dR/dz:
 *   [[-4z,   -2w,    2x],
 *    [2w,    -4z,    2y],
 *    [2x,     2y,    0 ]]
 *
 * d2R/dqa.dqb are constants (see inline computation below).
 *
 * Author: Dy164
 * Date: 2026-02-20
 */

#include "ParameterTransform.h"
#include "../include/common.h"
#include <cuda_runtime.h>

namespace gsplat2ct {

// ============================================================================
// Helper: dR/dq_a (returns 3x3 matrix in row-major as array of 9)
// ============================================================================

template <typename S>
__device__ __forceinline__ void dR_dq(
    const S w, const S x, const S y, const S z,
    int a,  // 0=w, 1=x, 2=y, 3=z
    S dR[9] // output: dR[row*3+col]
) {
    // Zero-initialize
    for (int i = 0; i < 9; ++i) dR[i] = static_cast<S>(0);

    const S two = static_cast<S>(2);
    const S four = static_cast<S>(4);

    switch (a) {
        case 0: // dR/dw
            dR[0*3+1] = -two*z;  dR[0*3+2] =  two*y;
            dR[1*3+0] =  two*z;  dR[1*3+2] = -two*x;
            dR[2*3+0] = -two*y;  dR[2*3+1] =  two*x;
            break;
        case 1: // dR/dx
            dR[0*3+1] =  two*y;  dR[0*3+2] =  two*z;
            dR[1*3+0] =  two*y;  dR[1*3+1] = -four*x; dR[1*3+2] = -two*w;
            dR[2*3+0] =  two*z;  dR[2*3+1] =  two*w;  dR[2*3+2] = -four*x;
            break;
        case 2: // dR/dy
            dR[0*3+0] = -four*y; dR[0*3+1] =  two*x;  dR[0*3+2] =  two*w;
            dR[1*3+0] =  two*x;  dR[1*3+2] =  two*z;
            dR[2*3+0] = -two*w;  dR[2*3+1] =  two*z;  dR[2*3+2] = -four*y;
            break;
        case 3: // dR/dz
            dR[0*3+0] = -four*z; dR[0*3+1] = -two*w;  dR[0*3+2] =  two*x;
            dR[1*3+0] =  two*w;  dR[1*3+1] = -four*z; dR[1*3+2] =  two*y;
            dR[2*3+0] =  two*x;  dR[2*3+1] =  two*y;
            break;
    }
}

// ============================================================================
// Helper: d2R/dq_a.dq_b (returns 3x3 constant matrix)
// These are constants (independent of q) since R is quadratic in q.
// ============================================================================

template <typename S>
__device__ __forceinline__ void d2R_dq2(
    int a, int b,  // quaternion component indices (0=w,1=x,2=y,3=z)
    S d2R[9]       // output: d2R[row*3+col]
) {
    for (int i = 0; i < 9; ++i) d2R[i] = static_cast<S>(0);

    const S two = static_cast<S>(2);
    const S four = static_cast<S>(4);

    // Ensure a <= b for symmetric access
    int aa = a, bb = b;
    if (aa > bb) { int tmp = aa; aa = bb; bb = tmp; }

    // (aa, bb) pairs: (0,0), (0,1), (0,2), (0,3),
    //                        (1,1), (1,2), (1,3),
    //                               (2,2), (2,3),
    //                                      (3,3)
    if (aa == 0 && bb == 0) {
        // d2R/dw^2 = 0 (dR/dw is linear in other components, not w)
        // Actually dR/dw has terms like -2z, 2y, 2x which are independent of w
        // so d2R/dw^2 = 0
    } else if (aa == 0 && bb == 1) {
        // d(dR/dw)/dx: dR/dw = [[0,-2z,2y],[2z,0,-2x],[-2y,2x,0]]
        // d/dx of above: only -2x -> d/dx = -2 at (1,2), 2 at (2,1)
        d2R[1*3+2] = -two;
        d2R[2*3+1] =  two;
    } else if (aa == 0 && bb == 2) {
        // d(dR/dw)/dy:
        d2R[0*3+2] =  two;
        d2R[2*3+0] = -two;
    } else if (aa == 0 && bb == 3) {
        // d(dR/dw)/dz:
        d2R[0*3+1] = -two;
        d2R[1*3+0] =  two;
    } else if (aa == 1 && bb == 1) {
        // d(dR/dx)/dx: dR/dx has -4x terms at (1,1) and (2,2)
        d2R[1*3+1] = -four;
        d2R[2*3+2] = -four;
    } else if (aa == 1 && bb == 2) {
        // d(dR/dx)/dy:
        // dR/dx = [[0,2y,2z],[2y,-4x,-2w],[2z,2w,-4x]]
        // d/dy: (0,1)=2, (1,0)=2
        d2R[0*3+1] =  two;
        d2R[1*3+0] =  two;
    } else if (aa == 1 && bb == 3) {
        // d(dR/dx)/dz:
        // dR/dx: (0,2)=2z -> d/dz=2; (2,0)=2z -> d/dz=2
        d2R[0*3+2] =  two;
        d2R[2*3+0] =  two;
    } else if (aa == 2 && bb == 2) {
        // d(dR/dy)/dy: dR/dy has -4y at (0,0) and (2,2)
        d2R[0*3+0] = -four;
        d2R[2*3+2] = -four;
    } else if (aa == 2 && bb == 3) {
        // d(dR/dy)/dz:
        // dR/dy = [[-4y,2x,2w],[2x,0,2z],[-2w,2z,-4y]]
        // d/dz: (1,2)=2, (2,1)=2
        d2R[1*3+2] =  two;
        d2R[2*3+1] =  two;
    } else if (aa == 3 && bb == 3) {
        // d(dR/dz)/dz: dR/dz has -4z at (0,0) and (1,1)
        d2R[0*3+0] = -four;
        d2R[1*3+1] = -four;
    }
}

// ============================================================================
// Helper: upper-tri index for symmetric NxN matrix
// ============================================================================

__device__ __forceinline__ int utri_idx(int i, int j, int n) {
    // For i <= j: index = i*n - i*(i+1)/2 + j
    if (i > j) { int t = i; i = j; j = t; }
    return i * n - i * (i + 1) / 2 + j;
}

// ============================================================================
// Helper: 6-element upper-tri index for 3x3 symmetric covariance
// Maps (row, col) with row<=col to flat index 0..5
// ============================================================================

__device__ __forceinline__ int cov_idx(int i, int j) {
    // (0,0)=0, (0,1)=1, (0,2)=2, (1,1)=3, (1,2)=4, (2,2)=5
    if (i > j) { int t = i; i = j; j = t; }
    if (i == 0) return j;        // 0,1,2
    if (i == 1) return j + 2;    // 3,4
    return 5;                     // (2,2)=5
}

// ============================================================================
// Second-Order Backward Kernel
// ============================================================================

template <typename S>
__global__ void parameter_transform_bwd2_kernel(
    const S* __restrict__ quats,             // [N, 4]
    const S* __restrict__ scale_ori,         // [N, 3]
    const S scale_min,
    const S scale_max,
    // downstream first-order
    const S* __restrict__ v_covars,          // [N, 6]
    // downstream covar-covar Hessian
    const S* __restrict__ h_covars,          // [N, 21] upper-tri 6x6
    // downstream covar-psi cross Hessians
    const S* __restrict__ h_covars_means,    // [N, 18] 6x3 row-major
    const S* __restrict__ h_covars_opacity,  // [N, 6]
    // outputs: first-order
    S* __restrict__ v_quats,                 // [N, 4]
    S* __restrict__ v_scale_ori,             // [N, 3]
    // outputs: theta-theta Hessians
    S* __restrict__ h_quats,                 // [N, 10] upper-tri 4x4
    S* __restrict__ h_scale_ori,             // [N, 6]  upper-tri 3x3
    S* __restrict__ h_quats_scale_ori,       // [N, 12] 4x3 row-major
    // outputs: theta-psi cross Hessians
    S* __restrict__ h_quats_means,           // [N, 12] 4x3 row-major
    S* __restrict__ h_quats_opacity,         // [N, 4]
    S* __restrict__ h_scale_ori_means,       // [N, 9]  3x3 row-major
    S* __restrict__ h_scale_ori_opacity,     // [N, 3]
    uint32_t N
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // ==================================================================
    // Read inputs
    // ==================================================================
    const S w = quats[idx * 4 + 0];
    const S x = quats[idx * 4 + 1];
    const S y = quats[idx * 4 + 2];
    const S z = quats[idx * 4 + 3];

    const S range = scale_max - scale_min;
    S s[3], sig[3], sp[3], spp[3]; // s, sigmoid, s', s''
    for (int i = 0; i < 3; ++i) {
        const S so = scale_ori[idx * 3 + i];
        sig[i] = static_cast<S>(1) / (static_cast<S>(1) + exp(-so));
        s[i] = sig[i] * range + scale_min;
        sp[i] = sig[i] * (static_cast<S>(1) - sig[i]) * range;  // ds/d(so)
        spp[i] = sig[i] * (static_cast<S>(1) - sig[i]) *
                 (static_cast<S>(1) - static_cast<S>(2) * sig[i]) * range;  // d2s/d(so)^2
    }

    // Rotation matrix (row-major)
    const S x2 = x*x, y2 = y*y, z2 = z*z;
    const S xy = x*y, xz = x*z, yz = y*z;
    const S wx = w*x, wy = w*y, wz = w*z;

    S R[9]; // R[row*3+col]
    R[0] = static_cast<S>(1) - static_cast<S>(2)*(y2+z2);
    R[1] = static_cast<S>(2)*(xy-wz);
    R[2] = static_cast<S>(2)*(xz+wy);
    R[3] = static_cast<S>(2)*(xy+wz);
    R[4] = static_cast<S>(1) - static_cast<S>(2)*(x2+z2);
    R[5] = static_cast<S>(2)*(yz-wx);
    R[6] = static_cast<S>(2)*(xz-wy);
    R[7] = static_cast<S>(2)*(yz+wx);
    R[8] = static_cast<S>(1) - static_cast<S>(2)*(x2+y2);

    // Downstream v_covars
    const S* vc = v_covars + idx * 6;

    // ==================================================================
    // Compute Jacobian J [6 x 7]
    // J[:, 0..3] = dC/dq_a   for a=0..3
    // J[:, 4..6] = dC/dso_m  for m=0..2
    // ==================================================================

    // dR/dq_a for each quaternion component
    S dRa[4][9]; // dRa[a][row*3+col]
    for (int a = 0; a < 4; ++a) {
        dR_dq(w, x, y, z, a, dRa[a]);
    }

    // J[k][a] = dC_k / dq_a, where k is the upper-tri index of covariance
    // C_ij = sum_m s_m^2 * R_im * R_jm
    // dC_ij/dq_a = sum_m s_m^2 * (dR_im/dq_a * R_jm + R_im * dR_jm/dq_a)
    //
    // We'll compute J in a flat array: J[k * 7 + col]
    // where k=0..5 (covar components), col=0..6 (q0..q3, so0..so2)

    S J[6 * 7]; // [6][7]
    for (int k = 0; k < 42; ++k) J[k] = static_cast<S>(0);

    // Upper-tri (i,j) pairs: (0,0),(0,1),(0,2),(1,1),(1,2),(2,2) -> k=0..5
    const int cov_i[6] = {0, 0, 0, 1, 1, 2};
    const int cov_j[6] = {0, 1, 2, 1, 2, 2};

    for (int k = 0; k < 6; ++k) {
        int ci = cov_i[k], cj = cov_j[k];
        // dC_ij/dq_a
        for (int a = 0; a < 4; ++a) {
            S val = static_cast<S>(0);
            for (int m = 0; m < 3; ++m) {
                val += s[m] * s[m] * (dRa[a][ci*3+m] * R[cj*3+m] + R[ci*3+m] * dRa[a][cj*3+m]);
            }
            J[k * 7 + a] = val;
        }
        // dC_ij/d(so_m)
        for (int m = 0; m < 3; ++m) {
            J[k * 7 + 4 + m] = static_cast<S>(2) * s[m] * R[ci*3+m] * R[cj*3+m] * sp[m];
        }
    }

    // ==================================================================
    // Compute first-order gradient: v_theta = J^T @ v_covars
    // ==================================================================
    S v_theta[7]; // [q0,q1,q2,q3,so0,so1,so2]
    for (int col = 0; col < 7; ++col) {
        S val = static_cast<S>(0);
        for (int k = 0; k < 6; ++k) {
            val += J[k * 7 + col] * vc[k];
        }
        v_theta[col] = val;
    }

    // Write first-order outputs
    S* vq_ptr = v_quats + idx * 4;
    S* vs_ptr = v_scale_ori + idx * 3;
    for (int i = 0; i < 4; ++i) vq_ptr[i] = v_theta[i];
    for (int i = 0; i < 3; ++i) vs_ptr[i] = v_theta[4 + i];

    // ==================================================================
    // Read downstream Hessian H_{covar,covar} [21 upper-tri of 6x6]
    // ==================================================================
    const S* hcc = h_covars + idx * 21;

    // Reconstruct full 6x6 symmetric matrix from upper-tri
    // H_full[i][j] where i,j in 0..5
    // We'll compute J^T @ H @ J directly without forming H_full explicitly.

    // ==================================================================
    // Compute H_{theta,theta} = J^T @ H_{cc} @ J + sum_k v_k * Hess_k
    // ==================================================================

    // Part 1: Gauss-Newton term: J^T @ H_{cc} @ J
    // First compute H @ J [6x7], then J^T @ (H@J) [7x7]

    // H_cc @ J [6 x 7]
    S HJ[6 * 7];
    for (int i = 0; i < 6; ++i) {
        for (int col = 0; col < 7; ++col) {
            S val = static_cast<S>(0);
            for (int j = 0; j < 6; ++j) {
                // Access H_cc[i][j] from upper-tri storage
                int ui = i, uj = j;
                if (ui > uj) { int t = ui; ui = uj; uj = t; }
                int hidx = ui * 6 - ui * (ui + 1) / 2 + uj;
                // ^^ This is utri_idx(i,j,6) but inlined
                val += hcc[hidx] * J[j * 7 + col];
            }
            HJ[i * 7 + col] = val;
        }
    }

    // J^T @ HJ [7 x 7]
    S JtHJ[7 * 7];
    for (int r = 0; r < 7; ++r) {
        for (int c = 0; c < 7; ++c) {
            S val = static_cast<S>(0);
            for (int k = 0; k < 6; ++k) {
                val += J[k * 7 + r] * HJ[k * 7 + c];
            }
            JtHJ[r * 7 + c] = val;
        }
    }

    // Part 2: Curvature term: sum_k v_k * Hess_k
    // Hess_k = d^2(covar_k) / d(theta)^2 [7x7]
    // This is symmetric, so we only need upper triangle.

    S curv[7 * 7]; // curvature contribution
    for (int i = 0; i < 49; ++i) curv[i] = static_cast<S>(0);

    // d2R/dq_a.dq_b
    S d2Rab[9];

    // For each covariance component k
    for (int k = 0; k < 6; ++k) {
        int ci = cov_i[k], cj = cov_j[k];
        S vk = vc[k];
        if (vk == static_cast<S>(0)) continue;

        // --- q-q block (4x4) ---
        for (int a = 0; a < 4; ++a) {
            for (int b = a; b < 4; ++b) {
                d2R_dq2(a, b, d2Rab);
                S val = static_cast<S>(0);
                for (int m = 0; m < 3; ++m) {
                    val += s[m] * s[m] * (
                        d2Rab[ci*3+m] * R[cj*3+m] +
                        dRa[a][ci*3+m] * dRa[b][cj*3+m] +
                        dRa[b][ci*3+m] * dRa[a][cj*3+m] +
                        R[ci*3+m] * d2Rab[cj*3+m]
                    );
                }
                curv[a * 7 + b] += vk * val;
                if (a != b) {
                    curv[b * 7 + a] += vk * val; // symmetry
                }
            }
        }

        // --- so-so block (3x3, diagonal only) ---
        for (int m = 0; m < 3; ++m) {
            // d2C_ij / d(so_m)^2 = 2 * R_im * R_jm * (sp_m^2 + s_m * spp_m)
            S val = static_cast<S>(2) * R[ci*3+m] * R[cj*3+m] * (sp[m]*sp[m] + s[m]*spp[m]);
            curv[(4+m) * 7 + (4+m)] += vk * val;
        }

        // --- q-so cross block (4x3) ---
        for (int a = 0; a < 4; ++a) {
            for (int m = 0; m < 3; ++m) {
                // d2C_ij / d(q_a) d(so_m) = 2*s_m*sp_m * (dR_im/dq_a * R_jm + R_im * dR_jm/dq_a)
                S val = static_cast<S>(2) * s[m] * sp[m] * (
                    dRa[a][ci*3+m] * R[cj*3+m] + R[ci*3+m] * dRa[a][cj*3+m]
                );
                curv[a * 7 + (4+m)] += vk * val;
                curv[(4+m) * 7 + a] += vk * val; // symmetry
            }
        }
    }

    // Combine: H_theta = JtHJ + curv
    S H_theta[7 * 7];
    for (int i = 0; i < 49; ++i) H_theta[i] = JtHJ[i] + curv[i];

    // ==================================================================
    // Write theta-theta Hessian outputs
    // ==================================================================

    // h_quats: upper-tri of 4x4 = 10 elements
    S* hq_ptr = h_quats + idx * 10;
    {
        int out_idx = 0;
        for (int i = 0; i < 4; ++i) {
            for (int j = i; j < 4; ++j) {
                hq_ptr[out_idx++] = H_theta[i * 7 + j];
            }
        }
    }

    // h_scale_ori: upper-tri of so-so block = 6 elements
    S* hs_ptr = h_scale_ori + idx * 6;
    {
        int out_idx = 0;
        for (int i = 0; i < 3; ++i) {
            for (int j = i; j < 3; ++j) {
                hs_ptr[out_idx++] = H_theta[(4+i) * 7 + (4+j)];
            }
        }
    }

    // h_quats_scale_ori: q-so cross block 4x3 row-major = 12 elements
    S* hqs_ptr = h_quats_scale_ori + idx * 12;
    for (int a = 0; a < 4; ++a) {
        for (int m = 0; m < 3; ++m) {
            hqs_ptr[a * 3 + m] = H_theta[a * 7 + (4+m)];
        }
    }

    // ==================================================================
    // Cross-Hessian: H_{theta, psi} = J^T @ H_{covar, psi}
    // ==================================================================

    // --- theta-mean cross: J^T @ h_covars_means [7x3] -> extract q[4x3], so[3x3] ---
    const S* hcm_ptr = h_covars_means + idx * 18; // [6x3 row-major]

    S JtHcm[7 * 3]; // [7x3]
    for (int r = 0; r < 7; ++r) {
        for (int c = 0; c < 3; ++c) {
            S val = static_cast<S>(0);
            for (int k = 0; k < 6; ++k) {
                val += J[k * 7 + r] * hcm_ptr[k * 3 + c];
            }
            JtHcm[r * 3 + c] = val;
        }
    }

    // h_quats_means [N, 12] = 4x3 row-major
    S* hqm_ptr = h_quats_means + idx * 12;
    for (int a = 0; a < 4; ++a) {
        for (int c = 0; c < 3; ++c) {
            hqm_ptr[a * 3 + c] = JtHcm[a * 3 + c];
        }
    }

    // h_scale_ori_means [N, 9] = 3x3 row-major
    S* hsm_ptr = h_scale_ori_means + idx * 9;
    for (int m = 0; m < 3; ++m) {
        for (int c = 0; c < 3; ++c) {
            hsm_ptr[m * 3 + c] = JtHcm[(4+m) * 3 + c];
        }
    }

    // --- theta-opacity cross: J^T @ h_covars_opacity [7x1] -> extract q[4], so[3] ---
    const S* hco_ptr = h_covars_opacity + idx * 6; // [6]

    // h_quats_opacity [N, 4]
    S* hqo_ptr = h_quats_opacity + idx * 4;
    for (int a = 0; a < 4; ++a) {
        S val = static_cast<S>(0);
        for (int k = 0; k < 6; ++k) {
            val += J[k * 7 + a] * hco_ptr[k];
        }
        hqo_ptr[a] = val;
    }

    // h_scale_ori_opacity [N, 3]
    S* hso_ptr = h_scale_ori_opacity + idx * 3;
    for (int m = 0; m < 3; ++m) {
        S val = static_cast<S>(0);
        for (int k = 0; k < 6; ++k) {
            val += J[k * 7 + (4+m)] * hco_ptr[k];
        }
        hso_ptr[m] = val;
    }
}

// ============================================================================
// Launch Function Implementation
// ============================================================================

template <typename S>
void launch_parameter_transform_bwd2(
    const S* __restrict__ quats,
    const S* __restrict__ scale_ori,
    S scale_min,
    S scale_max,
    const S* __restrict__ v_covars,
    const S* __restrict__ h_covars,
    const S* __restrict__ h_covars_means,
    const S* __restrict__ h_covars_opacity,
    S* __restrict__ v_quats,
    S* __restrict__ v_scale_ori,
    S* __restrict__ h_quats,
    S* __restrict__ h_scale_ori,
    S* __restrict__ h_quats_scale_ori,
    S* __restrict__ h_quats_means,
    S* __restrict__ h_quats_opacity,
    S* __restrict__ h_scale_ori_means,
    S* __restrict__ h_scale_ori_opacity,
    uint32_t N,
    cudaStream_t stream
) {
    if (N == 0) return;
    const int threads = GSPLAT2CT_N_THREADS;
    const int blocks = (N + threads - 1) / threads;

    parameter_transform_bwd2_kernel<S><<<blocks, threads, 0, stream>>>(
        quats, scale_ori, scale_min, scale_max,
        v_covars, h_covars, h_covars_means, h_covars_opacity,
        v_quats, v_scale_ori,
        h_quats, h_scale_ori, h_quats_scale_ori,
        h_quats_means, h_quats_opacity,
        h_scale_ori_means, h_scale_ori_opacity,
        N
    );
}

// Explicit template instantiations
template void launch_parameter_transform_bwd2<float>(
    const float*, const float*, float, float,
    const float*, const float*, const float*, const float*,
    float*, float*,
    float*, float*, float*,
    float*, float*, float*, float*,
    uint32_t, cudaStream_t
);
template void launch_parameter_transform_bwd2<double>(
    const double*, const double*, double, double,
    const double*, const double*, const double*, const double*,
    double*, double*,
    double*, double*, double*,
    double*, double*, double*, double*,
    uint32_t, cudaStream_t
);

} // namespace gsplat2ct
