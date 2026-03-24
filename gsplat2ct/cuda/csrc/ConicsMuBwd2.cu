/**
 * @file ConicsMuBwd2.cu
 * @brief Second-order backward CUDA kernel for ConicsMu module.
 *
 * Given the downstream first-order gradients and second-order Hessians
 * w.r.t. (conics, mu, means2d, opacity), computes the upstream gradients
 * and Hessians w.r.t. cov2d_3x3 (the 6 independent elements of the
 * projected 3x3 covariance), as well as cross-Hessians with means2d
 * and opacity.
 *
 * =========================================================================
 * Notation
 * =========================================================================
 *
 * Input parameters (theta):
 *   sigma = (sigma_0, sigma_1, sigma_2, sigma_3, sigma_4, sigma_5)
 *         = (a, b, c, d, e, f)  -- upper-triangular storage of tilde{Sigma}
 *
 * Output parameters (phi):
 *   phi = (u11, u12, u22, mu)  -- conics (3) + compensation factor (1)
 *
 * External parameters (psi), not transformed by this module:
 *   psi = (means2d_x, means2d_y, opacity)
 *
 * =========================================================================
 * Core Formulas
 * =========================================================================
 *
 * 1. First-order gradient (same as bwd1):
 *    v_sigma = J^T @ v_phi
 *
 * 2. Second-order Hessian (theta-theta):
 *    H_{theta,theta} = J^T @ H_{phi,phi} @ J
 *                     + sum_k v_phi_k * Hess_k
 *    where Hess_k = d^2(phi_k) / d(theta)^2
 *
 * 3. Cross-Hessian (theta-psi):
 *    H_{theta,psi} = J^T @ H_{phi,psi}
 *
 * =========================================================================
 * Jacobian J (4 x 6) structure
 * =========================================================================
 *
 * Rows: (u11, u12, u22, mu)
 * Cols: (a, b, c, d, e, f) = (sigma_0 ... sigma_5)
 *
 * Conics rows (only cols 0, 1, 3 are nonzero):
 *   du11/da = -u11^2           du11/db = -2*u11*u12              du11/dd = -u12^2
 *   du12/da = -u11*u12         du12/db = -(u11*u22 + u12^2)     du12/dd = -u12*u22
 *   du22/da = -u12^2           du22/db = -2*u12*u22              du22/dd = -u22^2
 *
 * Mu row (all 6 cols):
 *   dmu/d(sigma_i) = (mu/2) * r_i
 *   where r = (q00 - u11, 2(q01 - u12), 2*q02, q11 - u22, 2*q12, q22)
 *
 * =========================================================================
 * Hessians of each output (Hess_k, 6x6 each)
 * =========================================================================
 *
 * For conics (u11, u12, u22): Hess is nonzero only in the (a,b,d) sub-block
 * i.e., rows/cols {0,1,3}. These are the second derivatives of 2x2 matrix
 * inverse elements w.r.t. the input matrix elements.
 *
 * Using the formula: d^2(B_kl)/d(A_ij)d(A_mn) involves products of 3 elements
 * of B = A^{-1}. Specifically, from the chain rule on dB/dA = -B (dA) B:
 *
 * === Hessian of u11 (= b_11 of 2x2 inverse) ===
 * d2u11/da^2       = 2*u11^3
 * d2u11/da.db      = 4*u11^2*u12
 * d2u11/da.dd      = 2*u11*u12^2
 * d2u11/db^2       = 2*u11^2*u22 + 6*u11*u12^2    (*)
 * d2u11/db.dd      = 2*u12^3 + 2*u11*u12*u22
 * d2u11/dd^2       = 2*u12^2*u22
 *
 * === Hessian of u12 (= b_12 of 2x2 inverse) ===
 * d2u12/da^2       = 2*u11^2*u12
 * d2u12/da.db      = u11^2*u22 + 3*u11*u12^2      (*)
 * d2u12/da.dd      = u12^3 + u11*u12*u22
 * d2u12/db^2       = 2*u12^3 + 6*u11*u12*u22      (*)
 * d2u12/db.dd      = u11*u22^2 + 3*u12^2*u22      (*)
 * d2u12/dd^2       = 2*u12*u22^2
 *
 * === Hessian of u22 (= b_22 of 2x2 inverse) ===
 * d2u22/da^2       = 2*u11*u12^2
 * d2u22/da.db      = 2*u12^3 + 2*u11*u12*u22
 * d2u22/da.dd      = 2*u12^2*u22
 * d2u22/db^2       = 2*u11*u22^2 + 6*u12^2*u22    (*)
 * d2u22/db.dd      = 4*u12*u22^2
 * d2u22/dd^2       = 2*u22^3
 *
 * (*) marks entries where the off-diagonal factor-of-2 produces non-obvious
 *     coefficients. These come from the full derivation accounting for the
 *     symmetric matrix parameterization (b = sigma_1 appears twice in the
 *     2x2 matrix as off-diagonal).
 *
 * === Hessian of mu (full 6x6) ===
 *
 * From dmu/d(sigma_i) = (mu/2) * r_i, we get:
 *
 *   d2mu/d(sigma_i)d(sigma_j) = (1/2) * dmu/d(sigma_j) * r_i
 *                               + (mu/2) * dr_i/d(sigma_j)
 *                              = (mu/4) * r_i * r_j
 *                               + (mu/2) * dr_i/d(sigma_j)
 *
 * where dr_i/d(sigma_j) involves:
 *   - dq_{kl}/d(sigma_j): derivatives of 3x3 inverse elements, same formula
 *     as for 2x2 but generalized: dQ_kl/dS_ij = -Q_ki*Q_lj (diagonal)
 *     or -(Q_ki*Q_lj + Q_kj*Q_li) (off-diagonal, with factor 2 absorbed)
 *   - du_{kl}/d(sigma_j): derivatives of 2x2 inverse (conics Jacobian rows)
 *
 * =========================================================================
 *
 * Author: Dy164
 * Date: 2026-02-10
 */

#include "ConicsMu.h"
#include "../include/common.h"
#include <cuda_runtime.h>

namespace gsplat2ct {

// ============================================================================
// Second-Order Backward Kernel
// ============================================================================

template <typename S>
__global__ void conics_mu_bwd2_kernel(
    // ---- Forward inputs/outputs (for recomputation) ----
    const S* __restrict__ cov2d_3x3,   // [total, 6]  (a, b, c, d, e, f)
    const S eps,                       // blur epsilon
    const S* __restrict__ conics,      // [total, 3]  (u11, u12, u22)
    const S* __restrict__ mus,         // [total]
    // ---- Downstream first-order gradients ----
    const S* __restrict__ v_conics,    // [total, 3]  dL/d(u11, u12, u22)
    const S* __restrict__ v_mus,       // [total]     dL/d(mu)
    // ---- Downstream second-order: phi-phi block ----
    const S* __restrict__ h_conics,    // [total, 6]  H_{conics,conics} upper-tri (H00,H01,H02,H11,H12,H22)
    const S* __restrict__ h_conics_mu, // [total, 3]  H_{conics,mu} = (H_{u11,mu}, H_{u12,mu}, H_{u22,mu})
    const S* __restrict__ h_mu_mu,     // [total]     H_{mu,mu}
    // ---- Downstream second-order: phi-psi cross block ----
    const S* __restrict__ h_means2d_conics,  // [total, 6]  H_{means2d,conics} [2x3 row-major]
    const S* __restrict__ h_conics_opacity,  // [total, 3]  H_{conics,opacity}
    const S* __restrict__ h_mu_means2d,      // [total, 2]  H_{mu,means2d}
    const S* __restrict__ h_mu_opacity,      // [total]     H_{mu,opacity}
    // ---- Outputs: first-order gradient ----
    S* __restrict__ v_cov2d_3x3,      // [total, 6]  dL/d(sigma_0..5)
    // ---- Outputs: second-order theta-theta ----
    S* __restrict__ h_cov_cov,        // [total, 21] H_{sigma,sigma} upper-tri of 6x6
    // ---- Outputs: second-order theta-psi cross ----
    S* __restrict__ h_cov_means2d,    // [total, 12] H_{sigma,means2d} [6x2 row-major]
    S* __restrict__ h_cov_opacity,    // [total, 6]  H_{sigma,opacity}
    // ---- Dimensions ----
    uint32_t total
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // ====================================================================
    // Read forward outputs
    // ====================================================================
    const S u11 = conics[idx * 3 + 0];
    const S u12 = conics[idx * 3 + 1];
    const S u22 = conics[idx * 3 + 2];
    const S mu  = mus[idx];

    // Output pointers
    S* v_cov_ptr = v_cov2d_3x3 + idx * 6;
    S* h_cc_ptr  = h_cov_cov + idx * 21;
    S* h_cm_ptr  = h_cov_means2d + idx * 12;
    S* h_co_ptr  = h_cov_opacity + idx * 6;

    // Handle degenerate case
    if (u11 == static_cast<S>(0) && u12 == static_cast<S>(0) && u22 == static_cast<S>(0)) {
        for (int i = 0; i < 6; ++i) v_cov_ptr[i] = static_cast<S>(0);
        for (int i = 0; i < 21; ++i) h_cc_ptr[i] = static_cast<S>(0);
        for (int i = 0; i < 12; ++i) h_cm_ptr[i] = static_cast<S>(0);
        for (int i = 0; i < 6; ++i) h_co_ptr[i] = static_cast<S>(0);
        return;
    }

    // ====================================================================
    // Read downstream first-order gradients
    // ====================================================================
    const S vu11 = v_conics[idx * 3 + 0];
    const S vu12 = v_conics[idx * 3 + 1];
    const S vu22 = v_conics[idx * 3 + 2];
    const S vmu  = v_mus[idx];

    // ====================================================================
    // Read downstream phi-phi Hessian
    // ====================================================================
    // h_conics: upper-tri of 3x3 = [H_u11u11, H_u11u12, H_u11u22, H_u12u12, H_u12u22, H_u22u22]
    const S Hcc00 = h_conics[idx * 6 + 0];  // H_{u11,u11}
    const S Hcc01 = h_conics[idx * 6 + 1];  // H_{u11,u12}
    const S Hcc02 = h_conics[idx * 6 + 2];  // H_{u11,u22}
    const S Hcc11 = h_conics[idx * 6 + 3];  // H_{u12,u12}
    const S Hcc12 = h_conics[idx * 6 + 4];  // H_{u12,u22}
    const S Hcc22 = h_conics[idx * 6 + 5];  // H_{u22,u22}

    // h_conics_mu: [H_{u11,mu}, H_{u12,mu}, H_{u22,mu}]
    const S Hcm0 = h_conics_mu[idx * 3 + 0];
    const S Hcm1 = h_conics_mu[idx * 3 + 1];
    const S Hcm2 = h_conics_mu[idx * 3 + 2];

    // h_mu_mu: scalar
    const S Hmm = h_mu_mu[idx];

    // ====================================================================
    // Read downstream phi-psi cross Hessians
    // ====================================================================
    // h_means2d_conics: [2x3 row-major] = [H_{mx,u11}, H_{mx,u12}, H_{mx,u22}, H_{my,u11}, H_{my,u12}, H_{my,u22}]
    // Since H is symmetric, H_{mx,u11} = H_{u11,mx}, etc.
    const S Hcmx0 = h_means2d_conics[idx * 6 + 0];  // H_{mx,u11} = H_{u11,mx}
    const S Hcmx1 = h_means2d_conics[idx * 6 + 1];  // H_{mx,u12} = H_{u12,mx}
    const S Hcmx2 = h_means2d_conics[idx * 6 + 2];  // H_{mx,u22} = H_{u22,mx}
    const S Hcmy0 = h_means2d_conics[idx * 6 + 3];  // H_{my,u11} = H_{u11,my}
    const S Hcmy1 = h_means2d_conics[idx * 6 + 4];  // H_{my,u12} = H_{u12,my}
    const S Hcmy2 = h_means2d_conics[idx * 6 + 5];  // H_{my,u22} = H_{u22,my}

    // h_conics_opacity: [H_{u11,o}, H_{u12,o}, H_{u22,o}]
    const S Hco0 = h_conics_opacity[idx * 3 + 0];
    const S Hco1 = h_conics_opacity[idx * 3 + 1];
    const S Hco2 = h_conics_opacity[idx * 3 + 2];

    // h_mu_means2d: [H_{mu,mx}, H_{mu,my}]
    const S Hmmx = h_mu_means2d[idx * 2 + 0];
    const S Hmmy = h_mu_means2d[idx * 2 + 1];

    // h_mu_opacity: scalar
    const S Hmo = h_mu_opacity[idx];

    // ====================================================================
    // Compute Jacobian J (4 x 6)
    // ====================================================================
    // J[i][j] = d(phi_i)/d(sigma_j)
    //
    // Conics Jacobian (rows 0-2, nonzero only at cols 0, 1, 3):
    //   J[0][0] = -u11^2,         J[0][1] = -2*u11*u12,              J[0][3] = -u12^2
    //   J[1][0] = -u11*u12,       J[1][1] = -(u11*u22 + u12^2),     J[1][3] = -u12*u22
    //   J[2][0] = -u12^2,         J[2][1] = -2*u12*u22,              J[2][3] = -u22^2

    const S J00 = -u11 * u11;
    const S J01 = -static_cast<S>(2) * u11 * u12;
    const S J03 = -u12 * u12;

    const S J10 = -u11 * u12;
    const S J11 = -(u11 * u22 + u12 * u12);
    const S J13 = -u12 * u22;

    const S J20 = -u12 * u12;
    const S J21 = -static_cast<S>(2) * u12 * u22;
    const S J23 = -u22 * u22;

    // Mu Jacobian (row 3, all 6 cols):
    // Need Q = (tilde{Sigma}_b)^{-1}
    // Recompute from cov2d_3x3 + eps
    const S* cov = cov2d_3x3 + idx * 6;
    const S ab = cov[0] + eps;
    const S bb = cov[1];         // b (not blurred, off-diagonal)
    const S cc = cov[2];         // c
    const S db = cov[3] + eps;
    const S ee = cov[4];         // e
    const S fb = cov[5] + eps;

    const S D3b = ab * db * fb - ab * ee * ee - bb * bb * fb
                + static_cast<S>(2) * bb * cc * ee - cc * cc * db;

    // Q elements (3x3 inverse)
    S q00 = static_cast<S>(0), q01 = static_cast<S>(0), q02 = static_cast<S>(0);
    S q11 = static_cast<S>(0), q12 = static_cast<S>(0), q22 = static_cast<S>(0);

    const S validity_eps = static_cast<S>(1e-9);
    bool mu_valid = (mu > static_cast<S>(0)) && (D3b > validity_eps);

    if (mu_valid) {
        const S inv_D3b = static_cast<S>(1) / D3b;
        q00 = (db * fb - ee * ee)  * inv_D3b;
        q01 = (cc * ee - bb * fb)  * inv_D3b;
        q02 = (bb * ee - cc * db)  * inv_D3b;
        q11 = (ab * fb - cc * cc)  * inv_D3b;
        q12 = (bb * cc - ab * ee)  * inv_D3b;
        q22 = (ab * db - bb * bb)  * inv_D3b;
    }

    // r_i = components of (Q - pad(P)) with off-diagonal factor 2
    const S r0 = q00 - u11;
    const S r1 = static_cast<S>(2) * (q01 - u12);
    const S r2 = static_cast<S>(2) * q02;
    const S r3 = q11 - u22;
    const S r4 = static_cast<S>(2) * q12;
    const S r5 = q22;

    const S half_mu = static_cast<S>(0.5) * mu;

    // J[3][i] = (mu/2) * r_i, but zero if mu_valid is false
    const S J30 = mu_valid ? half_mu * r0 : static_cast<S>(0);
    const S J31 = mu_valid ? half_mu * r1 : static_cast<S>(0);
    const S J32 = mu_valid ? half_mu * r2 : static_cast<S>(0);
    const S J33 = mu_valid ? half_mu * r3 : static_cast<S>(0);
    const S J34 = mu_valid ? half_mu * r4 : static_cast<S>(0);
    const S J35 = mu_valid ? half_mu * r5 : static_cast<S>(0);

    // ====================================================================
    // Step 1: First-order gradient v_sigma = J^T @ v_phi
    // ====================================================================
    // v_phi = (vu11, vu12, vu22, vmu)
    // v_sigma[i] = J[0][i]*vu11 + J[1][i]*vu12 + J[2][i]*vu22 + J[3][i]*vmu

    v_cov_ptr[0] = J00 * vu11 + J10 * vu12 + J20 * vu22 + J30 * vmu;  // a
    v_cov_ptr[1] = J01 * vu11 + J11 * vu12 + J21 * vu22 + J31 * vmu;  // b
    v_cov_ptr[2] =                                          J32 * vmu;  // c
    v_cov_ptr[3] = J03 * vu11 + J13 * vu12 + J23 * vu22 + J33 * vmu;  // d
    v_cov_ptr[4] =                                          J34 * vmu;  // e
    v_cov_ptr[5] =                                          J35 * vmu;  // f

    // ====================================================================
    // Step 2: Compute J^T @ H_{phi,phi} @ J  (term1 of H_{theta,theta})
    // ====================================================================
    // H_{phi,phi} is 4x4 symmetric:
    //   [Hcc00 Hcc01 Hcc02 Hcm0]
    //   [Hcc01 Hcc11 Hcc12 Hcm1]
    //   [Hcc02 Hcc12 Hcc22 Hcm2]
    //   [Hcm0  Hcm1  Hcm2  Hmm ]
    //
    // We compute T = H_{phi,phi} @ J, then term1 = J^T @ T
    // T is 4x6, T[k][j] = sum_l H[k][l] * J[l][j]

    // T[k][j] for k=0..3, j=0..5
    // For efficiency, compute only the unique entries we need.
    // Since H_{theta,theta} is 6x6 symmetric with 21 independent elements,
    // and we need term1[i][j] = sum_k J[k][i] * T[k][j],
    // let's compute T row by row.

    // T[0][j] = Hcc00*J[0][j] + Hcc01*J[1][j] + Hcc02*J[2][j] + Hcm0*J[3][j]
    S T0[6];
    T0[0] = Hcc00 * J00 + Hcc01 * J10 + Hcc02 * J20 + Hcm0 * J30;
    T0[1] = Hcc00 * J01 + Hcc01 * J11 + Hcc02 * J21 + Hcm0 * J31;
    T0[2] =                                             Hcm0 * J32;
    T0[3] = Hcc00 * J03 + Hcc01 * J13 + Hcc02 * J23 + Hcm0 * J33;
    T0[4] =                                             Hcm0 * J34;
    T0[5] =                                             Hcm0 * J35;

    // T[1][j] = Hcc01*J[0][j] + Hcc11*J[1][j] + Hcc12*J[2][j] + Hcm1*J[3][j]
    S T1[6];
    T1[0] = Hcc01 * J00 + Hcc11 * J10 + Hcc12 * J20 + Hcm1 * J30;
    T1[1] = Hcc01 * J01 + Hcc11 * J11 + Hcc12 * J21 + Hcm1 * J31;
    T1[2] =                                             Hcm1 * J32;
    T1[3] = Hcc01 * J03 + Hcc11 * J13 + Hcc12 * J23 + Hcm1 * J33;
    T1[4] =                                             Hcm1 * J34;
    T1[5] =                                             Hcm1 * J35;

    // T[2][j] = Hcc02*J[0][j] + Hcc12*J[1][j] + Hcc22*J[2][j] + Hcm2*J[3][j]
    S T2[6];
    T2[0] = Hcc02 * J00 + Hcc12 * J10 + Hcc22 * J20 + Hcm2 * J30;
    T2[1] = Hcc02 * J01 + Hcc12 * J11 + Hcc22 * J21 + Hcm2 * J31;
    T2[2] =                                             Hcm2 * J32;
    T2[3] = Hcc02 * J03 + Hcc12 * J13 + Hcc22 * J23 + Hcm2 * J33;
    T2[4] =                                             Hcm2 * J34;
    T2[5] =                                             Hcm2 * J35;

    // T[3][j] = Hcm0*J[0][j] + Hcm1*J[1][j] + Hcm2*J[2][j] + Hmm*J[3][j]
    S T3[6];
    T3[0] = Hcm0 * J00 + Hcm1 * J10 + Hcm2 * J20 + Hmm * J30;
    T3[1] = Hcm0 * J01 + Hcm1 * J11 + Hcm2 * J21 + Hmm * J31;
    T3[2] =                                          Hmm * J32;
    T3[3] = Hcm0 * J03 + Hcm1 * J13 + Hcm2 * J23 + Hmm * J33;
    T3[4] =                                          Hmm * J34;
    T3[5] =                                          Hmm * J35;

    // term1[i][j] = J[0][i]*T0[j] + J[1][i]*T1[j] + J[2][i]*T2[j] + J[3][i]*T3[j]
    // We only compute the upper-triangular part (i <= j) in flat index order.
    // Flat storage of 6x6 upper triangle: 21 elements
    // Index mapping: (0,0)=0, (0,1)=1, (0,2)=2, (0,3)=3, (0,4)=4, (0,5)=5,
    //                (1,1)=6, (1,2)=7, (1,3)=8, (1,4)=9, (1,5)=10,
    //                (2,2)=11, (2,3)=12, (2,4)=13, (2,5)=14,
    //                (3,3)=15, (3,4)=16, (3,5)=17,
    //                (4,4)=18, (4,5)=19,
    //                (5,5)=20

    // For generic i: sum_k J[k][i] * T_k[j]
    // Let's define Jcol[i] = {J[0][i], J[1][i], J[2][i], J[3][i]}

    // Jcol0 = {J00, J10, J20, J30}  (sigma_0 = a)
    // Jcol1 = {J01, J11, J21, J31}  (sigma_1 = b)
    // Jcol2 = {0,   0,   0,   J32}  (sigma_2 = c)
    // Jcol3 = {J03, J13, J23, J33}  (sigma_3 = d)
    // Jcol4 = {0,   0,   0,   J34}  (sigma_4 = e)
    // Jcol5 = {0,   0,   0,   J35}  (sigma_5 = f)

    // term1[i][j] = Jcol_i . T_col_j  where T_col_j = (T0[j], T1[j], T2[j], T3[j])

    // For i in {2,4,5}: only J[3][i] is nonzero, so term1[i][j] = J[3][i] * T3[j]
    // For i in {0,1,3}: all 4 components

    S term1[21];

    // (0,0)
    term1[0]  = J00*T0[0] + J10*T1[0] + J20*T2[0] + J30*T3[0];
    // (0,1)
    term1[1]  = J00*T0[1] + J10*T1[1] + J20*T2[1] + J30*T3[1];
    // (0,2)
    term1[2]  = J00*T0[2] + J10*T1[2] + J20*T2[2] + J30*T3[2];
    // (0,3)
    term1[3]  = J00*T0[3] + J10*T1[3] + J20*T2[3] + J30*T3[3];
    // (0,4)
    term1[4]  = J00*T0[4] + J10*T1[4] + J20*T2[4] + J30*T3[4];
    // (0,5)
    term1[5]  = J00*T0[5] + J10*T1[5] + J20*T2[5] + J30*T3[5];
    // (1,1)
    term1[6]  = J01*T0[1] + J11*T1[1] + J21*T2[1] + J31*T3[1];
    // (1,2)
    term1[7]  = J01*T0[2] + J11*T1[2] + J21*T2[2] + J31*T3[2];
    // (1,3)
    term1[8]  = J01*T0[3] + J11*T1[3] + J21*T2[3] + J31*T3[3];
    // (1,4)
    term1[9]  = J01*T0[4] + J11*T1[4] + J21*T2[4] + J31*T3[4];
    // (1,5)
    term1[10] = J01*T0[5] + J11*T1[5] + J21*T2[5] + J31*T3[5];
    // (2,2)
    term1[11] = J32*T3[2];
    // (2,3)
    term1[12] = J32*T3[3];
    // (2,4)
    term1[13] = J32*T3[4];
    // (2,5)
    term1[14] = J32*T3[5];
    // (3,3)
    term1[15] = J03*T0[3] + J13*T1[3] + J23*T2[3] + J33*T3[3];
    // (3,4)
    term1[16] = J03*T0[4] + J13*T1[4] + J23*T2[4] + J33*T3[4];
    // (3,5)
    term1[17] = J03*T0[5] + J13*T1[5] + J23*T2[5] + J33*T3[5];
    // (4,4)
    term1[18] = J34*T3[4];
    // (4,5)
    term1[19] = J34*T3[5];
    // (5,5)
    term1[20] = J35*T3[5];

    // ====================================================================
    // Step 3: Compute term2 = sum_k v_phi_k * Hess_k
    // ====================================================================
    // Each Hess_k is 6x6 symmetric (21 independent elements).
    // We accumulate into term2[21].

    S term2[21];
    for (int i = 0; i < 21; ++i) term2[i] = static_cast<S>(0);

    // ------------------------------------------------------------------
    // 3a. Conics Hessians (u11, u12, u22)
    // ------------------------------------------------------------------
    // These are nonzero only in the {a,b,d} = {sigma_0, sigma_1, sigma_3} sub-block.
    // In the 21-element flat storage, the relevant indices are:
    //   (0,0)=0, (0,1)=1, (0,3)=3, (1,1)=6, (1,3)=8, (3,3)=15

    // Hessian of u11:
    {
        const S h2_00 = static_cast<S>(2) * u11 * u11 * u11;
        const S h2_01 = static_cast<S>(4) * u11 * u11 * u12;
        const S h2_03 = static_cast<S>(2) * u11 * u12 * u12;
        const S h2_11 = static_cast<S>(2) * u11 * u11 * u22 + static_cast<S>(6) * u11 * u12 * u12;
        const S h2_13 = static_cast<S>(2) * u12 * u12 * u12 + static_cast<S>(2) * u11 * u12 * u22;
        const S h2_33 = static_cast<S>(2) * u12 * u12 * u22;

        term2[0]  += vu11 * h2_00;
        term2[1]  += vu11 * h2_01;
        term2[3]  += vu11 * h2_03;
        term2[6]  += vu11 * h2_11;
        term2[8]  += vu11 * h2_13;
        term2[15] += vu11 * h2_33;
    }

    // Hessian of u12:
    {
        const S h2_00 = static_cast<S>(2) * u11 * u11 * u12;
        const S h2_01 = u11 * u11 * u22 + static_cast<S>(3) * u11 * u12 * u12;
        const S h2_03 = u12 * u12 * u12 + u11 * u12 * u22;
        const S h2_11 = static_cast<S>(2) * u12 * u12 * u12 + static_cast<S>(6) * u11 * u12 * u22;
        const S h2_13 = u11 * u22 * u22 + static_cast<S>(3) * u12 * u12 * u22;
        const S h2_33 = static_cast<S>(2) * u12 * u22 * u22;

        term2[0]  += vu12 * h2_00;
        term2[1]  += vu12 * h2_01;
        term2[3]  += vu12 * h2_03;
        term2[6]  += vu12 * h2_11;
        term2[8]  += vu12 * h2_13;
        term2[15] += vu12 * h2_33;
    }

    // Hessian of u22:
    {
        const S h2_00 = static_cast<S>(2) * u11 * u12 * u12;
        const S h2_01 = static_cast<S>(2) * u12 * u12 * u12 + static_cast<S>(2) * u11 * u12 * u22;
        const S h2_03 = static_cast<S>(2) * u12 * u12 * u22;
        const S h2_11 = static_cast<S>(2) * u11 * u22 * u22 + static_cast<S>(6) * u12 * u12 * u22;
        const S h2_13 = static_cast<S>(4) * u12 * u22 * u22;
        const S h2_33 = static_cast<S>(2) * u22 * u22 * u22;

        term2[0]  += vu22 * h2_00;
        term2[1]  += vu22 * h2_01;
        term2[3]  += vu22 * h2_03;
        term2[6]  += vu22 * h2_11;
        term2[8]  += vu22 * h2_13;
        term2[15] += vu22 * h2_33;
    }

    // ------------------------------------------------------------------
    // 3b. Mu Hessian (full 6x6)
    // ------------------------------------------------------------------
    // d2mu/d(sigma_i)d(sigma_j) = (mu/4) * r_i * r_j + (mu/2) * dr_i/d(sigma_j)
    //
    // dr_i/d(sigma_j) decomposes into:
    //   r_i = (Q_{kl} - pad(P)_{kl}) * factor_i
    //   where factor_i is 1 for diagonal, 2 for off-diagonal
    //
    // For Q derivatives: dQ_kl/d(sigma_j) follows the same formula as for P:
    //   For symmetric matrix, d(Q_kl)/d(sigma_j) = -Q_ki*Q_lj (diagonal sigma_j)
    //   or -Q_ki*Q_lj - Q_kj*Q_li (off-diagonal sigma_j, times appropriate factor)
    //
    // For P derivatives: du_kl/d(sigma_j), these are just the Jacobian entries
    //   (only nonzero for j in {0,1,3}).
    //
    // The full derivation for dr_i/d(sigma_j) is:
    //
    // r_0 = q00 - u11, dr_0/d(sigma_j):
    //   dq00/d(sigma_j) - du11/d(sigma_j)
    //
    // r_1 = 2*(q01 - u12), dr_1/d(sigma_j):
    //   2*(dq01/d(sigma_j) - du12/d(sigma_j))
    //
    // r_2 = 2*q02, dr_2/d(sigma_j): 2*dq02/d(sigma_j)
    //
    // r_3 = q11 - u22, dr_3/d(sigma_j): dq11/d(sigma_j) - du22/d(sigma_j)
    //
    // r_4 = 2*q12, dr_4/d(sigma_j): 2*dq12/d(sigma_j)
    //
    // r_5 = q22, dr_5/d(sigma_j): dq22/d(sigma_j)
    //
    // For Q (3x3 symmetric inverse), the derivatives dQ/d(sigma_j):
    //
    // sigma_0 = a (diagonal, affects position [0,0]):
    //   dq_kl/da = -q_k0 * q_l0
    // sigma_1 = b (off-diagonal, affects [0,1] and [1,0]):
    //   dq_kl/db = -(q_k0*q_l1 + q_k1*q_l0)
    // sigma_2 = c (off-diagonal, affects [0,2] and [2,0]):
    //   dq_kl/dc = -(q_k0*q_l2 + q_k2*q_l0)
    // sigma_3 = d (diagonal, affects [1,1]):
    //   dq_kl/dd = -q_k1 * q_l1
    // sigma_4 = e (off-diagonal, affects [1,2] and [2,1]):
    //   dq_kl/de = -(q_k1*q_l2 + q_k2*q_l1)
    // sigma_5 = f (diagonal, affects [2,2]):
    //   dq_kl/df = -q_k2 * q_l2
    //
    // For P (2x2 symmetric inverse, conics), du/d(sigma_j):
    //   du_kl/da = -u_k0*u_l0     (J row k,l at col 0)
    //   du_kl/db = -(u_k0*u_l1 + u_k1*u_l0)  (J row k,l at col 1)
    //   du_kl/dd = -u_k1*u_l1     (J row k,l at col 3)
    //   du_kl/dc = du_kl/de = du_kl/df = 0

    if (mu_valid) {
        const S qmu = static_cast<S>(0.25) * mu;  // mu/4
        const S hmu = static_cast<S>(0.5)  * mu;  // mu/2

        // Precompute all dr_i/d(sigma_j) values
        // We have 6 r values and 6 sigma values = 36 entries, but many simplify.
        //
        // dq_{kl}/d(sigma_j) values (symmetric in k,l):

        // --- dQ/d(sigma_0=a): dq_kl/da = -q_k0*q_l0 ---
        const S dq00_da = -q00 * q00;
        const S dq01_da = -q00 * q01;
        const S dq02_da = -q00 * q02;
        const S dq11_da = -q01 * q01;
        const S dq12_da = -q01 * q02;
        const S dq22_da = -q02 * q02;

        // --- dQ/d(sigma_1=b): dq_kl/db = -(q_k0*q_l1 + q_k1*q_l0) ---
        const S dq00_db = -static_cast<S>(2) * q00 * q01;
        const S dq01_db = -(q00 * q11 + q01 * q01);
        const S dq02_db = -(q00 * q12 + q02 * q01);
        const S dq11_db = -static_cast<S>(2) * q01 * q11;
        const S dq12_db = -(q01 * q12 + q02 * q11);
        const S dq22_db = -static_cast<S>(2) * q02 * q12;

        // --- dQ/d(sigma_2=c): dq_kl/dc = -(q_k0*q_l2 + q_k2*q_l0) ---
        const S dq00_dc = -static_cast<S>(2) * q00 * q02;
        const S dq01_dc = -(q00 * q12 + q02 * q01);
        const S dq02_dc = -(q00 * q22 + q02 * q02);
        const S dq11_dc = -static_cast<S>(2) * q01 * q12;
        const S dq12_dc = -(q01 * q22 + q02 * q12);
        const S dq22_dc = -static_cast<S>(2) * q02 * q22;

        // --- dQ/d(sigma_3=d): dq_kl/dd = -q_k1*q_l1 ---
        const S dq00_dd = -q01 * q01;
        const S dq01_dd = -q01 * q11;
        const S dq02_dd = -q01 * q12;
        const S dq11_dd = -q11 * q11;
        const S dq12_dd = -q11 * q12;
        const S dq22_dd = -q12 * q12;

        // --- dQ/d(sigma_4=e): dq_kl/de = -(q_k1*q_l2 + q_k2*q_l1) ---
        const S dq00_de = -static_cast<S>(2) * q01 * q02;
        const S dq01_de = -(q01 * q12 + q02 * q11);
        const S dq02_de = -(q01 * q22 + q02 * q12);
        const S dq11_de = -static_cast<S>(2) * q11 * q12;
        const S dq12_de = -(q11 * q22 + q12 * q12);
        const S dq22_de = -static_cast<S>(2) * q12 * q22;

        // --- dQ/d(sigma_5=f): dq_kl/df = -q_k2*q_l2 ---
        const S dq00_df = -q02 * q02;
        const S dq01_df = -q02 * q12;
        const S dq02_df = -q02 * q22;
        const S dq11_df = -q12 * q12;
        const S dq12_df = -q12 * q22;
        const S dq22_df = -q22 * q22;

        // du/d(sigma_j): from the conics Jacobian
        // du11/da = J00 = -u11^2,  du11/db = J01/1? No. J01 already has the factor.
        // Wait — the Jacobian rows are:
        //   J[0][j] = du11/d(sigma_j), J[1][j] = du12/d(sigma_j), J[2][j] = du22/d(sigma_j)
        // So:
        //   du11/da = J00, du11/db = J01, du11/dd = J03
        //   du12/da = J10, du12/db = J11, du12/dd = J13
        //   du22/da = J20, du22/db = J21, du22/dd = J23
        //   All others = 0.

        // Now compute dr_i/d(sigma_j) for all 36 combinations:
        //
        // dr_0/d(sigma_j) = dq00/d(sigma_j) - du11/d(sigma_j)
        const S dr0_da = dq00_da - J00;  // dq00/da - du11/da
        const S dr0_db = dq00_db - J01;
        const S dr0_dc = dq00_dc;
        const S dr0_dd = dq00_dd - J03;
        const S dr0_de = dq00_de;
        const S dr0_df = dq00_df;

        // dr_1/d(sigma_j) = 2 * (dq01/d(sigma_j) - du12/d(sigma_j))
        const S dr1_da = static_cast<S>(2) * (dq01_da - J10);
        const S dr1_db = static_cast<S>(2) * (dq01_db - J11);
        const S dr1_dc = static_cast<S>(2) * dq01_dc;
        const S dr1_dd = static_cast<S>(2) * (dq01_dd - J13);
        const S dr1_de = static_cast<S>(2) * dq01_de;
        const S dr1_df = static_cast<S>(2) * dq01_df;

        // dr_2/d(sigma_j) = 2 * dq02/d(sigma_j)
        const S dr2_da = static_cast<S>(2) * dq02_da;
        const S dr2_db = static_cast<S>(2) * dq02_db;
        const S dr2_dc = static_cast<S>(2) * dq02_dc;
        const S dr2_dd = static_cast<S>(2) * dq02_dd;
        const S dr2_de = static_cast<S>(2) * dq02_de;
        const S dr2_df = static_cast<S>(2) * dq02_df;

        // dr_3/d(sigma_j) = dq11/d(sigma_j) - du22/d(sigma_j)
        const S dr3_da = dq11_da - J20;
        const S dr3_db = dq11_db - J21;
        const S dr3_dc = dq11_dc;
        const S dr3_dd = dq11_dd - J23;
        const S dr3_de = dq11_de;
        const S dr3_df = dq11_df;

        // dr_4/d(sigma_j) = 2 * dq12/d(sigma_j)
        const S dr4_da = static_cast<S>(2) * dq12_da;
        const S dr4_db = static_cast<S>(2) * dq12_db;
        const S dr4_dc = static_cast<S>(2) * dq12_dc;
        const S dr4_dd = static_cast<S>(2) * dq12_dd;
        const S dr4_de = static_cast<S>(2) * dq12_de;
        const S dr4_df = static_cast<S>(2) * dq12_df;

        // dr_5/d(sigma_j) = dq22/d(sigma_j)
        const S dr5_da = dq22_da;
        const S dr5_db = dq22_db;
        const S dr5_dc = dq22_dc;
        const S dr5_dd = dq22_dd;
        const S dr5_de = dq22_de;
        const S dr5_df = dq22_df;

        // Now assemble mu Hessian contributions:
        // d2mu/d(sigma_i)d(sigma_j) = (mu/4)*r_i*r_j + (mu/2)*dr_i/d(sigma_j)
        //
        // Pack dr[i][j] into an array for convenience:
        // dr[6][6] where dr[i][j] = dr_i / d(sigma_j)
        const S dr[6][6] = {
            {dr0_da, dr0_db, dr0_dc, dr0_dd, dr0_de, dr0_df},
            {dr1_da, dr1_db, dr1_dc, dr1_dd, dr1_de, dr1_df},
            {dr2_da, dr2_db, dr2_dc, dr2_dd, dr2_de, dr2_df},
            {dr3_da, dr3_db, dr3_dc, dr3_dd, dr3_de, dr3_df},
            {dr4_da, dr4_db, dr4_dc, dr4_dd, dr4_de, dr4_df},
            {dr5_da, dr5_db, dr5_dc, dr5_dd, dr5_de, dr5_df},
        };

        const S rv[6] = {r0, r1, r2, r3, r4, r5};

        // Upper-triangle (i <= j) flat index
        int flat = 0;
        #pragma unroll
        for (int i = 0; i < 6; ++i) {
            #pragma unroll
            for (int j = i; j < 6; ++j) {
                // d2mu/d(sigma_i)d(sigma_j) = (mu/4)*r_i*r_j + (mu/2)*dr_i/d(sigma_j)
                // Note: dr_i/d(sigma_j) should equal dr_j/d(sigma_i) since Hessian is symmetric
                // We use dr[i][j] directly.
                const S h_mu_ij = qmu * rv[i] * rv[j] + hmu * dr[i][j];
                term2[flat] += vmu * h_mu_ij;
                ++flat;
            }
        }
    }

    // ====================================================================
    // Step 4: Combine H_{theta,theta} = term1 + term2
    // ====================================================================
    #pragma unroll
    for (int i = 0; i < 21; ++i) {
        h_cc_ptr[i] = term1[i] + term2[i];
    }

    // ====================================================================
    // Step 5: Cross-Hessian H_{theta,psi} = J^T @ H_{phi,psi}
    // ====================================================================
    // H_{phi,psi} is 4x3 matrix:
    //   col 0 = means2d_x: (Hcmx0, Hcmx1, Hcmx2, Hmmx)
    //   col 1 = means2d_y: (Hcmy0, Hcmy1, Hcmy2, Hmmy)
    //   col 2 = opacity:   (Hco0,  Hco1,  Hco2,  Hmo)
    //
    // H_{theta,psi}[i][j] = sum_k J[k][i] * H_{phi,psi}[k][j]

    // H_{theta,means2d}: [6 x 2] stored as 12 elements row-major
    // col 0 (mx): hpm = (Hcmx0, Hcmx1, Hcmx2, Hmmx)
    // col 1 (my): hpm = (Hcmy0, Hcmy1, Hcmy2, Hmmy)

    // Row i=0 (sigma_0 = a): Jcol0 = (J00, J10, J20, J30)
    h_cm_ptr[0]  = J00*Hcmx0 + J10*Hcmx1 + J20*Hcmx2 + J30*Hmmx;  // (a, mx)
    h_cm_ptr[1]  = J00*Hcmy0 + J10*Hcmy1 + J20*Hcmy2 + J30*Hmmy;  // (a, my)

    // Row i=1 (sigma_1 = b): Jcol1 = (J01, J11, J21, J31)
    h_cm_ptr[2]  = J01*Hcmx0 + J11*Hcmx1 + J21*Hcmx2 + J31*Hmmx;  // (b, mx)
    h_cm_ptr[3]  = J01*Hcmy0 + J11*Hcmy1 + J21*Hcmy2 + J31*Hmmy;  // (b, my)

    // Row i=2 (sigma_2 = c): Jcol2 = (0, 0, 0, J32)
    h_cm_ptr[4]  = J32*Hmmx;  // (c, mx)
    h_cm_ptr[5]  = J32*Hmmy;  // (c, my)

    // Row i=3 (sigma_3 = d): Jcol3 = (J03, J13, J23, J33)
    h_cm_ptr[6]  = J03*Hcmx0 + J13*Hcmx1 + J23*Hcmx2 + J33*Hmmx;  // (d, mx)
    h_cm_ptr[7]  = J03*Hcmy0 + J13*Hcmy1 + J23*Hcmy2 + J33*Hmmy;  // (d, my)

    // Row i=4 (sigma_4 = e): Jcol4 = (0, 0, 0, J34)
    h_cm_ptr[8]  = J34*Hmmx;  // (e, mx)
    h_cm_ptr[9]  = J34*Hmmy;  // (e, my)

    // Row i=5 (sigma_5 = f): Jcol5 = (0, 0, 0, J35)
    h_cm_ptr[10] = J35*Hmmx;  // (f, mx)
    h_cm_ptr[11] = J35*Hmmy;  // (f, my)

    // H_{theta,opacity}: [6] = J^T @ (Hco0, Hco1, Hco2, Hmo)
    h_co_ptr[0] = J00*Hco0 + J10*Hco1 + J20*Hco2 + J30*Hmo;  // a
    h_co_ptr[1] = J01*Hco0 + J11*Hco1 + J21*Hco2 + J31*Hmo;  // b
    h_co_ptr[2] =                                    J32*Hmo;  // c
    h_co_ptr[3] = J03*Hco0 + J13*Hco1 + J23*Hco2 + J33*Hmo;  // d
    h_co_ptr[4] =                                    J34*Hmo;  // e
    h_co_ptr[5] =                                    J35*Hmo;  // f
}

// ============================================================================
// Launch Function Implementation
// ============================================================================

template <typename S>
void launch_conics_mu_bwd2(
    const S* __restrict__ cov2d_3x3,
    S eps,
    const S* __restrict__ conics,
    const S* __restrict__ mus,
    const S* __restrict__ v_conics,
    const S* __restrict__ v_mus,
    const S* __restrict__ h_conics,
    const S* __restrict__ h_conics_mu,
    const S* __restrict__ h_mu_mu,
    const S* __restrict__ h_means2d_conics,
    const S* __restrict__ h_conics_opacity,
    const S* __restrict__ h_mu_means2d,
    const S* __restrict__ h_mu_opacity,
    S* __restrict__ v_cov2d_3x3,
    S* __restrict__ h_cov_cov,
    S* __restrict__ h_cov_means2d,
    S* __restrict__ h_cov_opacity,
    uint32_t total,
    cudaStream_t stream
) {
    if (total == 0) return;
    const int threads = GSPLAT2CT_N_THREADS;
    const int blocks = (total + threads - 1) / threads;

    conics_mu_bwd2_kernel<S><<<blocks, threads, 0, stream>>>(
        cov2d_3x3, eps, conics, mus,
        v_conics, v_mus,
        h_conics, h_conics_mu, h_mu_mu,
        h_means2d_conics, h_conics_opacity, h_mu_means2d, h_mu_opacity,
        v_cov2d_3x3,
        h_cov_cov,
        h_cov_means2d, h_cov_opacity,
        total
    );
}

// Explicit template instantiations
template void launch_conics_mu_bwd2<float>(
    const float*, float, const float*, const float*,
    const float*, const float*,
    const float*, const float*, const float*,
    const float*, const float*, const float*, const float*,
    float*, float*, float*, float*,
    uint32_t, cudaStream_t
);

template void launch_conics_mu_bwd2<double>(
    const double*, double, const double*, const double*,
    const double*, const double*,
    const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    double*, double*, double*, double*,
    uint32_t, cudaStream_t
);

} // namespace gsplat2ct
