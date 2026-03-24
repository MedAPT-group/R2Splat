/**
 * @file ConicsMu.h
 * @brief CUDA Kernel Launch Functions for Conics & Mu Computation (with Blur)
 * 
 * This header declares the launch functions that bridge between
 * C++ tensor processing (ConicsMu.cpp) and CUDA kernels
 * (ConicsMuFwd.cu, ConicsMuBwd.cu).
 * 
 * The module computes, from the projected 3x3 covariance matrix
 * (tilde{Sigma}), with optional blur (eps * I), the inverse of the
 * blurred upper-left 2x2 sub-block and the compensation factor mu.
 * 
 * Mathematical notation (with blur):
 *   tilde{Sigma} = [a, b, c; b, d, e; c, e, f]   (symmetric 3x3)
 *   hat{Sigma}   = [a, b; b, d]                    (upper-left 2x2)
 *   hat{Sigma}_b = hat{Sigma} + eps * I_2           (blurred 2x2)
 *   tilde{Sigma}_b = tilde{Sigma} + eps * I_3       (blurred 3x3)
 *   D2b = det(hat{Sigma}_b) = (a+eps)(d+eps) - b^2
 *   D3b = det(tilde{Sigma}_b)
 *   conics = hat{Sigma}_b^{-1} stored as (u11, u12, u22)
 *   mu = sqrt(2*pi * D3b / D2b)
 * 
 * The backward pass recomputes cov3d_inv = tilde{Sigma}_b^{-1} locally
 * from the original cov2d_3x3 and eps, avoiding the need to store it
 * between forward and backward (saving global memory bandwidth and
 * tensor allocation in the forward pass).
 * 
 * Input layout (upper-triangular storage):
 *   cov2d_3x3[..., 6] = (a, b, c, d, e, f)
 * 
 * Per-Gaussian parallelism: each thread processes one Gaussian.
 * 
 * Author: Dy164
 * Date: 2026-02-10
 */

#ifndef GSPLAT2CT_CONICS_MU_H
#define GSPLAT2CT_CONICS_MU_H

#include "../include/common.h"
#include <cuda_runtime.h>

namespace gsplat2ct {

// ============================================================================
// Forward Pass Launch Function
// ============================================================================

/**
 * Launch kernel to compute conics and mu from projected 3x3 covariance
 * with blur (eps * I) applied before inversion.
 * 
 * @tparam S Scalar type (float, double)
 * 
 * @param cov2d_3x3   Projected 3x3 covariance matrices [C*N, 6] upper-triangular
 *                     stored as (a, b, c, d, e, f)
 * @param eps          Blur epsilon added to diagonal before inversion
 * @param conics       Output inverse of blurred upper-left 2x2 [C*N, 3]
 *                     as (u11, u12, u22)
 * @param mus          Output compensation factor [C*N]
 *                     mu = sqrt(2*pi * det(Sigma_b) / det(hat{Sigma}_b))
 * @param total        Total number of elements (C * N)
 * @param stream       CUDA stream
 */
template <typename S>
void launch_conics_mu_fwd(
    const S* __restrict__ cov2d_3x3,   // [C*N, 6]
    S eps,
    S* __restrict__ conics,            // [C*N, 3]
    S* __restrict__ mus,               // [C*N]
    uint32_t total,
    cudaStream_t stream
);

// ============================================================================
// First-Order Backward Pass Launch Function
// ============================================================================

/**
 * Launch kernel for conics & mu backward pass (gradients only).
 * 
 * Recomputes cov3d_inv = (tilde{Sigma} + eps*I_3)^{-1} locally in each
 * thread from cov2d_3x3 and eps, then uses the efficient formulation:
 *   dL/d(cov2d_3x3) via conics and cov3d_inv.
 * 
 * Key formulas:
 *   d(conics)/d(Sigma) uses:  dP_ij/dS_kl = -P_ik * P_lj
 *   d(mu)/d(Sigma) = (mu/2) * (cov3d_inv - pad(conics))
 *   (with appropriate scaling for off-diagonal elements)
 * 
 * @tparam S Scalar type (float, double)
 * 
 * @param cov2d_3x3       Forward input: projected covariance [C*N, 6]
 * @param eps              Blur epsilon (same value used in forward)
 * @param conics           Forward output: hat{Sigma}_b^{-1} [C*N, 3] = (u11, u12, u22)
 * @param mus              Forward output: compensation factor [C*N]
 * @param v_conics         Downstream gradient dL/d(conics) [C*N, 3]
 * @param v_mus            Downstream gradient dL/d(mu) [C*N]
 * @param v_cov2d_3x3     Output gradient dL/d(cov2d_3x3) [C*N, 6]
 * @param total            Total number of elements (C * N)
 * @param stream           CUDA stream
 */
template <typename S>
void launch_conics_mu_bwd(
    const S* __restrict__ cov2d_3x3,   // [C*N, 6]
    S eps,
    const S* __restrict__ conics,      // [C*N, 3]
    const S* __restrict__ mus,         // [C*N]
    const S* __restrict__ v_conics,    // [C*N, 3]
    const S* __restrict__ v_mus,       // [C*N]
    S* __restrict__ v_cov2d_3x3,      // [C*N, 6]
    uint32_t total,
    cudaStream_t stream
);

// ============================================================================
// Second-Order Backward Pass Launch Function
// ============================================================================

/**
 * Launch kernel for conics & mu second-order backward pass (Hessians).
 *
 * Given downstream first-order gradients and second-order Hessians w.r.t.
 * (conics, mu, means2d, opacity), computes upstream gradients and Hessians
 * w.r.t. cov2d_3x3 (sigma_0..5), plus cross-Hessians with means2d & opacity.
 *
 * Core formulas (per-Gaussian):
 *   J = d(u11,u12,u22,mu) / d(sigma_0..5)           [4 x 6 Jacobian]
 *   v_sigma = J^T @ v_phi                             [first-order gradient]
 *   H_{sigma,sigma} = J^T H_{phi,phi} J + sum_k v_k Hess_k  [6x6 Hessian]
 *   H_{sigma,psi}   = J^T H_{phi,psi}                 [cross-Hessian]
 *
 * The Hess_k are the second derivatives of each output function
 * (u11, u12, u22, mu) w.r.t. the 6 input parameters. For conics these
 * are cubic polynomials in the inverse elements; for mu they involve
 * products of inverse elements of both 2x2 and 3x3 blurred matrices.
 *
 * H_{sigma,sigma} is stored as upper-triangular flat array of 21 elements:
 *   (0,0),(0,1),(0,2),(0,3),(0,4),(0,5),
 *         (1,1),(1,2),(1,3),(1,4),(1,5),
 *               (2,2),(2,3),(2,4),(2,5),
 *                     (3,3),(3,4),(3,5),
 *                           (4,4),(4,5),
 *                                 (5,5)
 *
 * @tparam S Scalar type (float, double)
 *
 * @param cov2d_3x3           Forward input: projected covariance [C*N, 6]
 * @param eps                  Blur epsilon
 * @param conics               Forward output: conics [C*N, 3]
 * @param mus                  Forward output: mu [C*N]
 * @param v_conics             Downstream dL/d(conics) [C*N, 3]
 * @param v_mus                Downstream dL/d(mu) [C*N]
 * @param h_conics             Downstream H_{conics,conics} upper-tri [C*N, 6]
 * @param h_conics_mu          Downstream H_{conics,mu} [C*N, 3]
 * @param h_mu_mu              Downstream H_{mu,mu} [C*N]
 * @param h_means2d_conics     Downstream H_{means2d,conics} [C*N, 6] (2x3 row-major)
 * @param h_conics_opacity     Downstream H_{conics,opacity} [C*N, 3]
 * @param h_mu_means2d         Downstream H_{mu,means2d} [C*N, 2]
 * @param h_mu_opacity         Downstream H_{mu,opacity} [C*N]
 * @param v_cov2d_3x3         Output dL/d(cov2d_3x3) [C*N, 6]
 * @param h_cov_cov            Output H_{sigma,sigma} upper-tri [C*N, 21]
 * @param h_cov_means2d        Output H_{sigma,means2d} [C*N, 12] (6x2 row-major)
 * @param h_cov_opacity        Output H_{sigma,opacity} [C*N, 6]
 * @param total                Total number of elements (C * N)
 * @param stream               CUDA stream
 */
template <typename S>
void launch_conics_mu_bwd2(
    const S* __restrict__ cov2d_3x3,   // [C*N, 6]
    S eps,
    const S* __restrict__ conics,      // [C*N, 3]
    const S* __restrict__ mus,         // [C*N]
    // downstream first-order
    const S* __restrict__ v_conics,    // [C*N, 3]
    const S* __restrict__ v_mus,       // [C*N]
    // downstream phi-phi Hessian
    const S* __restrict__ h_conics,          // [C*N, 6]  upper-tri 3x3
    const S* __restrict__ h_conics_mu,       // [C*N, 3]
    const S* __restrict__ h_mu_mu,           // [C*N]
    // downstream phi-psi cross Hessian
    const S* __restrict__ h_means2d_conics,  // [C*N, 6]  2x3 row-major
    const S* __restrict__ h_conics_opacity,  // [C*N, 3]
    const S* __restrict__ h_mu_means2d,      // [C*N, 2]
    const S* __restrict__ h_mu_opacity,      // [C*N]
    // outputs
    S* __restrict__ v_cov2d_3x3,       // [C*N, 6]
    S* __restrict__ h_cov_cov,         // [C*N, 21]  upper-tri 6x6
    S* __restrict__ h_cov_means2d,     // [C*N, 12]  6x2 row-major
    S* __restrict__ h_cov_opacity,     // [C*N, 6]
    uint32_t total,
    cudaStream_t stream
);

} // namespace gsplat2ct

#endif // GSPLAT2CT_CONICS_MU_H
