/**
 * @file ParameterTransform.h
 * @brief CUDA Kernel Launch Functions for Parameter Transform (quat, scale_ori) -> covar
 *
 * This header declares the launch functions that bridge between
 * C++ tensor processing (ParameterTransform.cpp) and CUDA kernels
 * (ParameterTransformFwd.cu, ParameterTransformBwd.cu,
 *  ParameterTransformBwd2.cu).
 *
 * =========================================================================
 * Module Overview
 * =========================================================================
 *
 * Transforms the raw Gaussian parameters (quaternion, scale_ori) into the
 * 3D world-space covariance matrix (upper-tri 6 elements).
 *
 *   Input  (theta): quats [N, 4] (unit quaternion, ||q||=1)
 *                    scale_ori [N, 3] (raw scale before activation)
 *   Output (phi):   covars [N, 6] (upper-triangular storage of 3x3 covariance)
 *
 * =========================================================================
 * Forward Pass Pipeline
 * =========================================================================
 *
 * F1. Scale activation:
 *       s_i = sigmoid(scale_ori_i) * (scale_max - scale_min) + scale_min
 *
 * F2. Quaternion to rotation matrix (unit quaternion, no normalization needed):
 *       q = (w, x, y, z),  ||q|| = 1
 *       R = rotation matrix from q
 *
 * F3. Covariance construction:
 *       S = diag(s_0, s_1, s_2)
 *       M = R * S
 *       C = M * M^T = R * S^2 * R^T
 *
 * F4. Store upper-triangular elements:
 *       covars = (C00, C01, C02, C11, C12, C22)
 *
 * =========================================================================
 * Backward Pass (First-Order VJP)
 * =========================================================================
 *
 * Given downstream dL/d(covars) [N, 6], compute:
 *   v_quats [N, 4]     = dL/d(quats)
 *   v_scale_ori [N, 3] = dL/d(scale_ori)
 *
 * Key chain:
 *   dL/dM = (v_C + v_C^T) * M
 *   dL/dR = dL/dM * S
 *   dL/ds_i = sum_j R_{ji} * dL_dM_{ji}
 *   dL/d(scale_ori_i) = dL/ds_i * s_i' (sigmoid derivative * range)
 *   dL/d(quat) = from R -> quat VJP (Euclidean, no projection)
 *
 * =========================================================================
 * Second-Order Backward Pass (Hessians)
 * =========================================================================
 *
 * Given downstream:
 *   - v_covars [N, 6]:            first-order gradient dL/d(covars)
 *   - h_covars [N, 21]:           H_{covar,covar} upper-tri of 6x6
 *   - h_covars_means [N, 18]:     H_{covar,mean} 6x3 row-major (cross-Hessian)
 *   - h_covars_opacity [N, 6]:    H_{covar,opacity} (cross-Hessian)
 *
 * Compute upstream:
 *   - v_quats [N, 4]:             first-order gradient dL/d(quats)
 *   - v_scale_ori [N, 3]:         first-order gradient dL/d(scale_ori)
 *   - h_quats [N, 10]:            H_{q,q} upper-tri of 4x4
 *   - h_scale_ori [N, 6]:         H_{s,s} upper-tri of 3x3
 *   - h_quats_scale_ori [N, 12]:  H_{q,s} 4x3 row-major
 *   - h_quats_means [N, 12]:      H_{q,mean} 4x3 row-major
 *   - h_quats_opacity [N, 4]:     H_{q,opacity}
 *   - h_scale_ori_means [N, 9]:   H_{s,mean} 3x3 row-major
 *   - h_scale_ori_opacity [N, 3]: H_{s,opacity}
 *
 * =========================================================================
 * Data Layout Details
 * =========================================================================
 *
 * Upper-triangular storage for symmetric 4x4 (quat Hessian): 10 elements
 *   (0,0),(0,1),(0,2),(0,3),
 *         (1,1),(1,2),(1,3),
 *               (2,2),(2,3),
 *                     (3,3)
 *
 * Upper-triangular storage for symmetric 6x6 (covar Hessian): 21 elements
 *   Same indexing as ConicsMu and WorldToImageProjection.
 *
 * =========================================================================
 * Gradient/Hessian variable naming
 * =========================================================================
 *
 *   v_  prefix: first-order gradient (VJP)
 *   h_  prefix: second-order Hessian
 *
 * Author: Dy164
 * Date: 2026-02-20
 */

#ifndef GSPLAT2CT_PARAMETER_TRANSFORM_H
#define GSPLAT2CT_PARAMETER_TRANSFORM_H

#include "../include/common.h"
#include <cuda_runtime.h>

namespace gsplat2ct {

// ============================================================================
// Forward Pass Launch Function
// ============================================================================

/**
 * Launch kernel for parameter transform forward pass.
 *
 * @tparam S Scalar type (float, double)
 *
 * @param quats       Unit quaternions [N, 4] (w, x, y, z), ||q||=1
 * @param scale_ori   Raw scale parameters [N, 3] (before sigmoid activation)
 * @param scale_min   Minimum scale after activation
 * @param scale_max   Maximum scale after activation
 * @param covars      Output covariance upper-tri [N, 6]
 * @param max_scales  Output max activated scale per Gaussian [N]
 * @param N           Number of Gaussians
 * @param stream      CUDA stream
 */
template <typename S>
void launch_parameter_transform_fwd(
    const S* __restrict__ quats,       // [N, 4]
    const S* __restrict__ scale_ori,   // [N, 3]
    S scale_min,
    S scale_max,
    S* __restrict__ covars,            // [N, 6]
    S* __restrict__ max_scales,        // [N]
    uint32_t N,
    cudaStream_t stream
);

// ============================================================================
// First-Order Backward Pass Launch Function
// ============================================================================

/**
 * Launch kernel for parameter transform backward pass (gradients).
 *
 * @tparam S Scalar type (float, double)
 *
 * @param quats       Forward input: unit quaternions [N, 4]
 * @param scale_ori   Forward input: raw scale [N, 3]
 * @param scale_min   Minimum scale after activation
 * @param scale_max   Maximum scale after activation
 * @param v_covars    Downstream dL/d(covars) [N, 6]
 * @param v_quats     Output dL/d(quats) [N, 4]
 * @param v_scale_ori Output dL/d(scale_ori) [N, 3]
 * @param N           Number of Gaussians
 * @param stream      CUDA stream
 */
template <typename S>
void launch_parameter_transform_bwd(
    const S* __restrict__ quats,       // [N, 4]
    const S* __restrict__ scale_ori,   // [N, 3]
    S scale_min,
    S scale_max,
    const S* __restrict__ v_covars,    // [N, 6]
    S* __restrict__ v_quats,           // [N, 4]
    S* __restrict__ v_scale_ori,       // [N, 3]
    uint32_t N,
    cudaStream_t stream
);

// ============================================================================
// Second-Order Backward Pass Launch Function
// ============================================================================

/**
 * Launch kernel for parameter transform second-order backward pass.
 *
 * Given downstream first-order gradients and second-order Hessians w.r.t.
 * covars, plus cross-Hessians with mean and opacity, compute upstream
 * gradients and Hessians w.r.t. (quats, scale_ori), plus cross-Hessians.
 *
 * Core formulas:
 *   J = d(covars_6) / d(quats_4, scale_ori_3)     [6 x 7 Jacobian]
 *   v_theta = J^T @ v_covars                        [first-order gradient]
 *   H_{theta,theta} = J^T H_{covar,covar} J + sum_k v_k Hess_k  [7x7 Hessian]
 *   H_{theta,psi}   = J^T H_{covar,psi}             [cross-Hessian]
 *
 * @tparam S Scalar type (float, double)
 *
 * @param quats             Forward input [N, 4]
 * @param scale_ori         Forward input [N, 3]
 * @param scale_min         Min scale
 * @param scale_max         Max scale
 * @param v_covars          Downstream dL/d(covars) [N, 6]
 * @param h_covars          Downstream H_{c,c} upper-tri [N, 21]
 * @param h_covars_means    Downstream H_{c,mean} [N, 18] 6x3 row-major
 * @param h_covars_opacity  Downstream H_{c,opacity} [N, 6]
 * @param v_quats           Output dL/d(quats) [N, 4]
 * @param v_scale_ori       Output dL/d(scale_ori) [N, 3]
 * @param h_quats           Output H_{q,q} upper-tri [N, 10]
 * @param h_scale_ori       Output H_{s,s} upper-tri [N, 6]
 * @param h_quats_scale_ori Output H_{q,s} [N, 12] 4x3 row-major
 * @param h_quats_means     Output H_{q,mean} [N, 12] 4x3 row-major
 * @param h_quats_opacity   Output H_{q,opacity} [N, 4]
 * @param h_scale_ori_means Output H_{s,mean} [N, 9] 3x3 row-major
 * @param h_scale_ori_opacity Output H_{s,opacity} [N, 3]
 * @param N                 Number of Gaussians
 * @param stream            CUDA stream
 */
template <typename S>
void launch_parameter_transform_bwd2(
    const S* __restrict__ quats,             // [N, 4]
    const S* __restrict__ scale_ori,         // [N, 3]
    S scale_min,
    S scale_max,
    // downstream first-order
    const S* __restrict__ v_covars,          // [N, 6]
    // downstream covar-covar Hessian
    const S* __restrict__ h_covars,          // [N, 21]
    // downstream covar-psi cross Hessians
    const S* __restrict__ h_covars_means,    // [N, 18]
    const S* __restrict__ h_covars_opacity,  // [N, 6]
    // outputs: first-order
    S* __restrict__ v_quats,                 // [N, 4]
    S* __restrict__ v_scale_ori,             // [N, 3]
    // outputs: theta-theta Hessians
    S* __restrict__ h_quats,                 // [N, 10]
    S* __restrict__ h_scale_ori,             // [N, 6]
    S* __restrict__ h_quats_scale_ori,       // [N, 12]
    // outputs: theta-psi cross Hessians
    S* __restrict__ h_quats_means,           // [N, 12]
    S* __restrict__ h_quats_opacity,         // [N, 4]
    S* __restrict__ h_scale_ori_means,       // [N, 9]
    S* __restrict__ h_scale_ori_opacity,     // [N, 3]
    // dimensions
    uint32_t N,
    cudaStream_t stream
);

} // namespace gsplat2ct

#endif // GSPLAT2CT_PARAMETER_TRANSFORM_H
