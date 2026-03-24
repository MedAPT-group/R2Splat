/**
 * @file WorldToImageProjection.h
 * @brief CUDA Kernel Launch Functions for World-to-Image Projection
 *
 * This header declares the launch functions that bridge between
 * C++ tensor processing (WorldToImageProjection.cpp) and CUDA kernels
 * (WorldToImageProjectionFwd.cu, WorldToImageProjectionBwd.cu,
 *  WorldToImageProjectionBwd2.cu).
 *
 * =========================================================================
 * Module Overview
 * =========================================================================
 *
 * Combines world-to-camera coordinate transform with perspective projection
 * into a single fused module:
 *
 *   World-space inputs:  p (mean3d), Sigma (covariance), R, t, fx, fy, cx, cy
 *   Image-space outputs: means2d (2D projected center),
 *                        cov2d_3x3 (projected 3x3 covariance, upper-tri 6)
 *
 * =========================================================================
 * Forward Pass Pipeline
 * =========================================================================
 *
 * F1. Camera transform: p_c = R * p + t
 * F2. Perspective projection:
 *       rz = 1/p_c.z,  rz2 = rz^2
 *       means2d = (fx * p_c.x * rz + cx,  fy * p_c.y * rz + cy)
 * F3. FOV clipping (hard clamp, following gsplat reference):
 *       tx = p_c.z * clamp(p_c.x / p_c.z, -lim_neg_x, lim_pos_x)
 *       ty = p_c.z * clamp(p_c.y / p_c.z, -lim_neg_y, lim_pos_y)
 * F4. Build J (3x3 Jacobian of projection):
 *       l = sqrt(tx^2 + ty^2 + z^2)
 *       J = [[fx*rz,   0,       -fx*tx*rz2],
 *            [0,       fy*rz,   -fy*ty*rz2],
 *            [tx/l,    ty/l,     z/l       ]]
 * F5. Camera covariance: Sigma_c = R * Sigma * R^T
 * F6. Projected covariance: cov2d_3x3 = J * Sigma_c * J^T
 *     Store as upper-triangular 6 elements: (a, b, c, d, e, f)
 *
 * =========================================================================
 * Data Layout
 * =========================================================================
 *
 * Input:
 *   means:     [N, 3]       -- world-space Gaussian centers (shared across cameras)
 *   covars:    [N, 6]       -- world-space covariance upper-tri (shared across cameras)
 *   viewmats:  [C, 4, 4]    -- camera extrinsic matrices [R|t; 0 1]
 *   Ks:        [C, 3, 3]    -- camera intrinsic matrices
 *
 * Output:
 *   means2d:   [C, N, 2]    -- projected 2D centers
 *   cov2d_3x3: [C, N, 6]    -- projected 3x3 covariance (upper-tri)
 *
 * Upper-triangular storage for 3x3 symmetric matrix:
 *   [a, b, c, d, e, f] = [S00, S01, S02, S11, S12, S22]
 *
 * Hessian storage for 6x6 symmetric: 21 elements upper-tri
 *   index(i,j) = i*(11-i)/2 + j,  i <= j
 *
 * Hessian storage for 3x3 symmetric (means2d, 2+6=8 params): varies
 *
 * =========================================================================
 * Gradient/Hessian variable naming
 * =========================================================================
 *
 *   v_  prefix: first-order gradient (VJP)
 *   h_  prefix: second-order Hessian
 *
 * Author: Dy164
 * Date: 2026-02-19
 */

#ifndef GSPLAT2CT_WORLD_TO_IMAGE_PROJECTION_H
#define GSPLAT2CT_WORLD_TO_IMAGE_PROJECTION_H

#include "../include/common.h"
#include <cuda_runtime.h>

namespace gsplat2ct {

// ============================================================================
// Forward Pass Launch Function
// ============================================================================

/**
 * Launch kernel for world-to-image projection forward pass.
 *
 * @tparam S Scalar type (float, double)
 *
 * @param means       World-space means [N, 3] (shared across cameras)
 * @param covars      World-space covariance upper-tri [N, 6] (shared across cameras)
 * @param viewmats    Camera extrinsic [C, 4, 4] (row-major)
 * @param Ks          Camera intrinsic [C, 3, 3] (row-major)
 * @param dim1  Image width in pixels
 * @param dim0 Image height in pixels
 * @param means2d     Output projected means [C*N, 2]
 * @param cov2d_3x3   Output projected covariance upper-tri [C*N, 6]
 * @param C           Number of cameras
 * @param N           Number of Gaussians
 * @param stream      CUDA stream
 */
template <typename S>
void launch_world_to_image_projection_fwd(
    const S* __restrict__ means,       // [N, 3]
    const S* __restrict__ covars,      // [N, 6]
    const S* __restrict__ viewmats,    // [C, 4, 4]
    const S* __restrict__ Ks,          // [C, 3, 3]
    uint32_t dim1,
    uint32_t dim0,
    S eps_blur,
    S* __restrict__ means2d,           // [C*N, 2]
    S* __restrict__ cov2d_3x3,        // [C*N, 6]
    int32_t* __restrict__ radii,       // [C*N]
    uint32_t C,
    uint32_t N,
    cudaStream_t stream
);

// ============================================================================
// First-Order Backward Pass Launch Function
// ============================================================================

/**
 * Launch kernel for world-to-image projection backward pass (gradients).
 *
 * Computes:
 *   v_means3d = dL/d(means3d)
 *   v_covars  = dL/d(covars)
 *
 * from downstream gradients v_means2d and v_cov2d_3x3.
 *
 * @tparam S Scalar type
 *
 * @param means        Forward input: world-space means [N, 3]
 * @param covars       Forward input: world-space covariance [N, 6]
 * @param viewmats     Camera extrinsic [C, 4, 4]
 * @param Ks           Camera intrinsic [C, 3, 3]
 * @param dim1  Image width
 * @param dim0 Image height
 * @param v_means2d    Downstream dL/d(means2d) [C*N, 2]
 * @param v_cov2d_3x3  Downstream dL/d(cov2d_3x3) [C*N, 6]
 * @param v_means      Output dL/d(means) [N, 3] (summed over cameras)
 * @param v_covars     Output dL/d(covars) [N, 6] (summed over cameras)
 * @param C            Number of cameras
 * @param N            Number of Gaussians
 * @param stream       CUDA stream
 */
template <typename S>
void launch_world_to_image_projection_bwd(
    const S* __restrict__ means,       // [N, 3]
    const S* __restrict__ covars,      // [N, 6]
    const S* __restrict__ viewmats,    // [C, 4, 4]
    const S* __restrict__ Ks,          // [C, 3, 3]
    uint32_t dim1,
    uint32_t dim0,
    const S* __restrict__ v_means2d,   // [C*N, 2]
    const S* __restrict__ v_cov2d_3x3, // [C*N, 6]
    S* __restrict__ v_means,           // [N, 3]
    S* __restrict__ v_covars,          // [N, 6]
    uint32_t C,
    uint32_t N,
    cudaStream_t stream
);

// ============================================================================
// Second-Order Backward Pass Launch Function
// ============================================================================

/**
 * Launch kernel for world-to-image projection second-order backward pass.
 *
 * Given downstream first-order gradients and second-order Hessians w.r.t.
 * (means2d, cov2d_3x3), computes upstream gradients and Hessians w.r.t.
 * (means3d, covars).
 *
 * Downstream Hessian structure (phi = means2d ++ cov2d_3x3, 2+6=8 outputs):
 *   h_means2d:            H_{m,m}           [C*N, 3]  upper-tri of 2x2
 *   h_cov2d_3x3:          H_{c,c}           [C*N, 21] upper-tri of 6x6
 *   h_means2d_cov2d_3x3:  H_{m,c}           [C*N, 12] 2x6 row-major
 *
 * Upstream Hessian structure (theta = means3d ++ covars, 3+6=9 inputs):
 *   h_means3d:             H_{p,p}          [C*N, 6]  upper-tri of 3x3
 *   h_covars:              H_{s,s}          [C*N, 21] upper-tri of 6x6
 *   h_means3d_covars:      H_{p,s}          [C*N, 18] 3x6 row-major
 *
 * Also propagates cross-Hessians with external parameters (psi):
 *   Downstream: h_means2d_psi [C*N, 2*Dpsi], h_cov2d_3x3_psi [C*N, 6*Dpsi]
 *   Upstream:   h_means3d_psi [C*N, 3*Dpsi], h_covars_psi    [C*N, 6*Dpsi]
 *
 * For this module, external psi = (opacity), Dpsi = 1.
 *
 * @tparam S Scalar type
 */
template <typename S>
void launch_world_to_image_projection_bwd2(
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
    // Outputs: first-order gradients (summed over cameras)
    S* __restrict__ v_means,           // [N, 3]
    S* __restrict__ v_covars,          // [N, 6]
    // Outputs: theta-theta Hessians (summed over cameras)
    S* __restrict__ h_means,           // [N, 6]  upper-tri 3x3
    S* __restrict__ h_covars,          // [N, 21] upper-tri 6x6
    S* __restrict__ h_means_covars,    // [N, 18] 3x6 row-major
    // Outputs: theta-psi cross Hessians (per-camera for opacity)
    S* __restrict__ h_means_opacity,   // [C*N, 3]
    S* __restrict__ h_covars_opacity,  // [C*N, 6]
    // Dimensions
    uint32_t C,
    uint32_t N,
    cudaStream_t stream
);

} // namespace gsplat2ct

#endif // GSPLAT2CT_WORLD_TO_IMAGE_PROJECTION_H
