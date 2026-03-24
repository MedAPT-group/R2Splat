/**
 * Loss Functions CUDA Kernel Launch Functions
 * 
 * This header declares the launch functions that bridge between
 * C++ tensor processing (Loss.cpp) and CUDA kernels
 * (ImageLossFwd.cu, ImageLossBwd.cu, VolumeLossFwd.cu, VolumeLossBwd.cu).
 * 
 * Following gsplat's pattern: 
 * - Template functions with typename S for scalar type
 * - Raw pointers with __restrict__ for no-aliasing optimization
 * 
 * Author: Dy164
 * Date: 2025-12-18
 */

#ifndef GSPLAT2CT_LOSS_H
#define GSPLAT2CT_LOSS_H

#include "../include/common.h"
#include <cuda_runtime.h>

namespace gsplat2ct {

// ============================================================================
// ImageLoss Forward Launch Functions
// ============================================================================

/**
 * Launch ImageLoss forward pass CUDA kernels
 * 
 * Computes L1 map and SSIM map.
 * Optionally outputs statistics for backward reuse.
 * 
 * @tparam S Scalar type (float, double)
 */
template <typename S>
void launch_image_loss_fwd(
    const S* __restrict__ prediction,       // (B, H, W)
    const S* __restrict__ target,           // (B, H, W)
    const S* __restrict__ gaussian_window,  // (window_size, window_size)
    S* __restrict__ l1_map,                 // (B, H, W) - |pred - target|
    S* __restrict__ mu_r,                   // (B, H, W) - optional
    S* __restrict__ mu_m,                   // (B, H, W) - optional
    S* __restrict__ f1,                     // (B, H, W) - optional
    S* __restrict__ f2,                     // (B, H, W) - optional
    S* __restrict__ f3,                     // (B, H, W) - optional
    S* __restrict__ f4,                     // (B, H, W) - optional
    S* __restrict__ ssim_map,               // (B, H, W)
    int B, int H, int W,
    int window_size,
    bool compute_statistics,
    cudaStream_t stream
);

// ============================================================================
// ImageLoss Backward Launch Functions (First Order)
// ============================================================================

/**
 * Launch ImageLoss first-order backward pass CUDA kernels
 * 
 * Computes gradient of L1 + D-SSIM loss w.r.t. prediction.
 * Uses pixel-parallel approach: each thread computes gradient for one pixel.
 * 
 * @tparam S Scalar type (float, double)
 */
template <typename S>
void launch_image_loss_bwd(
    const S* __restrict__ prediction,       // (B, H, W)
    const S* __restrict__ target,           // (B, H, W)
    const S* __restrict__ gaussian_window,  // (window_size, window_size)
    const S* __restrict__ mu_r,             // (B, H, W)
    const S* __restrict__ mu_m,             // (B, H, W)
    const S* __restrict__ f1,               // (B, H, W)
    const S* __restrict__ f2,               // (B, H, W)
    const S* __restrict__ f3,               // (B, H, W)
    const S* __restrict__ f4,               // (B, H, W)
    const S* __restrict__ grad_output,      // scalar
    S* __restrict__ grad_pred,              // (B, H, W) - combined gradient
    int B, int H, int W,
    int window_size,
    S lambda_l1,
    S lambda_ssim,
    S C1, S C2,
    cudaStream_t stream
);

// ============================================================================
// ImageLoss Backward Launch Functions (Second Order - Diagonal Hessian)
// ============================================================================

/**
 * Launch ImageLoss second-order backward pass CUDA kernels
 * 
 * Computes diagonal Hessian of D-SSIM loss w.r.t. prediction.
 * L1 Hessian is always zero.
 * 
 * @tparam S Scalar type (float, double)
 */
template <typename S>
void launch_image_loss_bwd2(
    const S* __restrict__ prediction,       // (B, H, W)
    const S* __restrict__ target,           // (B, H, W)
    const S* __restrict__ gaussian_window,  // (window_size, window_size)
    const S* __restrict__ mu_r,             // (B, H, W)
    const S* __restrict__ mu_m,             // (B, H, W)
    const S* __restrict__ f1,               // (B, H, W)
    const S* __restrict__ f2,               // (B, H, W)
    const S* __restrict__ f3,               // (B, H, W)
    const S* __restrict__ f4,               // (B, H, W)
    const S* __restrict__ grad_output,      // scalar
    S* __restrict__ hessian_diag,           // (B, H, W) - SSIM diagonal Hessian
    int B, int H, int W,
    int window_size,
    S lambda_l1,
    S lambda_ssim,
    S C1, S C2,
    cudaStream_t stream
);

// ============================================================================
// VolumeLoss Forward Launch Functions
// ============================================================================

/**
 * Launch VolumeLoss (TV) forward pass CUDA kernels
 * 
 * Computes Total Variation loss for 3D volumes.
 * 
 * @tparam S Scalar type (float, double)
 */
template <typename S>
void launch_volume_loss_fwd(
    const S* __restrict__ volume,           // (B, D, H, W)
    S* __restrict__ tv_loss,                // scalar output
    S* __restrict__ tv_map,                 // (B, D, H, W) - per-element TV
    int B, int D, int H, int W,
    S eps,                                  // Smoothing parameter
    cudaStream_t stream
);

// ============================================================================
// VolumeLoss Backward Launch Functions
// ============================================================================

/**
 * Launch VolumeLoss (TV) backward pass CUDA kernels
 * 
 * Computes gradient of TV loss w.r.t. volume.
 * 
 * @tparam S Scalar type (float, double)
 */
template <typename S>
void launch_volume_loss_bwd(
    const S* __restrict__ volume,           // (B, D, H, W)
    const S* __restrict__ grad_output,      // scalar
    S* __restrict__ grad_volume,            // (B, D, H, W)
    int B, int D, int H, int W,
    S eps,                                  // Smoothing parameter
    cudaStream_t stream
);

/**
 * Launch VolumeLoss (TV) second-order backward pass CUDA kernels
 * 
 * Computes diagonal Hessian of TV loss w.r.t. volume.
 * Uses smooth L1 approximation.
 * 
 * @tparam S Scalar type (float, double)
 */
template <typename S>
void launch_volume_loss_bwd2(
    const S* __restrict__ volume,           // (B, D, H, W)
    const S* __restrict__ grad_output,      // scalar
    S* __restrict__ hessian_diag,           // (B, D, H, W)
    int B, int D, int H, int W,
    S eps,                                  // Smoothing parameter
    cudaStream_t stream
);

} // namespace gsplat2ct

#endif // GSPLAT2CT_LOSS_H
