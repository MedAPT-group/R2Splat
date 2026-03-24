/**
 * VolumeLoss Backward CUDA Kernels
 * 
 * Implements backward pass for Smooth Total Variation (TV) loss on 3D volumes.
 * 
 * For Smooth TV Loss = mean(sqrt(∂x^2 + eps))
 * 
 * Gradient at voxel (b, d, h, w):
 * d(smooth_TV)/d(x) = x / sqrt(x^2 + eps)
 * 
 * Each voxel's gradient comes from:
 * 1. Forward differences starting at this voxel
 * 2. Forward differences from neighboring voxels ending at this voxel
 * 
 * Author: Dy164
 * Date: 2025-12-18
 */

#include "Loss.h"
#include "../include/common.h"
#include <cuda_runtime.h>
#include <cmath>

namespace gsplat2ct {

// ============================================================================
// Helper Functions
// ============================================================================

template <typename S>
__device__ __forceinline__ S smooth_sign_func(S x, S eps) {
    return x / sqrt(x * x + eps);
}

// ============================================================================
// TV Gradient Kernel
// ============================================================================

/**
 * Compute smooth TV gradient per voxel.
 * Each thread handles one voxel.
 * 
 * Uses smooth approximation: d(sqrt(x^2+eps))/dx = x/sqrt(x^2+eps)
 * 
 * The gradient comes from 6 potential contributions:
 * - Forward differences in d, h, w where this voxel is the "start"
 * - Forward differences in d, h, w where this voxel is the "end"
 */
template <typename S>
__global__ void compute_tv_grad_kernel(
    const S* __restrict__ volume,
    const S* __restrict__ grad_output,
    S* __restrict__ grad_volume,
    int B, int D, int H, int W,
    S eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_voxels = B * D * H * W;
    if (idx >= total_voxels) return;
    
    // Decode indices
    int b = idx / (D * H * W);
    int remainder = idx % (D * H * W);
    int d = remainder / (H * W);
    remainder = remainder % (H * W);
    int h = remainder / W;
    int w = remainder % W;
    
    S val = volume[idx];
    S grad = 0;
    S grad_out = grad_output[0] / static_cast<S>(total_voxels);
    
    // Contribution from forward difference in depth (this is start)
    if (d + 1 < D) {
        int idx_d_plus = b * D * H * W + (d + 1) * H * W + h * W + w;
        S diff = volume[idx_d_plus] - val;
        grad -= smooth_sign_func(diff, eps);  // d(sqrt((a-b)^2+eps))/db = -(a-b)/sqrt((a-b)^2+eps)
    }
    
    // Contribution from forward difference in depth (this is end)
    if (d > 0) {
        int idx_d_minus = b * D * H * W + (d - 1) * H * W + h * W + w;
        S diff = val - volume[idx_d_minus];
        grad += smooth_sign_func(diff, eps);  // d(sqrt((a-b)^2+eps))/da = (a-b)/sqrt((a-b)^2+eps)
    }
    
    // Contribution from forward difference in height (this is start)
    if (h + 1 < H) {
        int idx_h_plus = b * D * H * W + d * H * W + (h + 1) * W + w;
        S diff = volume[idx_h_plus] - val;
        grad -= smooth_sign_func(diff, eps);
    }
    
    // Contribution from forward difference in height (this is end)
    if (h > 0) {
        int idx_h_minus = b * D * H * W + d * H * W + (h - 1) * W + w;
        S diff = val - volume[idx_h_minus];
        grad += smooth_sign_func(diff, eps);
    }
    
    // Contribution from forward difference in width (this is start)
    if (w + 1 < W) {
        int idx_w_plus = b * D * H * W + d * H * W + h * W + (w + 1);
        S diff = volume[idx_w_plus] - val;
        grad -= smooth_sign_func(diff, eps);
    }
    
    // Contribution from forward difference in width (this is end)
    if (w > 0) {
        int idx_w_minus = b * D * H * W + d * H * W + h * W + (w - 1);
        S diff = val - volume[idx_w_minus];
        grad += smooth_sign_func(diff, eps);
    }
    
    grad_volume[idx] = grad * grad_out;
}

// ============================================================================
// TV Hessian Kernel (Second-Order Derivative)
// ============================================================================

/**
 * Compute Hessian diagonal for TV loss.
 * Since TV uses L1 norm (|x|), the second derivative is zero almost everywhere.
 * However, at discontinuities, it could be approximated as delta function.
 * 
 * For practical purposes, we use a smooth approximation:
 * Instead of |x|, use sqrt(x^2 + eps), which has:
 * - First derivative: x / sqrt(x^2 + eps)
 * - Second derivative: eps / (x^2 + eps)^(3/2)
 */
template <typename S>
__global__ void compute_tv_hessian_kernel(
    const S* __restrict__ volume,
    const S* __restrict__ grad_output,
    S* __restrict__ hessian_diag,
    int B, int D, int H, int W,
    S eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_voxels = B * D * H * W;
    if (idx >= total_voxels) return;
    
    int b = idx / (D * H * W);
    int remainder = idx % (D * H * W);
    int d = remainder / (H * W);
    remainder = remainder % (H * W);
    int h = remainder / W;
    int w = remainder % W;
    
    S val = volume[idx];
    S hess = 0;
    S grad_out = grad_output[0] / static_cast<S>(total_voxels);
    
    // For each neighboring difference, compute second derivative
    // Using smooth L1: sqrt(x^2 + eps)
    // d/dx = x / sqrt(x^2 + eps)
    // d^2/dx^2 = eps / (x^2 + eps)^(3/2)
    
    auto smooth_hess = [eps](S diff) -> S {
        S denom = diff * diff + eps;
        return eps / (denom * sqrt(denom));
    };
    
    // Depth direction
    if (d + 1 < D) {
        int idx_d_plus = b * D * H * W + (d + 1) * H * W + h * W + w;
        S diff = volume[idx_d_plus] - val;
        hess += smooth_hess(diff);  // Contribution as "start"
    }
    if (d > 0) {
        int idx_d_minus = b * D * H * W + (d - 1) * H * W + h * W + w;
        S diff = val - volume[idx_d_minus];
        hess += smooth_hess(diff);  // Contribution as "end"
    }
    
    // Height direction
    if (h + 1 < H) {
        int idx_h_plus = b * D * H * W + d * H * W + (h + 1) * W + w;
        S diff = volume[idx_h_plus] - val;
        hess += smooth_hess(diff);
    }
    if (h > 0) {
        int idx_h_minus = b * D * H * W + d * H * W + (h - 1) * W + w;
        S diff = val - volume[idx_h_minus];
        hess += smooth_hess(diff);
    }
    
    // Width direction
    if (w + 1 < W) {
        int idx_w_plus = b * D * H * W + d * H * W + h * W + (w + 1);
        S diff = volume[idx_w_plus] - val;
        hess += smooth_hess(diff);
    }
    if (w > 0) {
        int idx_w_minus = b * D * H * W + d * H * W + h * W + (w - 1);
        S diff = val - volume[idx_w_minus];
        hess += smooth_hess(diff);
    }
    
    hessian_diag[idx] = hess * grad_out;
}

// ============================================================================
// Launch Function Implementation
// ============================================================================

template <typename S>
void launch_volume_loss_bwd(
    const S* __restrict__ volume,
    const S* __restrict__ grad_output,
    S* __restrict__ grad_volume,
    int B, int D, int H, int W,
    S eps,
    cudaStream_t stream
) {
    int total_voxels = B * D * H * W;
    int threads = GSPLAT2CT_N_THREADS;
    int blocks = (total_voxels + threads - 1) / threads;
    
    compute_tv_grad_kernel<S><<<blocks, threads, 0, stream>>>(
        volume, grad_output, grad_volume, B, D, H, W, eps
    );
}

template <typename S>
void launch_volume_loss_bwd2(
    const S* __restrict__ volume,
    const S* __restrict__ grad_output,
    S* __restrict__ hessian_diag,
    int B, int D, int H, int W,
    S eps,
    cudaStream_t stream
) {
    int total_voxels = B * D * H * W;
    int threads = GSPLAT2CT_N_THREADS;
    int blocks = (total_voxels + threads - 1) / threads;
    
    compute_tv_hessian_kernel<S><<<blocks, threads, 0, stream>>>(
        volume, grad_output, hessian_diag, B, D, H, W, eps
    );
}

// Explicit template instantiations
template void launch_volume_loss_bwd<float>(
    const float*, const float*, float*, int, int, int, int, float, cudaStream_t
);

template void launch_volume_loss_bwd<double>(
    const double*, const double*, double*, int, int, int, int, double, cudaStream_t
);

template void launch_volume_loss_bwd2<float>(
    const float*, const float*, float*, int, int, int, int, float, cudaStream_t
);

template void launch_volume_loss_bwd2<double>(
    const double*, const double*, double*, int, int, int, int, double, cudaStream_t
);

} // namespace gsplat2ct
