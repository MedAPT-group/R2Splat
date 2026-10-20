/**
 * VolumeLoss Backward CUDA Kernels
 * 
 * Implements backward pass for anisotropic Total Variation (TV) loss on 3D volumes.
 * 
 * For TV Loss = edge-count mean(abs(diff)).
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

namespace r2splat {

// ============================================================================
// Helper Functions
// ============================================================================

template <typename S>
__device__ __forceinline__ S sign_func(S x) {
    return (x > static_cast<S>(0))
        ? static_cast<S>(1)
        : ((x < static_cast<S>(0)) ? static_cast<S>(-1) : static_cast<S>(0));
}

// ============================================================================
// TV Gradient Kernel
// ============================================================================

/**
 * Compute TV gradient per voxel.
 * Each thread handles one voxel.
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
    int B, int D, int H, int W
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
    int edge_count =
        B * ((D - 1) * H * W + D * (H - 1) * W + D * H * (W - 1));
    edge_count = max(edge_count, 1);
    S grad_out = grad_output[0] / static_cast<S>(edge_count);
    
    // Contribution from forward difference in depth (this is start)
    if (d + 1 < D) {
        int idx_d_plus = b * D * H * W + (d + 1) * H * W + h * W + w;
        S diff = volume[idx_d_plus] - val;
        grad -= sign_func(diff);
    }
    
    // Contribution from forward difference in depth (this is end)
    if (d > 0) {
        int idx_d_minus = b * D * H * W + (d - 1) * H * W + h * W + w;
        S diff = val - volume[idx_d_minus];
        grad += sign_func(diff);
    }
    
    // Contribution from forward difference in height (this is start)
    if (h + 1 < H) {
        int idx_h_plus = b * D * H * W + d * H * W + (h + 1) * W + w;
        S diff = volume[idx_h_plus] - val;
        grad -= sign_func(diff);
    }
    
    // Contribution from forward difference in height (this is end)
    if (h > 0) {
        int idx_h_minus = b * D * H * W + d * H * W + (h - 1) * W + w;
        S diff = val - volume[idx_h_minus];
        grad += sign_func(diff);
    }
    
    // Contribution from forward difference in width (this is start)
    if (w + 1 < W) {
        int idx_w_plus = b * D * H * W + d * H * W + h * W + (w + 1);
        S diff = volume[idx_w_plus] - val;
        grad -= sign_func(diff);
    }
    
    // Contribution from forward difference in width (this is end)
    if (w > 0) {
        int idx_w_minus = b * D * H * W + d * H * W + h * W + (w - 1);
        S diff = val - volume[idx_w_minus];
        grad += sign_func(diff);
    }
    
    grad_volume[idx] = grad * grad_out;
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
    cudaStream_t stream
) {
    int total_voxels = B * D * H * W;
    int threads = R2SPLAT_N_THREADS;
    int blocks = (total_voxels + threads - 1) / threads;
    
    compute_tv_grad_kernel<S><<<blocks, threads, 0, stream>>>(
        volume, grad_output, grad_volume, B, D, H, W
    );
}

// Explicit template instantiations
template void launch_volume_loss_bwd<float>(
    const float*, const float*, float*, int, int, int, int, cudaStream_t
);

template void launch_volume_loss_bwd<double>(
    const double*, const double*, double*, int, int, int, int, cudaStream_t
);

} // namespace r2splat
