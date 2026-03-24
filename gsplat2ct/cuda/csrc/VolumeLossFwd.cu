/**
 * VolumeLoss Forward CUDA Kernels
 * 
 * Implements forward pass for Smooth Total Variation (TV) loss on 3D volumes.
 * Input: (B, D, H, W) volume tensor
 * 
 * Smooth TV Loss = mean(sqrt(∂x/∂d² + eps) + sqrt(∂x/∂h² + eps) + sqrt(∂x/∂w² + eps))
 * 
 * Uses forward differences:
 * - ∂x/∂d = x[d+1] - x[d]
 * - ∂x/∂h = x[h+1] - x[h]
 * - ∂x/∂w = x[w+1] - x[w]
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
// TV Map Kernel
// ============================================================================

/**
 * Compute smooth TV value per voxel.
 * Each thread handles one voxel and computes the sum of sqrt(diff^2 + eps).
 */
template <typename S>
__global__ void compute_tv_map_kernel(
    const S* __restrict__ volume,
    S* __restrict__ tv_map,
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
    S tv = 0;
    
    // Forward difference in depth
    if (d + 1 < D) {
        int idx_d = b * D * H * W + (d + 1) * H * W + h * W + w;
        S diff = volume[idx_d] - val;
        tv += sqrt(diff * diff + eps);
    }
    
    // Forward difference in height
    if (h + 1 < H) {
        int idx_h = b * D * H * W + d * H * W + (h + 1) * W + w;
        S diff = volume[idx_h] - val;
        tv += sqrt(diff * diff + eps);
    }
    
    // Forward difference in width
    if (w + 1 < W) {
        int idx_w = b * D * H * W + d * H * W + h * W + (w + 1);
        S diff = volume[idx_w] - val;
        tv += sqrt(diff * diff + eps);
    }
    
    tv_map[idx] = tv;
}

// ============================================================================
// Parallel Reduction for Sum
// ============================================================================

template <typename S, int BLOCK_SIZE = 256>
__global__ void reduce_sum_kernel(
    const S* __restrict__ input,
    S* __restrict__ output,
    int N
) {
    __shared__ S s_data[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * (BLOCK_SIZE * 2) + tid;
    
    // Load and add two elements
    S sum = 0;
    if (idx < N) sum += input[idx];
    if (idx + BLOCK_SIZE < N) sum += input[idx + BLOCK_SIZE];
    s_data[tid] = sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
    GSPLAT2CT_PRAGMA_UNROLL
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }
    
    // Write block result
    if (tid == 0) {
        atomicAdd(output, s_data[0]);
    }
}

// ============================================================================
// Launch Function Implementation
// ============================================================================

template <typename S>
void launch_volume_loss_fwd(
    const S* __restrict__ volume,
    S* __restrict__ tv_loss,
    S* __restrict__ tv_map,
    int B, int D, int H, int W,
    S eps,
    cudaStream_t stream
) {
    int total_voxels = B * D * H * W;
    int threads = GSPLAT2CT_N_THREADS;
    int blocks = (total_voxels + threads - 1) / threads;
    
    // Kernel 1: Compute TV map
    compute_tv_map_kernel<S><<<blocks, threads, 0, stream>>>(
        volume, tv_map, B, D, H, W, eps
    );
    
    // Kernel 2: Reduce to get total TV loss
    // Initialize output to zero
    cudaMemsetAsync(tv_loss, 0, sizeof(S), stream);
    
    // Use parallel reduction
    int reduce_blocks = (total_voxels + 2 * threads - 1) / (2 * threads);
    reduce_sum_kernel<S, GSPLAT2CT_N_THREADS><<<reduce_blocks, threads, 0, stream>>>(
        tv_map, tv_loss, total_voxels
    );
    
    // Note: The final division by N should be done in the wrapper
}

// Explicit template instantiations
template void launch_volume_loss_fwd<float>(
    const float*, float*, float*, int, int, int, int, float, cudaStream_t
);

template void launch_volume_loss_fwd<double>(
    const double*, double*, double*, int, int, int, int, double, cudaStream_t
);

} // namespace gsplat2ct
