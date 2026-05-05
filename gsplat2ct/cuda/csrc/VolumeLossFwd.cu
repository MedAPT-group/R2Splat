/**
 * VolumeLoss Forward CUDA Kernels
 * 
 * Implements forward pass for anisotropic Total Variation (TV) loss on 3D volumes.
 * Input: (B, D, H, W) volume tensor
 * 
 * TV Loss = mean over forward-difference edges of abs(diff),
 * matching r2_gaussian's edge-count normalization.
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

template <typename S>
__device__ __forceinline__ S abs_func(S x) {
    return x >= static_cast<S>(0) ? x : -x;
}

// ============================================================================
// TV Map Kernel
// ============================================================================

/**
 * Compute TV value per voxel.
 * Each thread handles one voxel and computes the sum of abs(diff).
 */
template <typename S>
__global__ void compute_tv_map_kernel(
    const S* __restrict__ volume,
    S* __restrict__ tv_map,
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
    S tv = 0;
    
    // Forward difference in depth
    if (d + 1 < D) {
        int idx_d = b * D * H * W + (d + 1) * H * W + h * W + w;
        S diff = volume[idx_d] - val;
        tv += abs_func(diff);
    }
    
    // Forward difference in height
    if (h + 1 < H) {
        int idx_h = b * D * H * W + d * H * W + (h + 1) * W + w;
        S diff = volume[idx_h] - val;
        tv += abs_func(diff);
    }
    
    // Forward difference in width
    if (w + 1 < W) {
        int idx_w = b * D * H * W + d * H * W + h * W + (w + 1);
        S diff = volume[idx_w] - val;
        tv += abs_func(diff);
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
    cudaStream_t stream
) {
    int total_voxels = B * D * H * W;
    int threads = GSPLAT2CT_N_THREADS;
    int blocks = (total_voxels + threads - 1) / threads;
    
    // Kernel 1: Compute TV map
    compute_tv_map_kernel<S><<<blocks, threads, 0, stream>>>(
        volume, tv_map, B, D, H, W
    );
    
    // Kernel 2: Reduce to get total TV loss
    // Initialize output to zero
    cudaMemsetAsync(tv_loss, 0, sizeof(S), stream);
    
    // Use parallel reduction
    int reduce_blocks = (total_voxels + 2 * threads - 1) / (2 * threads);
    reduce_sum_kernel<S, GSPLAT2CT_N_THREADS><<<reduce_blocks, threads, 0, stream>>>(
        tv_map, tv_loss, total_voxels
    );
    
    // Note: The final division by edge count is done in the wrapper.
}

// Explicit template instantiations
template void launch_volume_loss_fwd<float>(
    const float*, float*, float*, int, int, int, int, cudaStream_t
);

template void launch_volume_loss_fwd<double>(
    const double*, double*, double*, int, int, int, int, cudaStream_t
);

} // namespace gsplat2ct
