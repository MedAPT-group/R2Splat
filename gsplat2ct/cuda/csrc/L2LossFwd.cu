/**
 * L2 Loss Forward CUDA Kernel
 * 
 * Implements forward pass for L2 (Mean Squared Error) loss.
 * 
 * Kernel:
 *   compute_l2_map_kernel: Computes (prediction - target)^2 per element
 * 
 * Supports arbitrary 3D tensors - works for both:
 *   - Batched images: (B, H, W)
 *   - Unbatched volumes: (D, H, W)
 * 
 * Author: Dy164
 * Date: 2026-02-06
 */

#include "L2Loss.h"
#include "../include/common.h"
#include <cuda_runtime.h>

namespace gsplat2ct {

// ============================================================================
// L2 Map Kernel
// ============================================================================

template <typename S>
__global__ void compute_l2_map_kernel(
    const S* __restrict__ prediction,
    const S* __restrict__ target,
    S* __restrict__ l2_map,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    S diff = prediction[idx] - target[idx];
    l2_map[idx] = diff * diff;
}

// ============================================================================
// Launch Function Implementation
// ============================================================================

template <typename S>
void launch_l2_loss_fwd(
    const S* __restrict__ prediction,
    const S* __restrict__ target,
    S* __restrict__ l2_map,
    int total_elements,
    cudaStream_t stream
) {
    int threads = GSPLAT2CT_N_THREADS;
    int blocks = (total_elements + threads - 1) / threads;
    
    compute_l2_map_kernel<S><<<blocks, threads, 0, stream>>>(
        prediction, target, l2_map, total_elements
    );
}

// Explicit template instantiations
template void launch_l2_loss_fwd<float>(
    const float*, const float*, float*, int, cudaStream_t
);

template void launch_l2_loss_fwd<double>(
    const double*, const double*, double*, int, cudaStream_t
);

} // namespace gsplat2ct
