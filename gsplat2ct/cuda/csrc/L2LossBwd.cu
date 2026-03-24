/**
 * L2 Loss Backward CUDA Kernels
 * 
 * Implements backward pass for L2 (Mean Squared Error) loss.
 * 
 * Kernels:
 * 1. compute_l2_grad_kernel: Computes first-order gradient d(L2)/d(pred) = 2*(pred - target)/N
 * 2. compute_l2_hessian_kernel: Computes second-order derivative (constant) = 2/N
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
// L2 Gradient Kernel (First Order)
// ============================================================================

template <typename S>
__global__ void compute_l2_grad_kernel(
    const S* __restrict__ prediction,
    const S* __restrict__ target,
    const S* __restrict__ grad_output,
    S* __restrict__ grad_pred,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    S diff = prediction[idx] - target[idx];
    
    // d(MSE)/d(pred) = 2 * (pred - target) / N
    // grad_output chains the upstream gradient
    grad_pred[idx] = grad_output[0] * static_cast<S>(2) * diff / static_cast<S>(total_elements);
}

// ============================================================================
// L2 Hessian Kernel (Second Order)
// ============================================================================

template <typename S>
__global__ void compute_l2_hessian_kernel(
    const S* __restrict__ grad_output,
    S* __restrict__ hessian_diag,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    // d^2(MSE)/d(pred)^2 = 2 / N (constant for all elements)
    hessian_diag[idx] = grad_output[0] * static_cast<S>(2) / static_cast<S>(total_elements);
}

// ============================================================================
// Launch Functions Implementation
// ============================================================================

template <typename S>
void launch_l2_loss_bwd(
    const S* __restrict__ prediction,
    const S* __restrict__ target,
    const S* __restrict__ grad_output,
    S* __restrict__ grad_pred,
    int total_elements,
    cudaStream_t stream
) {
    int threads = GSPLAT2CT_N_THREADS;
    int blocks = (total_elements + threads - 1) / threads;
    
    compute_l2_grad_kernel<S><<<blocks, threads, 0, stream>>>(
        prediction, target, grad_output, grad_pred, total_elements
    );
}

template <typename S>
void launch_l2_loss_bwd2(
    const S* __restrict__ grad_output,
    S* __restrict__ hessian_diag,
    int total_elements,
    cudaStream_t stream
) {
    int threads = GSPLAT2CT_N_THREADS;
    int blocks = (total_elements + threads - 1) / threads;
    
    compute_l2_hessian_kernel<S><<<blocks, threads, 0, stream>>>(
        grad_output, hessian_diag, total_elements
    );
}

// Explicit template instantiations
template void launch_l2_loss_bwd<float>(
    const float*, const float*, const float*, float*, int, cudaStream_t
);

template void launch_l2_loss_bwd<double>(
    const double*, const double*, const double*, double*, int, cudaStream_t
);

template void launch_l2_loss_bwd2<float>(
    const float*, float*, int, cudaStream_t
);

template void launch_l2_loss_bwd2<double>(
    const double*, double*, int, cudaStream_t
);

} // namespace gsplat2ct
