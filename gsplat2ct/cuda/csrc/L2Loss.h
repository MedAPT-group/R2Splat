/**
 * L2 Loss CUDA Kernel Launch Functions
 * 
 * This header declares the launch functions that bridge between
 * C++ tensor processing (L2Loss.cpp) and CUDA kernels
 * (L2LossFwd.cu, L2LossBwd.cu).
 * 
 * Following gsplat's pattern:
 * - Template functions with typename S for scalar type
 * - Raw pointers with __restrict__ for no-aliasing optimization
 * 
 * Author: Dy164
 * Date: 2026-02-06
 */

#ifndef GSPLAT2CT_L2LOSS_H
#define GSPLAT2CT_L2LOSS_H

#include "../include/common.h"
#include <cuda_runtime.h>

namespace gsplat2ct {

// ============================================================================
// L2 Loss Forward Launch Function
// ============================================================================

/**
 * Launch L2 Loss forward pass CUDA kernel
 * 
 * Computes per-element squared difference: (prediction - target)^2
 * 
 * @tparam S Scalar type (float, double)
 * @param prediction Input prediction tensor (flattened)
 * @param target Input target tensor (flattened)
 * @param l2_map Output per-element squared error
 * @param total_elements Total number of elements
 * @param stream CUDA stream
 */
template <typename S>
void launch_l2_loss_fwd(
    const S* __restrict__ prediction,
    const S* __restrict__ target,
    S* __restrict__ l2_map,
    int total_elements,
    cudaStream_t stream
);

// ============================================================================
// L2 Loss Backward Launch Function (First Order)
// ============================================================================

/**
 * Launch L2 Loss first-order backward pass CUDA kernel
 * 
 * Computes gradient: d(MSE)/d(pred) = 2 * (pred - target) / N
 * 
 * @tparam S Scalar type (float, double)
 * @param prediction Input prediction tensor (flattened)
 * @param target Input target tensor (flattened)
 * @param grad_output Upstream gradient (scalar)
 * @param grad_pred Output gradient w.r.t. prediction
 * @param total_elements Total number of elements
 * @param stream CUDA stream
 */
template <typename S>
void launch_l2_loss_bwd(
    const S* __restrict__ prediction,
    const S* __restrict__ target,
    const S* __restrict__ grad_output,
    S* __restrict__ grad_pred,
    int total_elements,
    cudaStream_t stream
);

// ============================================================================
// L2 Loss Backward Launch Function (Second Order - Diagonal Hessian)
// ============================================================================

/**
 * Launch L2 Loss second-order backward pass CUDA kernel
 * 
 * Computes diagonal Hessian: d^2(MSE)/d(pred)^2 = 2 / N (constant)
 * 
 * @tparam S Scalar type (float, double)
 * @param grad_output Upstream gradient (scalar)
 * @param hessian_diag Output diagonal Hessian
 * @param total_elements Total number of elements
 * @param stream CUDA stream
 */
template <typename S>
void launch_l2_loss_bwd2(
    const S* __restrict__ grad_output,
    S* __restrict__ hessian_diag,
    int total_elements,
    cudaStream_t stream
);

} // namespace gsplat2ct

#endif // GSPLAT2CT_L2LOSS_H
