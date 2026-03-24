/**
 * L2 Loss C++ Wrapper
 * 
 * This file provides the C++ tensor wrapper functions that:
 * 1. Validate input tensors (CUDA, contiguous, shape)
 * 2. Allocate output tensors
 * 3. Dispatch to type-specific CUDA kernels
 * 
 * Supports 3D tensors for both:
 *   - Batched images: (B, H, W)
 *   - Unbatched volumes: (D, H, W)
 * 
 * Following gsplat's pattern for tensor handling.
 * 
 * Author: Dy164
 * Date: 2026-02-06
 */

#include "L2Loss.h"
#include "../include/Ops.h"
#include <torch/extension.h>
#include <tuple>

namespace gsplat2ct {

// ============================================================================
// L2 Loss Forward
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor>
l2_loss_fwd(
    const torch::Tensor& prediction,      // 3D tensor
    const torch::Tensor& target           // 3D tensor (same shape)
) {
    GSPLAT2CT_CHECK_INPUT(prediction);
    GSPLAT2CT_CHECK_INPUT(target);
    
    TORCH_CHECK(prediction.dim() == 3,
                "prediction must be 3D (B, H, W) or (D, H, W), got dim=", prediction.dim());
    TORCH_CHECK(target.dim() == 3,
                "target must be 3D (B, H, W) or (D, H, W), got dim=", target.dim());
    TORCH_CHECK(prediction.sizes() == target.sizes(),
                "prediction and target must have same shape");
    
    int total_elements = prediction.numel();
    
    // Allocate output
    auto options = prediction.options();
    auto l2_map = torch::empty_like(prediction);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    AT_DISPATCH_FLOATING_TYPES(prediction.scalar_type(), "l2_loss_fwd", [&] {
        launch_l2_loss_fwd<scalar_t>(
            prediction.data_ptr<scalar_t>(),
            target.data_ptr<scalar_t>(),
            l2_map.data_ptr<scalar_t>(),
            total_elements,
            stream
        );
    });
    
    // Compute scalar loss: MSE = mean of squared differences
    auto loss = l2_map.mean();
    
    return std::make_tuple(loss, l2_map);
}

// ============================================================================
// L2 Loss Backward (First Order)
// ============================================================================

torch::Tensor l2_loss_bwd(
    const torch::Tensor& prediction,      // 3D tensor
    const torch::Tensor& target,          // 3D tensor
    const torch::Tensor& grad_output      // scalar
) {
    GSPLAT2CT_CHECK_INPUT(prediction);
    GSPLAT2CT_CHECK_INPUT(target);
    
    int total_elements = prediction.numel();
    
    // Allocate output
    auto grad_pred = torch::empty_like(prediction);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    AT_DISPATCH_FLOATING_TYPES(prediction.scalar_type(), "l2_loss_bwd", [&] {
        launch_l2_loss_bwd<scalar_t>(
            prediction.data_ptr<scalar_t>(),
            target.data_ptr<scalar_t>(),
            grad_output.data_ptr<scalar_t>(),
            grad_pred.data_ptr<scalar_t>(),
            total_elements,
            stream
        );
    });
    
    return grad_pred;
}

// ============================================================================
// L2 Loss Backward (Second Order - Hessian Diagonal)
// ============================================================================

torch::Tensor l2_loss_bwd2(
    const torch::Tensor& prediction,      // 3D tensor (for shape/dtype reference)
    const torch::Tensor& grad_output      // scalar
) {
    GSPLAT2CT_CHECK_INPUT(prediction);
    
    int total_elements = prediction.numel();
    
    // Allocate output
    auto hessian_diag = torch::empty_like(prediction);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    AT_DISPATCH_FLOATING_TYPES(prediction.scalar_type(), "l2_loss_bwd2", [&] {
        launch_l2_loss_bwd2<scalar_t>(
            grad_output.data_ptr<scalar_t>(),
            hessian_diag.data_ptr<scalar_t>(),
            total_elements,
            stream
        );
    });
    
    return hessian_diag;
}

} // namespace gsplat2ct
