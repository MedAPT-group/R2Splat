/**
 * Loss Functions C++ Wrapper
 * 
 * This file provides the C++ tensor wrapper functions that:
 * 1. Validate input tensors (CUDA, contiguous, shape)
 * 2. Allocate output tensors
 * 3. Dispatch to type-specific CUDA kernels
 * 
 * Following gsplat's pattern for tensor handling.
 * 
 * Author: Dy164
 * Date: 2025-12-18
 */

#include "Loss.h"
#include "../include/Ops.h"
#include <torch/extension.h>
#include <tuple>
#include <cmath>

namespace gsplat2ct {

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Create a 1D Gaussian kernel
 */
torch::Tensor create_gaussian_kernel_1d(int window_size, float sigma) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto x = torch::arange(window_size, options) - (window_size - 1) / 2.0f;
    auto kernel = torch::exp(-x.pow(2) / (2.0f * sigma * sigma));
    return kernel / kernel.sum();
}

/**
 * Create a 2D Gaussian kernel from outer product
 */
torch::Tensor create_gaussian_kernel_2d(int window_size, float sigma) {
    auto kernel_1d = create_gaussian_kernel_1d(window_size, sigma);
    auto kernel_2d = torch::outer(kernel_1d, kernel_1d);
    return kernel_2d;
}

// ============================================================================
// ImageLoss Forward
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
image_loss_fwd(
    const torch::Tensor& prediction,      // (B, H, W)
    const torch::Tensor& target,          // (B, H, W)
    int window_size,
    float lambda_l2,
    float lambda_ssim,
    bool compute_statistics
) {
    GSPLAT2CT_CHECK_INPUT(prediction);
    GSPLAT2CT_CHECK_INPUT(target);
    
    TORCH_CHECK(prediction.dim() == 3, "prediction must be 3D (B, H, W)");
    TORCH_CHECK(target.dim() == 3, "target must be 3D (B, H, W)");
    TORCH_CHECK(prediction.sizes() == target.sizes(), 
                "prediction and target must have same shape");
    TORCH_CHECK(window_size % 2 == 1, "window_size must be odd");
    
    int B = prediction.size(0);
    int H = prediction.size(1);
    int W = prediction.size(2);
    
    // Create Gaussian window and move to device
    float sigma = 1.5f;
    auto gaussian = create_gaussian_kernel_2d(window_size, sigma)
                    .to(prediction.device(), prediction.dtype())
                    .contiguous();
    
    // Allocate outputs
    auto options = prediction.options();
    auto l2_map = torch::empty({B, H, W}, options);
    auto ssim_map = torch::empty({B, H, W}, options);
    
    // Optional statistics for backward reuse
    torch::Tensor mu_r, mu_m, f1, f2, f3, f4;
    if (compute_statistics) {
        mu_r = torch::empty({B, H, W}, options);
        mu_m = torch::empty({B, H, W}, options);
        f1 = torch::empty({B, H, W}, options);
        f2 = torch::empty({B, H, W}, options);
        f3 = torch::empty({B, H, W}, options);
        f4 = torch::empty({B, H, W}, options);
    } else {
        // Create empty tensors
        mu_r = torch::empty({0}, options);
        mu_m = torch::empty({0}, options);
        f1 = torch::empty({0}, options);
        f2 = torch::empty({0}, options);
        f3 = torch::empty({0}, options);
        f4 = torch::empty({0}, options);
    }
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Dispatch based on dtype
    AT_DISPATCH_FLOATING_TYPES(prediction.scalar_type(), "image_loss_fwd", [&] {
        launch_image_loss_fwd<scalar_t>(
            prediction.data_ptr<scalar_t>(),
            target.data_ptr<scalar_t>(),
            gaussian.data_ptr<scalar_t>(),
            l2_map.data_ptr<scalar_t>(),
            compute_statistics ? mu_r.data_ptr<scalar_t>() : nullptr,
            compute_statistics ? mu_m.data_ptr<scalar_t>() : nullptr,
            compute_statistics ? f1.data_ptr<scalar_t>() : nullptr,
            compute_statistics ? f2.data_ptr<scalar_t>() : nullptr,
            compute_statistics ? f3.data_ptr<scalar_t>() : nullptr,
            compute_statistics ? f4.data_ptr<scalar_t>() : nullptr,
            ssim_map.data_ptr<scalar_t>(),
            B, H, W, window_size,
            compute_statistics,
            stream
        );
    });
    
    // Compute scalar losses
    auto l2_loss = l2_map.mean();
    auto ssim_loss = (1.0f - ssim_map.mean());  // D-SSIM
    auto loss = lambda_l2 * l2_loss + lambda_ssim * ssim_loss;
    
    return std::make_tuple(
        loss, l2_loss, ssim_loss,
        mu_r, mu_m, f1, f2, f3, f4
    );
}

// ============================================================================
// ImageLoss Backward (First Order)
// ============================================================================

torch::Tensor image_loss_bwd(
    const torch::Tensor& prediction,      // (B, H, W)
    const torch::Tensor& target,          // (B, H, W)
    const torch::Tensor& mu_r,            // (B, H, W)
    const torch::Tensor& mu_m,            // (B, H, W)
    const torch::Tensor& f1,              // (B, H, W)
    const torch::Tensor& f2,              // (B, H, W)
    const torch::Tensor& f3,              // (B, H, W)
    const torch::Tensor& f4,              // (B, H, W)
    const torch::Tensor& grad_output,     // scalar
    int window_size,
    float lambda_l2,
    float lambda_ssim
) {
    GSPLAT2CT_CHECK_INPUT(prediction);
    GSPLAT2CT_CHECK_INPUT(target);
    
    int B = prediction.size(0);
    int H = prediction.size(1);
    int W = prediction.size(2);
    
    // Create Gaussian window
    float sigma = 1.5f;
    auto gaussian = create_gaussian_kernel_2d(window_size, sigma)
                    .to(prediction.device(), prediction.dtype())
                    .contiguous();
    
    // Allocate output
    auto grad_pred = torch::zeros({B, H, W}, prediction.options());
    
    // SSIM constants
    float C1 = 0.0001f;  // 0.01^2
    float C2 = 0.0009f;  // 0.03^2
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    AT_DISPATCH_FLOATING_TYPES(prediction.scalar_type(), "image_loss_bwd", [&] {
        launch_image_loss_bwd<scalar_t>(
            prediction.data_ptr<scalar_t>(),
            target.data_ptr<scalar_t>(),
            gaussian.data_ptr<scalar_t>(),
            mu_r.data_ptr<scalar_t>(),
            mu_m.data_ptr<scalar_t>(),
            f1.data_ptr<scalar_t>(),
            f2.data_ptr<scalar_t>(),
            f3.data_ptr<scalar_t>(),
            f4.data_ptr<scalar_t>(),
            grad_output.data_ptr<scalar_t>(),
            grad_pred.data_ptr<scalar_t>(),
            B, H, W, window_size,
            static_cast<scalar_t>(lambda_l2),
            static_cast<scalar_t>(lambda_ssim),
            static_cast<scalar_t>(C1),
            static_cast<scalar_t>(C2),
            stream
        );
    });
    
    return grad_pred;
}

// ============================================================================
// ImageLoss Backward (Second Order - Hessian Diagonal)
// ============================================================================

torch::Tensor image_loss_bwd2(
    const torch::Tensor& prediction,      // (B, H, W)
    const torch::Tensor& target,          // (B, H, W)
    const torch::Tensor& mu_r,            // (B, H, W)
    const torch::Tensor& mu_m,            // (B, H, W)
    const torch::Tensor& f1,              // (B, H, W)
    const torch::Tensor& f2,              // (B, H, W)
    const torch::Tensor& f3,              // (B, H, W)
    const torch::Tensor& f4,              // (B, H, W)
    const torch::Tensor& grad_output,     // scalar
    int window_size,
    float lambda_l2,
    float lambda_ssim
) {
    GSPLAT2CT_CHECK_INPUT(prediction);
    GSPLAT2CT_CHECK_INPUT(target);
    
    int B = prediction.size(0);
    int H = prediction.size(1);
    int W = prediction.size(2);
    
    // Create Gaussian window
    float sigma = 1.5f;
    auto gaussian = create_gaussian_kernel_2d(window_size, sigma)
                    .to(prediction.device(), prediction.dtype())
                    .contiguous();
    
    // Allocate output
    auto hessian_diag = torch::zeros({B, H, W}, prediction.options());
    
    // SSIM constants
    float C1 = 0.0001f;
    float C2 = 0.0009f;
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    AT_DISPATCH_FLOATING_TYPES(prediction.scalar_type(), "image_loss_bwd2", [&] {
        launch_image_loss_bwd2<scalar_t>(
            prediction.data_ptr<scalar_t>(),
            target.data_ptr<scalar_t>(),
            gaussian.data_ptr<scalar_t>(),
            mu_r.data_ptr<scalar_t>(),
            mu_m.data_ptr<scalar_t>(),
            f1.data_ptr<scalar_t>(),
            f2.data_ptr<scalar_t>(),
            f3.data_ptr<scalar_t>(),
            f4.data_ptr<scalar_t>(),
            grad_output.data_ptr<scalar_t>(),
            hessian_diag.data_ptr<scalar_t>(),
            B, H, W, window_size,
            static_cast<scalar_t>(lambda_l2),
            static_cast<scalar_t>(lambda_ssim),
            static_cast<scalar_t>(C1),
            static_cast<scalar_t>(C2),
            stream
        );
    });
    
    return hessian_diag;
}

// ============================================================================
// VolumeLoss Forward
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
volume_loss_fwd(
    const torch::Tensor& volume,           // (B, D, H, W)
    float eps,
    float lambda_tv
) {
    GSPLAT2CT_CHECK_INPUT(volume);
    TORCH_CHECK(volume.dim() == 4, "volume must be 4D (B, D, H, W)");
    
    int B = volume.size(0);
    int D = volume.size(1);
    int H = volume.size(2);
    int W = volume.size(3);
    
    // Allocate outputs
    auto options = volume.options();
    auto tv_map = torch::empty({B, D, H, W}, options);
    auto tv_loss = torch::zeros({1}, options);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    AT_DISPATCH_FLOATING_TYPES(volume.scalar_type(), "volume_loss_fwd", [&] {
        launch_volume_loss_fwd<scalar_t>(
            volume.data_ptr<scalar_t>(),
            tv_loss.data_ptr<scalar_t>(),
            tv_map.data_ptr<scalar_t>(),
            B, D, H, W,
            static_cast<scalar_t>(eps),
            stream
        );
    });
    
    // Normalize by number of elements
    int N = B * D * H * W;
    tv_loss = tv_loss / static_cast<float>(N);
    
    auto loss = lambda_tv * tv_loss;
    
    return std::make_tuple(loss.squeeze(), tv_loss.squeeze(), tv_map);
}

// ============================================================================
// VolumeLoss Backward (First Order)
// ============================================================================

torch::Tensor volume_loss_bwd(
    const torch::Tensor& volume,          // (B, D, H, W)
    const torch::Tensor& grad_output,     // scalar
    float eps,
    float lambda_tv
) {
    GSPLAT2CT_CHECK_INPUT(volume);
    
    int B = volume.size(0);
    int D = volume.size(1);
    int H = volume.size(2);
    int W = volume.size(3);
    
    // Allocate output
    auto grad_volume = torch::zeros({B, D, H, W}, volume.options());
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    AT_DISPATCH_FLOATING_TYPES(volume.scalar_type(), "volume_loss_bwd", [&] {
        launch_volume_loss_bwd<scalar_t>(
            volume.data_ptr<scalar_t>(),
            grad_output.data_ptr<scalar_t>(),
            grad_volume.data_ptr<scalar_t>(),
            B, D, H, W,
            static_cast<scalar_t>(eps),
            stream
        );
    });
    
    return grad_volume * lambda_tv;
}

// ============================================================================
// VolumeLoss Backward (Second Order - Hessian Diagonal)
// ============================================================================

torch::Tensor volume_loss_bwd2(
    const torch::Tensor& volume,          // (B, D, H, W)
    const torch::Tensor& grad_output,     // scalar
    float eps,
    float lambda_tv
) {
    GSPLAT2CT_CHECK_INPUT(volume);
    
    int B = volume.size(0);
    int D = volume.size(1);
    int H = volume.size(2);
    int W = volume.size(3);
    
    // Allocate output
    auto hessian_diag = torch::zeros({B, D, H, W}, volume.options());
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    AT_DISPATCH_FLOATING_TYPES(volume.scalar_type(), "volume_loss_bwd2", [&] {
        launch_volume_loss_bwd2<scalar_t>(
            volume.data_ptr<scalar_t>(),
            grad_output.data_ptr<scalar_t>(),
            hessian_diag.data_ptr<scalar_t>(),
            B, D, H, W,
            static_cast<scalar_t>(eps),
            stream
        );
    });
    
    return hessian_diag * lambda_tv;
}

} // namespace gsplat2ct
