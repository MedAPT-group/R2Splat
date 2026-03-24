/**
 * MLP Classification Tensor Processing Layer
 * 
 * This file handles PyTorch tensor operations and calls CUDA kernel
 * launch functions. It serves as the bridge between Python-exposed
 * functions (declared in Ops.h) and CUDA kernels.
 * 
 * Following gsplat's pattern: tensor validation, allocation, and
 * delegation to launch functions.
 * 
 * Author: Dy164
 * Date: 2025-12-03
 */

#include <cuda_runtime.h>
#include "../include/Ops.h"
#include "MLPClassification.h"

namespace gsplat2ct {

// ============================================================================
// Forward Pass
// ============================================================================

std::tuple<
    torch::Tensor,  // sigma
    torch::Tensor,  // loss
    torch::Tensor,  // a1
    torch::Tensor,  // z1
    torch::Tensor   // a2
> mlp_classification_fwd(
    const torch::Tensor &w1,
    const torch::Tensor &b1,
    const torch::Tensor &w2,
    const torch::Tensor &b2,
    const torch::Tensor &x,
    const torch::Tensor &y,
    bool save_intermediates,
    bool compute_loss
) {
    // Check device
    TORCH_CHECK(w1.is_cuda(), "w1 must be a CUDA tensor");
    TORCH_CHECK(b1.is_cuda(), "b1 must be a CUDA tensor");
    TORCH_CHECK(w2.is_cuda(), "w2 must be a CUDA tensor");
    TORCH_CHECK(b2.is_cuda(), "b2 must be a CUDA tensor");
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    
    // Check contiguous
    auto w1_c = w1.contiguous();
    auto b1_c = b1.contiguous();
    auto w2_c = w2.contiguous();
    auto b2_c = b2.contiguous();
    auto x_c = x.contiguous();
    auto y_c = y.contiguous();
    
    // Get dimensions
    int m = w1_c.size(0);  // hidden dimension
    int i = w1_c.size(1);  // input dimension
    int o = w2_c.size(0);  // output dimension
    
    // Check shapes
    TORCH_CHECK(b1_c.size(0) == m, "b1 shape mismatch");
    TORCH_CHECK(w2_c.size(1) == m, "w2 shape mismatch");
    TORCH_CHECK(b2_c.size(0) == o, "b2 shape mismatch");
    
    // Handle both single sample and batch
    bool is_batch = x_c.dim() == 2;
    int batch_size = is_batch ? x_c.size(0) : 1;
    
    if (!is_batch) {
        TORCH_CHECK(x_c.size(0) == i, "x shape mismatch");
        TORCH_CHECK(y_c.size(0) == o, "y shape mismatch");
    } else {
        TORCH_CHECK(x_c.size(1) == i, "x shape mismatch");
        TORCH_CHECK(y_c.size(1) == o, "y shape mismatch");
    }
    
    // Prepare output tensors
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x_c.device());
    torch::Tensor sigma = torch::empty({batch_size, o}, options);
    torch::Tensor loss = compute_loss ? torch::zeros({}, options) : torch::Tensor();
    
    torch::Tensor a1, z1, a2;
    if (save_intermediates) {
        a1 = torch::empty({batch_size, m}, options);
        z1 = torch::empty({batch_size, m}, options);
        a2 = torch::empty({batch_size, o}, options);
    } else {
        a1 = torch::Tensor();
        z1 = torch::Tensor();
        a2 = torch::Tensor();
    }
    
    // Allocate device memory for sample loss if needed
    torch::Tensor sample_loss_tensor;
    if (compute_loss) {
        sample_loss_tensor = torch::zeros({}, options);
    }
    
    // Process each sample in the batch
    for (int b = 0; b < batch_size; b++) {
        const float* x_ptr = is_batch ? x_c[b].data_ptr<float>() : x_c.data_ptr<float>();
        const float* y_ptr = is_batch ? y_c[b].data_ptr<float>() : y_c.data_ptr<float>();
        float* sigma_ptr = (batch_size == 1) ? sigma.data_ptr<float>() : sigma[b].data_ptr<float>();
        
        float* a1_ptr = save_intermediates ? ((batch_size == 1) ? a1.data_ptr<float>() : a1[b].data_ptr<float>()) : nullptr;
        float* z1_ptr = save_intermediates ? ((batch_size == 1) ? z1.data_ptr<float>() : z1[b].data_ptr<float>()) : nullptr;
        float* a2_ptr = save_intermediates ? ((batch_size == 1) ? a2.data_ptr<float>() : a2[b].data_ptr<float>()) : nullptr;
        
        float* loss_ptr = compute_loss ? sample_loss_tensor.data_ptr<float>() : nullptr;
        
        // Call CUDA kernel launch function
        launch_mlp_classification_fwd(
            w1_c.data_ptr<float>(), b1_c.data_ptr<float>(),
            w2_c.data_ptr<float>(), b2_c.data_ptr<float>(),
            x_ptr, y_ptr,
            i, m, o,
            sigma_ptr, loss_ptr,
            a1_ptr, z1_ptr, a2_ptr,
            save_intermediates, compute_loss
        );
        
        if (compute_loss) {
            // Synchronize to ensure loss is computed before copying
            cudaDeviceSynchronize();
            loss += sample_loss_tensor;
        }
    }
    
    // Average loss over batch
    if (compute_loss && batch_size > 1) {
        loss /= batch_size;
    }
    
    // Return tuple (squeeze if single sample)
    if (!is_batch) {
        sigma = sigma.squeeze(0);
        if (save_intermediates) {
            a1 = a1.squeeze(0);
            z1 = z1.squeeze(0);
            a2 = a2.squeeze(0);
        }
    }
    
    return std::make_tuple(sigma, loss, a1, z1, a2);
}

// ============================================================================
// Backward Pass (Gradients Only)
// ============================================================================

std::tuple<
    torch::Tensor,  // grad_w1
    torch::Tensor,  // grad_b1
    torch::Tensor,  // grad_w2
    torch::Tensor   // grad_b2
> mlp_classification_bwd(
    const torch::Tensor &w1,
    const torch::Tensor &b1,
    const torch::Tensor &w2,
    const torch::Tensor &b2,
    const torch::Tensor &x,
    const torch::Tensor &y,
    const torch::Tensor &a1,
    const torch::Tensor &z1,
    const torch::Tensor &a2,
    const torch::Tensor &sigma,
    bool accumulate_grad
) {
    // Check device
    TORCH_CHECK(w1.is_cuda(), "w1 must be a CUDA tensor");
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    
    // Get dimensions
    int m = w1.size(0);
    int i = w1.size(1);
    int o = w2.size(0);
    
    // Handle batch
    bool is_batch = x.dim() == 2;
    int batch_size = is_batch ? x.size(0) : 1;
    
    // Allocate gradient tensors
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    torch::Tensor grad_w1 = torch::zeros({m, i}, options);
    torch::Tensor grad_b1 = torch::zeros({m}, options);
    torch::Tensor grad_w2 = torch::zeros({o, m}, options);
    torch::Tensor grad_b2 = torch::zeros({o}, options);
    
    // Process each sample
    for (int b = 0; b < batch_size; b++) {
        const float* x_ptr = is_batch ? x[b].data_ptr<float>() : x.data_ptr<float>();
        const float* y_ptr = is_batch ? y[b].data_ptr<float>() : y.data_ptr<float>();
        const float* a1_ptr = is_batch ? a1[b].data_ptr<float>() : a1.data_ptr<float>();
        const float* z1_ptr = is_batch ? z1[b].data_ptr<float>() : z1.data_ptr<float>();
        const float* a2_ptr = is_batch ? a2[b].data_ptr<float>() : a2.data_ptr<float>();
        const float* sigma_ptr = is_batch ? sigma[b].data_ptr<float>() : sigma.data_ptr<float>();
        
        // Call CUDA kernel launch function
        launch_mlp_classification_bwd(
            a1_ptr, z1_ptr, a2_ptr, sigma_ptr,
            w1.data_ptr<float>(), w2.data_ptr<float>(),
            x_ptr, y_ptr,
            i, m, o,
            grad_w1.data_ptr<float>(), grad_b1.data_ptr<float>(),
            grad_w2.data_ptr<float>(), grad_b2.data_ptr<float>(),
            accumulate_grad || (b > 0)  // Accumulate for all samples after first
        );
    }
    
    // Average gradients over batch
    if (batch_size > 1) {
        grad_w1 /= batch_size;
        grad_b1 /= batch_size;
        grad_w2 /= batch_size;
        grad_b2 /= batch_size;
    }
    
    return std::make_tuple(grad_w1, grad_b1, grad_w2, grad_b2);
}

// ============================================================================
// Backward Pass (Gradients + Hessians)
// ============================================================================

std::tuple<
    torch::Tensor,  // grad_w1
    torch::Tensor,  // grad_b1
    torch::Tensor,  // grad_w2
    torch::Tensor,  // grad_b2
    torch::Tensor,  // hess_b2_full
    torch::Tensor,  // hess_b1_full
    torch::Tensor,  // hess_w2_diag
    torch::Tensor   // hess_w1_diag
> mlp_classification_bwd2(
    const torch::Tensor &w1,
    const torch::Tensor &b1,
    const torch::Tensor &w2,
    const torch::Tensor &b2,
    const torch::Tensor &x,
    const torch::Tensor &y,
    const torch::Tensor &a1,
    const torch::Tensor &z1,
    const torch::Tensor &a2,
    const torch::Tensor &sigma,
    bool accumulate_grad
) {
    // Check device
    TORCH_CHECK(w1.is_cuda(), "w1 must be a CUDA tensor");
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    
    // Get dimensions
    int m = w1.size(0);
    int i = w1.size(1);
    int o = w2.size(0);
    
    // Handle batch (support multiple samples for Hessian)
    bool is_batch = x.dim() == 2;
    int batch_size = is_batch ? x.size(0) : 1;
    
    // Allocate gradient tensors
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    torch::Tensor grad_w1 = torch::zeros({m, i}, options);
    torch::Tensor grad_b1 = torch::zeros({m}, options);
    torch::Tensor grad_w2 = torch::zeros({o, m}, options);
    torch::Tensor grad_b2 = torch::zeros({o}, options);
    
    // Allocate Hessian tensors (will accumulate over batch)
    torch::Tensor hess_b2_full = torch::zeros({o, o}, options);
    torch::Tensor hess_b1_full = torch::zeros({m, m}, options);
    torch::Tensor hess_w2_full = torch::zeros({o * m, o * m}, options);
    torch::Tensor hess_w1_full = torch::zeros({m * i, m * i}, options);
    
    // Allocate temporary Hessian buffers for single sample computation
    // These are reused for each sample to avoid repeated allocation
    torch::Tensor hess_b2_temp = torch::zeros({o, o}, options);
    torch::Tensor hess_b1_temp = torch::zeros({m, m}, options);
    torch::Tensor hess_w2_temp = torch::zeros({o * m, o * m}, options);
    torch::Tensor hess_w1_temp = torch::zeros({m * i, m * i}, options);
    
    // Process each sample in the batch
    for (int b = 0; b < batch_size; b++) {
        // Get data pointers for sample b
        const float* x_ptr = is_batch ? x[b].data_ptr<float>() : x.data_ptr<float>();
        const float* y_ptr = is_batch ? y[b].data_ptr<float>() : y.data_ptr<float>();
        const float* a1_ptr = is_batch ? a1[b].data_ptr<float>() : a1.data_ptr<float>();
        const float* z1_ptr = is_batch ? z1[b].data_ptr<float>() : z1.data_ptr<float>();
        const float* a2_ptr = is_batch ? a2[b].data_ptr<float>() : a2.data_ptr<float>();
        const float* sigma_ptr = is_batch ? sigma[b].data_ptr<float>() : sigma.data_ptr<float>();
        
        // Compute gradients and Hessians for this sample into temp buffers
        // Always use temp buffers to avoid dependency issues between Hessian layers
        // Gradients: accumulate in kernel if not first sample or if accumulate_grad is true
        bool should_accumulate_grad = accumulate_grad || (b > 0);
        
        launch_mlp_classification_bwd2(
            a1_ptr, z1_ptr, a2_ptr, sigma_ptr,
            w1.data_ptr<float>(), w2.data_ptr<float>(),
            x_ptr, y_ptr,
            i, m, o,
            grad_w1.data_ptr<float>(), grad_b1.data_ptr<float>(),
            grad_w2.data_ptr<float>(), grad_b2.data_ptr<float>(),
            hess_b2_temp.data_ptr<float>(),
            hess_b1_temp.data_ptr<float>(),
            hess_w2_temp.data_ptr<float>(),
            hess_w1_temp.data_ptr<float>(),
            should_accumulate_grad  // Accumulate gradients in kernel
        );
        
        // Manually accumulate Hessians (outside kernel to avoid cross-layer dependencies)
        hess_b2_full += hess_b2_temp;
        hess_b1_full += hess_b1_temp;
        hess_w2_full += hess_w2_temp;
        hess_w1_full += hess_w1_temp;
    }
    
    // Average gradients and Hessians over batch
    if (batch_size > 1) {
        grad_w1 /= batch_size;
        grad_b1 /= batch_size;
        grad_w2 /= batch_size;
        grad_b2 /= batch_size;
        
        hess_b2_full /= batch_size;
        hess_b1_full /= batch_size;
        hess_w2_full /= batch_size;
        hess_w1_full /= batch_size;
    }
    
    return std::make_tuple(
        grad_w1, grad_b1, grad_w2, grad_b2,
        hess_b2_full, hess_b1_full, hess_w2_full, hess_w1_full
    );
}

}  // namespace gsplat2ct
