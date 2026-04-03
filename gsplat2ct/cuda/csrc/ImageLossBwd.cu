/**
 * ImageLoss Backward CUDA Kernels
 * 
 * Implements backward pass for L2 + SSIM image loss.
 * Uses pixel-parallel strategy: each thread computes gradient for one pixel
 * by iterating through all windows that contain this pixel.
 * 
 * Kernels:
 * 1. compute_l2_grad_kernel: Computes L2 gradient d_l2 / d_pred
 * 2. compute_ssim_grad_pixel_parallel_kernel: Computes SSIM gradient per pixel
 * 3. compute_ssim_hessian_kernel: Computes second-order derivative (Hessian diagonal)
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
// L2/L1 Gradient Kernels
// ============================================================================

template <typename S>
__global__ void compute_l2_grad_kernel(
    const S* __restrict__ prediction,
    const S* __restrict__ target,
    const S* __restrict__ grad_output,
    S* __restrict__ grad_pred,
    S lambda_l2,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    S pred_val = prediction[idx];
    S tgt_val = target[idx];
    S diff = pred_val - tgt_val;
    
    // d((x-t)^2)/dx = 2*(x-t)
    S grad = grad_output[0] * lambda_l2 * static_cast<S>(2) * diff / static_cast<S>(total_elements);
    grad_pred[idx] += grad;
}

template <typename S>
__global__ void compute_l1_grad_kernel(
    const S* __restrict__ prediction,
    const S* __restrict__ target,
    const S* __restrict__ grad_output,
    S* __restrict__ grad_pred,
    S lambda_l1,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    S diff = prediction[idx] - target[idx];
    S sign = (diff > static_cast<S>(0)) ? static_cast<S>(1) :
             ((diff < static_cast<S>(0)) ? static_cast<S>(-1) : static_cast<S>(0));
    S grad = grad_output[0] * lambda_l1 * sign / static_cast<S>(total_elements);
    grad_pred[idx] += grad;
}

// ============================================================================
// SSIM Gradient Kernel (Pixel-Parallel)
// ============================================================================

/**
 * Pixel-parallel SSIM gradient computation.
 * Each thread handles one pixel and accumulates gradients from all windows
 * that contain this pixel. This avoids atomicAdd conflicts entirely.
 * 
 * For pixel (i, j), it contributes to windows at positions:
 *   (out_i, out_j) where:
 *   out_i in [max(0, i - window_size + 1), min(H, i + 1)]
 *   out_j in [max(0, j - window_size + 1), min(W, j + 1)]
 * 
 * Within each window, the pixel is at local position:
 *   (wi, wj) = (i - out_i, j - out_j)
 */
template <typename S, int MAX_WINDOW_SIZE = 32>
__global__ void compute_ssim_grad_pixel_parallel_kernel(
    const S* __restrict__ prediction,
    const S* __restrict__ target,
    const S* __restrict__ gaussian_window,
    const S* __restrict__ mu_r,
    const S* __restrict__ mu_m,
    const S* __restrict__ f1,
    const S* __restrict__ f2,
    const S* __restrict__ f3,
    const S* __restrict__ f4,
    const S* __restrict__ grad_output,
    S* __restrict__ grad_pred,
    int B, int H, int W,
    int window_size,
    S lambda_ssim,
    S C1, S C2
) {
    // Shared memory for Gaussian kernel
    __shared__ S s_gaussian[MAX_WINDOW_SIZE * MAX_WINDOW_SIZE];
    
    // Cooperative loading of Gaussian kernel
    int tid = threadIdx.x;
    int kernel_size = window_size * window_size;
    for (int i = tid; i < kernel_size; i += blockDim.x) {
        s_gaussian[i] = gaussian_window[i];
    }
    __syncthreads();
    
    // Each thread handles one pixel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = B * H * W;
    if (idx >= total_pixels) return;
    
    // Decode batch and spatial indices
    int b = idx / (H * W);
    int spatial_idx = idx % (H * W);
    int i = spatial_idx / W;
    int j = spatial_idx % W;
    
    int pad_size = window_size / 2;
    int out_H = H;
    int out_W = W;
    
    // Get values at this pixel
    S pred_val = prediction[idx];
    S tgt_val = target[idx];
    S grad_out = grad_output[0];
    
    S grad_sum = 0;
    
    // Iterate over all windows that contain this pixel
    // In padded coordinates: pixel is at (i + pad_size, j + pad_size)
    // Window at (out_i, out_j) covers padded coords [out_i, out_i + window_size) x [out_j, out_j + window_size)
    // So pixel is in window if:
    //   out_i <= i + pad_size < out_i + window_size
    //   out_j <= j + pad_size < out_j + window_size
    // Which means:
    //   i + pad_size - window_size + 1 <= out_i <= i + pad_size
    //   j + pad_size - window_size + 1 <= out_j <= j + pad_size
    
    int out_i_min = max(0, i - pad_size);
    int out_i_max = min(out_H - 1, i + pad_size);
    int out_j_min = max(0, j - pad_size);
    int out_j_max = min(out_W - 1, j + pad_size);
    
    for (int out_i = out_i_min; out_i <= out_i_max; ++out_i) {
        for (int out_j = out_j_min; out_j <= out_j_max; ++out_j) {
            // Local position within the window
            int wi = i - (out_i - pad_size);
            int wj = j - (out_j - pad_size);
            
            // Check if local position is valid
            if (wi < 0 || wi >= window_size || wj < 0 || wj >= window_size) continue;
            
            // Get statistics for this window
            int out_idx = b * out_H * out_W + out_i * out_W + out_j;
            S mu_r_val = mu_r[out_idx];
            S mu_m_val = mu_m[out_idx];
            S f1_val = f1[out_idx];
            S f2_val = f2[out_idx];
            S f3_val = f3[out_idx];
            S f4_val = f4[out_idx];
            
            // Gaussian weight at this local position
            S w = s_gaussian[wi * window_size + wj];
            
            // Compute gradient of SSIM w.r.t. this pixel
            // SSIM = (f1 * f2) / (f3 * f4)
            // f1 = 2 * mu_r * mu_m + C1
            // f2 = 2 * sigma_rm + C2
            // f3 = mu_r^2 + mu_m^2 + C1
            // f4 = sigma_r_sq + sigma_m_sq + C2
            
            // d(mu_r)/d(pred) = w
            // d(sigma_r_sq)/d(pred) = 2 * w * (pred - mu_r)
            // d(sigma_rm)/d(pred) = w * (tgt - mu_m)
            
            S sigma_r_sq = f4_val - C2 - (mu_m_val * mu_m_val - mu_r_val * mu_r_val);
            
            // Intermediate derivatives
            S d_f1_d_pred = 2 * w * mu_m_val;
            S d_f2_d_pred = 2 * w * (tgt_val - mu_m_val);
            S d_f3_d_pred = 2 * mu_r_val * w;
            S d_f4_d_pred = 2 * w * (pred_val - mu_r_val);
            
            // Quotient rule: d(f1*f2)/(f3*f4) = (f3*f4*(d_f1*f2 + f1*d_f2) - f1*f2*(d_f3*f4 + f3*d_f4)) / (f3*f4)^2
            S numer = f1_val * f2_val;
            S denom = f3_val * f4_val;
            S d_numer = d_f1_d_pred * f2_val + f1_val * d_f2_d_pred;
            S d_denom = d_f3_d_pred * f4_val + f3_val * d_f4_d_pred;
            
            S d_ssim = (d_numer * denom - numer * d_denom) / (denom * denom);
            
            // SSIM loss is 1 - mean(ssim_map), so gradient is -1/N * d_ssim
            grad_sum -= d_ssim / static_cast<S>(out_H * out_W);
        }
    }
    
    // Apply chain rule with lambda and grad_output
    grad_pred[idx] += grad_out * lambda_ssim * grad_sum / static_cast<S>(B);
}

// ============================================================================
// SSIM Hessian Kernel (Second-Order Derivative - Full Implementation)
// ============================================================================

/**
 * Computes the diagonal of the Hessian matrix for SSIM loss using complete formula.
 * 
 * This kernel implements the full Hessian computation based on the formula:
 * H = -SSIM * (term1 + term2)
 * 
 * where:
 *   term1 = (d(log SSIM)/d(pred))^2
 *   term2 = Σ_i (h_i * f_i - g_i^2) / f_i^2
 * 
 * This matches the PyTorch reference implementation in ImageLossTorch._compute_ssim_hess_by_window
 * 
 * Each thread handles one pixel and accumulates Hessian contributions from all windows
 * that contain this pixel (pixel-parallel approach).
 */
template <typename S, int MAX_WINDOW_SIZE = 32>
__global__ void compute_ssim_hessian_kernel(
    const S* __restrict__ prediction,
    const S* __restrict__ target,
    const S* __restrict__ gaussian_window,
    const S* __restrict__ mu_r,
    const S* __restrict__ mu_m,
    const S* __restrict__ f1,
    const S* __restrict__ f2,
    const S* __restrict__ f3,
    const S* __restrict__ f4,
    const S* __restrict__ grad_output,
    S* __restrict__ hessian_diag,
    int B, int H, int W,
    int window_size,
    S lambda_ssim,
    S C1, S C2
) {
    // Shared memory for Gaussian kernel
    __shared__ S s_gaussian[MAX_WINDOW_SIZE * MAX_WINDOW_SIZE];
    
    int tid = threadIdx.x;
    int kernel_size = window_size * window_size;
    for (int i = tid; i < kernel_size; i += blockDim.x) {
        s_gaussian[i] = gaussian_window[i];
    }
    __syncthreads();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = B * H * W;
    if (idx >= total_pixels) return;
    
    // Decode batch and spatial indices
    int b = idx / (H * W);
    int spatial_idx = idx % (H * W);
    int i = spatial_idx / W;
    int j = spatial_idx % W;
    
    int pad_size = window_size / 2;
    int out_H = H;
    int out_W = W;
    
    S pred_val = prediction[idx];
    S tgt_val = target[idx];
    S grad_out = grad_output[0];
    
    S hess_sum = 0;
    
    // Iterate over all windows that contain this pixel
    int out_i_min = max(0, i - pad_size);
    int out_i_max = min(out_H - 1, i + pad_size);
    int out_j_min = max(0, j - pad_size);
    int out_j_max = min(out_W - 1, j + pad_size);
    
    for (int out_i = out_i_min; out_i <= out_i_max; ++out_i) {
        for (int out_j = out_j_min; out_j <= out_j_max; ++out_j) {
            // Local position within the window
            int wi = i - (out_i - pad_size);
            int wj = j - (out_j - pad_size);
            
            if (wi < 0 || wi >= window_size || wj < 0 || wj >= window_size) continue;
            
            // Get statistics for this window
            int out_idx = b * out_H * out_W + out_i * out_W + out_j;
            S mu_r_val = mu_r[out_idx];
            S mu_m_val = mu_m[out_idx];
            S f1_val = f1[out_idx];
            S f2_val = f2[out_idx];
            S f3_val = f3[out_idx];
            S f4_val = f4[out_idx];
            
            // SSIM value for this window
            S numer = f1_val * f2_val;
            S denom = f3_val * f4_val;
            S ssim_val = numer / denom;
            
            // Gaussian weight at this local position
            S w = s_gaussian[wi * window_size + wj];
            
            // ========== First-order derivatives (g_i = d(f_i)/d(pred)) ==========
            // g1 = d(2*mu_r*mu_m)/d(pred) = 2*mu_m*w
            S g1 = 2 * w * mu_m_val;
            // g2 = d(2*sigma_rm)/d(pred) = 2*w*(tgt - mu_m)
            S g2 = 2 * w * (tgt_val - mu_m_val);
            // g3 = d(mu_r²)/d(pred) = 2*mu_r*w
            S g3 = 2 * w * mu_r_val;
            // g4 = d(sigma_r_sq)/d(pred) = 2*w*(pred - mu_r)
            S g4 = 2 * w * (pred_val - mu_r_val);
            
            // ========== Second-order derivatives (h_i = d²(f_i)/d(pred)²) ==========
            S h1 = 0;  // d²(2*mu_r*mu_m)/d(pred)² = 0
            S h2 = 0;  // d²(2*sigma_rm)/d(pred)² = 0
            S h3 = 2 * w * w;  // d²(mu_r²)/d(pred)² = 2*w²
            S h4 = 2 * w * (1 - w);  // d²(sigma_r_sq)/d(pred)² = 2*w*(1-w)
            
            // ========== Term 1: (log_deriv)² ==========
            S log_deriv = g1/f1_val + g2/f2_val - g3/f3_val - g4/f4_val;
            S term1 = log_deriv * log_deriv;
            
            // ========== Term 2: Σ_i (h_i*f_i - g_i²)/f_i² ==========
            S term2 = (h1 * f1_val - g1 * g1) / (f1_val * f1_val)
                    + (h2 * f2_val - g2 * g2) / (f2_val * f2_val)
                    - (h3 * f3_val - g3 * g3) / (f3_val * f3_val)
                    - (h4 * f4_val - g4 * g4) / (f4_val * f4_val);
            
            // ========== Full Hessian for this window ==========
            // hess = -SSIM * (term1 + term2)
            S window_hess = -ssim_val * (term1 + term2);
            
            hess_sum += window_hess / static_cast<S>(out_H * out_W);
        }
    }
    
    // Apply chain rule with lambda and grad_output (accumulate, don't overwrite)
    hessian_diag[idx] += grad_out * lambda_ssim * hess_sum / static_cast<S>(B);
}

// ============================================================================
// L2 Hessian Kernel
// ============================================================================

template <typename S>
__global__ void compute_l2_hessian_kernel(
    const S* __restrict__ grad_output,
    S* __restrict__ hessian_diag,
    S lambda_l2,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    // d^2((x-t)^2)/dx^2 = 2, so Hessian = grad_output * lambda_l2 * 2/N
    hessian_diag[idx] += grad_output[0] * lambda_l2 * static_cast<S>(2) / static_cast<S>(total_elements);
}

// ============================================================================
// Launch Functions Implementation
// ============================================================================

template <typename S>
void launch_image_loss_l2_bwd(
    const S* __restrict__ prediction,
    const S* __restrict__ target,
    const S* __restrict__ gaussian_window,
    const S* __restrict__ mu_r,
    const S* __restrict__ mu_m,
    const S* __restrict__ f1,
    const S* __restrict__ f2,
    const S* __restrict__ f3,
    const S* __restrict__ f4,
    const S* __restrict__ grad_output,
    S* __restrict__ grad_pred,
    int B, int H, int W,
    int window_size,
    S lambda_l2,
    S lambda_ssim,
    S C1, S C2,
    cudaStream_t stream
) {
    int total_pixels = B * H * W;
    int threads = GSPLAT2CT_N_THREADS;
    int blocks = (total_pixels + threads - 1) / threads;
    
    // Initialize grad_pred to zero
    cudaMemsetAsync(grad_pred, 0, total_pixels * sizeof(S), stream);
    
    // Kernel 1: L2 gradient
    compute_l2_grad_kernel<S><<<blocks, threads, 0, stream>>>(
        prediction, target, grad_output, grad_pred, lambda_l2, total_pixels
    );
    
    // Kernel 2: SSIM gradient (pixel-parallel)
    if (window_size <= 32) {
        compute_ssim_grad_pixel_parallel_kernel<S, 32><<<blocks, threads, 0, stream>>>(
            prediction, target, gaussian_window,
            mu_r, mu_m, f1, f2, f3, f4,
            grad_output, grad_pred,
            B, H, W, window_size,
            lambda_ssim, C1, C2
        );
    }
}

template <typename S>
void launch_image_loss_l2_bwd2(
    const S* __restrict__ prediction,
    const S* __restrict__ target,
    const S* __restrict__ gaussian_window,
    const S* __restrict__ mu_r,
    const S* __restrict__ mu_m,
    const S* __restrict__ f1,
    const S* __restrict__ f2,
    const S* __restrict__ f3,
    const S* __restrict__ f4,
    const S* __restrict__ grad_output,
    S* __restrict__ hessian_diag,
    int B, int H, int W,
    int window_size,
    S lambda_l2,
    S lambda_ssim,
    S C1, S C2,
    cudaStream_t stream
) {
    int total_pixels = B * H * W;
    int threads = GSPLAT2CT_N_THREADS;
    int blocks = (total_pixels + threads - 1) / threads;
    
    // Initialize hessian_diag to zero
    cudaMemsetAsync(hessian_diag, 0, total_pixels * sizeof(S), stream);
    
    // L2 contributes 2/N to Hessian
    compute_l2_hessian_kernel<S><<<blocks, threads, 0, stream>>>(
        grad_output, hessian_diag, lambda_l2, total_pixels
    );
    
    // Kernel: SSIM Hessian
    if (window_size <= 32) {
        compute_ssim_hessian_kernel<S, 32><<<blocks, threads, 0, stream>>>(
            prediction, target, gaussian_window,
            mu_r, mu_m, f1, f2, f3, f4,
            grad_output, hessian_diag,
            B, H, W, window_size,
            lambda_ssim, C1, C2
        );
    }
}

// Explicit template instantiations
template void launch_image_loss_l2_bwd<float>(
    const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const float*, const float*,
    const float*, float*, int, int, int, int, float, float, float, float, cudaStream_t
);

template void launch_image_loss_l2_bwd<double>(
    const double*, const double*, const double*,
    const double*, const double*, const double*, const double*, const double*, const double*,
    const double*, double*, int, int, int, int, double, double, double, double, cudaStream_t
);

template void launch_image_loss_l2_bwd2<float>(
    const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const float*, const float*,
    const float*, float*, int, int, int, int, float, float, float, float, cudaStream_t
);

template void launch_image_loss_l2_bwd2<double>(
    const double*, const double*, const double*,
    const double*, const double*, const double*, const double*, const double*, const double*,
    const double*, double*, int, int, int, int, double, double, double, double, cudaStream_t
);

template <typename S>
void launch_image_loss_l1_bwd(
    const S* __restrict__ prediction,
    const S* __restrict__ target,
    const S* __restrict__ gaussian_window,
    const S* __restrict__ mu_r,
    const S* __restrict__ mu_m,
    const S* __restrict__ f1,
    const S* __restrict__ f2,
    const S* __restrict__ f3,
    const S* __restrict__ f4,
    const S* __restrict__ grad_output,
    S* __restrict__ grad_pred,
    int B, int H, int W,
    int window_size,
    S lambda_l1,
    S lambda_ssim,
    S C1, S C2,
    cudaStream_t stream
) {
    int total_pixels = B * H * W;
    int threads = GSPLAT2CT_N_THREADS;
    int blocks = (total_pixels + threads - 1) / threads;

    cudaMemsetAsync(grad_pred, 0, total_pixels * sizeof(S), stream);

    compute_l1_grad_kernel<S><<<blocks, threads, 0, stream>>>(
        prediction, target, grad_output, grad_pred, lambda_l1, total_pixels
    );

    if (window_size <= 32) {
        compute_ssim_grad_pixel_parallel_kernel<S, 32><<<blocks, threads, 0, stream>>>(
            prediction, target, gaussian_window,
            mu_r, mu_m, f1, f2, f3, f4,
            grad_output, grad_pred,
            B, H, W, window_size,
            lambda_ssim, C1, C2
        );
    }
}

template <typename S>
void launch_image_loss_l1_bwd2(
    const S* __restrict__ prediction,
    const S* __restrict__ target,
    const S* __restrict__ gaussian_window,
    const S* __restrict__ mu_r,
    const S* __restrict__ mu_m,
    const S* __restrict__ f1,
    const S* __restrict__ f2,
    const S* __restrict__ f3,
    const S* __restrict__ f4,
    const S* __restrict__ grad_output,
    S* __restrict__ hessian_diag,
    int B, int H, int W,
    int window_size,
    S lambda_l1,
    S lambda_ssim,
    S C1, S C2,
    cudaStream_t stream
) {
    int total_pixels = B * H * W;
    int threads = GSPLAT2CT_N_THREADS;
    int blocks = (total_pixels + threads - 1) / threads;

    cudaMemsetAsync(hessian_diag, 0, total_pixels * sizeof(S), stream);

    if (window_size <= 32) {
        compute_ssim_hessian_kernel<S, 32><<<blocks, threads, 0, stream>>>(
            prediction, target, gaussian_window,
            mu_r, mu_m, f1, f2, f3, f4,
            grad_output, hessian_diag,
            B, H, W, window_size,
            lambda_ssim, C1, C2
        );
    }
}

template void launch_image_loss_l1_bwd<float>(
    const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const float*, const float*,
    const float*, float*, int, int, int, int, float, float, float, float, cudaStream_t
);

template void launch_image_loss_l1_bwd<double>(
    const double*, const double*, const double*,
    const double*, const double*, const double*, const double*, const double*, const double*,
    const double*, double*, int, int, int, int, double, double, double, double, cudaStream_t
);

template void launch_image_loss_l1_bwd2<float>(
    const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const float*, const float*,
    const float*, float*, int, int, int, int, float, float, float, float, cudaStream_t
);

template void launch_image_loss_l1_bwd2<double>(
    const double*, const double*, const double*,
    const double*, const double*, const double*, const double*, const double*, const double*,
    const double*, double*, int, int, int, int, double, double, double, double, cudaStream_t
);

} // namespace gsplat2ct
