/**
 * ImageLoss Forward CUDA Kernels
 * 
 * Implements forward pass for L2 + SSIM image loss.
 * 
 * Kernels:
 * 1. compute_l2_map_kernel: Computes (prediction - target)^2 per pixel
 * 2. compute_ssim_statistics_kernel: Computes window statistics (mu, sigma, f1-f4)
 * 3. compute_ssim_map_kernel: Computes SSIM map from statistics
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
    
    S pred_val = prediction[idx];
    S tgt_val = target[idx];
    S diff = pred_val - tgt_val;
    l2_map[idx] = diff * diff;
}

// ============================================================================
// SSIM Statistics Kernel
// ============================================================================

/**
 * Compute SSIM window statistics per output position.
 * Each thread processes one output window position.
 * 
 * Uses shared memory to cache Gaussian kernel for fast access.
 */
template <typename S, int MAX_WINDOW_SIZE = 32>
__global__ void compute_ssim_statistics_kernel(
    const S* __restrict__ prediction,
    const S* __restrict__ target,
    const S* __restrict__ gaussian_window,
    S* __restrict__ mu_r,
    S* __restrict__ mu_m,
    S* __restrict__ f1,
    S* __restrict__ f2,
    S* __restrict__ f3,
    S* __restrict__ f4,
    S* __restrict__ ssim_map,
    int B, int H, int W,
    int window_size,
    int pad_size,
    int out_H, int out_W,
    bool compute_statistics
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
    
    // Each thread handles one output position
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = B * out_H * out_W;
    if (idx >= total_outputs) return;
    
    // Decode batch and spatial indices
    int b = idx / (out_H * out_W);
    int spatial_idx = idx % (out_H * out_W);
    int out_i = spatial_idx / out_W;
    int out_j = spatial_idx % out_W;
    
    // Constants
    const S C1 = static_cast<S>(0.0001);  // 0.01^2
    const S C2 = static_cast<S>(0.0009);  // 0.03^2
    
    // Compute window statistics
    S sum_pred = 0, sum_tgt = 0;
    S sum_pred_sq = 0, sum_tgt_sq = 0;
    S sum_pred_tgt = 0;
    
    GSPLAT2CT_PRAGMA_UNROLL
    for (int wi = 0; wi < window_size; ++wi) {
        GSPLAT2CT_PRAGMA_UNROLL
        for (int wj = 0; wj < window_size; ++wj) {
            int img_i = out_i + wi;
            int img_j = out_j + wj;
            
            // Handle padding (zero padding)
            S pred_val = 0, tgt_val = 0;
            if (img_i >= pad_size && img_i < H + pad_size &&
                img_j >= pad_size && img_j < W + pad_size) {
                int real_i = img_i - pad_size;
                int real_j = img_j - pad_size;
                int pixel_idx = b * H * W + real_i * W + real_j;
                pred_val = prediction[pixel_idx];
                tgt_val = target[pixel_idx];
            }
            
            S w = s_gaussian[wi * window_size + wj];
            sum_pred += w * pred_val;
            sum_tgt += w * tgt_val;
            sum_pred_sq += w * pred_val * pred_val;
            sum_tgt_sq += w * tgt_val * tgt_val;
            sum_pred_tgt += w * pred_val * tgt_val;
        }
    }
    
    // Compute statistics
    S mu_r_val = sum_pred;
    S mu_m_val = sum_tgt;
    S sigma_r_sq = sum_pred_sq - mu_r_val * mu_r_val;
    S sigma_m_sq = sum_tgt_sq - mu_m_val * mu_m_val;
    S sigma_rm = sum_pred_tgt - mu_r_val * mu_m_val;
    
    // Compute f1, f2, f3, f4
    S f1_val = 2 * mu_r_val * mu_m_val + C1;
    S f2_val = 2 * sigma_rm + C2;
    S f3_val = mu_r_val * mu_r_val + mu_m_val * mu_m_val + C1;
    S f4_val = sigma_r_sq + sigma_m_sq + C2;
    
    // Compute SSIM
    S ssim_val = (f1_val * f2_val) / (f3_val * f4_val);
    
    // Store results
    ssim_map[idx] = ssim_val;
    
    if (compute_statistics) {
        mu_r[idx] = mu_r_val;
        mu_m[idx] = mu_m_val;
        f1[idx] = f1_val;
        f2[idx] = f2_val;
        f3[idx] = f3_val;
        f4[idx] = f4_val;
    }
}

// ============================================================================
// Launch Function Implementation
// ============================================================================

template <typename S>
void launch_image_loss_fwd(
    const S* __restrict__ prediction,
    const S* __restrict__ target,
    const S* __restrict__ gaussian_window,
    S* __restrict__ l2_map,
    S* __restrict__ mu_r,
    S* __restrict__ mu_m,
    S* __restrict__ f1,
    S* __restrict__ f2,
    S* __restrict__ f3,
    S* __restrict__ f4,
    S* __restrict__ ssim_map,
    int B, int H, int W,
    int window_size,
    bool compute_statistics,
    cudaStream_t stream
) {
    int total_pixels = B * H * W;
    int pad_size = window_size / 2;
    int out_H = H;  // With same padding
    int out_W = W;
    int total_outputs = B * out_H * out_W;
    
    // Kernel 1: Compute L2 (squared error) map
    {
        int threads = GSPLAT2CT_N_THREADS;
        int blocks = (total_pixels + threads - 1) / threads;
        compute_l2_map_kernel<S><<<blocks, threads, 0, stream>>>(
            prediction, target, l2_map, total_pixels
        );
    }
    
    // Kernel 2: Compute SSIM statistics and map
    {
        int threads = GSPLAT2CT_N_THREADS;
        int blocks = (total_outputs + threads - 1) / threads;
        
        // Choose kernel based on window size for optimal unrolling
        if (window_size <= 32) {
            compute_ssim_statistics_kernel<S, 32><<<blocks, threads, 0, stream>>>(
                prediction, target, gaussian_window,
                mu_r, mu_m, f1, f2, f3, f4, ssim_map,
                B, H, W, window_size, pad_size, out_H, out_W,
                compute_statistics
            );
        }
    }
}

// Explicit template instantiations
template void launch_image_loss_fwd<float>(
    const float*, const float*, const float*,
    float*, float*, float*, float*, float*, float*, float*, float*,
    int, int, int, int, bool, cudaStream_t
);

template void launch_image_loss_fwd<double>(
    const double*, const double*, const double*,
    double*, double*, double*, double*, double*, double*, double*, double*,
    int, int, int, int, bool, cudaStream_t
);

} // namespace gsplat2ct
