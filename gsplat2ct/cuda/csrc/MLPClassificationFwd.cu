/**
 * MLP Classification Forward Pass - CUDA Kernels Only
 * 
 * This file contains only CUDA kernel implementations and launch functions.
 * Tensor processing logic has been moved to MLPClassification.cpp.
 * 
 * Following gsplat's pattern: 
 * - Template kernels with typename S for scalar type
 * - __restrict__ pointers for no-aliasing optimization
 * - GSPLAT2CT_PRAGMA_UNROLL for loop unrolling
 * 
 * Author: Dy164
 * Date: 2025-12-17
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include "MLPClassification.h"

namespace gsplat2ct {

// ============================================================================
// CUDA Kernels (Template Version)
// ============================================================================

// ============================================================================
// CUDA Kernel: First Layer (Linear + Tanh)
// ============================================================================

/**
 * Compute first layer: a1 = W1 * x + b1, then z1 = tanh(a1)
 * 
 * Grid/Block config: Each thread computes one element of a1/z1
 * - blockDim.x = min(m, 256)
 * - gridDim.x = (m + blockDim.x - 1) / blockDim.x
 */
template <typename S>
__global__ void first_layer_fwd_kernel(
    const S* __restrict__ w1,      // (m, i) - row-major
    const S* __restrict__ b1,      // (m, 1)
    const S* __restrict__ x,       // (i, 1)
    S* __restrict__ a1,            // (m, 1) - output
    S* __restrict__ z1,            // (m, 1) - output
    const int m,                   // hidden size
    const int i                    // input size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < m) {
        // Compute a1[idx] = sum(w1[idx, :] * x[:]) + b1[idx]
        S sum = S(0);
        for (int j = 0; j < i; j++) {
            sum += w1[idx * i + j] * x[j];
        }
        sum += b1[idx];
        a1[idx] = sum;
        
        // Compute z1[idx] = tanh(a1[idx])
        z1[idx] = tanhf(sum);
    }
}

// ============================================================================
// CUDA Kernel: Second Layer (Linear)
// ============================================================================

/**
 * Compute second layer: a2 = W2 * z1 + b2
 * 
 * Grid/Block config: Each thread computes one element of a2
 */
template <typename S>
__global__ void second_layer_fwd_kernel(
    const S* __restrict__ w2,      // (o, m) - row-major
    const S* __restrict__ b2,      // (o, 1)
    const S* __restrict__ z1,      // (m, 1)
    S* __restrict__ a2,            // (o, 1) - output (logits)
    const int o,                   // output size (num classes)
    const int m                    // hidden size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < o) {
        // Compute a2[idx] = sum(w2[idx, :] * z1[:]) + b2[idx]
        S sum = S(0);
        for (int j = 0; j < m; j++) {
            sum += w2[idx * m + j] * z1[j];
        }
        sum += b2[idx];
        a2[idx] = sum;
    }
}

// ============================================================================
// CUDA Kernel: Softmax + Cross-Entropy Loss
// ============================================================================

/**
 * Compute softmax with numerical stability: σ = exp(a2 - max(a2)) / sum(exp(a2 - max(a2)))
 * Then compute cross-entropy loss: L = -sum(y * log(σ))
 * 
 * This is a single-threaded kernel for simplicity (o is typically small, e.g., 3-1000)
 * For larger o, use warp reduction for finding max and sum.
 */
template <typename S>
__global__ void softmax_crossentropy_kernel(
    const S* __restrict__ a2,      // (o, 1) - logits
    const S* __restrict__ y,       // (o, 1) - one-hot labels
    S* __restrict__ sigma,         // (o, 1) - output probabilities
    S* __restrict__ loss,          // scalar - output loss
    const int o                    // output size
) {
    // Step 1: Find max(a2) for numerical stability
    S z_max = a2[0];
    for (int i = 1; i < o; i++) {
        if (a2[i] > z_max) {
            z_max = a2[i];
        }
    }
    
    // Step 2: Compute exp(a2 - z_max) and sum
    S exp_sum = S(0);
    for (int i = 0; i < o; i++) {
        S exp_val = expf(a2[i] - z_max);
        sigma[i] = exp_val;
        exp_sum += exp_val;
    }
    
    // Step 3: Normalize to get softmax probabilities
    for (int i = 0; i < o; i++) {
        sigma[i] /= exp_sum;
    }
    
    // Step 4: Compute cross-entropy loss
    if (loss != nullptr) {
        S loss_val = S(0);
        for (int i = 0; i < o; i++) {
            if (y[i] > S(0)) {  // Only compute for non-zero labels
                loss_val -= y[i] * logf(sigma[i] + S(1e-10));
            }
        }
        *loss = loss_val;
    }
}

// ============================================================================
// Optimized Version: Fused Softmax for Moderate o
// ============================================================================

/**
 * Optimized softmax using warp reduction for moderate o (e.g., 32-256 classes)
 * This version uses parallel threads within a warp for reduction operations.
 */
template <typename S>
__global__ void softmax_crossentropy_optimized_kernel(
    const S* __restrict__ a2,
    const S* __restrict__ y,
    S* __restrict__ sigma,
    S* __restrict__ loss,
    const int o
) {
    __shared__ S shared_max;
    __shared__ S shared_sum;
    
    const int tid = threadIdx.x;
    
    // Step 1: Find max using warp reduction
    S local_max = -INFINITY;
    for (int i = tid; i < o; i += blockDim.x) {
        local_max = fmaxf(local_max, a2[i]);
    }
    
    // Warp-level reduction for max
    GSPLAT2CT_PRAGMA_UNROLL
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    
    if (tid == 0) {
        shared_max = local_max;
    }
    __syncthreads();
    
    const S z_max = shared_max;
    
    // Step 2: Compute exp and sum
    S local_sum = S(0);
    for (int i = tid; i < o; i += blockDim.x) {
        S exp_val = expf(a2[i] - z_max);
        sigma[i] = exp_val;
        local_sum += exp_val;
    }
    
    // Warp-level reduction for sum
    GSPLAT2CT_PRAGMA_UNROLL
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    
    if (tid == 0) {
        shared_sum = local_sum;
    }
    __syncthreads();
    
    const S exp_sum = shared_sum;
    
    // Step 3: Normalize and compute loss
    S local_loss = S(0);
    for (int i = tid; i < o; i += blockDim.x) {
        sigma[i] /= exp_sum;
        if (loss != nullptr && y[i] > S(0)) {
            local_loss -= y[i] * logf(sigma[i] + S(1e-10));
        }
    }
    
    // Warp-level reduction for loss
    if (loss != nullptr) {
        GSPLAT2CT_PRAGMA_UNROLL
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_loss += __shfl_down_sync(0xffffffff, local_loss, offset);
        }
        
        if (tid == 0) {
            *loss = local_loss;
        }
    }
}

// ============================================================================
// Launch Function: Forward Pass (Template Version)
// ============================================================================

/**
 * Launch function for MLP classification forward pass.
 * This wraps the CUDA kernel launches and is called from MLPClassification.cpp.
 * 
 * All pointers are device pointers (already on GPU).
 * Tensor validation and allocation are handled in the .cpp layer.
 */
template <typename S>
void launch_mlp_classification_fwd(
    const S* __restrict__ w1, 
    const S* __restrict__ b1,
    const S* __restrict__ w2, 
    const S* __restrict__ b2,
    const S* __restrict__ x, 
    const S* __restrict__ y,
    int i, int m, int o,
    S* __restrict__ sigma,
    S* __restrict__ loss,
    S* __restrict__ a1, 
    S* __restrict__ z1, 
    S* __restrict__ a2,
    bool save_intermediates,
    bool compute_loss
) {
    // Temporary buffers if not saving intermediates
    S *a1_temp = nullptr, *z1_temp = nullptr, *a2_temp = nullptr;
    bool need_cleanup = false;
    
    if (!save_intermediates) {
        // Allocate temporary buffers
        cudaMalloc(&a1_temp, m * sizeof(S));
        cudaMalloc(&z1_temp, m * sizeof(S));
        cudaMalloc(&a2_temp, o * sizeof(S));
        a1 = a1_temp;
        z1 = z1_temp;
        a2 = a2_temp;
        need_cleanup = true;
    }
    
    // Launch first layer kernel
    const int block_size_1 = min(m, GSPLAT2CT_N_THREADS);
    const int grid_size_1 = (m + block_size_1 - 1) / block_size_1;
    first_layer_fwd_kernel<S><<<grid_size_1, block_size_1>>>(
        w1, b1, x, a1, z1, m, i
    );
    
    // Launch second layer kernel
    const int block_size_2 = min(o, GSPLAT2CT_N_THREADS);
    const int grid_size_2 = (o + block_size_2 - 1) / block_size_2;
    second_layer_fwd_kernel<S><<<grid_size_2, block_size_2>>>(
        w2, b2, z1, a2, o, m
    );
    
    // Launch softmax + loss kernel
    // Use simple version for small o, optimized for moderate o
    if (o <= 32) {
        softmax_crossentropy_kernel<S><<<1, 1>>>(
            a2, y, sigma, compute_loss ? loss : nullptr, o
        );
    } else {
        softmax_crossentropy_optimized_kernel<S><<<1, 32>>>(
            a2, y, sigma, compute_loss ? loss : nullptr, o
        );
    }
    
    // Synchronize to ensure all kernels complete
    cudaDeviceSynchronize();
    
    // Cleanup temporary buffers
    if (need_cleanup) {
        cudaFree(a1_temp);
        cudaFree(z1_temp);
        cudaFree(a2_temp);
    }
}

// ============================================================================
// Explicit Template Instantiations
// ============================================================================

template void launch_mlp_classification_fwd<float>(
    const float* __restrict__, const float* __restrict__,
    const float* __restrict__, const float* __restrict__,
    const float* __restrict__, const float* __restrict__,
    int, int, int,
    float* __restrict__, float* __restrict__,
    float* __restrict__, float* __restrict__, float* __restrict__,
    bool, bool
);

template void launch_mlp_classification_fwd<double>(
    const double* __restrict__, const double* __restrict__,
    const double* __restrict__, const double* __restrict__,
    const double* __restrict__, const double* __restrict__,
    int, int, int,
    double* __restrict__, double* __restrict__,
    double* __restrict__, double* __restrict__, double* __restrict__,
    bool, bool
);

}  // namespace gsplat2ct
