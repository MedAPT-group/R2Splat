/**
 * MLP Classification Backward Pass - CUDA Kernels Only
 * 
 * This file contains only CUDA kernel implementations and launch functions.
 * Tensor processing logic has been moved to MLPClassification.cpp.
 * 
 * Following gsplat's pattern: 
 * - Template kernels with typename S for scalar type
 * - __restrict__ pointers for no-aliasing optimization
 * - GSPLAT2CT_PRAGMA_UNROLL for loop unrolling
 * 
 * Implements both first-order gradients and second-order Hessians.
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
// CUDA Kernel: Compute Output Layer Error (σ - y)
// ============================================================================

template <typename S>
__global__ void compute_error_kernel(
    const S* __restrict__ sigma,   // (o, 1)
    const S* __restrict__ y,       // (o, 1)
    S* __restrict__ error,         // (o, 1) - output
    const int o
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < o) {
        error[idx] = sigma[idx] - y[idx];
    }
}

// ============================================================================
// CUDA Kernel: Second Layer Gradients (w2, b2)
// ============================================================================

template <typename S>
__global__ void second_layer_grad_kernel(
    const S* __restrict__ error,   // (o, 1)
    const S* __restrict__ z1,      // (m, 1)
    S* __restrict__ grad_w2,       // (o, m) - output
    S* __restrict__ grad_b2,       // (o, 1) - output
    const int o,
    const int m,
    const bool accumulate
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;  // output index
    const int col = blockIdx.x * blockDim.x + threadIdx.x;  // hidden index
    
    // Compute grad_w2
    if (row < o && col < m) {
        const int idx = row * m + col;
        const S grad_val = error[row] * z1[col];
        if (accumulate) {
            grad_w2[idx] += grad_val;
        } else {
            grad_w2[idx] = grad_val;
        }
    }
    
    // Compute grad_b2 (only one thread per row)
    if (col == 0 && row < o) {
        if (accumulate) {
            grad_b2[row] += error[row];
        } else {
            grad_b2[row] = error[row];
        }
    }
}

// ============================================================================
// CUDA Kernel: Backprop Through Tanh
// ============================================================================

template <typename S>
__global__ void backprop_tanh_kernel(
    const S* __restrict__ w2,      // (o, m)
    const S* __restrict__ error,   // (o, 1)
    const S* __restrict__ z1,      // (m, 1)
    S* __restrict__ grad_a1,       // (m, 1) - output
    const int m,
    const int o
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < m) {
        // Compute grad_z1[idx] = sum(w2[:, idx] * error[:])
        S grad_z1_val = S(0);
        for (int i = 0; i < o; i++) {
            grad_z1_val += w2[i * m + idx] * error[i];
        }
        
        // Apply tanh derivative: (1 - z1^2)
        const S z1_val = z1[idx];
        const S tanh_deriv = S(1) - z1_val * z1_val;
        grad_a1[idx] = grad_z1_val * tanh_deriv;
    }
}

// ============================================================================
// CUDA Kernel: First Layer Gradients (w1, b1)
// ============================================================================

template <typename S>
__global__ void first_layer_grad_kernel(
    const S* __restrict__ grad_a1, // (m, 1)
    const S* __restrict__ x,       // (i, 1)
    S* __restrict__ grad_w1,       // (m, i) - output
    S* __restrict__ grad_b1,       // (m, 1) - output
    const int m,
    const int i,
    const bool accumulate
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;  // hidden index
    const int col = blockIdx.x * blockDim.x + threadIdx.x;  // input index
    
    // Compute grad_w1
    if (row < m && col < i) {
        const int idx = row * i + col;
        const S grad_val = grad_a1[row] * x[col];
        if (accumulate) {
            grad_w1[idx] += grad_val;
        } else {
            grad_w1[idx] = grad_val;
        }
    }
    
    // Compute grad_b1 (only one thread per row)
    if (col == 0 && row < m) {
        if (accumulate) {
            grad_b1[row] += grad_a1[row];
        } else {
            grad_b1[row] = grad_a1[row];
        }
    }
}

// ============================================================================
// CUDA Kernel: Hessian for b2 (Full Matrix)
// ============================================================================

template <typename S>
__global__ void hessian_b2_kernel(
    const S* __restrict__ sigma,   // (o, 1)
    S* __restrict__ hess_b2,       // (o, o) - output
    const int o,
    const bool accumulate
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < o && col < o) {
        S hess_val = -sigma[row] * sigma[col];
        if (row == col) {
            hess_val += sigma[row];  // Add diagonal term
        }
        const int idx = row * o + col;
        if (accumulate) {
            hess_b2[idx] += hess_val;
        } else {
            hess_b2[idx] = hess_val;
        }
    }
}

// ============================================================================
// CUDA Kernel: Hessian for b1 (Full Matrix)
// ============================================================================

template <typename S>
__global__ void hessian_b1_kernel(
    const S* __restrict__ w2,      // (o, m)
    const S* __restrict__ hess_b2, // (o, o)
    const S* __restrict__ z1,      // (m, 1)
    const S* __restrict__ sigma,   // (o, 1)
    const S* __restrict__ y,       // (o, 1)
    S* __restrict__ hess_b1,       // (m, m) - output
    const int m,
    const int o,
    const bool accumulate
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < m) {
        // Compute first term: D * (W2^T * H_b2 * W2) * D
        S w2T_h_w2 = S(0);
        for (int i = 0; i < o; i++) {
            for (int j = 0; j < o; j++) {
                w2T_h_w2 += w2[i * m + row] * hess_b2[i * o + j] * w2[j * m + col];
            }
        }
        
        // Multiply by D = diag(1 - z1^2)
        const S d_row = S(1) - z1[row] * z1[row];
        const S d_col = S(1) - z1[col] * z1[col];
        const S first_term = d_row * w2T_h_w2 * d_col;
        
        // Compute second term: -2 * diag(s * z1 * d) where s = W2^T @ (σ - y)
        S second_term = S(0);
        if (row == col) {
            S s_val = S(0);
            for (int i = 0; i < o; i++) {
                s_val += w2[i * m + row] * (sigma[i] - y[i]);
            }
            second_term = S(-2) * s_val * z1[row] * d_row;
        }
        
        const int idx = row * m + col;
        const S hess_val = first_term + second_term;
        if (accumulate) {
            hess_b1[idx] += hess_val;
        } else {
            hess_b1[idx] = hess_val;
        }
    }
}

// ============================================================================
// CUDA Kernel: Full Hessian for w2 using Kronecker product
// H_w2 = kron(H_b2, z1*z1^T)
// Output size: (o*m, o*m)
// ============================================================================

template <typename S>
__global__ void hessian_w2_full_kernel(
    const S* __restrict__ hess_b2,  // (o, o)
    const S* __restrict__ z1,       // (m, 1)
    S* __restrict__ hess_w2_full,   // (o*m, o*m) - output
    const int o,
    const int m,
    const bool accumulate
) {
    // Each thread computes one element of the (o*m) x (o*m) matrix
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    const int total_size = o * m;
    if (row < total_size && col < total_size) {
        // Decompose indices: row = i1*m + j1, col = i2*m + j2
        const int i1 = row / m;  // output neuron index for row
        const int j1 = row % m;  // hidden neuron index for row
        const int i2 = col / m;  // output neuron index for col
        const int j2 = col % m;  // hidden neuron index for col
        
        // Kronecker product: kron(A, B)[i1*n+j1, i2*n+j2] = A[i1,i2] * B[j1,j2]
        // Here: H_w2[i1*m+j1, i2*m+j2] = H_b2[i1,i2] * (z1[j1] * z1[j2])
        const S h_b2_val = hess_b2[i1 * o + i2];
        const S zzT_val = z1[j1] * z1[j2];
        
        const int idx = row * total_size + col;
        const S hess_val = h_b2_val * zzT_val;
        if (accumulate) {
            hess_w2_full[idx] += hess_val;
        } else {
            hess_w2_full[idx] = hess_val;
        }
    }
}

// ============================================================================
// CUDA Kernel: Full Hessian for w1 using Kronecker product
// H_w1 = kron(H_b1, x*x^T)
// Output size: (m*i, m*i)
// ============================================================================

template <typename S>
__global__ void hessian_w1_full_kernel(
    const S* __restrict__ hess_b1,  // (m, m)
    const S* __restrict__ x,        // (i, 1)
    S* __restrict__ hess_w1_full,   // (m*i, m*i) - output
    const int m,
    const int i,
    const bool accumulate
) {
    // Each thread computes one element of the (m*i) x (m*i) matrix
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    const int total_size = m * i;
    if (row < total_size && col < total_size) {
        // Decompose indices: row = j1*i + k1, col = j2*i + k2
        const int j1 = row / i;  // hidden neuron index for row
        const int k1 = row % i;  // input feature index for row
        const int j2 = col / i;  // hidden neuron index for col
        const int k2 = col % i;  // input feature index for col
        
        // Kronecker product: H_w1[j1*i+k1, j2*i+k2] = H_b1[j1,j2] * (x[k1] * x[k2])
        const S h_b1_val = hess_b1[j1 * m + j2];
        const S xxT_val = x[k1] * x[k2];
        
        const int idx = row * total_size + col;
        const S hess_val = h_b1_val * xxT_val;
        if (accumulate) {
            hess_w1_full[idx] += hess_val;
        } else {
            hess_w1_full[idx] = hess_val;
        }
    }
}

// ============================================================================
// Launch Function: Backward Pass (First-Order Only)
// ============================================================================

template <typename S>
void launch_mlp_classification_bwd(
    const S* __restrict__ a1, 
    const S* __restrict__ z1,
    const S* __restrict__ a2, 
    const S* __restrict__ sigma,
    const S* __restrict__ w1, 
    const S* __restrict__ w2,
    const S* __restrict__ x, 
    const S* __restrict__ y,
    int i, int m, int o,
    S* __restrict__ grad_w1, 
    S* __restrict__ grad_b1,
    S* __restrict__ grad_w2, 
    S* __restrict__ grad_b2,
    bool accumulate_grad
) {
    // Allocate temporary buffer for error
    S* error;
    cudaMalloc(&error, o * sizeof(S));
    
    // Allocate temporary buffer for grad_a1
    S* grad_a1;
    cudaMalloc(&grad_a1, m * sizeof(S));
    
    // Step 1: Compute error = σ - y
    const int block_size = min(o, GSPLAT2CT_N_THREADS);
    const int grid_size = (o + block_size - 1) / block_size;
    compute_error_kernel<S><<<grid_size, block_size>>>(
        sigma, y, error, o
    );
    
    // Step 2: Compute second layer gradients
    dim3 block_2d(16, 16);
    dim3 grid_w2((m + 15) / 16, (o + 15) / 16);
    second_layer_grad_kernel<S><<<grid_w2, block_2d>>>(
        error, z1, grad_w2, grad_b2, o, m, accumulate_grad
    );
    
    // Step 3: Backprop through tanh
    const int block_size_m = min(m, GSPLAT2CT_N_THREADS);
    const int grid_size_m = (m + block_size_m - 1) / block_size_m;
    backprop_tanh_kernel<S><<<grid_size_m, block_size_m>>>(
        w2, error, z1, grad_a1, m, o
    );
    
    // Step 4: Compute first layer gradients
    dim3 grid_w1((i + 15) / 16, (m + 15) / 16);
    first_layer_grad_kernel<S><<<grid_w1, block_2d>>>(
        grad_a1, x, grad_w1, grad_b1, m, i, accumulate_grad
    );
    
    cudaDeviceSynchronize();
    
    // Cleanup
    cudaFree(error);
    cudaFree(grad_a1);
}

// ============================================================================
// Launch Function: Backward Pass (First + Second Order)
// ============================================================================

template <typename S>
void launch_mlp_classification_bwd2(
    const S* __restrict__ a1, 
    const S* __restrict__ z1,
    const S* __restrict__ a2, 
    const S* __restrict__ sigma,
    const S* __restrict__ w1, 
    const S* __restrict__ w2,
    const S* __restrict__ x, 
    const S* __restrict__ y,
    int i, int m, int o,
    S* __restrict__ grad_w1, 
    S* __restrict__ grad_b1,
    S* __restrict__ grad_w2, 
    S* __restrict__ grad_b2,
    S* __restrict__ hess_b2_full,
    S* __restrict__ hess_b1_full,
    S* __restrict__ hess_w2_full,
    S* __restrict__ hess_w1_full,
    bool accumulate_grad
) {
    // First, compute first-order gradients
    launch_mlp_classification_bwd<S>(
        a1, z1, a2, sigma, w1, w2, x, y, i, m, o,
        grad_w1, grad_b1, grad_w2, grad_b2, accumulate_grad
    );
    
    // Step 5: Compute Hessian for b2
    // Note: For Hessians, always write to buffer (no accumulation)
    // Accumulation is handled outside kernel to avoid cross-layer dependencies
    dim3 block_2d(16, 16);
    dim3 grid_b2((o + 15) / 16, (o + 15) / 16);
    hessian_b2_kernel<S><<<grid_b2, block_2d>>>(
        sigma, hess_b2_full, o, false
    );
    
    // Step 6: Compute Hessian for b1
    dim3 grid_b1((m + 15) / 16, (m + 15) / 16);
    hessian_b1_kernel<S><<<grid_b1, block_2d>>>(
        w2, hess_b2_full, z1, sigma, y, hess_b1_full, m, o, false
    );
    
    // Step 7: Compute full Hessian for w2 (Kronecker product)
    const int w2_size = o * m;
    dim3 grid_w2_full((w2_size + 15) / 16, (w2_size + 15) / 16);
    hessian_w2_full_kernel<S><<<grid_w2_full, block_2d>>>(
        hess_b2_full, z1, hess_w2_full, o, m, false
    );
    
    // Step 8: Compute full Hessian for w1 (Kronecker product)
    const int w1_size = m * i;
    dim3 grid_w1_full((w1_size + 15) / 16, (w1_size + 15) / 16);
    hessian_w1_full_kernel<S><<<grid_w1_full, block_2d>>>(
        hess_b1_full, x, hess_w1_full, m, i, false
    );
    
    cudaDeviceSynchronize();
}

// ============================================================================
// Explicit Template Instantiations
// ============================================================================

template void launch_mlp_classification_bwd<float>(
    const float* __restrict__, const float* __restrict__,
    const float* __restrict__, const float* __restrict__,
    const float* __restrict__, const float* __restrict__,
    const float* __restrict__, const float* __restrict__,
    int, int, int,
    float* __restrict__, float* __restrict__,
    float* __restrict__, float* __restrict__,
    bool
);

template void launch_mlp_classification_bwd<double>(
    const double* __restrict__, const double* __restrict__,
    const double* __restrict__, const double* __restrict__,
    const double* __restrict__, const double* __restrict__,
    const double* __restrict__, const double* __restrict__,
    int, int, int,
    double* __restrict__, double* __restrict__,
    double* __restrict__, double* __restrict__,
    bool
);

template void launch_mlp_classification_bwd2<float>(
    const float* __restrict__, const float* __restrict__,
    const float* __restrict__, const float* __restrict__,
    const float* __restrict__, const float* __restrict__,
    const float* __restrict__, const float* __restrict__,
    int, int, int,
    float* __restrict__, float* __restrict__,
    float* __restrict__, float* __restrict__,
    float* __restrict__, float* __restrict__,
    float* __restrict__, float* __restrict__,
    bool
);

template void launch_mlp_classification_bwd2<double>(
    const double* __restrict__, const double* __restrict__,
    const double* __restrict__, const double* __restrict__,
    const double* __restrict__, const double* __restrict__,
    const double* __restrict__, const double* __restrict__,
    int, int, int,
    double* __restrict__, double* __restrict__,
    double* __restrict__, double* __restrict__,
    double* __restrict__, double* __restrict__,
    double* __restrict__, double* __restrict__,
    bool
);

}  // namespace gsplat2ct
