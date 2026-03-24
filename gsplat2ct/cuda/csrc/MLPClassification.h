/**
 * MLP Classification CUDA Kernel Launch Functions
 * 
 * This header declares the launch functions that bridge between
 * C++ tensor processing (MLPClassification.cpp) and CUDA kernels
 * (MLPClassificationFwd.cu, MLPClassificationBwd.cu).
 * 
 * Following gsplat's pattern: 
 * - Template functions with typename S for scalar type
 * - Raw pointers with __restrict__ for no-aliasing optimization
 * 
 * Author: Dy164
 * Date: 2025-12-17
 */

#ifndef GSPLAT2CT_MLP_CLASSIFICATION_H
#define GSPLAT2CT_MLP_CLASSIFICATION_H

#include "../include/common.h"

namespace gsplat2ct {

// ============================================================================
// Forward Pass Launch Functions (Template Version)
// ============================================================================

/**
 * Launch forward pass CUDA kernels for MLP classification
 * 
 * @tparam S Scalar type (float, double, half, etc.)
 * @param w1, b1 First layer parameters (device pointers)
 * @param w2, b2 Second layer parameters (device pointers)
 * @param x Input features (device pointer)
 * @param y One-hot labels (device pointer)
 * @param i, m, o Dimensions (input, hidden, output)
 * @param sigma Output probabilities (device pointer, output)
 * @param loss Output loss (device pointer, output)
 * @param a1, z1, a2 Intermediate variables (device pointers, output, can be nullptr)
 * @param save_intermediates Whether to save intermediate variables
 * @param compute_loss Whether to compute loss
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
);

// ============================================================================
// Backward Pass Launch Functions (Template Version)
// ============================================================================

/**
 * Launch backward pass CUDA kernels for first-order gradients
 * 
 * @tparam S Scalar type
 * @param a1, z1, a2, sigma Cached forward pass intermediates (device pointers)
 * @param w1, w2 Network parameters (device pointers)
 * @param x, y Input and labels (device pointers)
 * @param i, m, o Dimensions
 * @param grad_w1, grad_b1, grad_w2, grad_b2 Output gradients (device pointers, output)
 * @param accumulate_grad If true, add to existing gradients
 */
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
);

/**
 * Launch backward pass CUDA kernels for gradients + Hessians
 * 
 * @tparam S Scalar type
 * @param Same as launch_mlp_classification_bwd
 * @param hess_b2_full Full Hessian matrix for b2, size (o, o) (device pointer, output)
 * @param hess_b1_full Full Hessian matrix for b1, size (m, m) (device pointer, output)
 * @param hess_w2_full Full Hessian matrix for w2, size (o*m, o*m) (device pointer, output)
 * @param hess_w1_full Full Hessian matrix for w1, size (m*i, m*i) (device pointer, output)
 */
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
);

}  // namespace gsplat2ct

#endif  // GSPLAT2CT_MLP_CLASSIFICATION_H
