/**
 * Operations Header
 * 
 * This header declares all C++ wrapper functions that are registered
 * to Python via PyBind11 in ext.cpp.
 * 
 * Following gsplat's pattern: each function accepts torch::Tensor
 * and returns std::tuple for type safety.
 * 
 * Author: Dy164
 * Date: 2025-12-03
 */

#ifndef GSPLAT2CT_OPS_H
#define GSPLAT2CT_OPS_H

#include <torch/extension.h>
#include <tuple>

namespace gsplat2ct {

// ============================================================================
// MLP Classification Operations
// ============================================================================

/**
 * MLP Classification Forward Pass
 * 
 * Computes forward pass for 2-layer MLP with softmax + cross-entropy loss.
 * 
 * @param w1, b1 First layer parameters (m×i), (m,)
 * @param w2, b2 Second layer parameters (o×m), (o,)
 * @param x Input features (batch_size, i) or (i,)
 * @param y One-hot labels (batch_size, o) or (o,)
 * @param save_intermediates Whether to save intermediate values for backward
 * @param compute_loss Whether to compute cross-entropy loss
 * 
 * @return std::tuple with:
 *   - sigma: Output probabilities (batch_size, o) or (o,)
 *   - loss: Scalar loss (if compute_loss=True, else empty tensor)
 *   - a1, z1, a2: Intermediates (if save_intermediates=True, else empty tensors)
 */
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
);

/**
 * MLP Classification Backward Pass (Gradients Only)
 * 
 * Computes first-order gradients w.r.t. all parameters.
 * 
 * @param w1, b1, w2, b2 Network parameters
 * @param x Input features (batch_size, i) or (i,)
 * @param y One-hot labels (batch_size, o) or (o,)
 * @param a1, z1, a2, sigma Cached forward pass intermediates
 * @param accumulate_grad If True, add to existing gradients (for mini-batch)
 * 
 * @return std::tuple with gradient tensors:
 *   - grad_w1, grad_b1, grad_w2, grad_b2
 */
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
);

/**
 * MLP Classification Backward Pass (Gradients + Hessians)
 * 
 * Computes both first-order gradients and second-order Hessians.
 * 
 * @param Same as mlp_classification_bwd
 * 
 * @return std::tuple with:
 *   - grad_w1, grad_b1, grad_w2, grad_b2 (gradients)
 *   - hess_b2_full (o×o), hess_b1_full (m×m) (full Hessians)
 *   - hess_w2_diag (o×m), hess_w1_diag (m×i) (diagonal Hessians)
 */
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
);

// ============================================================================
// Image Loss Operations
// ============================================================================

/**
 * Image Loss Forward Pass
 * 
 * Computes combined L1 + D-SSIM loss for batched single-channel images.
 * 
 * @param prediction Predicted images (B, H, W)
 * @param target Target images (B, H, W)
 * @param window_size SSIM window size (odd number, typically 11)
 * @param lambda_l2 Weight for L2 loss ((pred - target)^2)
 * @param lambda_ssim Weight for D-SSIM loss
 * @param compute_statistics Whether to save statistics for backward pass
 * 
 * @return std::tuple with:
 *   - loss: Combined scalar loss
 *   - l2_loss: L2 component
 *   - ssim_loss: D-SSIM component (1 - SSIM)
 *   - mu_r, mu_m, f1, f2, f3, f4: Statistics (if compute_statistics=True)
 */
std::tuple<
    torch::Tensor,  // loss
    torch::Tensor,  // l2_loss
    torch::Tensor,  // ssim_loss
    torch::Tensor,  // mu_r
    torch::Tensor,  // mu_m
    torch::Tensor,  // f1
    torch::Tensor,  // f2
    torch::Tensor,  // f3
    torch::Tensor   // f4
> image_loss_fwd(
    const torch::Tensor& prediction,
    const torch::Tensor& target,
    int window_size,
    float lambda_l2,
    float lambda_ssim,
    bool compute_statistics
);

std::tuple<
    torch::Tensor,  // loss
    torch::Tensor,  // l2_loss
    torch::Tensor,  // ssim_loss
    torch::Tensor,  // mu_r
    torch::Tensor,  // mu_m
    torch::Tensor,  // f1
    torch::Tensor,  // f2
    torch::Tensor,  // f3
    torch::Tensor   // f4
> image_loss_l2_fwd(
    const torch::Tensor& prediction,
    const torch::Tensor& target,
    int window_size,
    float lambda_l2,
    float lambda_ssim,
    bool compute_statistics
);

std::tuple<
    torch::Tensor,  // loss
    torch::Tensor,  // l1_loss
    torch::Tensor,  // ssim_loss
    torch::Tensor,  // mu_r
    torch::Tensor,  // mu_m
    torch::Tensor,  // f1
    torch::Tensor,  // f2
    torch::Tensor,  // f3
    torch::Tensor   // f4
> image_loss_l1_fwd(
    const torch::Tensor& prediction,
    const torch::Tensor& target,
    int window_size,
    float lambda_l1,
    float lambda_ssim,
    bool compute_statistics
);

/**
 * Image Loss Backward Pass
 * 
 * Computes gradient of combined L2 + D-SSIM loss w.r.t. prediction.
 * Uses pixel-parallel strategy for conflict-free gradient computation.
 * 
 * @param prediction Predicted images (B, H, W)
 * @param target Target images (B, H, W)
 * @param mu_r, mu_m, f1, f2, f3, f4 Statistics from forward pass
 * @param grad_output Incoming gradient (scalar, typically 1.0)
 * @param window_size SSIM window size
 * @param lambda_l2 Weight for L2 loss
 * @param lambda_ssim Weight for D-SSIM loss
 * 
 * @return grad_pred: Gradient w.r.t. prediction (B, H, W)
 */
torch::Tensor image_loss_bwd(
    const torch::Tensor& prediction,
    const torch::Tensor& target,
    const torch::Tensor& mu_r,
    const torch::Tensor& mu_m,
    const torch::Tensor& f1,
    const torch::Tensor& f2,
    const torch::Tensor& f3,
    const torch::Tensor& f4,
    const torch::Tensor& grad_output,
    int window_size,
    float lambda_l2,
    float lambda_ssim
);

torch::Tensor image_loss_l2_bwd(
    const torch::Tensor& prediction,
    const torch::Tensor& target,
    const torch::Tensor& mu_r,
    const torch::Tensor& mu_m,
    const torch::Tensor& f1,
    const torch::Tensor& f2,
    const torch::Tensor& f3,
    const torch::Tensor& f4,
    const torch::Tensor& grad_output,
    int window_size,
    float lambda_l2,
    float lambda_ssim
);

torch::Tensor image_loss_l1_bwd(
    const torch::Tensor& prediction,
    const torch::Tensor& target,
    const torch::Tensor& mu_r,
    const torch::Tensor& mu_m,
    const torch::Tensor& f1,
    const torch::Tensor& f2,
    const torch::Tensor& f3,
    const torch::Tensor& f4,
    const torch::Tensor& grad_output,
    int window_size,
    float lambda_l1,
    float lambda_ssim
);

/**
 * Image Loss Second-Order Backward Pass
 * 
 * Computes diagonal Hessian of D-SSIM loss w.r.t. prediction.
 * L2 contributes constant diagonal Hessian (same curvature per pixel).
 * 
 * @param Same as image_loss_bwd
 * 
 * @return hessian_diag: Diagonal Hessian (B, H, W)
 */
torch::Tensor image_loss_bwd2(
    const torch::Tensor& prediction,
    const torch::Tensor& target,
    const torch::Tensor& mu_r,
    const torch::Tensor& mu_m,
    const torch::Tensor& f1,
    const torch::Tensor& f2,
    const torch::Tensor& f3,
    const torch::Tensor& f4,
    const torch::Tensor& grad_output,
    int window_size,
    float lambda_l2,
    float lambda_ssim
);

torch::Tensor image_loss_l2_bwd2(
    const torch::Tensor& prediction,
    const torch::Tensor& target,
    const torch::Tensor& mu_r,
    const torch::Tensor& mu_m,
    const torch::Tensor& f1,
    const torch::Tensor& f2,
    const torch::Tensor& f3,
    const torch::Tensor& f4,
    const torch::Tensor& grad_output,
    int window_size,
    float lambda_l2,
    float lambda_ssim
);

torch::Tensor image_loss_l1_bwd2(
    const torch::Tensor& prediction,
    const torch::Tensor& target,
    const torch::Tensor& mu_r,
    const torch::Tensor& mu_m,
    const torch::Tensor& f1,
    const torch::Tensor& f2,
    const torch::Tensor& f3,
    const torch::Tensor& f4,
    const torch::Tensor& grad_output,
    int window_size,
    float lambda_l1,
    float lambda_ssim
);

// ============================================================================
// L2 Loss Operations
// ============================================================================

/**
 * L2 Loss Forward Pass
 * 
 * Computes Mean Squared Error loss for 3D tensors.
 * Supports both batched images (B, H, W) and unbatched volumes (D, H, W).
 * 
 * @param prediction Predicted tensor (dim0, dim1, dim2)
 * @param target Target tensor (dim0, dim1, dim2)
 * 
 * @return std::tuple with:
 *   - loss: Scalar MSE loss
 *   - l2_map: Per-element squared error (dim0, dim1, dim2)
 */
std::tuple<torch::Tensor, torch::Tensor> l2_loss_fwd(
    const torch::Tensor& prediction,
    const torch::Tensor& target
);

/**
 * L2 Loss Backward Pass
 * 
 * Computes gradient of MSE loss w.r.t. prediction.
 * grad = 2 * (prediction - target) / N
 * 
 * @param prediction Predicted tensor (dim0, dim1, dim2)
 * @param target Target tensor (dim0, dim1, dim2)
 * @param grad_output Incoming gradient (scalar, typically 1.0)
 * 
 * @return grad_pred: Gradient w.r.t. prediction (dim0, dim1, dim2)
 */
torch::Tensor l2_loss_bwd(
    const torch::Tensor& prediction,
    const torch::Tensor& target,
    const torch::Tensor& grad_output
);

/**
 * L2 Loss Second-Order Backward Pass
 * 
 * Computes diagonal Hessian of MSE loss w.r.t. prediction.
 * Hessian = 2 / N (constant for all elements)
 * 
 * @param prediction Predicted tensor (for shape/dtype reference)
 * @param grad_output Incoming gradient (scalar)
 * 
 * @return hessian_diag: Diagonal Hessian (dim0, dim1, dim2)
 */
torch::Tensor l2_loss_bwd2(
    const torch::Tensor& prediction,
    const torch::Tensor& grad_output
);

// ============================================================================
// Volume Loss Operations
// ============================================================================

/**
 * Volume Loss Forward Pass
 * 
 * Computes Smooth Total Variation (TV) loss for 3D volumes.
 * 
 * @param volume Input volume (B, D, H, W)
 * @param eps Smoothing parameter for smooth TV
 * 
 * @return std::tuple with:
 *   - tv_loss: Scalar TV loss
 *   - tv_map: Per-voxel TV values (B, D, H, W)
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> volume_loss_fwd(
    const torch::Tensor& volume,
    float eps,
    float lambda_tv
);

/**
 * Volume Loss Backward Pass
 * 
 * Computes gradient of lambda_tv * smooth_TV loss w.r.t. volume.
 * 
 * @param volume Input volume (B, D, H, W)
 * @param grad_output Incoming gradient (scalar)
 * @param eps Smoothing parameter for smooth TV
 * @param lambda_tv Weight for TV loss contribution
 * 
 * @return grad_volume: Gradient w.r.t. volume (B, D, H, W)
 */
torch::Tensor volume_loss_bwd(
    const torch::Tensor& volume,
    const torch::Tensor& grad_output,
    float eps,
    float lambda_tv
);

/**
 * Volume Loss Second-Order Backward Pass
 * 
 * Computes diagonal Hessian of lambda_tv * smooth_TV loss w.r.t. volume.
 * 
 * @param volume Input volume (B, D, H, W)
 * @param grad_output Incoming gradient (scalar)
 * @param eps Smoothing parameter for smooth TV
 * @param lambda_tv Weight for TV loss contribution
 * 
 * @return hessian_diag: Diagonal Hessian (B, D, H, W)
 */
torch::Tensor volume_loss_bwd2(
    const torch::Tensor& volume,
    const torch::Tensor& grad_output,
    float eps,
    float lambda_tv
);

// ============================================================================
// Isect Tiles Operations
// ============================================================================

/**
 * Compute tile-Gaussian intersections
 * 
 * Two-pass algorithm:
 * 1. Count tiles per Gaussian
 * 2. Generate intersection records
 * 
 * Optimized for CT reconstruction (no depth sorting needed).
 * Results are always sorted by (camera_id, tile_id) for rasterization.
 * 
 * Dimension convention:
 * - dim1 = x = columns (tile_id varies fastest)
 * - dim0 = y = rows
 * 
 * @param means2d 2D projected Gaussian centers (C, N, 2), format: (x, y) = (dim1, dim0)
 * @param radii Gaussian radii in pixels (C, N)
 * @param tile_size Size of each tile in pixels
 * @param n_tiles_dim1 Number of tiles in x dimension (columns)
 * @param n_tiles_dim0 Number of tiles in y dimension (rows)
 * 
 * @return std::tuple with:
 *   - tiles_per_gauss: Number of tiles per Gaussian (C, N)
 *   - isect_ids: Sorted encoded (camera_id, tile_id) for each intersection [n_isects]
 *   - flatten_ids: Sorted original Gaussian index for each intersection [n_isects]
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
isect_tiles_tensor(
    const torch::Tensor& means2d,
    const torch::Tensor& radii,
    const uint32_t tile_size,
    const uint32_t n_tiles_dim1,
    const uint32_t n_tiles_dim0
);

/**
 * Encode tile offsets from sorted isect_ids
 * 
 * Creates an offset array for efficient per-tile access.
 * offsets[c, dim0, dim1] gives the starting index in sorted flatten_ids
 * for tile (c, dim0, dim1).
 * 
 * @param isect_ids Sorted encoded (camera_id, tile_id) [n_isects]
 * @param C Number of cameras
 * @param n_tiles_dim1 Number of tiles in x dimension (columns)
 * @param n_tiles_dim0 Number of tiles in y dimension (rows)
 * 
 * @return offsets: Per-tile starting offsets (C, n_tiles_dim0, n_tiles_dim1)
 */
torch::Tensor isect_offset_encode_tensor(
    const torch::Tensor& isect_ids,
    const uint32_t C,
    const uint32_t n_tiles_dim1,
    const uint32_t n_tiles_dim0
);

// ============================================================================
// Isect Tiles 3D Operations
// ============================================================================

/**
 * Compute tile-Gaussian intersections in 3D
 * 
 * Two-pass algorithm:
 * 1. Count tiles per Gaussian
 * 2. Generate intersection records
 * 
 * Optimized for CT reconstruction (no depth sorting needed).
 * Results are always sorted by tile_id for rasterization.
 * 
 * Dimension convention:
 * - dim2 = x (fastest varying in memory)
 * - dim1 = y (middle dimension)
 * - dim0 = z (slowest varying in memory)
 * 
 * @param means3d 3D Gaussian centers (N, 3), stored as (x, y, z)
 * @param radii Gaussian radii in each dimension (N, 3), stored as (rx, ry, rz)
 * @param tile_size Size of each tile in voxels
 * @param n_tiles_dim2 Number of tiles in x dimension (fastest)
 * @param n_tiles_dim1 Number of tiles in y dimension (middle)
 * @param n_tiles_dim0 Number of tiles in z dimension (slowest)
 * 
 * @return std::tuple with:
 *   - tiles_per_gauss: Number of tiles per Gaussian (N,)
 *   - isect_ids: Sorted tile_id for each intersection [n_isects]
 *   - flatten_ids: Sorted original Gaussian index for each intersection [n_isects]
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
isect_tiles_3d_tensor(
    const torch::Tensor& means3d,
    const torch::Tensor& radii,
    const uint32_t tile_size,
    const uint32_t n_tiles_dim2,
    const uint32_t n_tiles_dim1,
    const uint32_t n_tiles_dim0
);

/**
 * Encode tile offsets from sorted isect_ids for 3D
 * 
 * Creates an offset array for efficient per-tile access.
 * offsets[dim0, dim1, dim2] gives the starting index in sorted flatten_ids
 * for tile at position (dim0, dim1, dim2).
 * 
 * @param isect_ids Sorted tile_ids [n_isects]
 * @param n_tiles_dim2 Number of tiles in x dimension (fastest)
 * @param n_tiles_dim1 Number of tiles in y dimension (middle)
 * @param n_tiles_dim0 Number of tiles in z dimension (slowest)
 * 
 * @return offsets: Per-tile starting offsets (n_tiles_dim0, n_tiles_dim1, n_tiles_dim2)
 */
torch::Tensor isect_offset_encode_3d_tensor(
    const torch::Tensor& isect_ids,
    const uint32_t n_tiles_dim2,
    const uint32_t n_tiles_dim1,
    const uint32_t n_tiles_dim0
);

// ============================================================================
// VoxelizeToVoxels Operations
// ============================================================================

/**
 * Voxelize 3D Gaussians to a voxel grid for CT reconstruction.
 * 
 * Tile-based voxelization with shared memory optimization.
 * CT formula: voxel_value = Σ (opacity × exp(-sigma))
 * where sigma = 0.5 × delta^T × conic × delta
 * 
 * This function performs the forward pass of CT voxelization,
 * accumulating contributions from all intersecting Gaussians to each voxel.
 * 
 * Dimension convention:
 * - dim2: x-axis (fastest varying, corresponds to means3d[..., 0])
 * - dim1: y-axis (middle dimension, corresponds to means3d[..., 1])
 * - dim0: z-axis (slowest varying, corresponds to means3d[..., 2])
 * - Output shape: (dim0, dim1, dim2)
 * 
 * @param means3d 3D Gaussian centers (N, 3), format: (x, y, z)
 * @param conics Inverse covariance matrices (N, 6), stored as upper triangular
 *               elements [c00, c01, c02, c11, c12, c22]
 * @param opacities Gaussian opacities (N)
 * @param tile_offsets Per-tile starting offsets (n_tiles_dim0, n_tiles_dim1, n_tiles_dim2)
 *                     from isect_offset_encode_3d
 * @param flatten_ids Original Gaussian index for each intersection [n_isects]
 *                    from isect_tiles_3d
 * @param dim2 Output size in dimension 2 (x-axis, fastest)
 * @param dim1 Output size in dimension 1 (y-axis, middle)
 * @param dim0 Output size in dimension 0 (z-axis, slowest)
 * @param tile_size Size of each tile in voxels
 * 
 * @return render_output: Voxelized volume (dim0, dim1, dim2)
 */
torch::Tensor voxelize_to_voxels_fwd_tensor(
    const torch::Tensor& means3d,       // [N, 3]
    const torch::Tensor& conics,        // [N, 6]
    const torch::Tensor& opacities,     // [N]
    const torch::Tensor& tile_offsets,  // [n_tiles_dim0, n_tiles_dim1, n_tiles_dim2]
    const torch::Tensor& flatten_ids,   // [n_isects]
    const uint32_t dim2,                // x size (fastest)
    const uint32_t dim1,                // y size (middle)
    const uint32_t dim0,                // z size (slowest)
    const uint32_t tile_size
);

/**
 * Voxelize to Voxels Backward Pass (First-Order Gradients)
 * 
 * Computes gradients of the loss w.r.t. Gaussian parameters.
 * 
 * @param means3d 3D Gaussian centers [N, 3]
 * @param conics Inverse covariance matrices [N, 6]
 * @param opacities Gaussian opacities [N]
 * @param tile_offsets Per-tile starting offsets
 * @param flatten_ids Gaussian indices for each intersection
 * @param v_render_output Upstream gradient [dim0, dim1, dim2]
 * @param dim2, dim1, dim0 Output dimensions
 * @param tile_size Size of each tile
 * 
 * @return std::tuple with:
 *   - v_means3d: [N, 3]
 *   - v_conics: [N, 6]
 *   - v_opacities: [N]
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
voxelize_to_voxels_bwd_tensor(
    const torch::Tensor& means3d,
    const torch::Tensor& conics,
    const torch::Tensor& opacities,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& flatten_ids,
    const torch::Tensor& v_render_output,
    const uint32_t dim2,
    const uint32_t dim1,
    const uint32_t dim0,
    const uint32_t tile_size
);

/**
 * Voxelize to Voxels Backward Pass (Second-Order: Gradients + Hessians)
 * 
 * Computes both first-order gradients and second-order Hessians.
 * 
 * Hessian storage uses upper triangular elements:
 *   h_means3d: [N, 6] upper triangle of 3x3 matrix
 *   h_conics: [N, 21] upper triangle of 6x6 matrix
 *   h_opacities: [N] scalar
 *   h_op_mean: [N, 3] cross term opacity-means3d
 *   h_op_conic: [N, 6] cross term opacity-conics
 *   h_mean_conic: [N, 18] = [N, 3, 6] cross term means3d-conics
 * 
 * @return std::tuple with:
 *   - v_means3d, v_conics, v_opacities (first-order)
 *   - h_means3d, h_conics, h_opacities (Hessian diagonals)
 *   - h_op_mean, h_op_conic, h_mean_conic (cross-terms)
 */
std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor,  // First-order
    torch::Tensor, torch::Tensor, torch::Tensor,  // Hessian diagonals
    torch::Tensor, torch::Tensor, torch::Tensor   // Cross-terms
>
voxelize_to_voxels_bwd2_tensor(
    const torch::Tensor& means3d,
    const torch::Tensor& conics,
    const torch::Tensor& opacities,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& flatten_ids,
    const torch::Tensor& v_render_output,
    const torch::Tensor& h_render_output,
    const uint32_t dim2,
    const uint32_t dim1,
    const uint32_t dim0,
    const uint32_t tile_size
);

// ============================================================================
// ConicsMu Operations
// ============================================================================

/**
 * ConicsMu Forward Pass
 * 
 * Computes conics (inverse of blurred upper-left 2x2 sub-block) and mu
 * (compensation factor) from projected 3x3 covariance matrices.
 * 
 * Mathematical notation (with blur):
 *   hat{Sigma}_b = hat{Sigma} + eps * I_2
 *   conics = hat{Sigma}_b^{-1}  stored as (u11, u12, u22)
 *   mu = sqrt(2*pi * det(tilde{Sigma}_b) / det(hat{Sigma}_b))
 * 
 * @param cov2d_3x3 Projected 3x3 covariance matrices (C, N, 6)
 *                   stored as upper triangular (a, b, c, d, e, f)
 * @param eps Blur epsilon added to diagonal before inversion
 * 
 * @return std::tuple with:
 *   - conics: Inverse of blurred 2x2 (C, N, 3) as (u11, u12, u22)
 *   - mus: Compensation factor (C, N)
 */
std::tuple<torch::Tensor, torch::Tensor>
conics_mu_fwd_tensor(
    const torch::Tensor& cov2d_3x3,   // [C, N, 6]
    double eps
);

/**
 * ConicsMu Backward Pass (First-Order Gradients)
 * 
 * Computes gradient of loss w.r.t. cov2d_3x3 given upstream gradients
 * dL/d(conics) and dL/d(mus).
 * 
 * Recomputes cov3d_inv = (tilde{Sigma} + eps*I_3)^{-1} locally.
 * 
 * Key formulas:
 *   d(conics)/d(Sigma) uses: dP_ij/dS_kl = -P_ik * P_lj
 *   d(mu)/d(Sigma) = (mu/2) * (cov3d_inv - pad(conics))
 * 
 * @param cov2d_3x3 Forward input: projected covariance (C, N, 6)
 * @param eps Blur epsilon (same value used in forward)
 * @param conics Forward output: hat{Sigma}_b^{-1} (C, N, 3)
 * @param mus Forward output: compensation factor (C, N)
 * @param v_conics Upstream gradient dL/d(conics) (C, N, 3)
 * @param v_mus Upstream gradient dL/d(mus) (C, N)
 * 
 * @return v_cov2d_3x3: Gradient dL/d(cov2d_3x3) (C, N, 6)
 */
torch::Tensor
conics_mu_bwd_tensor(
    const torch::Tensor& cov2d_3x3,   // [C, N, 6]
    double eps,
    const torch::Tensor& conics,       // [C, N, 3]
    const torch::Tensor& mus,          // [C, N]
    const torch::Tensor& v_conics,     // [C, N, 3]
    const torch::Tensor& v_mus         // [C, N]
);

/**
 * ConicsMu Second-Order Backward Pass (Hessians)
 *
 * Given downstream first-order gradients and second-order Hessians w.r.t.
 * (conics, mu, means2d, opacity), computes upstream gradients and Hessians
 * w.r.t. cov2d_3x3, plus cross-Hessians with means2d & opacity.
 *
 * @return std::tuple with:
 *   - v_cov2d_3x3: Gradient dL/d(cov2d_3x3) (C, N, 6)
 *   - h_cov_cov: H_{sigma,sigma} upper-tri (C, N, 21)
 *   - h_cov_means2d: H_{sigma,means2d} (C, N, 12)
 *   - h_cov_opacity: H_{sigma,opacity} (C, N, 6)
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
conics_mu_bwd2_tensor(
    const torch::Tensor& cov2d_3x3,          // [C, N, 6]
    double eps,
    const torch::Tensor& conics,             // [C, N, 3]
    const torch::Tensor& mus,                // [C, N]
    const torch::Tensor& v_conics,           // [C, N, 3]
    const torch::Tensor& v_mus,              // [C, N]
    const torch::Tensor& h_conics,           // [C, N, 6]
    const torch::Tensor& h_conics_mu,        // [C, N, 3]
    const torch::Tensor& h_mu_mu,            // [C, N]
    const torch::Tensor& h_means2d_conics,   // [C, N, 6]
    const torch::Tensor& h_conics_opacity,   // [C, N, 3]
    const torch::Tensor& h_mu_means2d,       // [C, N, 2]
    const torch::Tensor& h_mu_opacity        // [C, N]
);

// ============================================================================
// Rasterize to Pixels Operations
// ============================================================================

/**
 * Rasterize Gaussians to pixels for CT reconstruction.
 * 
 * Tile-based rasterization with shared memory optimization.
 * CT formula: pixel_value = Σ (opacity × mu × exp(-sigma))
 * 
 * Dimension convention:
 * - dim0: Number of rows (y-axis), corresponds to means2d[..., 1]
 * - dim1: Number of columns (x-axis), corresponds to means2d[..., 0]
 * 
 * @param means2d 2D projected Gaussian centers (C, N, 2), format: (x, y) = (col, row)
 * @param conics Inverse covariance matrices (C, N, 3)
 * @param opacities Gaussian opacities (C, N), can be in [-1, 1]
 * @param mus Gaussian mu values (CT attenuation) (C, N)
 * @param tile_offsets Per-tile starting offsets (C, n_tiles_dim0, n_tiles_dim1)
 * @param flatten_ids Original Gaussian index for each intersection [n_isects]
 * @param dim1 Output size in dimension 1 (columns, x-axis)
 * @param dim0 Output size in dimension 0 (rows, y-axis)
 * @param tile_size Size of each tile in pixels
 * 
 * @return render_output: Rendered image (C, dim0, dim1)
 */
torch::Tensor rasterize_to_pixels_fwd_tensor(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& opacities,
    const torch::Tensor& mus,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& flatten_ids,
    const uint32_t dim1,
    const uint32_t dim0,
    const uint32_t tile_size
);

/**
 * CT Rasterization Backward Pass
 * 
 * Computes gradients of the loss w.r.t. Gaussian parameters.
 * 
 * CT Formula: O = Σ (opacity × mu × exp(-sigma))
 * where sigma = 0.5 × (a×δx² + c×δy²) + b×δx×δy
 * 
 * Unlike 3DGS which requires back-to-front processing due to alpha blending,
 * CT uses simple summation so gradient computation is order-independent.
 * 
 * @param means2d 2D projected centers (C, N, 2)
 * @param conics Inverse covariance matrices (C, N, 3)
 * @param opacities Gaussian opacities (C, N)
 * @param mus Gaussian mu values (C, N)
 * @param tile_offsets Per-tile starting offsets (C, n_tiles_dim0, n_tiles_dim1)
 * @param flatten_ids Original Gaussian index for each intersection [n_isects]
 * @param v_render_output Upstream gradient (C, dim0, dim1)
 * @param dim1 Output size in dimension 1 (columns, x-axis)
 * @param dim0 Output size in dimension 0 (rows, y-axis)
 * @param tile_size Size of each tile in pixels
 * 
 * @return std::tuple with:
 *   - v_means2d: Gradient w.r.t. means2d (C, N, 2)
 *   - v_conics: Gradient w.r.t. conics (C, N, 3)
 *   - v_opacities: Gradient w.r.t. opacities (C, N)
 *   - v_mus: Gradient w.r.t. mus (C, N)
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_to_pixels_bwd_tensor(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& opacities,
    const torch::Tensor& mus,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& flatten_ids,
    const torch::Tensor& v_render_output,
    const uint32_t dim1,
    const uint32_t dim0,
    const uint32_t tile_size,
    const bool absgrad
);

/**
 * CT Rasterization Backward Pass (Second-Order)
 * 
 * Computes both first-order gradients and second-order Hessian.
 * 
 * For CT reconstruction: O = Σ_i G_i where G_i = μ_i × o_i × exp(-σ_i)
 * 
 * Uses chain rule:
 *   v_θ = v_O × ∂G/∂θ
 *   h_{θi,θj} = h_O × (∂G/∂θi)(∂G/∂θj) + v_O × ∂²G/∂θi∂θj
 * 
 * Hessian storage uses upper triangular elements only (same order as conics):
 *   - means2d Hessian: [H00, H01, H11] (3 elements)
 *   - conics Hessian: [H00, H01, H02, H11, H12, H22] (6 elements)
 * 
 * @param means2d 2D projected centers (C, N, 2)
 * @param conics Inverse covariance matrices (C, N, 3)
 * @param opacities Gaussian opacities (C, N)
 * @param mus Gaussian mu values (C, N)
 * @param tile_offsets Per-tile starting offsets (C, n_tiles_dim0, n_tiles_dim1)
 * @param flatten_ids Original Gaussian index for each intersection [n_isects]
 * @param v_render_output First-order upstream gradient (C, dim0, dim1)
 * @param h_render_output Second-order upstream gradient (C, dim0, dim1)
 * @param dim1 Output size in dimension 1 (columns, x-axis)
 * @param dim0 Output size in dimension 0 (rows, y-axis)
 * @param tile_size Size of each tile in pixels
 * 
 * @return std::tuple with:
 *   - v_means2d: Gradient w.r.t. means2d (C, N, 2)
 *   - v_conics: Gradient w.r.t. conics (C, N, 3)
 *   - v_opacities: Gradient w.r.t. opacities (C, N)
 *   - v_mus: Gradient w.r.t. mus (C, N)
 *   - h_means2d: Hessian upper triangle w.r.t. means2d (C, N, 3)
 *   - h_conics: Hessian upper triangle w.r.t. conics (C, N, 6)
 *   - h_mu_mean: Cross-term Hessian mu-mean (C, N, 2)
 *   - h_mu_conic: Cross-term Hessian mu-conic (C, N, 3)
 *   - h_mean_conic: Cross-term Hessian mean-conic (C, N, 2, 3)
 *   - h_mu_opacity: Cross-term Hessian mu-opacity (C, N)
 *   - h_conic_opacity: Cross-term Hessian conic-opacity (C, N, 3)
 *   - h_mean_opacity: Cross-term Hessian mean-opacity (C, N, 2)
 *   - h_mus: Scalar Hessian w.r.t. mus (C, N)
 *   - h_opacities: Scalar Hessian w.r.t. opacities (C, N)
 */
std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,   // First-order gradients: v_means2d, v_conics, v_opacities, v_mus
    torch::Tensor, torch::Tensor,                                 // Full Hessians
    torch::Tensor, torch::Tensor, torch::Tensor,                  // Cross-term Hessians (mu-mean, mu-conic, mean-conic)
    torch::Tensor, torch::Tensor, torch::Tensor,                  // Cross-term Hessians involving opacity
    torch::Tensor, torch::Tensor                                  // Scalar Hessians
>
rasterize_to_pixels_bwd2_tensor(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& opacities,
    const torch::Tensor& mus,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& flatten_ids,
    const torch::Tensor& v_render_output,
    const torch::Tensor& h_render_output,
    const uint32_t dim1,
    const uint32_t dim0,
    const uint32_t tile_size
);

// ============================================================================
// WorldToImageProjection Operations
// ============================================================================

/**
 * WorldToImageProjection Forward Pass
 *
 * Fused world-to-camera transform + perspective projection.
 * Computes 2D projected means and 3x3 projected covariance.
 *
 * Pipeline: p_c = R*p+t → perspective proj → FOV clamp → J → cov2d_3x3 = J*R*Σ*R^T*J^T
 *
 * @param means World-space Gaussian centers [N, 3] (shared across cameras)\n * @param covars World-space covariance upper-tri [N, 6] (shared across cameras)
 * @param viewmats Camera extrinsic matrices [C, 4, 4] (row-major)
 * @param Ks Camera intrinsic matrices [C, 3, 3] (row-major)
 * @param dim1 Image width in pixels
 * @param dim0 Image height in pixels
 *
 * @return std::tuple with:
 *   - means2d: Projected 2D centers [C, N, 2]
 *   - cov2d_3x3: Projected 3x3 covariance upper-tri [C, N, 6]
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
world_to_image_projection_fwd_tensor(
    const torch::Tensor& means,       // [N, 3]
    const torch::Tensor& covars,      // [N, 6]
    const torch::Tensor& viewmats,    // [C, 4, 4]
    const torch::Tensor& Ks,          // [C, 3, 3]
    uint32_t dim1,
    uint32_t dim0,
    float eps_blur
);

/**
 * WorldToImageProjection Backward Pass (First-Order Gradients)
 *
 * Computes gradients dL/d(means) and dL/d(covars) from downstream
 * gradients dL/d(means2d) and dL/d(cov2d_3x3).
 * Gradients are accumulated over C cameras via atomicAdd.
 *
 * @param means Forward input: world-space means [N, 3] (shared across cameras)
 * @param covars Forward input: world-space covariance [N, 6] (shared across cameras)
 * @param viewmats Camera extrinsic [C, 4, 4]
 * @param Ks Camera intrinsic [C, 3, 3]
 * @param dim1 Image width
 * @param dim0 Image height
 * @param v_means2d Upstream dL/d(means2d) [C, N, 2]
 * @param v_cov2d_3x3 Upstream dL/d(cov2d_3x3) [C, N, 6]
 *
 * @return std::tuple with:
 *   - v_means: Gradient w.r.t. means [N, 3] (summed over C cameras)
 *   - v_covars: Gradient w.r.t. covars [N, 6] (summed over C cameras)
 */
std::tuple<torch::Tensor, torch::Tensor>
world_to_image_projection_bwd_tensor(
    const torch::Tensor& means,       // [N, 3]
    const torch::Tensor& covars,      // [N, 6]
    const torch::Tensor& viewmats,    // [C, 4, 4]
    const torch::Tensor& Ks,          // [C, 3, 3]
    uint32_t dim1,
    uint32_t dim0,
    const torch::Tensor& v_means2d,   // [C, N, 2]
    const torch::Tensor& v_cov2d_3x3  // [C, N, 6]
);

/**
 * WorldToImageProjection Second-Order Backward Pass (Hessians)
 *
 * Given downstream first-order gradients and second-order Hessians w.r.t.
 * (means2d, cov2d_3x3), computes upstream gradients and Hessians w.r.t.
 * (means, covars), plus cross-Hessians with opacity.
 *
 * Gradients and Hessians for means/covars are accumulated over C cameras.
 * Cross-Hessians with opacity remain per-camera since opacity is per-camera.
 *
 * Uses Gauss-Newton approximation for the position (means) Hessian
 * block, which drops second-order terms of the projection Jacobian.
 * The covars Hessian block is exact (linear map, no second-order terms).
 *
 * @param means Forward input [N, 3] (shared across cameras)
 * @param covars Forward input [N, 6] (shared across cameras)
 * @param viewmats Camera extrinsic [C, 4, 4]
 * @param Ks Camera intrinsic [C, 3, 3]
 * @param dim1 Image width
 * @param dim0 Image height
 * @param v_means2d Downstream gradient [C, N, 2]
 * @param v_cov2d_3x3 Downstream gradient [C, N, 6]
 * @param h_means2d Downstream Hessian H_{m,m} upper-tri [C, N, 3]
 * @param h_cov2d_3x3 Downstream Hessian H_{c,c} upper-tri [C, N, 21]
 * @param h_means2d_cov2d_3x3 Downstream cross-Hessian H_{m,c} [C, N, 12]
 * @param h_means2d_opacity Downstream cross-Hessian H_{m,opacity} [C, N, 2]
 * @param h_cov2d_3x3_opacity Downstream cross-Hessian H_{c,opacity} [C, N, 6]
 *
 * @return std::tuple with:
 *   - v_means: Gradient [N, 3] (summed over C)
 *   - v_covars: Gradient [N, 6] (summed over C)
 *   - h_means: Hessian H_{p,p} upper-tri [N, 6] (summed over C)
 *   - h_covars: Hessian H_{s,s} upper-tri [N, 21] (summed over C)
 *   - h_means_covars: Cross-Hessian H_{p,s} [N, 18] (summed over C)
 *   - h_means_opacity: Cross-Hessian H_{p,opacity} [C, N, 3] (per-camera)
 *   - h_covars_opacity: Cross-Hessian H_{s,opacity} [C, N, 6] (per-camera)
 */
std::tuple<
    torch::Tensor, torch::Tensor,   // v_means, v_covars
    torch::Tensor, torch::Tensor, torch::Tensor,  // h_means, h_covars, h_means_covars
    torch::Tensor, torch::Tensor    // h_means_opacity, h_covars_opacity
>
world_to_image_projection_bwd2_tensor(
    const torch::Tensor& means,                // [N, 3]
    const torch::Tensor& covars,               // [N, 6]
    const torch::Tensor& viewmats,             // [C, 4, 4]
    const torch::Tensor& Ks,                   // [C, 3, 3]
    uint32_t dim1,
    uint32_t dim0,
    const torch::Tensor& v_means2d,            // [C, N, 2]
    const torch::Tensor& v_cov2d_3x3,         // [C, N, 6]
    const torch::Tensor& h_means2d,            // [C, N, 3]
    const torch::Tensor& h_cov2d_3x3,         // [C, N, 21]
    const torch::Tensor& h_means2d_cov2d_3x3, // [C, N, 12]
    const torch::Tensor& h_means2d_opacity,    // [C, N, 2]
    const torch::Tensor& h_cov2d_3x3_opacity  // [C, N, 6]
);

// ============================================================================
// ParameterTransform Operations
// ============================================================================

/**
 * ParameterTransform Forward Pass
 *
 * Transforms unit quaternions and raw scale parameters into world-space
 * 3D covariance matrices.
 *
 * Pipeline:
 *   s = sigmoid(scale_ori) * (scale_max - scale_min) + scale_min
 *   R = quat_to_rotmat(quats)   (unit quaternion, no normalization)
 *   M = R * diag(s)
 *   covars = M * M^T  (stored as upper-tri 6 elements)
 *
 * @param quats     Unit quaternions [N, 4] (w, x, y, z), ||q||=1
 * @param scale_ori Raw scale parameters [N, 3] (before sigmoid)
 * @param scale_min Minimum scale after activation
 * @param scale_max Maximum scale after activation
 *
 * @return std::tuple with:
 *   - covars: World-space covariance upper-tri [N, 6]
 *   - max_scales: Max activated scale per Gaussian [N]
 */
std::tuple<torch::Tensor, torch::Tensor>
parameter_transform_fwd_tensor(
    const torch::Tensor& quats,       // [N, 4]
    const torch::Tensor& scale_ori,   // [N, 3]
    double scale_min,
    double scale_max
);

/**
 * ParameterTransform Backward Pass (First-Order Gradients)
 *
 * Computes dL/d(quats) and dL/d(scale_ori) from downstream dL/d(covars).
 *
 * @param quats     Forward input [N, 4]
 * @param scale_ori Forward input [N, 3]
 * @param scale_min Min scale
 * @param scale_max Max scale
 * @param v_covars  Downstream dL/d(covars) [N, 6]
 *
 * @return std::tuple with:
 *   - v_quats: Gradient w.r.t. quats [N, 4]
 *   - v_scale_ori: Gradient w.r.t. scale_ori [N, 3]
 */
std::tuple<torch::Tensor, torch::Tensor>
parameter_transform_bwd_tensor(
    const torch::Tensor& quats,       // [N, 4]
    const torch::Tensor& scale_ori,   // [N, 3]
    double scale_min,
    double scale_max,
    const torch::Tensor& v_covars     // [N, 6]
);

/**
 * ParameterTransform Second-Order Backward Pass (Hessians)
 *
 * Given downstream first-order gradients and second-order Hessians w.r.t.
 * covars, plus cross-Hessians with mean and opacity, compute upstream
 * gradients and Hessians w.r.t. (quats, scale_ori), plus cross-Hessians.
 *
 * @param quats             Forward input [N, 4]
 * @param scale_ori         Forward input [N, 3]
 * @param scale_min         Min scale
 * @param scale_max         Max scale
 * @param v_covars          Downstream dL/d(covars) [N, 6]
 * @param h_covars          Downstream H_{c,c} upper-tri [N, 21]
 * @param h_covars_means    Downstream H_{c,mean} [N, 18] 6x3 row-major
 * @param h_covars_opacity  Downstream H_{c,opacity} [N, 6]
 *
 * @return std::tuple with:
 *   - v_quats [N, 4], v_scale_ori [N, 3]   (first-order)
 *   - h_quats [N, 10], h_scale_ori [N, 3], h_quats_scale_ori [N, 12]  (theta-theta Hessians)
 *   - h_quats_means [N, 12], h_quats_opacity [N, 4]   (quat cross-Hessians)
 *   - h_scale_ori_means [N, 9], h_scale_ori_opacity [N, 3]  (scale cross-Hessians)
 */
std::tuple<
    torch::Tensor, torch::Tensor,   // v_quats, v_scale_ori
    torch::Tensor, torch::Tensor, torch::Tensor,  // h_quats, h_scale_ori, h_quats_scale_ori
    torch::Tensor, torch::Tensor,   // h_quats_means, h_quats_opacity
    torch::Tensor, torch::Tensor    // h_scale_ori_means, h_scale_ori_opacity
>
parameter_transform_bwd2_tensor(
    const torch::Tensor& quats,              // [N, 4]
    const torch::Tensor& scale_ori,          // [N, 3]
    double scale_min,
    double scale_max,
    const torch::Tensor& v_covars,           // [N, 6]
    const torch::Tensor& h_covars,           // [N, 21]
    const torch::Tensor& h_covars_means,     // [N, 18]
    const torch::Tensor& h_covars_opacity    // [N, 6]
);

// ============================================================================
// WorldToVolumeMapping Operations
// ============================================================================

/**
 * WorldToVolumeMapping Forward Pass
 *
 * Transforms world-space Gaussian parameters into volume (voxel) space:
 *   means → means3d_v  (affine transform)
 *   covars → conics     (diagonal scale + blur + inverse)
 *   max_scales → radii  (non-differentiable integer radii)
 *
 * @param means       World-space Gaussian centers [N, 3]
 * @param covars      World-space covariance upper-tri [N, 6]
 * @param max_scales  Max activated scale per Gaussian [N]
 * @param ranges      Physical extent per axis [3] (rx, ry, rz)
 * @param centers     Physical center per axis [3] (cx, cy, cz)
 * @param dim2        Voxel grid x dimension (fastest)
 * @param dim1        Voxel grid y dimension (middle)
 * @param dim0        Voxel grid z dimension (slowest)
 * @param eps3d       Blur epsilon (default 1e-4)
 * @param radius_clip Radius multiplier (default 3.0)
 *
 * @return std::tuple with:
 *   - means3d_v: Voxel-space centers [N, 3]
 *   - conics: Inverse covariance upper-tri [N, 6]
 *   - radii: Integer radii per axis [N, 3] (int32)
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
world_to_volume_mapping_fwd_tensor(
    const torch::Tensor& means,       // [N, 3]
    const torch::Tensor& covars,      // [N, 6]
    const torch::Tensor& max_scales,  // [N]
    const torch::Tensor& ranges,      // [3]
    const torch::Tensor& centers,     // [3]
    uint32_t dim2,
    uint32_t dim1,
    uint32_t dim0,
    double eps3d,
    double radius_clip
);

/**
 * WorldToVolumeMapping Backward Pass (First-Order Gradients)
 *
 * Computes dL/d(means) and dL/d(covars) from downstream gradients
 * dL/d(means3d_v) and dL/d(conics).
 *
 * @param means       Forward input [N, 3]
 * @param covars      Forward input [N, 6]
 * @param ranges      Physical extent [3]
 * @param centers     Physical center [3]
 * @param dim2,dim1,dim0 Voxel dimensions
 * @param eps3d       Blur epsilon
 * @param v_means3d_v Upstream dL/d(means3d_v) [N, 3]
 * @param v_conics    Upstream dL/d(conics) [N, 6]
 *
 * @return std::tuple with:
 *   - v_means: Gradient w.r.t. means [N, 3]
 *   - v_covars: Gradient w.r.t. covars [N, 6]
 */
std::tuple<torch::Tensor, torch::Tensor>
world_to_volume_mapping_bwd_tensor(
    const torch::Tensor& means,       // [N, 3]
    const torch::Tensor& covars,      // [N, 6]
    const torch::Tensor& ranges,      // [3]
    const torch::Tensor& centers,     // [3]
    uint32_t dim2,
    uint32_t dim1,
    uint32_t dim0,
    double eps3d,
    const torch::Tensor& v_means3d_v, // [N, 3]
    const torch::Tensor& v_conics     // [N, 6]
);

/**
 * WorldToVolumeMapping Second-Order Backward Pass (Hessians)
 *
 * Given downstream first-order gradients and second-order Hessians w.r.t.
 * (means3d_v, conics), plus cross-Hessians with opacity, compute upstream
 * gradients and Hessians w.r.t. (means, covars), plus cross-Hessians.
 *
 * @param means              Forward input [N, 3]
 * @param covars             Forward input [N, 6]
 * @param ranges             Physical extent [3]
 * @param centers            Physical center [3]
 * @param dim2,dim1,dim0     Voxel dimensions
 * @param eps3d              Blur epsilon
 * @param v_means3d_v        Downstream gradient [N, 3]
 * @param v_conics           Downstream gradient [N, 6]
 * @param h_means3d_v        Downstream Hessian H_{m,m} upper-tri [N, 6]
 * @param h_conics           Downstream Hessian H_{c,c} upper-tri [N, 21]
 * @param h_means3d_v_conics Downstream cross-Hessian H_{m,c} [N, 18]
 * @param h_means3d_v_opacity Downstream cross-Hessian H_{m,opacity} [N, 3]
 * @param h_conics_opacity   Downstream cross-Hessian H_{c,opacity} [N, 6]
 *
 * @return std::tuple with:
 *   - v_means: Gradient [N, 3]
 *   - v_covars: Gradient [N, 6]
 *   - h_means: Hessian H_{p,p} upper-tri [N, 6]
 *   - h_covars: Hessian H_{s,s} upper-tri [N, 21]
 *   - h_means_covars: Cross-Hessian H_{p,s} [N, 18]
 *   - h_means_opacity: Cross-Hessian H_{p,opacity} [N, 3]
 *   - h_covars_opacity: Cross-Hessian H_{s,opacity} [N, 6]
 */
std::tuple<
    torch::Tensor, torch::Tensor,   // v_means, v_covars
    torch::Tensor, torch::Tensor, torch::Tensor,  // h_means, h_covars, h_means_covars
    torch::Tensor, torch::Tensor    // h_means_opacity, h_covars_opacity
>
world_to_volume_mapping_bwd2_tensor(
    const torch::Tensor& means,                    // [N, 3]
    const torch::Tensor& covars,                   // [N, 6]
    const torch::Tensor& ranges,                   // [3]
    const torch::Tensor& centers,                  // [3]
    uint32_t dim2,
    uint32_t dim1,
    uint32_t dim0,
    double eps3d,
    const torch::Tensor& v_means3d_v,              // [N, 3]
    const torch::Tensor& v_conics,                 // [N, 6]
    const torch::Tensor& h_means3d_v,              // [N, 6]
    const torch::Tensor& h_conics,                 // [N, 21]
    const torch::Tensor& h_means3d_v_conics,       // [N, 18]
    const torch::Tensor& h_means3d_v_opacity,      // [N, 3]
    const torch::Tensor& h_conics_opacity          // [N, 6]
);

}  // namespace gsplat2ct

#endif  // GSPLAT2CT_OPS_H
