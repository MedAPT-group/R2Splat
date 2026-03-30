/**
 * @file RasterizeToPixels.h
 * @brief CUDA Kernel Launch Functions for CT Rasterization
 * 
 * This header declares the launch functions that bridge between
 * C++ tensor processing (RasterizeToPixels.cpp) and CUDA kernels 
 * (RasterizeToPixelsFwd.cu, RasterizeToPixelsBwd.cu).
 * 
 * The module implements tile-based rasterization for CT reconstruction.
 * 
 * Key differences from standard 3DGS rasterization:
 * - No packed mode support: input is always [C, N, ...]
 * - No color channels: CT uses opacity × mu directly (single channel)
 * - Simpler output: just accumulated projection value per pixel
 * - Uses cooperative loading with shared memory for efficiency
 * 
 * CT Formula:
 *   pixel_value = Σ_i (opacity_i × mu_i × exp(-sigma_i))
 *   where sigma_i = 0.5 × (x - mean_i)^T × conic_i × (x - mean_i)
 * 
 * Author: Dy164
 * Date: 2025-01-12
 */

#ifndef GSPLAT2CT_RASTERIZE_TO_PIXELS_H
#define GSPLAT2CT_RASTERIZE_TO_PIXELS_H

#include "../include/common.h"
#include <cuda_runtime.h>

namespace gsplat2ct {

// ============================================================================
// Forward Pass: Rasterize Gaussians to Pixels
// ============================================================================

/**
 * Launch kernel for CT rasterization forward pass.
 * 
 * Each block processes one tile (TILE_SIZE × TILE_SIZE pixels).
 * Threads within a block cooperatively load Gaussian data into shared memory,
 * then each thread accumulates contributions from all Gaussians to its pixel.
 * 
 * Dimension convention:
 * - dim0: Number of rows (y-axis), corresponds to means2d[..., 1]
 * - dim1: Number of columns (x-axis), corresponds to means2d[..., 0]
 * - Output shape: [C, dim0, dim1]
 * 
 * @tparam S Scalar type (float, double)
 * 
 * @param means2d 2D projected Gaussian centers [C * N, 2], format: (x, y) = (col, row)
 * @param conics Inverse covariance matrices (upper triangular) [C * N, 3]
 * @param opacities Gaussian opacities [C * N]
 * @param mus Gaussian mu values (CT attenuation) [C * N]
 * @param tile_offsets Offset for each tile [C, n_tiles_dim0, n_tiles_dim1]
 * @param flatten_ids Flattened Gaussian indices [n_isects]
 * @param render_output Output rendered image [C, dim0, dim1]
 * @param C Number of cameras
 * @param N Number of Gaussians per camera
 * @param n_isects Number of intersections
 * @param dim1 Output size in dimension 1 (columns, x-axis)
 * @param dim0 Output size in dimension 0 (rows, y-axis)
 * @param tile_size Size of each tile in pixels
 * @param n_tiles_dim1 Number of tiles in dimension 1
 * @param n_tiles_dim0 Number of tiles in dimension 0
 * @param stream CUDA stream
 */
template <typename S>
void launch_rasterize_to_pixels_fwd(
    const S* __restrict__ means2d,          // [C * N, 2]
    const S* __restrict__ conics,           // [C * N, 3]
    const S* __restrict__ opacities,        // [C * N]
    const S* __restrict__ mus,              // [C * N]
    const int32_t* __restrict__ tile_offsets,  // [C, n_tiles_dim0, n_tiles_dim1]
    const int32_t* __restrict__ flatten_ids,   // [n_isects]
    S* __restrict__ render_output,          // [C, dim0, dim1]
    uint32_t C,
    uint32_t N,
    uint32_t n_isects,
    uint32_t dim1,
    uint32_t dim0,
    uint32_t tile_size,
    uint32_t n_tiles_dim1,
    uint32_t n_tiles_dim0,
    cudaStream_t stream
);

// ============================================================================
// Backward Pass: First-Order Gradients
// ============================================================================

/**
 * Launch kernel for CT rasterization backward pass (gradients only).
 * 
 * Computes gradients of the loss w.r.t.:
 * - means2d: ∂L/∂mean [C * N, 2]
 * - conics: ∂L/∂conic [C * N, 3]
 * - opacities: ∂L/∂opacity [C * N]
 * - mus: ∂L/∂mu [C * N]
 * 
 * Uses reverse-mode automatic differentiation.
 * Unlike 3DGS which requires back-to-front processing due to alpha blending,
 * CT uses simple summation so gradient computation is order-independent.
 * 
 * @tparam S Scalar type (float, double)
 * 
 * @param means2d 2D projected Gaussian centers [C * N, 2]
 * @param conics Inverse covariance matrices [C * N, 3]
 * @param opacities Gaussian opacities [C * N]
 * @param mus Gaussian mu values [C * N]
 * @param tile_offsets Offset for each tile [C, n_tiles_dim0, n_tiles_dim1]
 * @param flatten_ids Flattened Gaussian indices [n_isects]
 * @param v_render_output Gradient from upstream [C, dim0, dim1] (∂L/∂O)
 * @param v_means2d Output gradient w.r.t. means2d [C * N, 2]
 * @param v_conics Output gradient w.r.t. conics [C * N, 3]
 * @param v_opacities Output gradient w.r.t. opacities [C * N]
 * @param v_mus Output gradient w.r.t. mus [C * N]
 * @param C Number of cameras
 * @param N Number of Gaussians per camera
 * @param n_isects Total number of tile-Gaussian intersections
 * @param dim1 Output size in dimension 1 (columns, x-axis)
 * @param dim0 Output size in dimension 0 (rows, y-axis)
 * @param tile_size Size of each tile in pixels
 * @param n_tiles_dim1 Number of tiles in dimension 1
 * @param n_tiles_dim0 Number of tiles in dimension 0
 * @param stream CUDA stream
 */
template <typename S>
void launch_rasterize_to_pixels_bwd(
    const S* __restrict__ means2d,
    const S* __restrict__ conics,
    const S* __restrict__ opacities,
    const S* __restrict__ mus,
    const int32_t* __restrict__ tile_offsets,
    const int32_t* __restrict__ flatten_ids,
    const S* __restrict__ v_render_output,
    // Optional: absolute gradient accumulator for means2d (AbsGS)
    // If nullptr, absgrad is disabled.
    S* __restrict__ v_means2d_abs,
    S* __restrict__ v_means2d,
    S* __restrict__ v_conics,
    S* __restrict__ v_opacities,
    S* __restrict__ v_mus,
    uint32_t C,
    uint32_t N,
    uint32_t n_isects,
    uint32_t dim1,
    uint32_t dim0,
    uint32_t tile_size,
    uint32_t n_tiles_dim1,
    uint32_t n_tiles_dim0,
    cudaStream_t stream
);

// ============================================================================
// Second-Order Backward Pass: Hessian Computation
// ============================================================================

/**
 * Launch kernel for CT rasterization second-order backward pass.
 * 
 * Computes both first-order gradients and second-order derivatives (Hessian).
 * For CT reconstruction: O = Σ_i G_i where G_i = μ_i × o_i × exp(-σ_i)
 * 
 * Given upstream:
 *   v_O = ∂L/∂O (first derivative)
 *   h_O = ∂²L/∂O² (second derivative / Hessian diagonal)
 * 
 * We compute using chain rule:
 *   v_θ = v_O × ∂G/∂θ
 *   h_{θi,θj} = h_O × (∂G/∂θi)(∂G/∂θj) + v_O × ∂²G/∂θi∂θj
 * 
 * Hessian storage uses upper triangular elements only (consistent with conics storage):
 *   h_means2d: [C * N, 3] upper triangle [H00, H01, H11]
 *   h_conics: [C * N, 6] upper triangle [H00, H01, H02, H11, H12, H22]
 * 
 * @tparam S Scalar type (float, double)
 * 
 * @param means2d 2D projected Gaussian centers [C * N, 2]
 * @param conics Inverse covariance matrices [C * N, 3]
 * @param opacities Gaussian opacities [C * N]
 * @param mus Gaussian mu values [C * N]
 * @param tile_offsets Offset for each tile [C, n_tiles_dim0, n_tiles_dim1]
 * @param flatten_ids Flattened Gaussian indices [n_isects]
 * @param v_render_output First-order gradient from upstream [C, dim0, dim1]
 * @param h_render_output Second-order gradient from upstream [C, dim0, dim1]
 * @param v_means2d Output first-order gradient w.r.t. means2d [C * N, 2]
 * @param v_conics Output first-order gradient w.r.t. conics [C * N, 3]
 * @param v_opacities Output first-order gradient w.r.t. opacities [C * N]
 * @param v_mus Output first-order gradient w.r.t. mus [C * N]
 * @param h_means2d Output Hessian upper triangle w.r.t. means2d [C * N, 3]
 * @param h_conics Output Hessian upper triangle w.r.t. conics [C * N, 6]
 * @param h_mu_mean Output cross-term Hessian mu-mean [C * N, 2]
 * @param h_mu_conic Output cross-term Hessian mu-conic [C * N, 3]
 * @param h_mean_conic Output cross-term Hessian mean-conic [C * N, 2, 3]
 * @param h_mu_opacity Output cross-term Hessian mu-opacity [C * N]
 * @param h_conic_opacity Output cross-term Hessian conic-opacity [C * N, 3]
 * @param h_mean_opacity Output cross-term Hessian mean-opacity [C * N, 2]
 * @param h_mus Output scalar Hessian w.r.t. mus [C * N]
 * @param h_opacities Output scalar Hessian w.r.t. opacities [C * N]
 * @param C Number of cameras
 * @param N Number of Gaussians per camera
 * @param n_isects Total number of tile-Gaussian intersections
 * @param dim1 Output size in dimension 1 (columns, x-axis)
 * @param dim0 Output size in dimension 0 (rows, y-axis)
 * @param tile_size Size of each tile in pixels
 * @param n_tiles_dim1 Number of tiles in dimension 1
 * @param n_tiles_dim0 Number of tiles in dimension 0
 * @param stream CUDA stream
 */
template <typename S>
void launch_rasterize_to_pixels_bwd2(
    const S* __restrict__ means2d,
    const S* __restrict__ conics,
    const S* __restrict__ opacities,
    const S* __restrict__ mus,
    const int32_t* __restrict__ tile_offsets,
    const int32_t* __restrict__ flatten_ids,
    const S* __restrict__ v_render_output,
    const S* __restrict__ h_render_output,
    S* __restrict__ v_means2d,
    S* __restrict__ v_conics,
    S* __restrict__ v_opacities,
    S* __restrict__ v_mus,
    S* __restrict__ h_means2d,
    S* __restrict__ h_conics,
    S* __restrict__ h_mu_mean,
    S* __restrict__ h_mu_conic,
    S* __restrict__ h_mean_conic,
    S* __restrict__ h_mu_opacity,
    S* __restrict__ h_conic_opacity,
    S* __restrict__ h_mean_opacity,
    S* __restrict__ h_mus,
    S* __restrict__ h_opacities,
    uint32_t C,
    uint32_t N,
    uint32_t n_isects,
    uint32_t dim1,
    uint32_t dim0,
    uint32_t tile_size,
    uint32_t n_tiles_dim1,
    uint32_t n_tiles_dim0,
    cudaStream_t stream
);

}  // namespace gsplat2ct

#endif  // GSPLAT2CT_RASTERIZE_TO_PIXELS_H
