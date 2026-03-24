/**
 * Voxelize To Voxels CUDA Kernel Launch Functions
 * 
 * This header declares the launch functions that bridge between
 * C++ tensor processing (VoxelizeToVoxels.cpp) and CUDA kernels (VoxelizeToVoxelsFwd.cu).
 * 
 * The module voxelizes 3D Gaussians to a regular voxel grid for CT reconstruction.
 * Unlike rendering, CT reconstruction uses simple ray summation, so no alpha blending.
 * 
 * Key features:
 * - Single channel output (grayscale for CT)
 * - No color array (removed from original gsplat)
 * - No mask support (removed)
 * - Conics merged to [N, 6] for inverse covariance matrix
 * - Tile-based with shared memory optimization
 * 
 * Dimension convention (following RasterizeToPixelsFwd.cu pattern):
 *   - px -- k -- dim2 -- x (fastest varying, columns)
 *   - py -- j -- dim1 -- y (middle dimension)
 *   - pz -- i -- dim0 -- z (slowest varying, rows)
 *   - Output shape: [dim0, dim1, dim2]
 *   - Voxel index: vox_id = i * (dim1 * dim2) + j * dim2 + k
 * 
 * Author: Dy164
 * Date: 2026-02-02
 */

#ifndef GSPLAT2CT_VOXELIZE_TO_VOXELS_H
#define GSPLAT2CT_VOXELIZE_TO_VOXELS_H

#include "../include/common.h"
#include <cuda_runtime.h>

namespace gsplat2ct {

// ============================================================================
// Forward Pass Launch Function
// ============================================================================

/**
 * Launch kernel to voxelize Gaussians to a 3D grid for CT reconstruction.
 * 
 * Each voxel accumulates contributions from all intersecting Gaussians.
 * CT formula: voxel_value = Σ (opacity × exp(-sigma))
 * where sigma = 0.5 × delta^T × conic × delta
 *       delta = (px - mean_x, py - mean_y, pz - mean_z)
 * 
 * @tparam S Scalar type (float, double)
 * 
 * @param means3d 3D Gaussian centers [N, 3], stored as (x, y, z)
 * @param conics Inverse covariance matrices [N, 6], stored as upper triangular
 *               elements [c00, c01, c02, c11, c12, c22]
 * @param opacities Gaussian opacities [N]
 * @param tile_offsets Per-tile starting offsets [n_tiles_dim0, n_tiles_dim1, n_tiles_dim2]
 * @param flatten_ids Original Gaussian index for each intersection [n_isects]
 * @param render_output Output voxel grid [dim0, dim1, dim2]
 * @param N Number of Gaussians
 * @param n_isects Total number of tile-Gaussian intersections
 * @param dim2 Output size in dimension 2 (x, fastest varying)
 * @param dim1 Output size in dimension 1 (y, middle)
 * @param dim0 Output size in dimension 0 (z, slowest varying)
 * @param tile_size Size of each tile in voxels
 * @param n_tiles_dim2 Number of tiles in x dimension (fastest)
 * @param n_tiles_dim1 Number of tiles in y dimension (middle)
 * @param n_tiles_dim0 Number of tiles in z dimension (slowest)
 * @param stream CUDA stream
 */
template <typename S>
void launch_voxelize_to_voxels_fwd(
    const S* __restrict__ means3d,       // [N, 3]
    const S* __restrict__ conics,        // [N, 6]
    const S* __restrict__ opacities,     // [N]
    const int32_t* __restrict__ tile_offsets,  // [n_tiles_dim0, n_tiles_dim1, n_tiles_dim2]
    const int32_t* __restrict__ flatten_ids,   // [n_isects]
    S* __restrict__ render_output,       // [dim0, dim1, dim2]
    uint32_t N,
    uint32_t n_isects,
    uint32_t dim2,
    uint32_t dim1,
    uint32_t dim0,
    uint32_t tile_size,
    uint32_t n_tiles_dim2,
    uint32_t n_tiles_dim1,
    uint32_t n_tiles_dim0,
    cudaStream_t stream
);

// ============================================================================
// First-Order Backward Pass Launch Function
// ============================================================================

/**
 * Launch kernel for CT voxelization backward pass (gradients only).
 * 
 * Computes gradients of the loss w.r.t.:
 * - means3d: ∂L/∂mean [N, 3]
 * - conics: ∂L/∂conic [N, 6]
 * - opacities: ∂L/∂opacity [N]
 * 
 * @tparam S Scalar type (float, double)
 * 
 * @param means3d 3D Gaussian centers [N, 3]
 * @param conics Inverse covariance matrices [N, 6]
 * @param opacities Gaussian opacities [N]
 * @param tile_offsets Per-tile starting offsets [n_tiles_dim0, n_tiles_dim1, n_tiles_dim2]
 * @param flatten_ids Original Gaussian index for each intersection [n_isects]
 * @param v_render_output Gradient from upstream [dim0, dim1, dim2]
 * @param v_means3d Output gradient w.r.t. means3d [N, 3]
 * @param v_conics Output gradient w.r.t. conics [N, 6]
 * @param v_opacities Output gradient w.r.t. opacities [N]
 * @param N Number of Gaussians
 * @param n_isects Total number of tile-Gaussian intersections
 * @param dim2 Output size in dimension 2 (x, fastest varying)
 * @param dim1 Output size in dimension 1 (y, middle)
 * @param dim0 Output size in dimension 0 (z, slowest varying)
 * @param tile_size Size of each tile in voxels
 * @param n_tiles_dim2 Number of tiles in x dimension
 * @param n_tiles_dim1 Number of tiles in y dimension
 * @param n_tiles_dim0 Number of tiles in z dimension
 * @param stream CUDA stream
 */
template <typename S>
void launch_voxelize_to_voxels_bwd(
    const S* __restrict__ means3d,
    const S* __restrict__ conics,
    const S* __restrict__ opacities,
    const int32_t* __restrict__ tile_offsets,
    const int32_t* __restrict__ flatten_ids,
    const S* __restrict__ v_render_output,
    S* __restrict__ v_means3d,
    S* __restrict__ v_conics,
    S* __restrict__ v_opacities,
    uint32_t N,
    uint32_t n_isects,
    uint32_t dim2,
    uint32_t dim1,
    uint32_t dim0,
    uint32_t tile_size,
    uint32_t n_tiles_dim2,
    uint32_t n_tiles_dim1,
    uint32_t n_tiles_dim0,
    cudaStream_t stream
);

// ============================================================================
// Second-Order Backward Pass Launch Function
// ============================================================================

/**
 * Launch kernel for CT voxelization second-order backward pass.
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
 * @tparam S Scalar type (float, double)
 */
template <typename S>
void launch_voxelize_to_voxels_bwd2(
    const S* __restrict__ means3d,
    const S* __restrict__ conics,
    const S* __restrict__ opacities,
    const int32_t* __restrict__ tile_offsets,
    const int32_t* __restrict__ flatten_ids,
    const S* __restrict__ v_render_output,
    const S* __restrict__ h_render_output,
    S* __restrict__ v_means3d,
    S* __restrict__ v_conics,
    S* __restrict__ v_opacities,
    S* __restrict__ h_means3d,
    S* __restrict__ h_conics,
    S* __restrict__ h_opacities,
    S* __restrict__ h_op_mean,
    S* __restrict__ h_op_conic,
    S* __restrict__ h_mean_conic,
    uint32_t N,
    uint32_t n_isects,
    uint32_t dim2,
    uint32_t dim1,
    uint32_t dim0,
    uint32_t tile_size,
    uint32_t n_tiles_dim2,
    uint32_t n_tiles_dim1,
    uint32_t n_tiles_dim0,
    cudaStream_t stream
);

}  // namespace gsplat2ct

#endif  // GSPLAT2CT_VOXELIZE_TO_VOXELS_H
