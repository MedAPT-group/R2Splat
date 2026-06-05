/**
 * CUDA kernel launch declarations for CT voxelization.
 */

#ifndef GSPLAT2CT_VOXELIZE_TO_VOXELS_H
#define GSPLAT2CT_VOXELIZE_TO_VOXELS_H

#include "../include/common.h"
#include <cuda_runtime.h>

namespace gsplat2ct {

template <typename S>
void launch_voxelize_to_voxels_fwd(
    const S* __restrict__ means3d,
    const S* __restrict__ conics,
    const S* __restrict__ opacities,
    const int32_t* __restrict__ tile_offsets,
    const int32_t* __restrict__ flatten_ids,
    S* __restrict__ render_output,
    uint32_t N,
    uint32_t n_isects,
    uint32_t dim2,
    uint32_t dim1,
    uint32_t dim0,
    uint32_t tile_size,
    uint32_t n_tiles_dim2,
    uint32_t n_tiles_dim1,
    uint32_t n_tiles_dim0,
    cudaStream_t stream);

template <typename S>
void launch_voxelize_to_voxels_mean_var_fwd(
    const S* __restrict__ means3d,
    const S* __restrict__ conics,
    const S* __restrict__ opacities,
    const S* __restrict__ uncertainties,
    const int32_t* __restrict__ tile_offsets,
    const int32_t* __restrict__ flatten_ids,
    S* __restrict__ render_output,
    S* __restrict__ render_var,
    uint32_t N,
    uint32_t n_isects,
    uint32_t dim2,
    uint32_t dim1,
    uint32_t dim0,
    uint32_t tile_size,
    uint32_t n_tiles_dim2,
    uint32_t n_tiles_dim1,
    uint32_t n_tiles_dim0,
    cudaStream_t stream);

template <typename S>
void launch_beta_voxelize_to_voxels_fwd(
    const S* __restrict__ means3d,
    const S* __restrict__ conics,
    const S* __restrict__ opacities,
    const S* __restrict__ betas,
    const int32_t* __restrict__ tile_offsets,
    const int32_t* __restrict__ flatten_ids,
    S* __restrict__ render_output,
    uint32_t N,
    uint32_t n_isects,
    uint32_t dim2,
    uint32_t dim1,
    uint32_t dim0,
    uint32_t tile_size,
    uint32_t n_tiles_dim2,
    uint32_t n_tiles_dim1,
    uint32_t n_tiles_dim0,
    cudaStream_t stream);

template <typename S>
void launch_beta_voxelize_to_voxels_mean_var_fwd(
    const S* __restrict__ means3d,
    const S* __restrict__ conics,
    const S* __restrict__ opacities,
    const S* __restrict__ betas,
    const S* __restrict__ uncertainties,
    const int32_t* __restrict__ tile_offsets,
    const int32_t* __restrict__ flatten_ids,
    S* __restrict__ render_output,
    S* __restrict__ render_var,
    uint32_t N,
    uint32_t n_isects,
    uint32_t dim2,
    uint32_t dim1,
    uint32_t dim0,
    uint32_t tile_size,
    uint32_t n_tiles_dim2,
    uint32_t n_tiles_dim1,
    uint32_t n_tiles_dim0,
    cudaStream_t stream);

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
    cudaStream_t stream);

template <typename S>
void launch_voxelize_to_voxels_bwd_per_gaussian(
    const S* __restrict__ means3d,
    const S* __restrict__ conics,
    const S* __restrict__ opacities,
    const int64_t* __restrict__ isect_ids,
    const int32_t* __restrict__ flatten_ids,
    const S* __restrict__ v_render_output,
    const S* __restrict__ v_render_var,
    const S* __restrict__ uncertainties,
    S* __restrict__ v_means3d,
    S* __restrict__ v_conics,
    S* __restrict__ v_opacities,
    S* __restrict__ v_uncertainties,
    uint32_t N,
    uint32_t n_isects,
    uint32_t dim2,
    uint32_t dim1,
    uint32_t dim0,
    uint32_t tile_size,
    uint32_t n_tiles_dim2,
    uint32_t n_tiles_dim1,
    uint32_t n_tiles_dim0,
    cudaStream_t stream);

template <typename S>
void launch_beta_voxelize_to_voxels_bwd(
    const S* __restrict__ means3d,
    const S* __restrict__ conics,
    const S* __restrict__ opacities,
    const S* __restrict__ betas,
    const int32_t* __restrict__ tile_offsets,
    const int32_t* __restrict__ flatten_ids,
    const S* __restrict__ v_render_output,
    S* __restrict__ v_means3d,
    S* __restrict__ v_conics,
    S* __restrict__ v_opacities,
    S* __restrict__ v_betas,
    uint32_t N,
    uint32_t n_isects,
    uint32_t dim2,
    uint32_t dim1,
    uint32_t dim0,
    uint32_t tile_size,
    uint32_t n_tiles_dim2,
    uint32_t n_tiles_dim1,
    uint32_t n_tiles_dim0,
    cudaStream_t stream);

template <typename S>
void launch_beta_voxelize_to_voxels_bwd_per_gaussian(
    const S* __restrict__ means3d,
    const S* __restrict__ conics,
    const S* __restrict__ opacities,
    const S* __restrict__ betas,
    const int64_t* __restrict__ isect_ids,
    const int32_t* __restrict__ flatten_ids,
    const S* __restrict__ v_render_output,
    const S* __restrict__ v_render_var,
    const S* __restrict__ uncertainties,
    S* __restrict__ v_means3d,
    S* __restrict__ v_conics,
    S* __restrict__ v_opacities,
    S* __restrict__ v_betas,
    S* __restrict__ v_uncertainties,
    uint32_t N,
    uint32_t n_isects,
    uint32_t dim2,
    uint32_t dim1,
    uint32_t dim0,
    uint32_t tile_size,
    uint32_t n_tiles_dim2,
    uint32_t n_tiles_dim1,
    uint32_t n_tiles_dim0,
    cudaStream_t stream);

} // namespace gsplat2ct

#endif // GSPLAT2CT_VOXELIZE_TO_VOXELS_H
