/**
 * C++ tensor wrapper declarations registered by ext.cpp.
 */

#ifndef GSPLAT2CT_OPS_H
#define GSPLAT2CT_OPS_H

#include <torch/extension.h>
#include <tuple>

namespace gsplat2ct {

using ImageLossResult = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>;

ImageLossResult image_loss_fwd(
    const torch::Tensor& prediction,
    const torch::Tensor& target,
    float lambda_l1);

torch::Tensor image_loss_bwd(
    const torch::Tensor& prediction,
    const torch::Tensor& target,
    const torch::Tensor& grad_output,
    float lambda_l1);

ImageLossResult image_loss_l1_fwd(
    const torch::Tensor& prediction,
    const torch::Tensor& target,
    float lambda_l1);

torch::Tensor image_loss_l1_bwd(
    const torch::Tensor& prediction,
    const torch::Tensor& target,
    const torch::Tensor& grad_output,
    float lambda_l1);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> volume_loss_fwd(
    const torch::Tensor& volume,
    float lambda_tv);

torch::Tensor volume_loss_bwd(
    const torch::Tensor& volume,
    const torch::Tensor& grad_output,
    float lambda_tv);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
isect_tiles_tensor(
    const torch::Tensor& means2d,
    const torch::Tensor& radii,
    const uint32_t tile_size,
    const uint32_t n_tiles_dim1,
    const uint32_t n_tiles_dim0);

torch::Tensor isect_offset_encode_tensor(
    const torch::Tensor& isect_ids,
    const uint32_t C,
    const uint32_t n_tiles_dim1,
    const uint32_t n_tiles_dim0);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
isect_tiles_3d_tensor(
    const torch::Tensor& means3d,
    const torch::Tensor& radii,
    const uint32_t tile_size,
    const uint32_t n_tiles_dim2,
    const uint32_t n_tiles_dim1,
    const uint32_t n_tiles_dim0);

torch::Tensor isect_offset_encode_3d_tensor(
    const torch::Tensor& isect_ids,
    const uint32_t n_tiles_dim2,
    const uint32_t n_tiles_dim1,
    const uint32_t n_tiles_dim0);

torch::Tensor voxelize_to_voxels_fwd_tensor(
    const torch::Tensor& means3d,
    const torch::Tensor& conics,
    const torch::Tensor& opacities,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& flatten_ids,
    const uint32_t dim2,
    const uint32_t dim1,
    const uint32_t dim0,
    const uint32_t tile_size);

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
    const uint32_t tile_size);

torch::Tensor beta_voxelize_to_voxels_fwd_tensor(
    const torch::Tensor& means3d,
    const torch::Tensor& conics,
    const torch::Tensor& opacities,
    const torch::Tensor& betas,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& flatten_ids,
    const uint32_t dim2,
    const uint32_t dim1,
    const uint32_t dim0,
    const uint32_t tile_size);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
beta_voxelize_to_voxels_bwd_tensor(
    const torch::Tensor& means3d,
    const torch::Tensor& conics,
    const torch::Tensor& opacities,
    const torch::Tensor& betas,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& flatten_ids,
    const torch::Tensor& v_render_output,
    const uint32_t dim2,
    const uint32_t dim1,
    const uint32_t dim0,
    const uint32_t tile_size);

std::tuple<torch::Tensor, torch::Tensor>
conics_mu_fwd_tensor(
    const torch::Tensor& cov2d_3x3,
    double eps);

torch::Tensor conics_mu_bwd_tensor(
    const torch::Tensor& cov2d_3x3,
    double eps,
    const torch::Tensor& conics,
    const torch::Tensor& mus,
    const torch::Tensor& v_conics,
    const torch::Tensor& v_mus);

torch::Tensor rasterize_to_pixels_fwd_tensor(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& opacities,
    const torch::Tensor& mus,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& flatten_ids,
    const uint32_t dim1,
    const uint32_t dim0,
    const uint32_t tile_size);

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
    const bool absgrad);

torch::Tensor beta_rasterize_to_pixels_fwd_tensor(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& opacities,
    const torch::Tensor& mus,
    const torch::Tensor& betas,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& flatten_ids,
    const uint32_t dim1,
    const uint32_t dim0,
    const uint32_t tile_size);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
beta_rasterize_to_pixels_bwd_tensor(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& opacities,
    const torch::Tensor& mus,
    const torch::Tensor& betas,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& flatten_ids,
    const torch::Tensor& v_render_output,
    const uint32_t dim1,
    const uint32_t dim0,
    const uint32_t tile_size,
    const bool absgrad);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
world_to_image_projection_fwd_tensor(
    const torch::Tensor& means,
    const torch::Tensor& covars,
    const torch::Tensor& viewmats,
    const torch::Tensor& Ks,
    uint32_t dim1,
    uint32_t dim0,
    float eps_blur);

std::tuple<torch::Tensor, torch::Tensor>
world_to_image_projection_bwd_tensor(
    const torch::Tensor& means,
    const torch::Tensor& covars,
    const torch::Tensor& viewmats,
    const torch::Tensor& Ks,
    uint32_t dim1,
    uint32_t dim0,
    const torch::Tensor& v_means2d,
    const torch::Tensor& v_cov2d_3x3);

std::tuple<torch::Tensor, torch::Tensor>
parameter_transform_fwd_tensor(
    const torch::Tensor& quats,
    const torch::Tensor& scale_ori,
    double scale_min,
    double scale_max);

std::tuple<torch::Tensor, torch::Tensor>
parameter_transform_bwd_tensor(
    const torch::Tensor& quats,
    const torch::Tensor& scale_ori,
    double scale_min,
    double scale_max,
    const torch::Tensor& v_covars);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
world_to_volume_mapping_fwd_tensor(
    const torch::Tensor& means,
    const torch::Tensor& covars,
    const torch::Tensor& max_scales,
    const torch::Tensor& ranges,
    const torch::Tensor& centers,
    uint32_t dim2,
    uint32_t dim1,
    uint32_t dim0,
    double eps3d,
    double radius_clip);

std::tuple<torch::Tensor, torch::Tensor>
world_to_volume_mapping_bwd_tensor(
    const torch::Tensor& means,
    const torch::Tensor& covars,
    const torch::Tensor& ranges,
    const torch::Tensor& centers,
    uint32_t dim2,
    uint32_t dim1,
    uint32_t dim0,
    double eps3d,
    const torch::Tensor& v_means3d_v,
    const torch::Tensor& v_conics);

}  // namespace gsplat2ct

#endif  // GSPLAT2CT_OPS_H
