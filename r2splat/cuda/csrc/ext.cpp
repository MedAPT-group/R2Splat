/**
 * PyBind11 binding entry point for CUDA extensions.
 */

#include <torch/extension.h>
#include "../include/Ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA extensions for first-order CT reconstruction";

    m.def("image_loss_fwd", &r2splat::image_loss_fwd,
          "Image loss (mean L1) forward pass");
    m.def("image_loss_bwd", &r2splat::image_loss_bwd,
          "Image loss (mean L1) backward pass");
    m.def("image_loss_l1_fwd", &r2splat::image_loss_l1_fwd,
          "Image loss (mean L1) forward pass");
    m.def("image_loss_l1_bwd", &r2splat::image_loss_l1_bwd,
          "Image loss (mean L1) backward pass");

    m.def("volume_loss_fwd", &r2splat::volume_loss_fwd,
          "Volume loss (anisotropic TV) forward pass");
    m.def("volume_loss_bwd", &r2splat::volume_loss_bwd,
          "Volume loss (anisotropic TV) backward pass");

    m.def("isect_tiles", &r2splat::isect_tiles_tensor,
          "Compute 2D tile-Gaussian intersections");
    m.def("isect_offset_encode", &r2splat::isect_offset_encode_tensor,
          "Encode 2D tile offsets");
    m.def("isect_tiles_3d", &r2splat::isect_tiles_3d_tensor,
          "Compute 3D tile-Gaussian intersections");
    m.def("isect_offset_encode_3d", &r2splat::isect_offset_encode_3d_tensor,
          "Encode 3D tile offsets");

    m.def("voxelize_to_voxels_fwd", &r2splat::voxelize_to_voxels_fwd_tensor,
          "Voxelize 3D Gaussians to a voxel grid");
    m.def("voxelize_to_voxels_mean_var_fwd", &r2splat::voxelize_to_voxels_mean_var_fwd_tensor,
          "Voxelize 3D Gaussians to voxel mean and propagated variance grids");
    m.def("voxelize_to_voxels_bwd", &r2splat::voxelize_to_voxels_bwd_tensor,
          "Voxelization backward pass");
    m.def("voxelize_to_voxels_bwd_per_gaussian", &r2splat::voxelize_to_voxels_bwd_per_gaussian_tensor,
          "Voxelization per-Gaussian-tile backward pass");
    m.def("voxelize_to_voxels_mean_var_bwd_per_gaussian",
          [](const torch::Tensor& means3d,
             const torch::Tensor& conics,
             const torch::Tensor& opacities,
             const torch::Tensor& tile_offsets,
             const torch::Tensor& isect_ids,
             const torch::Tensor& flatten_ids,
             const torch::Tensor& v_render_output,
             const torch::Tensor& v_render_var,
             const uint32_t dim2,
             const uint32_t dim1,
             const uint32_t dim0,
             const uint32_t tile_size) {
              auto empty_uncertainties = torch::empty({0}, opacities.options());
              return r2splat::voxelize_to_voxels_mean_var_bwd_per_gaussian_tensor(
                  means3d, conics, opacities, tile_offsets, isect_ids, flatten_ids,
                  v_render_output, v_render_var, empty_uncertainties, true,
                  dim2, dim1, dim0, tile_size);
          },
          "Voxelization per-Gaussian-tile backward pass with detached variance contribution");
    m.def("voxelize_to_voxels_mean_var_bwd_per_gaussian", &r2splat::voxelize_to_voxels_mean_var_bwd_per_gaussian_tensor,
          "Voxelization per-Gaussian-tile backward pass with fused uncertainty gradient");
    m.def("beta_voxelize_to_voxels_fwd", &r2splat::beta_voxelize_to_voxels_fwd_tensor,
          "Voxelize 3D Beta primitives to a voxel grid");
    m.def("beta_voxelize_to_voxels_mean_var_fwd", &r2splat::beta_voxelize_to_voxels_mean_var_fwd_tensor,
          "Voxelize 3D Beta primitives to voxel mean and propagated variance grids");
    m.def("beta_voxelize_to_voxels_bwd", &r2splat::beta_voxelize_to_voxels_bwd_tensor,
          "Beta voxelization backward pass");
    m.def("beta_voxelize_to_voxels_bwd_per_gaussian", &r2splat::beta_voxelize_to_voxels_bwd_per_gaussian_tensor,
          "Beta voxelization per-Gaussian-tile backward pass");
    m.def("beta_voxelize_to_voxels_mean_var_bwd_per_gaussian",
          [](const torch::Tensor& means3d,
             const torch::Tensor& conics,
             const torch::Tensor& opacities,
             const torch::Tensor& betas,
             const torch::Tensor& tile_offsets,
             const torch::Tensor& isect_ids,
             const torch::Tensor& flatten_ids,
             const torch::Tensor& v_render_output,
             const torch::Tensor& v_render_var,
             const uint32_t dim2,
             const uint32_t dim1,
             const uint32_t dim0,
             const uint32_t tile_size) {
              auto empty_uncertainties = torch::empty({0}, opacities.options());
              return r2splat::beta_voxelize_to_voxels_mean_var_bwd_per_gaussian_tensor(
                  means3d, conics, opacities, betas, tile_offsets, isect_ids,
                  flatten_ids, v_render_output, v_render_var, empty_uncertainties,
                  true, dim2, dim1, dim0, tile_size);
          },
          "Beta voxelization per-Gaussian-tile backward pass with detached variance contribution");
    m.def("beta_voxelize_to_voxels_mean_var_bwd_per_gaussian", &r2splat::beta_voxelize_to_voxels_mean_var_bwd_per_gaussian_tensor,
          "Beta voxelization per-Gaussian-tile backward pass with fused uncertainty gradient");

    m.def("rasterize_to_pixels_fwd", &r2splat::rasterize_to_pixels_fwd_tensor,
          "Projection rasterization forward pass");
    m.def("rasterize_to_pixels_mean_var_fwd", &r2splat::rasterize_to_pixels_mean_var_fwd_tensor,
          "Projection rasterization forward pass with propagated variance");
    m.def("rasterize_to_pixels_uncertainty_bwd", &r2splat::rasterize_to_pixels_uncertainty_bwd_tensor,
          "Projection variance backward pass w.r.t. activated primitive uncertainty");
    m.def("rasterize_to_pixels_bwd", &r2splat::rasterize_to_pixels_bwd_tensor,
          "Projection rasterization backward pass");
    m.def("rasterize_to_pixels_bwd_per_gaussian", &r2splat::rasterize_to_pixels_bwd_per_gaussian_tensor,
          "Projection rasterization per-Gaussian-tile backward pass");
    m.def("rasterize_to_pixels_mean_var_bwd_per_gaussian",
          [](const torch::Tensor& means2d,
             const torch::Tensor& conics,
             const torch::Tensor& opacities,
             const torch::Tensor& mus,
             const torch::Tensor& tile_offsets,
             const torch::Tensor& isect_ids,
             const torch::Tensor& flatten_ids,
             const torch::Tensor& v_render_output,
             const torch::Tensor& v_render_var,
             const uint32_t dim1,
             const uint32_t dim0,
             const uint32_t tile_size,
             const bool absgrad) {
              auto empty_uncertainties = torch::empty({0}, opacities.options());
              return r2splat::rasterize_to_pixels_mean_var_bwd_per_gaussian_tensor(
                  means2d, conics, opacities, mus, tile_offsets, isect_ids,
                  flatten_ids, v_render_output, v_render_var, empty_uncertainties,
                  true, dim1, dim0, tile_size, absgrad);
          },
          "Projection rasterization per-Gaussian-tile backward pass with detached variance contribution");
    m.def("rasterize_to_pixels_mean_var_bwd_per_gaussian", &r2splat::rasterize_to_pixels_mean_var_bwd_per_gaussian_tensor,
          "Projection rasterization per-Gaussian-tile backward pass with fused uncertainty gradient");
    m.def("beta_rasterize_to_pixels_fwd", &r2splat::beta_rasterize_to_pixels_fwd_tensor,
          "Beta projection rasterization forward pass");
    m.def("beta_rasterize_to_pixels_mean_var_fwd", &r2splat::beta_rasterize_to_pixels_mean_var_fwd_tensor,
          "Beta projection rasterization forward pass with propagated variance");
    m.def("beta_rasterize_to_pixels_uncertainty_bwd", &r2splat::beta_rasterize_to_pixels_uncertainty_bwd_tensor,
          "Beta projection variance backward pass w.r.t. activated primitive uncertainty");
    m.def("beta_rasterize_to_pixels_bwd", &r2splat::beta_rasterize_to_pixels_bwd_tensor,
          "Beta projection rasterization backward pass");
    m.def("beta_rasterize_to_pixels_bwd_per_gaussian", &r2splat::beta_rasterize_to_pixels_bwd_per_gaussian_tensor,
          "Beta projection rasterization per-Gaussian-tile backward pass");
    m.def("beta_rasterize_to_pixels_mean_var_bwd_per_gaussian",
          [](const torch::Tensor& means2d,
             const torch::Tensor& conics,
             const torch::Tensor& opacities,
             const torch::Tensor& mus,
             const torch::Tensor& betas,
             const torch::Tensor& tile_offsets,
             const torch::Tensor& isect_ids,
             const torch::Tensor& flatten_ids,
             const torch::Tensor& v_render_output,
             const torch::Tensor& v_render_var,
             const uint32_t dim1,
             const uint32_t dim0,
             const uint32_t tile_size,
             const bool absgrad) {
              auto empty_uncertainties = torch::empty({0}, opacities.options());
              return r2splat::beta_rasterize_to_pixels_mean_var_bwd_per_gaussian_tensor(
                  means2d, conics, opacities, mus, betas, tile_offsets, isect_ids,
                  flatten_ids, v_render_output, v_render_var, empty_uncertainties,
                  true, dim1, dim0, tile_size, absgrad);
          },
          "Beta projection rasterization per-Gaussian-tile backward pass with detached variance contribution");
    m.def("beta_rasterize_to_pixels_mean_var_bwd_per_gaussian", &r2splat::beta_rasterize_to_pixels_mean_var_bwd_per_gaussian_tensor,
          "Beta projection rasterization per-Gaussian-tile backward pass with fused uncertainty gradient");

    m.def("conics_mu_fwd", &r2splat::conics_mu_fwd_tensor,
          "Conics/mu forward pass");
    m.def("conics_mu_bwd", &r2splat::conics_mu_bwd_tensor,
          "Conics/mu backward pass");

    m.def("world_to_image_projection_fwd", &r2splat::world_to_image_projection_fwd_tensor,
          "World-to-image projection forward pass");
    m.def("world_to_image_projection_bwd", &r2splat::world_to_image_projection_bwd_tensor,
          "World-to-image projection backward pass");

    m.def("parameter_transform_fwd", &r2splat::parameter_transform_fwd_tensor,
          "Normalize wxyz quaternions and compute covariance from physical scales");
    m.def("parameter_transform_bwd", &r2splat::parameter_transform_bwd_tensor,
          "Covariance VJP w.r.t. raw nonunit quaternions and physical scales");

    m.def("world_to_volume_mapping_fwd", &r2splat::world_to_volume_mapping_fwd_tensor,
          "World-to-volume mapping forward pass");
    m.def("world_to_volume_mapping_bwd", &r2splat::world_to_volume_mapping_bwd_tensor,
          "World-to-volume mapping backward pass");
}
