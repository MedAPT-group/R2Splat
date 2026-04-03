/**
 * PyBind11 binding entry point for CUDA extensions
 * 
 * Following gsplat style: minimal registration file that only
 * binds functions. All Tensor processing logic is in .cpp files.
 * 
 * Author: Dy164
 * Date: 2025-12-03
 */

#include <torch/extension.h>
#include "../include/Ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA extensions for MLP classification and loss functions with second-order derivatives";
    
    // MLP Classification
    m.def("mlp_classification_fwd", &gsplat2ct::mlp_classification_fwd,
          "MLP classification forward pass");
    
    m.def("mlp_classification_bwd", &gsplat2ct::mlp_classification_bwd,
          "MLP classification backward pass (gradients only)");
    
    m.def("mlp_classification_bwd2", &gsplat2ct::mlp_classification_bwd2,
          "MLP classification backward pass (gradients + Hessians)");
    
    // Image Loss
    m.def("image_loss_fwd", &gsplat2ct::image_loss_fwd,
          "Image loss (L2 + D-SSIM) forward pass (backward-compatible alias)");
    
    m.def("image_loss_bwd", &gsplat2ct::image_loss_bwd,
          "Image loss (L2 + D-SSIM) backward pass (backward-compatible alias)");
    
    m.def("image_loss_bwd2", &gsplat2ct::image_loss_bwd2,
          "Image loss (L2 + D-SSIM) second-order backward pass (backward-compatible alias)");

    // Explicit Image Loss variants
    m.def("image_loss_l2_fwd", &gsplat2ct::image_loss_l2_fwd,
          "Image loss (L2 + D-SSIM) forward pass");
    m.def("image_loss_l2_bwd", &gsplat2ct::image_loss_l2_bwd,
          "Image loss (L2 + D-SSIM) backward pass");
    m.def("image_loss_l2_bwd2", &gsplat2ct::image_loss_l2_bwd2,
          "Image loss (L2 + D-SSIM) second-order backward pass (Hessian diagonal)");

    m.def("image_loss_l1_fwd", &gsplat2ct::image_loss_l1_fwd,
          "Image loss (L1 + D-SSIM) forward pass");
    m.def("image_loss_l1_bwd", &gsplat2ct::image_loss_l1_bwd,
          "Image loss (L1 + D-SSIM) backward pass");
    m.def("image_loss_l1_bwd2", &gsplat2ct::image_loss_l1_bwd2,
          "Image loss (L1 + D-SSIM) second-order backward pass (Hessian diagonal)");
    
    // L2 Loss
    m.def("l2_loss_fwd", &gsplat2ct::l2_loss_fwd,
          "L2 (MSE) loss forward pass");
    
    m.def("l2_loss_bwd", &gsplat2ct::l2_loss_bwd,
          "L2 (MSE) loss backward pass");
    
    m.def("l2_loss_bwd2", &gsplat2ct::l2_loss_bwd2,
          "L2 (MSE) loss second-order backward pass (Hessian diagonal)");
    
    // Volume Loss
    m.def("volume_loss_fwd", &gsplat2ct::volume_loss_fwd,
          "Volume loss (Smooth TV) forward pass");
    
    m.def("volume_loss_bwd", &gsplat2ct::volume_loss_bwd,
          "Volume loss (Smooth TV) backward pass");
    
    m.def("volume_loss_bwd2", &gsplat2ct::volume_loss_bwd2,
          "Volume loss (Smooth TV) second-order backward pass (Hessian diagonal)");
    
    // Isect Tiles
    m.def("isect_tiles", &gsplat2ct::isect_tiles_tensor,
          "Compute tile-Gaussian intersections");
    
    m.def("isect_offset_encode", &gsplat2ct::isect_offset_encode_tensor,
          "Encode tile offsets from sorted isect_ids");
    
    // Isect Tiles 3D
    m.def("isect_tiles_3d", &gsplat2ct::isect_tiles_3d_tensor,
          "Compute tile-Gaussian intersections in 3D");
    
    m.def("isect_offset_encode_3d", &gsplat2ct::isect_offset_encode_3d_tensor,
          "Encode tile offsets from sorted isect_ids for 3D");
    
    // VoxelizeToVoxels
    m.def("voxelize_to_voxels_fwd", &gsplat2ct::voxelize_to_voxels_fwd_tensor,
          "Voxelize 3D Gaussians to voxel grid for CT reconstruction");
    
    m.def("voxelize_to_voxels_bwd", &gsplat2ct::voxelize_to_voxels_bwd_tensor,
          "Voxelize backward pass (first-order gradients)");
    
    m.def("voxelize_to_voxels_bwd2", &gsplat2ct::voxelize_to_voxels_bwd2_tensor,
          "Voxelize backward pass (gradients + Hessian)");
    
    // Rasterize to Pixels
    m.def("rasterize_to_pixels_fwd", &gsplat2ct::rasterize_to_pixels_fwd_tensor,
          "CT rasterization forward pass");
    
    m.def("rasterize_to_pixels_bwd", &gsplat2ct::rasterize_to_pixels_bwd_tensor,
          "CT rasterization backward pass (gradients only)");
    
    m.def("rasterize_to_pixels_bwd2", &gsplat2ct::rasterize_to_pixels_bwd2_tensor,
          "CT rasterization backward pass (gradients + Hessian)");
    
    // ConicsMu
    m.def("conics_mu_fwd", &gsplat2ct::conics_mu_fwd_tensor,
          "ConicsMu forward pass (conics and mu from projected 3x3 covariance)");
    
    m.def("conics_mu_bwd", &gsplat2ct::conics_mu_bwd_tensor,
          "ConicsMu backward pass (gradients w.r.t. cov2d_3x3)");
    
    m.def("conics_mu_bwd2", &gsplat2ct::conics_mu_bwd2_tensor,
          "ConicsMu second-order backward pass (gradients + Hessians)");
    
    // WorldToImageProjection
    m.def("world_to_image_projection_fwd", &gsplat2ct::world_to_image_projection_fwd_tensor,
          "WorldToImageProjection forward pass (world-space to 2D projection)");
    
    m.def("world_to_image_projection_bwd", &gsplat2ct::world_to_image_projection_bwd_tensor,
          "WorldToImageProjection backward pass (gradients w.r.t. means3d, covars)");
    
    m.def("world_to_image_projection_bwd2", &gsplat2ct::world_to_image_projection_bwd2_tensor,
          "WorldToImageProjection second-order backward pass (gradients + Hessians)");
    
    // ParameterTransform
    m.def("parameter_transform_fwd", &gsplat2ct::parameter_transform_fwd_tensor,
          "ParameterTransform forward pass (quats, scale_ori -> covar)");
    
    m.def("parameter_transform_bwd", &gsplat2ct::parameter_transform_bwd_tensor,
          "ParameterTransform backward pass (gradients w.r.t. quats, scale_ori)");
    
    m.def("parameter_transform_bwd2", &gsplat2ct::parameter_transform_bwd2_tensor,
          "ParameterTransform second-order backward pass (gradients + Hessians)");
    
    // WorldToVolumeMapping
    m.def("world_to_volume_mapping_fwd", &gsplat2ct::world_to_volume_mapping_fwd_tensor,
          "WorldToVolumeMapping forward pass (world-space to volume-space)");
    
    m.def("world_to_volume_mapping_bwd", &gsplat2ct::world_to_volume_mapping_bwd_tensor,
          "WorldToVolumeMapping backward pass (gradients w.r.t. means, covars)");
    
    m.def("world_to_volume_mapping_bwd2", &gsplat2ct::world_to_volume_mapping_bwd2_tensor,
          "WorldToVolumeMapping second-order backward pass (gradients + Hessians)");
}