/**
 * PyBind11 binding entry point for CUDA extensions.
 */

#include <torch/extension.h>
#include "../include/Ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA extensions for first-order CT reconstruction";

    m.def("image_loss_fwd", &gsplat2ct::image_loss_fwd,
          "Image loss (L2 + D-SSIM) forward pass");
    m.def("image_loss_bwd", &gsplat2ct::image_loss_bwd,
          "Image loss (L2 + D-SSIM) backward pass");
    m.def("image_loss_l2_fwd", &gsplat2ct::image_loss_l2_fwd,
          "Image loss (L2 + D-SSIM) forward pass");
    m.def("image_loss_l2_bwd", &gsplat2ct::image_loss_l2_bwd,
          "Image loss (L2 + D-SSIM) backward pass");
    m.def("image_loss_l1_fwd", &gsplat2ct::image_loss_l1_fwd,
          "Image loss (L1 + D-SSIM) forward pass");
    m.def("image_loss_l1_bwd", &gsplat2ct::image_loss_l1_bwd,
          "Image loss (L1 + D-SSIM) backward pass");

    m.def("volume_loss_fwd", &gsplat2ct::volume_loss_fwd,
          "Volume loss (Smooth TV) forward pass");
    m.def("volume_loss_bwd", &gsplat2ct::volume_loss_bwd,
          "Volume loss (Smooth TV) backward pass");

    m.def("isect_tiles", &gsplat2ct::isect_tiles_tensor,
          "Compute 2D tile-Gaussian intersections");
    m.def("isect_offset_encode", &gsplat2ct::isect_offset_encode_tensor,
          "Encode 2D tile offsets");
    m.def("isect_tiles_3d", &gsplat2ct::isect_tiles_3d_tensor,
          "Compute 3D tile-Gaussian intersections");
    m.def("isect_offset_encode_3d", &gsplat2ct::isect_offset_encode_3d_tensor,
          "Encode 3D tile offsets");

    m.def("voxelize_to_voxels_fwd", &gsplat2ct::voxelize_to_voxels_fwd_tensor,
          "Voxelize 3D Gaussians to a voxel grid");
    m.def("voxelize_to_voxels_bwd", &gsplat2ct::voxelize_to_voxels_bwd_tensor,
          "Voxelization backward pass");

    m.def("rasterize_to_pixels_fwd", &gsplat2ct::rasterize_to_pixels_fwd_tensor,
          "Projection rasterization forward pass");
    m.def("rasterize_to_pixels_bwd", &gsplat2ct::rasterize_to_pixels_bwd_tensor,
          "Projection rasterization backward pass");

    m.def("conics_mu_fwd", &gsplat2ct::conics_mu_fwd_tensor,
          "Conics/mu forward pass");
    m.def("conics_mu_bwd", &gsplat2ct::conics_mu_bwd_tensor,
          "Conics/mu backward pass");

    m.def("world_to_image_projection_fwd", &gsplat2ct::world_to_image_projection_fwd_tensor,
          "World-to-image projection forward pass");
    m.def("world_to_image_projection_bwd", &gsplat2ct::world_to_image_projection_bwd_tensor,
          "World-to-image projection backward pass");

    m.def("parameter_transform_fwd", &gsplat2ct::parameter_transform_fwd_tensor,
          "Raw parameter to covariance transform");
    m.def("parameter_transform_bwd", &gsplat2ct::parameter_transform_bwd_tensor,
          "Parameter transform backward pass");

    m.def("world_to_volume_mapping_fwd", &gsplat2ct::world_to_volume_mapping_fwd_tensor,
          "World-to-volume mapping forward pass");
    m.def("world_to_volume_mapping_bwd", &gsplat2ct::world_to_volume_mapping_bwd_tensor,
          "World-to-volume mapping backward pass");
}
