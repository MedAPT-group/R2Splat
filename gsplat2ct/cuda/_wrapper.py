"""Python wrappers for the installed CUDA extension."""

from __future__ import annotations

from typing import Callable


def _make_cuda_func(name: str) -> Callable:
    def call_cuda(*args, **kwargs):
        from ._backend import _C

        return getattr(_C, name)(*args, **kwargs)

    call_cuda.__name__ = name
    call_cuda.__qualname__ = name
    return call_cuda


image_loss_fwd = _make_cuda_func("image_loss_fwd")
image_loss_bwd = _make_cuda_func("image_loss_bwd")
image_loss_l1_fwd = _make_cuda_func("image_loss_l1_fwd")
image_loss_l1_bwd = _make_cuda_func("image_loss_l1_bwd")

volume_loss_fwd = _make_cuda_func("volume_loss_fwd")
volume_loss_bwd = _make_cuda_func("volume_loss_bwd")

isect_tiles = _make_cuda_func("isect_tiles")
isect_offset_encode = _make_cuda_func("isect_offset_encode")
isect_tiles_3d = _make_cuda_func("isect_tiles_3d")
isect_offset_encode_3d = _make_cuda_func("isect_offset_encode_3d")

voxelize_to_voxels_fwd = _make_cuda_func("voxelize_to_voxels_fwd")
voxelize_to_voxels_bwd = _make_cuda_func("voxelize_to_voxels_bwd")
voxelize_to_voxels_bwd_per_gaussian = _make_cuda_func(
    "voxelize_to_voxels_bwd_per_gaussian"
)
beta_voxelize_to_voxels_fwd = _make_cuda_func("beta_voxelize_to_voxels_fwd")
beta_voxelize_to_voxels_bwd = _make_cuda_func("beta_voxelize_to_voxels_bwd")
beta_voxelize_to_voxels_bwd_per_gaussian = _make_cuda_func(
    "beta_voxelize_to_voxels_bwd_per_gaussian"
)
rasterize_to_pixels_fwd = _make_cuda_func("rasterize_to_pixels_fwd")
rasterize_to_pixels_bwd = _make_cuda_func("rasterize_to_pixels_bwd")
rasterize_to_pixels_bwd_per_gaussian = _make_cuda_func(
    "rasterize_to_pixels_bwd_per_gaussian"
)
beta_rasterize_to_pixels_fwd = _make_cuda_func("beta_rasterize_to_pixels_fwd")
beta_rasterize_to_pixels_bwd = _make_cuda_func("beta_rasterize_to_pixels_bwd")
beta_rasterize_to_pixels_bwd_per_gaussian = _make_cuda_func(
    "beta_rasterize_to_pixels_bwd_per_gaussian"
)

conics_mu_fwd = _make_cuda_func("conics_mu_fwd")
conics_mu_bwd = _make_cuda_func("conics_mu_bwd")
world_to_image_projection_fwd = _make_cuda_func("world_to_image_projection_fwd")
world_to_image_projection_bwd = _make_cuda_func("world_to_image_projection_bwd")
parameter_transform_fwd = _make_cuda_func("parameter_transform_fwd")
parameter_transform_bwd = _make_cuda_func("parameter_transform_bwd")
world_to_volume_mapping_fwd = _make_cuda_func("world_to_volume_mapping_fwd")
world_to_volume_mapping_bwd = _make_cuda_func("world_to_volume_mapping_bwd")


def is_cuda_extension_available() -> bool:
    try:
        from ._backend import is_cuda_available

        return is_cuda_available()
    except ImportError:
        return False


__all__ = [
    "image_loss_fwd",
    "image_loss_bwd",
    "image_loss_l1_fwd",
    "image_loss_l1_bwd",
    "volume_loss_fwd",
    "volume_loss_bwd",
    "isect_tiles",
    "isect_offset_encode",
    "isect_tiles_3d",
    "isect_offset_encode_3d",
    "voxelize_to_voxels_fwd",
    "voxelize_to_voxels_bwd",
    "voxelize_to_voxels_bwd_per_gaussian",
    "beta_voxelize_to_voxels_fwd",
    "beta_voxelize_to_voxels_bwd",
    "beta_voxelize_to_voxels_bwd_per_gaussian",
    "rasterize_to_pixels_fwd",
    "rasterize_to_pixels_bwd",
    "rasterize_to_pixels_bwd_per_gaussian",
    "beta_rasterize_to_pixels_fwd",
    "beta_rasterize_to_pixels_bwd",
    "beta_rasterize_to_pixels_bwd_per_gaussian",
    "conics_mu_fwd",
    "conics_mu_bwd",
    "world_to_image_projection_fwd",
    "world_to_image_projection_bwd",
    "parameter_transform_fwd",
    "parameter_transform_bwd",
    "world_to_volume_mapping_fwd",
    "world_to_volume_mapping_bwd",
    "is_cuda_extension_available",
]
