"""PyTorch autograd wrappers for the CUDA CT splatting kernels.

The low-level CUDA extension exposes explicit forward/backward functions.
This module turns those pairs into standard ``torch.autograd.Function`` nodes
and keeps ordinary tensor math, losses, and optimizers in PyTorch land.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from ..model.beta import activated_beta_terms


_LAST_RASTER_ABSGRAD: Optional[torch.Tensor] = None


def pop_last_raster_absgrad() -> Optional[torch.Tensor]:
    """Return and clear the most recent rasterizer absgrad auxiliary tensor."""
    global _LAST_RASTER_ABSGRAD
    value = _LAST_RASTER_ABSGRAD
    _LAST_RASTER_ABSGRAD = None
    return value


def _covars_from_quats_scales(quats: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Return upper-triangular world covariances from unit quaternions and scales."""
    q = torch.nn.functional.normalize(quats, dim=-1)
    w, x, y, z = q.unbind(dim=-1)

    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - w * z)
    r02 = 2.0 * (x * z + w * y)
    r10 = 2.0 * (x * y + w * z)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - w * x)
    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)

    sx2, sy2, sz2 = (scales * scales).unbind(dim=-1)
    c00 = r00 * r00 * sx2 + r01 * r01 * sy2 + r02 * r02 * sz2
    c01 = r00 * r10 * sx2 + r01 * r11 * sy2 + r02 * r12 * sz2
    c02 = r00 * r20 * sx2 + r01 * r21 * sy2 + r02 * r22 * sz2
    c11 = r10 * r10 * sx2 + r11 * r11 * sy2 + r12 * r12 * sz2
    c12 = r10 * r20 * sx2 + r11 * r21 * sy2 + r12 * r22 * sz2
    c22 = r20 * r20 * sx2 + r21 * r21 * sy2 + r22 * r22 * sz2
    return torch.stack((c00, c01, c02, c11, c12, c22), dim=-1).contiguous()


def activate_gaussian_params(
    raw_quats: torch.Tensor,
    raw_scales: torch.Tensor,
    raw_opacity: torch.Tensor,
    *,
    scale_min: float,
    scale_max: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Activate raw parameters using differentiable PyTorch operations."""
    quats = torch.nn.functional.normalize(raw_quats, dim=-1)
    scales = torch.sigmoid(raw_scales) * (float(scale_max) - float(scale_min)) + float(scale_min)
    opacity = torch.nn.functional.softplus(raw_opacity)
    covars = _covars_from_quats_scales(quats, scales)
    max_scales = scales.max(dim=-1).values
    return quats, scales, opacity, covars, max_scales


class _ProjectionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means, covars, viewmats, Ks, img_w, img_h, eps_blur, radius_clip):
        from ..cuda._wrapper import conics_mu_fwd, world_to_image_projection_fwd

        means2d, cov2d_3x3, radii = world_to_image_projection_fwd(
            means, covars, viewmats, Ks, img_w, img_h, eps_blur, radius_clip
        )
        conics, mus = conics_mu_fwd(cov2d_3x3, eps_blur)
        ctx.save_for_backward(means, covars, viewmats, Ks, cov2d_3x3, conics, mus)
        ctx.img_w = int(img_w)
        ctx.img_h = int(img_h)
        ctx.eps_blur = float(eps_blur)
        return means2d, conics, mus, radii

    @staticmethod
    def backward(ctx, v_means2d, v_conics, v_mus, _v_radii):
        from ..cuda._wrapper import conics_mu_bwd, world_to_image_projection_bwd

        means, covars, viewmats, Ks, cov2d_3x3, conics, mus = ctx.saved_tensors
        if v_means2d is None:
            v_means2d = torch.zeros((*conics.shape[:2], 2), device=means.device, dtype=means.dtype)
        if v_conics is None:
            v_conics = torch.zeros_like(conics)
        if v_mus is None:
            v_mus = torch.zeros_like(mus)

        v_cov2d = conics_mu_bwd(
            cov2d_3x3,
            ctx.eps_blur,
            conics,
            mus,
            v_conics.contiguous(),
            v_mus.contiguous(),
        )
        v_means, v_covars = world_to_image_projection_bwd(
            means,
            covars,
            viewmats,
            Ks,
            ctx.img_w,
            ctx.img_h,
            v_means2d.contiguous(),
            v_cov2d.contiguous(),
        )
        return v_means, v_covars, None, None, None, None, None, None


class _MappingFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means,
        covars,
        max_scales,
        ranges,
        centers,
        vol_dim2,
        vol_dim1,
        vol_dim0,
        eps3d,
        radius_clip,
    ):
        from ..cuda._wrapper import world_to_volume_mapping_fwd

        means3d, conics, radii = world_to_volume_mapping_fwd(
            means,
            covars,
            max_scales,
            ranges,
            centers,
            vol_dim2,
            vol_dim1,
            vol_dim0,
            eps3d,
            radius_clip,
        )
        ctx.save_for_backward(means, covars, ranges, centers, conics)
        ctx.vol_dim2 = int(vol_dim2)
        ctx.vol_dim1 = int(vol_dim1)
        ctx.vol_dim0 = int(vol_dim0)
        ctx.eps3d = float(eps3d)
        return means3d, conics, radii

    @staticmethod
    def backward(ctx, v_means3d, v_conics, _v_radii):
        from ..cuda._wrapper import world_to_volume_mapping_bwd

        means, covars, ranges, centers, conics = ctx.saved_tensors
        if v_means3d is None:
            v_means3d = torch.zeros((*conics.shape[:1], 3), device=means.device, dtype=means.dtype)
        if v_conics is None:
            v_conics = torch.zeros_like(conics)
        v_means, v_covars = world_to_volume_mapping_bwd(
            means,
            covars,
            ranges,
            centers,
            ctx.vol_dim2,
            ctx.vol_dim1,
            ctx.vol_dim0,
            ctx.eps3d,
            v_means3d.contiguous(),
            v_conics.contiguous(),
        )
        return v_means, v_covars, None, None, None, None, None, None, None, None


class _RasterizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means2d,
        conics,
        opacity,
        mus,
        beta2d,
        tile_offsets,
        flatten_ids,
        isect_ids,
        img_w,
        img_h,
        tile_size,
        primitive_type,
        absgrad,
        per_primitive,
    ):
        from ..cuda._wrapper import beta_rasterize_to_pixels_fwd, rasterize_to_pixels_fwd

        primitive_type = str(primitive_type).lower()
        if primitive_type == "beta":
            output = beta_rasterize_to_pixels_fwd(
                means2d, conics, opacity, mus, beta2d, tile_offsets, flatten_ids,
                img_w, img_h, tile_size
            )
        else:
            output = rasterize_to_pixels_fwd(
                means2d, conics, opacity, mus, tile_offsets, flatten_ids,
                img_w, img_h, tile_size
            )
            beta2d = torch.empty(0, device=means2d.device, dtype=means2d.dtype)
        ctx.save_for_backward(
            means2d, conics, opacity, mus, beta2d, tile_offsets, flatten_ids, isect_ids
        )
        ctx.img_w = int(img_w)
        ctx.img_h = int(img_h)
        ctx.tile_size = int(tile_size)
        ctx.primitive_type = primitive_type
        ctx.absgrad = bool(absgrad)
        ctx.per_primitive = bool(per_primitive)
        return output

    @staticmethod
    def backward(ctx, v_output):
        global _LAST_RASTER_ABSGRAD
        from ..cuda._wrapper import (
            beta_rasterize_to_pixels_bwd,
            beta_rasterize_to_pixels_bwd_per_gaussian,
            rasterize_to_pixels_bwd,
            rasterize_to_pixels_bwd_per_gaussian,
        )

        means2d, conics, opacity, mus, beta2d, tile_offsets, flatten_ids, isect_ids = ctx.saved_tensors
        if ctx.primitive_type == "beta":
            if ctx.per_primitive:
                grads = beta_rasterize_to_pixels_bwd_per_gaussian(
                    means2d, conics, opacity, mus, beta2d, tile_offsets, isect_ids,
                    flatten_ids, v_output.contiguous(), ctx.img_w, ctx.img_h,
                    ctx.tile_size, ctx.absgrad
                )
            else:
                grads = beta_rasterize_to_pixels_bwd(
                    means2d, conics, opacity, mus, beta2d, tile_offsets, flatten_ids,
                    v_output.contiguous(), ctx.img_w, ctx.img_h, ctx.tile_size,
                    ctx.absgrad
                )
            v_means_abs, v_means, v_conics, v_opacity, v_mus, v_beta = grads
        else:
            if ctx.per_primitive:
                grads = rasterize_to_pixels_bwd_per_gaussian(
                    means2d, conics, opacity, mus, tile_offsets, isect_ids,
                    flatten_ids, v_output.contiguous(), ctx.img_w, ctx.img_h,
                    ctx.tile_size, ctx.absgrad
                )
            else:
                grads = rasterize_to_pixels_bwd(
                    means2d, conics, opacity, mus, tile_offsets, flatten_ids,
                    v_output.contiguous(), ctx.img_w, ctx.img_h, ctx.tile_size,
                    ctx.absgrad
                )
            v_means_abs, v_means, v_conics, v_opacity, v_mus = grads
            v_beta = None
        _LAST_RASTER_ABSGRAD = v_means_abs.detach() if v_means_abs is not None else None
        return (
            v_means,
            v_conics,
            v_opacity,
            v_mus,
            v_beta,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _VoxelizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3d,
        conics,
        opacity,
        gamma,
        tile_offsets,
        flatten_ids,
        isect_ids,
        vol_dim2,
        vol_dim1,
        vol_dim0,
        tile_size,
        primitive_type,
        per_primitive,
    ):
        from ..cuda._wrapper import beta_voxelize_to_voxels_fwd, voxelize_to_voxels_fwd

        primitive_type = str(primitive_type).lower()
        if primitive_type == "beta":
            output = beta_voxelize_to_voxels_fwd(
                means3d, conics, opacity, gamma, tile_offsets, flatten_ids,
                vol_dim2, vol_dim1, vol_dim0, tile_size
            )
        else:
            output = voxelize_to_voxels_fwd(
                means3d, conics, opacity, tile_offsets, flatten_ids,
                vol_dim2, vol_dim1, vol_dim0, tile_size
            )
            gamma = torch.empty(0, device=means3d.device, dtype=means3d.dtype)
        ctx.save_for_backward(means3d, conics, opacity, gamma, tile_offsets, flatten_ids, isect_ids)
        ctx.vol_dim2 = int(vol_dim2)
        ctx.vol_dim1 = int(vol_dim1)
        ctx.vol_dim0 = int(vol_dim0)
        ctx.tile_size = int(tile_size)
        ctx.primitive_type = primitive_type
        ctx.per_primitive = bool(per_primitive)
        return output

    @staticmethod
    def backward(ctx, v_output):
        from ..cuda._wrapper import (
            beta_voxelize_to_voxels_bwd,
            beta_voxelize_to_voxels_bwd_per_gaussian,
            voxelize_to_voxels_bwd,
            voxelize_to_voxels_bwd_per_gaussian,
        )

        means3d, conics, opacity, gamma, tile_offsets, flatten_ids, isect_ids = ctx.saved_tensors
        if ctx.primitive_type == "beta":
            if ctx.per_primitive:
                v_means, v_conics, v_opacity, v_gamma = beta_voxelize_to_voxels_bwd_per_gaussian(
                    means3d, conics, opacity, gamma, tile_offsets, isect_ids,
                    flatten_ids, v_output.contiguous(), ctx.vol_dim2, ctx.vol_dim1,
                    ctx.vol_dim0, ctx.tile_size
                )
            else:
                v_means, v_conics, v_opacity, v_gamma = beta_voxelize_to_voxels_bwd(
                    means3d, conics, opacity, gamma, tile_offsets, flatten_ids,
                    v_output.contiguous(), ctx.vol_dim2, ctx.vol_dim1, ctx.vol_dim0,
                    ctx.tile_size
                )
        else:
            if ctx.per_primitive:
                v_means, v_conics, v_opacity = voxelize_to_voxels_bwd_per_gaussian(
                    means3d, conics, opacity, tile_offsets, isect_ids, flatten_ids,
                    v_output.contiguous(), ctx.vol_dim2, ctx.vol_dim1, ctx.vol_dim0,
                    ctx.tile_size
                )
            else:
                v_means, v_conics, v_opacity = voxelize_to_voxels_bwd(
                    means3d, conics, opacity, tile_offsets, flatten_ids,
                    v_output.contiguous(), ctx.vol_dim2, ctx.vol_dim1, ctx.vol_dim0,
                    ctx.tile_size
                )
            v_gamma = None
        return (
            v_means,
            v_conics,
            v_opacity,
            v_gamma,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@dataclass
class ProjectionRasterizationResult:
    image: torch.Tensor
    means2d: torch.Tensor
    radii: torch.Tensor


@dataclass
class MappingVoxelizationResult:
    volume: torch.Tensor
    means3d: torch.Tensor
    radii: torch.Tensor


class GaussianProjectionRasterizer(nn.Module):
    def __init__(
        self,
        *,
        img_w: int,
        img_h: int,
        tile_size: int = 16,
        eps_blur: float = 0.3,
        radius_clip: float = 3.0,
        primitive_type: str = "gaussian",
        absgrad: bool = False,
        backward_mode: str = "pixel",
    ):
        super().__init__()
        self.img_w = int(img_w)
        self.img_h = int(img_h)
        self.tile_size = int(tile_size)
        self.eps_blur = float(eps_blur)
        self.radius_clip = float(radius_clip)
        self.primitive_type = str(primitive_type).lower()
        self.absgrad = bool(absgrad)
        self.backward_mode = str(backward_mode).lower()

    def forward(
        self,
        means: torch.Tensor,
        covars: torch.Tensor,
        opacity: torch.Tensor,
        viewmats: torch.Tensor,
        Ks: torch.Tensor,
        *,
        beta: Optional[torch.Tensor] = None,
        beta_gamma_init: float = 3.5,
    ) -> ProjectionRasterizationResult:
        from ..cuda._wrapper import isect_offset_encode, isect_tiles

        means2d, conics, mus, radii = _ProjectionFunction.apply(
            means,
            covars,
            viewmats,
            Ks,
            self.img_w,
            self.img_h,
            self.eps_blur,
            self.radius_clip,
        )
        C = viewmats.shape[0]
        opacity2d = opacity.unsqueeze(0).expand(C, -1).contiguous()
        beta2d = None
        if self.primitive_type == "beta":
            if beta is None:
                raise ValueError("primitive_type='beta' requires beta")
            gamma, projected_beta, path_ratio, _ = activated_beta_terms(
                beta, gamma_init=beta_gamma_init
            )
            mus = mus * path_ratio.unsqueeze(0)
            beta2d = projected_beta.unsqueeze(0).expand(C, -1).contiguous()

        n_tiles_h = (self.img_h + self.tile_size - 1) // self.tile_size
        n_tiles_w = (self.img_w + self.tile_size - 1) // self.tile_size
        _, isect_ids, flatten_ids = isect_tiles(
            means2d, radii, self.tile_size, n_tiles_w, n_tiles_h
        )
        tile_offsets = isect_offset_encode(isect_ids, C, n_tiles_w, n_tiles_h)
        image = _RasterizeFunction.apply(
            means2d,
            conics,
            opacity2d,
            mus,
            beta2d,
            tile_offsets,
            flatten_ids,
            isect_ids,
            self.img_w,
            self.img_h,
            self.tile_size,
            self.primitive_type,
            self.absgrad,
            self.backward_mode == "primitive",
        )
        return ProjectionRasterizationResult(image=image, means2d=means2d, radii=radii)


class GaussianMappingVoxelizer(nn.Module):
    def __init__(
        self,
        *,
        vol_dim2: int,
        vol_dim1: int,
        vol_dim0: int,
        tile_size: int = 4,
        eps3d: float = 1.0e-4,
        radius_clip: float = 3.0,
        primitive_type: str = "gaussian",
        backward_mode: str = "pixel",
    ):
        super().__init__()
        self.vol_dim2 = int(vol_dim2)
        self.vol_dim1 = int(vol_dim1)
        self.vol_dim0 = int(vol_dim0)
        self.tile_size = int(tile_size)
        self.eps3d = float(eps3d)
        self.radius_clip = float(radius_clip)
        self.primitive_type = str(primitive_type).lower()
        self.backward_mode = str(backward_mode).lower()

    def forward(
        self,
        means: torch.Tensor,
        covars: torch.Tensor,
        max_scales: torch.Tensor,
        opacity: torch.Tensor,
        ranges: torch.Tensor,
        centers: torch.Tensor,
        *,
        beta: Optional[torch.Tensor] = None,
        beta_gamma_init: float = 3.5,
    ) -> MappingVoxelizationResult:
        from ..cuda._wrapper import isect_offset_encode_3d, isect_tiles_3d

        means3d, conics, radii = _MappingFunction.apply(
            means,
            covars,
            max_scales,
            ranges,
            centers,
            self.vol_dim2,
            self.vol_dim1,
            self.vol_dim0,
            self.eps3d,
            self.radius_clip,
        )
        gamma = None
        if self.primitive_type == "beta":
            if beta is None:
                raise ValueError("primitive_type='beta' requires beta")
            gamma = activated_beta_terms(beta, gamma_init=beta_gamma_init)[0]

        n_tiles_z = (self.vol_dim2 + self.tile_size - 1) // self.tile_size
        n_tiles_y = (self.vol_dim1 + self.tile_size - 1) // self.tile_size
        n_tiles_x = (self.vol_dim0 + self.tile_size - 1) // self.tile_size
        _, isect_ids, flatten_ids = isect_tiles_3d(
            means3d, radii, self.tile_size, n_tiles_z, n_tiles_y, n_tiles_x
        )
        tile_offsets = isect_offset_encode_3d(isect_ids, n_tiles_z, n_tiles_y, n_tiles_x)
        volume = _VoxelizeFunction.apply(
            means3d,
            conics,
            opacity,
            gamma,
            tile_offsets,
            flatten_ids,
            isect_ids,
            self.vol_dim2,
            self.vol_dim1,
            self.vol_dim0,
            self.tile_size,
            self.primitive_type,
            self.backward_mode == "primitive",
        )
        return MappingVoxelizationResult(volume=volume, means3d=means3d, radii=radii)


__all__ = [
    "GaussianMappingVoxelizer",
    "GaussianProjectionRasterizer",
    "MappingVoxelizationResult",
    "ProjectionRasterizationResult",
    "activate_gaussian_params",
    "pop_last_raster_absgrad",
]
