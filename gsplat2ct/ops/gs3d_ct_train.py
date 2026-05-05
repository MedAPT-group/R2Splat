"""End-to-end first-order CT loss operator."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from .base import ForwardBackwardBase
from .data import Config, Data, ForwardResult, GradResult, Params, get_lambda_l2


class GS3DCTLossCUDA(ForwardBackwardBase):
    """Full CT loss operator for Adam training.

    The pipeline combines a projection-fitting term and a TV regularizer:

    - raw parameters -> covariance transform
    - rasterize projections -> image loss
    - voxelize a sampled sub-volume -> smooth TV loss
    """

    def __init__(self, image_loss_type: Optional[str] = None):
        self.initialized = False
        self._default_image_loss_type = image_loss_type

    def _image_loss_fwd(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        window_size: int,
        lambda_l2: float,
        lambda_ssim: float,
        compute_statistics: bool,
        image_loss_type: str,
    ):
        if image_loss_type == "l1":
            from ..cuda._wrapper import image_loss_l1_fwd as fwd
        else:
            from ..cuda._wrapper import image_loss_l2_fwd as fwd
        return fwd(
            prediction,
            target,
            window_size,
            lambda_l2,
            lambda_ssim,
            compute_statistics,
        )

    def _image_loss_bwd(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mu_r: torch.Tensor,
        mu_m: torch.Tensor,
        f1: torch.Tensor,
        f2: torch.Tensor,
        f3: torch.Tensor,
        f4: torch.Tensor,
        grad_output: torch.Tensor,
        window_size: int,
        lambda_l2: float,
        lambda_ssim: float,
        image_loss_type: str,
    ):
        if image_loss_type == "l1":
            from ..cuda._wrapper import image_loss_l1_bwd as bwd
        else:
            from ..cuda._wrapper import image_loss_l2_bwd as bwd
        return bwd(
            prediction,
            target,
            mu_r,
            mu_m,
            f1,
            f2,
            f3,
            f4,
            grad_output,
            window_size,
            lambda_l2,
            lambda_ssim,
        )

    def _resolve_image_loss_type(self, config: Optional[Config]) -> str:
        mode = self._default_image_loss_type
        if mode is None and config is not None:
            mode = config.get("image_loss_type", None)
        mode = "l2" if mode is None else str(mode).lower()
        if mode not in ("l1", "l2"):
            raise ValueError(f"image_loss_type must be 'l1' or 'l2', got: {mode}")
        return mode

    def initialize(self, params: Params, config: Optional[Config] = None) -> None:
        for name in ["means", "quats", "scale_ori", "opacity"]:
            if params.get(name) is None:
                raise ValueError(f"Missing required parameter: {name}")
        if not params.get("means").data.is_cuda:
            raise ValueError("Parameters must be CUDA tensors")
        self.initialized = True

    @staticmethod
    def _extract(params: Params, data: Data, config: Config):
        means = params.get("means").data
        quats = params.get("quats").data
        scale_ori = params.get("scale_ori").data
        opacity = params.get("opacity").data

        return (
            means,
            quats,
            scale_ori,
            opacity,
            data.get("viewmats"),
            data.get("Ks"),
            data.get("targets"),
            data.get("vol_ranges"),
            data.get("vol_centers"),
            config.get("img_dim1"),
            config.get("img_dim0"),
            config.get("img_tile_size", 16),
            config.get("eps_blur", 0.3),
            get_lambda_l2(config, 1.0),
            config.get("lambda_ssim", 0.0),
            config.get("window_size", 11),
            config.get("vol_dim2"),
            config.get("vol_dim1"),
            config.get("vol_dim0"),
            config.get("vol_tile_size", 4),
            config.get("eps3d", 1e-4),
            config.get("radius_clip", 3.0),
            config.get("vol_loss_eps", 1e-4),
            config.get("lambda_tv", 1.0),
            config.get("scale_min", 0.01),
            config.get("scale_max", 2.0),
        )

    def forward_only(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> ForwardResult:
        from ..cuda._wrapper import (
            conics_mu_fwd,
            isect_offset_encode,
            isect_offset_encode_3d,
            isect_tiles,
            isect_tiles_3d,
            parameter_transform_fwd,
            rasterize_to_pixels_fwd,
            volume_loss_fwd,
            voxelize_to_voxels_fwd,
            world_to_image_projection_fwd,
            world_to_volume_mapping_fwd,
        )

        (
            means,
            quats,
            scale_ori,
            opacity,
            viewmats,
            Ks,
            targets,
            vol_ranges,
            vol_centers,
            img_dim1,
            img_dim0,
            img_tile_size,
            eps_blur,
            lambda_l2,
            lambda_ssim,
            window_size,
            vol_dim2,
            vol_dim1,
            vol_dim0,
            vol_tile_size,
            eps3d,
            radius_clip,
            vol_loss_eps,
            lambda_tv,
            scale_min,
            scale_max,
        ) = self._extract(params, data, config)
        image_loss_type = self._resolve_image_loss_type(config)

        C = viewmats.shape[0]
        opacity_2d = opacity.unsqueeze(0).expand(C, -1).contiguous()

        covars, max_scales = parameter_transform_fwd(quats, scale_ori, scale_min, scale_max)
        means2d, cov2d_3x3, radii_2d = world_to_image_projection_fwd(
            means, covars, viewmats, Ks, img_dim1, img_dim0, eps_blur
        )
        conics_2d, mus = conics_mu_fwd(cov2d_3x3, eps_blur)

        n_tiles_h = (img_dim0 + img_tile_size - 1) // img_tile_size
        n_tiles_w = (img_dim1 + img_tile_size - 1) // img_tile_size
        _, isect_ids_2d, flatten_ids_2d = isect_tiles(
            means2d, radii_2d, img_tile_size, n_tiles_w, n_tiles_h
        )
        tile_offsets_2d = isect_offset_encode(isect_ids_2d, C, n_tiles_w, n_tiles_h)
        render_img = rasterize_to_pixels_fwd(
            means2d,
            conics_2d,
            opacity_2d,
            mus,
            tile_offsets_2d,
            flatten_ids_2d,
            img_dim1,
            img_dim0,
            img_tile_size,
        )

        img_loss = self._image_loss_fwd(
            render_img,
            targets,
            window_size,
            lambda_l2,
            lambda_ssim,
            compute_statistics=False,
            image_loss_type=image_loss_type,
        )[0]

        means3d_v, conics_3d, radii_3d = world_to_volume_mapping_fwd(
            means,
            covars,
            max_scales,
            vol_ranges,
            vol_centers,
            vol_dim2,
            vol_dim1,
            vol_dim0,
            eps3d,
            radius_clip,
        )
        n_tiles_vz = (vol_dim2 + vol_tile_size - 1) // vol_tile_size
        n_tiles_vy = (vol_dim1 + vol_tile_size - 1) // vol_tile_size
        n_tiles_vx = (vol_dim0 + vol_tile_size - 1) // vol_tile_size
        _, isect_ids_3d, flatten_ids_3d = isect_tiles_3d(
            means3d_v, radii_3d, vol_tile_size, n_tiles_vz, n_tiles_vy, n_tiles_vx
        )
        tile_offsets_3d = isect_offset_encode_3d(
            isect_ids_3d, n_tiles_vz, n_tiles_vy, n_tiles_vx
        )
        render_vol = voxelize_to_voxels_fwd(
            means3d_v,
            conics_3d,
            opacity,
            tile_offsets_3d,
            flatten_ids_3d,
            vol_dim2,
            vol_dim1,
            vol_dim0,
            vol_tile_size,
        )
        vol_loss_total, _tv_loss, _tv_map = volume_loss_fwd(
            render_vol.unsqueeze(0), vol_loss_eps, lambda_tv
        )
        loss = img_loss + vol_loss_total

        return ForwardResult(
            outputs={
                "loss": loss,
                "loss_image": img_loss.detach(),
                "loss_tv": vol_loss_total.detach(),
                "render_image": render_img.detach(),
                "render_volume": render_vol.detach(),
            }
        )

    def forward_backward1(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> Tuple[ForwardResult, GradResult]:
        from ..cuda._wrapper import (
            conics_mu_bwd,
            conics_mu_fwd,
            isect_offset_encode,
            isect_offset_encode_3d,
            isect_tiles,
            isect_tiles_3d,
            parameter_transform_bwd,
            parameter_transform_fwd,
            rasterize_to_pixels_bwd,
            rasterize_to_pixels_fwd,
            volume_loss_bwd,
            volume_loss_fwd,
            voxelize_to_voxels_bwd,
            voxelize_to_voxels_fwd,
            world_to_image_projection_bwd,
            world_to_image_projection_fwd,
            world_to_volume_mapping_bwd,
            world_to_volume_mapping_fwd,
        )

        (
            means,
            quats,
            scale_ori,
            opacity,
            viewmats,
            Ks,
            targets,
            vol_ranges,
            vol_centers,
            img_dim1,
            img_dim0,
            img_tile_size,
            eps_blur,
            lambda_l2,
            lambda_ssim,
            window_size,
            vol_dim2,
            vol_dim1,
            vol_dim0,
            vol_tile_size,
            eps3d,
            radius_clip,
            vol_loss_eps,
            lambda_tv,
            scale_min,
            scale_max,
        ) = self._extract(params, data, config)
        image_loss_type = self._resolve_image_loss_type(config)

        C = viewmats.shape[0]
        opacity_2d = opacity.unsqueeze(0).expand(C, -1).contiguous()

        covars, max_scales = parameter_transform_fwd(quats, scale_ori, scale_min, scale_max)
        means2d, cov2d_3x3, radii_2d = world_to_image_projection_fwd(
            means, covars, viewmats, Ks, img_dim1, img_dim0, eps_blur
        )
        conics_2d, mus = conics_mu_fwd(cov2d_3x3, eps_blur)

        n_tiles_h = (img_dim0 + img_tile_size - 1) // img_tile_size
        n_tiles_w = (img_dim1 + img_tile_size - 1) // img_tile_size
        _, isect_ids_2d, flatten_ids_2d = isect_tiles(
            means2d, radii_2d, img_tile_size, n_tiles_w, n_tiles_h
        )
        tile_offsets_2d = isect_offset_encode(isect_ids_2d, C, n_tiles_w, n_tiles_h)
        render_img = rasterize_to_pixels_fwd(
            means2d,
            conics_2d,
            opacity_2d,
            mus,
            tile_offsets_2d,
            flatten_ids_2d,
            img_dim1,
            img_dim0,
            img_tile_size,
        )

        img_loss_result = self._image_loss_fwd(
            render_img,
            targets,
            window_size,
            lambda_l2,
            lambda_ssim,
            compute_statistics=True,
            image_loss_type=image_loss_type,
        )
        img_loss = img_loss_result[0]
        mu_r, mu_m = img_loss_result[3], img_loss_result[4]
        f1, f2, f3, f4 = (
            img_loss_result[5],
            img_loss_result[6],
            img_loss_result[7],
            img_loss_result[8],
        )

        means3d_v, conics_3d, radii_3d = world_to_volume_mapping_fwd(
            means,
            covars,
            max_scales,
            vol_ranges,
            vol_centers,
            vol_dim2,
            vol_dim1,
            vol_dim0,
            eps3d,
            radius_clip,
        )
        n_tiles_vz = (vol_dim2 + vol_tile_size - 1) // vol_tile_size
        n_tiles_vy = (vol_dim1 + vol_tile_size - 1) // vol_tile_size
        n_tiles_vx = (vol_dim0 + vol_tile_size - 1) // vol_tile_size
        _, isect_ids_3d, flatten_ids_3d = isect_tiles_3d(
            means3d_v, radii_3d, vol_tile_size, n_tiles_vz, n_tiles_vy, n_tiles_vx
        )
        tile_offsets_3d = isect_offset_encode_3d(
            isect_ids_3d, n_tiles_vz, n_tiles_vy, n_tiles_vx
        )
        render_vol = voxelize_to_voxels_fwd(
            means3d_v,
            conics_3d,
            opacity,
            tile_offsets_3d,
            flatten_ids_3d,
            vol_dim2,
            vol_dim1,
            vol_dim0,
            vol_tile_size,
        )

        render_vol_4d = render_vol.unsqueeze(0)
        vol_loss_total, _tv_loss, _tv_map = volume_loss_fwd(
            render_vol_4d, vol_loss_eps, lambda_tv
        )
        loss = img_loss + vol_loss_total
        grad_one = torch.ones(1, device=means.device, dtype=means.dtype)

        v_render_img = self._image_loss_bwd(
            render_img,
            targets,
            mu_r,
            mu_m,
            f1,
            f2,
            f3,
            f4,
            grad_one,
            window_size,
            lambda_l2,
            lambda_ssim,
            image_loss_type=image_loss_type,
        )

        absgrad = bool(config.get("absgrad", False)) if config is not None else False
        v_means2d_abs, v_means2d, v_conics_2d, v_opacity_rast, v_mus = (
            rasterize_to_pixels_bwd(
                means2d,
                conics_2d,
                opacity_2d,
                mus,
                tile_offsets_2d,
                flatten_ids_2d,
                v_render_img,
                img_dim1,
                img_dim0,
                img_tile_size,
                absgrad,
            )
        )
        v_cov2d_3x3 = conics_mu_bwd(
            cov2d_3x3, eps_blur, conics_2d, mus, v_conics_2d, v_mus
        )
        v_means_rast, v_covars_rast = world_to_image_projection_bwd(
            means,
            covars,
            viewmats,
            Ks,
            img_dim1,
            img_dim0,
            v_means2d,
            v_cov2d_3x3,
        )

        v_render_vol = volume_loss_bwd(
            render_vol_4d, grad_one, vol_loss_eps, lambda_tv
        ).squeeze(0)
        v_means3d_v, v_conics_3d_vol, v_opacity_vol = voxelize_to_voxels_bwd(
            means3d_v,
            conics_3d,
            opacity,
            tile_offsets_3d,
            flatten_ids_3d,
            v_render_vol,
            vol_dim2,
            vol_dim1,
            vol_dim0,
            vol_tile_size,
        )
        v_means_vol, v_covars_vol = world_to_volume_mapping_bwd(
            means,
            covars,
            vol_ranges,
            vol_centers,
            vol_dim2,
            vol_dim1,
            vol_dim0,
            eps3d,
            v_means3d_v,
            v_conics_3d_vol,
        )

        v_covars_total = v_covars_rast + v_covars_vol
        v_means_total = v_means_rast + v_means_vol
        v_opacity_total = v_opacity_rast.sum(dim=0) + v_opacity_vol
        v_quats, v_scale_ori = parameter_transform_bwd(
            quats, scale_ori, scale_min, scale_max, v_covars_total
        )

        forward_result = ForwardResult(
            outputs={
                "loss": loss,
                "loss_image": img_loss.detach(),
                "loss_tv": vol_loss_total.detach(),
                "render_image": render_img.detach(),
                "render_volume": render_vol.detach(),
            }
        )
        grad_result = GradResult(
            param_grads={
                "means": v_means_total,
                "quats": v_quats,
                "scale_ori": v_scale_ori,
                "opacity": v_opacity_total,
            },
            aux={
                "v_means2d": v_means2d,
                "v_means2d_abs": v_means2d_abs,
                "radii_2d": radii_2d,
            },
        )
        return forward_result, grad_result


__all__ = ["GS3DCTLossCUDA"]
