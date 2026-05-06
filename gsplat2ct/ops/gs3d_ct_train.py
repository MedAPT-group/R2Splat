"""End-to-end first-order CT loss operator."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ..eval.metric import ssim2d
from ..model.beta import activated_beta_terms, beta_path_log_grad
from ..model.opacity import opacity_activation, opacity_activation_grad
from .base import ForwardBackwardBase
from .data import Config, Data, ForwardResult, GradResult, Params


class GS3DCTLossCUDA(ForwardBackwardBase):
    """Full CT loss operator for Adam training.

    The pipeline combines a projection-fitting term and a TV regularizer:

    - raw parameters -> covariance transform
    - rasterize projections -> image loss
    - voxelize a sampled sub-volume -> anisotropic TV loss

    ``scale_min`` / ``scale_max`` in the config must already be world-space
    activated scale bounds.  High-level scripts convert the r2_gaussian-style
    relative config values by multiplying ``max(scanner.sVoxel)``.
    """

    def __init__(self):
        self.initialized = False

    def _image_loss_fwd(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        lambda_l1: float,
        lambda_dssim: float,
        ssim_window: int,
    ):
        l1_loss = (prediction - target).abs().mean()
        if float(lambda_dssim) > 0.0:
            ssim_val = ssim2d(
                prediction.unsqueeze(1),
                target.unsqueeze(1),
                window_size=int(ssim_window),
            )
            dssim_loss = 1.0 - ssim_val
        else:
            dssim_loss = torch.zeros((), device=prediction.device, dtype=prediction.dtype)
        loss = float(lambda_l1) * l1_loss + float(lambda_dssim) * dssim_loss
        return loss, l1_loss, dssim_loss

    def _image_loss_fwd_bwd(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        lambda_l1: float,
        lambda_dssim: float,
        ssim_window: int,
    ):
        pred_leaf = prediction.detach().requires_grad_(True)
        loss, l1_loss, dssim_loss = self._image_loss_fwd(
            pred_leaf, target, lambda_l1, lambda_dssim, ssim_window
        )
        loss.backward()
        return (
            loss.detach(),
            l1_loss.detach(),
            dssim_loss.detach(),
            pred_leaf.grad.detach().contiguous(),
        )

    def initialize(self, params: Params, config: Optional[Config] = None) -> None:
        for name in ["means", "quats", "scale_ori", "opacity"]:
            if params.get(name) is None:
                raise ValueError(f"Missing required parameter: {name}")
        primitive_type = "gaussian"
        if config is not None:
            primitive_type = str(config.get("primitive_type", primitive_type)).lower()
        if params.get("beta") is not None:
            primitive_type = "beta"
        if primitive_type == "beta" and params.get("beta") is None:
            raise ValueError("primitive_type='beta' requires parameter: beta")
        if not params.get("means").data.is_cuda:
            raise ValueError("Parameters must be CUDA tensors")
        self.initialized = True

    @staticmethod
    def _extract(params: Params, data: Data, config: Config):
        means = params.get("means").data
        quats = params.get("quats").data
        scale_ori = params.get("scale_ori").data
        raw_opacity = params.get("opacity").data
        beta_param = params.get("beta")
        beta = beta_param.data if beta_param is not None else None
        primitive_type = str(
            config.get("primitive_type", "beta" if beta is not None else "gaussian")
        ).lower()
        radius_clip_default = 1.0 if primitive_type == "beta" else 3.0
        img_radius_clip_default = 1.0 if primitive_type == "beta" else 3.0

        return (
            means,
            quats,
            scale_ori,
            raw_opacity,
            beta,
            primitive_type,
            data.get("viewmats"),
            data.get("Ks"),
            data.get("targets"),
            data.get("vol_ranges"),
            data.get("vol_centers"),
            config.get("img_dim1"),
            config.get("img_dim0"),
            config.get("img_tile_size", 16),
            config.get("eps_blur", 0.3),
            config.get("img_radius_clip", img_radius_clip_default),
            float(config.get("lambda_l1", 1.0)),
            float(config.get("lambda_dssim", 0.0)),
            int(config.get("ssim_window", 11)),
            config.get("vol_dim2"),
            config.get("vol_dim1"),
            config.get("vol_dim0"),
            config.get("vol_tile_size", 4),
            config.get("eps3d", 1e-4),
            config.get("radius_clip", radius_clip_default),
            config.get("lambda_tv", 1.0),
            config.get("scale_min", 0.01),
            config.get("scale_max", 2.0),
            config.get("beta_gamma_init", 3.5),
        )

    def forward_only(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> ForwardResult:
        from ..cuda._wrapper import (
            beta_rasterize_to_pixels_fwd,
            beta_voxelize_to_voxels_fwd,
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
            raw_opacity,
            beta,
            primitive_type,
            viewmats,
            Ks,
            targets,
            vol_ranges,
            vol_centers,
            img_dim1,
            img_dim0,
            img_tile_size,
            eps_blur,
            img_radius_clip,
            lambda_l1,
            lambda_dssim,
            ssim_window,
            vol_dim2,
            vol_dim1,
            vol_dim0,
            vol_tile_size,
            eps3d,
            radius_clip,
            lambda_tv,
            scale_min,
            scale_max,
            beta_gamma_init,
        ) = self._extract(params, data, config)

        C = viewmats.shape[0]
        opacity = opacity_activation(raw_opacity)
        opacity_2d = opacity.unsqueeze(0).expand(C, -1).contiguous()

        covars, max_scales = parameter_transform_fwd(quats, scale_ori, scale_min, scale_max)
        means2d, cov2d_3x3, radii_2d = world_to_image_projection_fwd(
            means, covars, viewmats, Ks, img_dim1, img_dim0, eps_blur, img_radius_clip
        )
        conics_2d, mus = conics_mu_fwd(cov2d_3x3, eps_blur)
        if primitive_type == "beta":
            if beta is None:
                raise ValueError("primitive_type='beta' requires parameter beta")
            gamma, projected_beta, path_ratio, _activation_grad = activated_beta_terms(
                beta, gamma_init=beta_gamma_init
            )
            mus_render = mus * path_ratio.unsqueeze(0)
            beta_2d = projected_beta.unsqueeze(0).expand(C, -1).contiguous()
        else:
            gamma = projected_beta = path_ratio = beta_2d = None
            mus_render = mus

        n_tiles_h = (img_dim0 + img_tile_size - 1) // img_tile_size
        n_tiles_w = (img_dim1 + img_tile_size - 1) // img_tile_size
        _, isect_ids_2d, flatten_ids_2d = isect_tiles(
            means2d, radii_2d, img_tile_size, n_tiles_w, n_tiles_h
        )
        tile_offsets_2d = isect_offset_encode(isect_ids_2d, C, n_tiles_w, n_tiles_h)
        if primitive_type == "beta":
            render_img = beta_rasterize_to_pixels_fwd(
                means2d,
                conics_2d,
                opacity_2d,
                mus_render,
                beta_2d,
                tile_offsets_2d,
                flatten_ids_2d,
                img_dim1,
                img_dim0,
                img_tile_size,
            )
        else:
            render_img = rasterize_to_pixels_fwd(
                means2d,
                conics_2d,
                opacity_2d,
                mus_render,
                tile_offsets_2d,
                flatten_ids_2d,
                img_dim1,
                img_dim0,
                img_tile_size,
            )

        img_loss, l1_loss, dssim_loss = self._image_loss_fwd(
            render_img, targets, lambda_l1, lambda_dssim, ssim_window
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
        if primitive_type == "beta":
            render_vol = beta_voxelize_to_voxels_fwd(
                means3d_v,
                conics_3d,
                opacity,
                gamma,
                tile_offsets_3d,
                flatten_ids_3d,
                vol_dim2,
                vol_dim1,
                vol_dim0,
                vol_tile_size,
            )
        else:
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
            render_vol.unsqueeze(0), lambda_tv
        )
        loss = img_loss + vol_loss_total

        return ForwardResult(
            outputs={
                "loss": loss,
                "loss_image": img_loss.detach(),
                "loss_l1": l1_loss.detach(),
                "loss_dssim": dssim_loss.detach(),
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
            beta_rasterize_to_pixels_bwd,
            beta_rasterize_to_pixels_fwd,
            beta_voxelize_to_voxels_bwd,
            beta_voxelize_to_voxels_fwd,
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
            raw_opacity,
            beta,
            primitive_type,
            viewmats,
            Ks,
            targets,
            vol_ranges,
            vol_centers,
            img_dim1,
            img_dim0,
            img_tile_size,
            eps_blur,
            img_radius_clip,
            lambda_l1,
            lambda_dssim,
            ssim_window,
            vol_dim2,
            vol_dim1,
            vol_dim0,
            vol_tile_size,
            eps3d,
            radius_clip,
            lambda_tv,
            scale_min,
            scale_max,
            beta_gamma_init,
        ) = self._extract(params, data, config)

        C = viewmats.shape[0]
        opacity = opacity_activation(raw_opacity)
        opacity_2d = opacity.unsqueeze(0).expand(C, -1).contiguous()

        covars, max_scales = parameter_transform_fwd(quats, scale_ori, scale_min, scale_max)
        means2d, cov2d_3x3, radii_2d = world_to_image_projection_fwd(
            means, covars, viewmats, Ks, img_dim1, img_dim0, eps_blur, img_radius_clip
        )
        conics_2d, mus = conics_mu_fwd(cov2d_3x3, eps_blur)
        if primitive_type == "beta":
            if beta is None:
                raise ValueError("primitive_type='beta' requires parameter beta")
            gamma, projected_beta, path_ratio, beta_act_grad = activated_beta_terms(
                beta, gamma_init=beta_gamma_init
            )
            mus_render = mus * path_ratio.unsqueeze(0)
            beta_2d = projected_beta.unsqueeze(0).expand(C, -1).contiguous()
        else:
            gamma = projected_beta = path_ratio = beta_act_grad = beta_2d = None
            mus_render = mus

        n_tiles_h = (img_dim0 + img_tile_size - 1) // img_tile_size
        n_tiles_w = (img_dim1 + img_tile_size - 1) // img_tile_size
        _, isect_ids_2d, flatten_ids_2d = isect_tiles(
            means2d, radii_2d, img_tile_size, n_tiles_w, n_tiles_h
        )
        tile_offsets_2d = isect_offset_encode(isect_ids_2d, C, n_tiles_w, n_tiles_h)
        if primitive_type == "beta":
            render_img = beta_rasterize_to_pixels_fwd(
                means2d,
                conics_2d,
                opacity_2d,
                mus_render,
                beta_2d,
                tile_offsets_2d,
                flatten_ids_2d,
                img_dim1,
                img_dim0,
                img_tile_size,
            )
        else:
            render_img = rasterize_to_pixels_fwd(
                means2d,
                conics_2d,
                opacity_2d,
                mus_render,
                tile_offsets_2d,
                flatten_ids_2d,
                img_dim1,
                img_dim0,
                img_tile_size,
            )

        img_loss, l1_loss, dssim_loss, v_render_img = self._image_loss_fwd_bwd(
            render_img,
            targets,
            lambda_l1,
            lambda_dssim,
            ssim_window,
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
        if primitive_type == "beta":
            render_vol = beta_voxelize_to_voxels_fwd(
                means3d_v,
                conics_3d,
                opacity,
                gamma,
                tile_offsets_3d,
                flatten_ids_3d,
                vol_dim2,
                vol_dim1,
                vol_dim0,
                vol_tile_size,
            )
        else:
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
        vol_loss_total, _tv_loss, _tv_map = volume_loss_fwd(render_vol_4d, lambda_tv)
        loss = img_loss + vol_loss_total
        grad_one = torch.ones(1, device=means.device, dtype=means.dtype)

        absgrad = bool(config.get("absgrad", False)) if config is not None else False
        v_beta_raw = None
        if primitive_type == "beta":
            (
                v_means2d_abs,
                v_means2d,
                v_conics_2d,
                v_opacity_rast,
                v_mus_render,
                v_projected_beta,
            ) = beta_rasterize_to_pixels_bwd(
                means2d,
                conics_2d,
                opacity_2d,
                mus_render,
                beta_2d,
                tile_offsets_2d,
                flatten_ids_2d,
                v_render_img,
                img_dim1,
                img_dim0,
                img_tile_size,
                absgrad,
            )
            v_mus = v_mus_render * path_ratio.unsqueeze(0)
            v_gamma_path = (
                v_mus_render
                * mus
                * path_ratio.unsqueeze(0)
                * beta_path_log_grad(gamma).unsqueeze(0)
            ).sum(dim=0)
            v_gamma_proj = v_projected_beta.sum(dim=0)
            v_beta_raw = (v_gamma_path + v_gamma_proj) * beta_act_grad
        else:
            v_means2d_abs, v_means2d, v_conics_2d, v_opacity_rast, v_mus = (
                rasterize_to_pixels_bwd(
                    means2d,
                    conics_2d,
                    opacity_2d,
                    mus_render,
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

        v_render_vol = volume_loss_bwd(render_vol_4d, grad_one, lambda_tv).squeeze(0)
        if primitive_type == "beta":
            v_means3d_v, v_conics_3d_vol, v_opacity_vol, v_gamma_vol = (
                beta_voxelize_to_voxels_bwd(
                    means3d_v,
                    conics_3d,
                    opacity,
                    gamma,
                    tile_offsets_3d,
                    flatten_ids_3d,
                    v_render_vol,
                    vol_dim2,
                    vol_dim1,
                    vol_dim0,
                    vol_tile_size,
                )
            )
            v_beta_raw = v_beta_raw + v_gamma_vol * beta_act_grad
        else:
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
        v_opacity_raw = v_opacity_total * opacity_activation_grad(raw_opacity)
        v_quats, v_scale_ori = parameter_transform_bwd(
            quats, scale_ori, scale_min, scale_max, v_covars_total
        )

        forward_result = ForwardResult(
            outputs={
                "loss": loss,
                "loss_image": img_loss.detach(),
                "loss_l1": l1_loss.detach(),
                "loss_dssim": dssim_loss.detach(),
                "loss_tv": vol_loss_total.detach(),
                "render_image": render_img.detach(),
                "render_volume": render_vol.detach(),
            }
        )
        param_grads = {
            "means": v_means_total,
            "quats": v_quats,
            "scale_ori": v_scale_ori,
            "opacity": v_opacity_raw,
        }
        if primitive_type == "beta" and v_beta_raw is not None:
            param_grads["beta"] = v_beta_raw

        grad_result = GradResult(
            param_grads=param_grads,
            aux={
                "v_means2d": v_means2d,
                "v_means2d_abs": v_means2d_abs,
                "radii_2d": radii_2d,
            },
        )
        return forward_result, grad_result


__all__ = ["GS3DCTLossCUDA"]
