"""
GS3DCTDebugCUDA / GS3DCTDebugAutograd: Debug operators for numerical validation.

Same architecture as GS3DCTTrainCUDA, but with two key changes for testability:
  1. Volume loss uses L2 (MSE) instead of TV, so the volume Hessian is exactly
     diagonal and the CUDA second-order chain rule is exact.
  2. Image loss should be called with lambda_ssim=0 so that the image Hessian
     is also exactly diagonal (no cross-pixel second-order terms).

This allows exact numerical comparison between the CUDA chain and autograd.

The Autograd version uses pure PyTorch operations with double autograd for
ground-truth Hessian computation.

Author: Dy164
Date: 2026-03-19
"""

import torch
from typing import Optional, Tuple
from .base import SecondForwardBackwardBase
from .data import Params, Data, Config, ForwardResult, GradResult, HessResult


class _ParamWrapper:
    """Simple wrapper to satisfy ParameterLike protocol."""
    def __init__(self, data: torch.Tensor):
        self.data = data


# ============================================================================
# CUDA Implementation
# ============================================================================

class GS3DCTDebugCUDA(SecondForwardBackwardBase):
    """Debug CUDA operator: ImageLoss(L1 only) + L2 volume loss.

    Same interface as GS3DCTTrainCUDA but replaces VolumeLoss(TV) with L2Loss
    for exact second-order numerical validation.

    Expected params/data/config: same as GS3DCTTrainCUDA, plus:
        data['vol_target']: (dim0, dim1, dim2) target volume for L2 loss
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, params: Params, config: Optional[Config] = None) -> None:
        for name in ['means', 'quats', 'scale_ori', 'opacity', 'opacity_vol']:
            if params.get(name) is None:
                raise ValueError(f"Missing required parameter: {name}")
        self.initialized = True

    @staticmethod
    def _extract(params, data, config):
        means = params.get('means').data
        quats = params.get('quats').data
        scale_ori = params.get('scale_ori').data
        opacity = params.get('opacity').data
        opacity_vol = params.get('opacity_vol').data

        viewmats = data.get('viewmats')
        Ks = data.get('Ks')
        targets = data.get('targets')
        vol_ranges = data.get('vol_ranges')
        vol_centers = data.get('vol_centers')
        vol_target = data.get('vol_target')

        img_dim1 = config.get('img_dim1')
        img_dim0 = config.get('img_dim0')
        img_tile_size = config.get('img_tile_size', 16)
        eps_blur = config.get('eps_blur', 0.3)
        lambda_l1 = config.get('lambda_l1', 1.0)
        lambda_ssim = config.get('lambda_ssim', 0.0)
        window_size = config.get('window_size', 11)

        vol_dim2 = config.get('vol_dim2')
        vol_dim1 = config.get('vol_dim1')
        vol_dim0 = config.get('vol_dim0')
        vol_tile_size = config.get('vol_tile_size', 4)
        eps3d = config.get('eps3d', 1e-4)
        radius_clip = config.get('radius_clip', 3.0)
        lambda_vol = config.get('lambda_vol', 1.0)

        scale_min = config.get('scale_min', 0.01)
        scale_max = config.get('scale_max', 2.0)

        return (means, quats, scale_ori, opacity, opacity_vol,
                viewmats, Ks, targets, vol_ranges, vol_centers, vol_target,
                img_dim1, img_dim0, img_tile_size, eps_blur,
                lambda_l1, lambda_ssim, window_size,
                vol_dim2, vol_dim1, vol_dim0, vol_tile_size,
                eps3d, radius_clip, lambda_vol,
                scale_min, scale_max)

    def forward_only(self, params, data, config=None):
        from ..cuda._wrapper import (
            parameter_transform_fwd,
            world_to_image_projection_fwd, conics_mu_fwd,
            isect_tiles, isect_offset_encode,
            rasterize_to_pixels_fwd, image_loss_fwd,
            world_to_volume_mapping_fwd,
            isect_tiles_3d, isect_offset_encode_3d,
            voxelize_to_voxels_fwd, l2_loss_fwd,
        )

        (means, quats, scale_ori, opacity, opacity_vol,
         viewmats, Ks, targets, vol_ranges, vol_centers, vol_target,
         img_dim1, img_dim0, img_tile_size, eps_blur,
         lambda_l1, lambda_ssim, window_size,
         vol_dim2, vol_dim1, vol_dim0, vol_tile_size,
         eps3d, radius_clip, lambda_vol,
         scale_min, scale_max) = self._extract(params, data, config)

        C = viewmats.shape[0]

        # Shared: ParameterTransform
        covars, max_scales = parameter_transform_fwd(
            quats, scale_ori, scale_min, scale_max
        )

        # Rasterize path
        means2d, cov2d_3x3, radii_2d = world_to_image_projection_fwd(
            means, covars, viewmats, Ks, img_dim1, img_dim0
        )
        conics_2d, mus = conics_mu_fwd(cov2d_3x3, eps_blur)

        n_th = (img_dim0 + img_tile_size - 1) // img_tile_size
        n_tw = (img_dim1 + img_tile_size - 1) // img_tile_size
        _, isect_ids_2d, flatten_ids_2d = isect_tiles(
            means2d, radii_2d, img_tile_size, n_tw, n_th
        )
        tile_offsets_2d = isect_offset_encode(isect_ids_2d, C, n_tw, n_th)

        render_img = rasterize_to_pixels_fwd(
            means2d, conics_2d, opacity, mus,
            tile_offsets_2d, flatten_ids_2d,
            img_dim1, img_dim0, img_tile_size
        )
        img_loss = image_loss_fwd(
            render_img, targets, window_size,
            lambda_l1, lambda_ssim, compute_statistics=False
        )[0]

        # Voxelize path (L2 loss)
        means3d_v, conics_3d, radii_3d = world_to_volume_mapping_fwd(
            means, covars, max_scales, vol_ranges, vol_centers,
            vol_dim2, vol_dim1, vol_dim0, eps3d, radius_clip
        )
        n_tvz = (vol_dim2 + vol_tile_size - 1) // vol_tile_size
        n_tvy = (vol_dim1 + vol_tile_size - 1) // vol_tile_size
        n_tvx = (vol_dim0 + vol_tile_size - 1) // vol_tile_size
        _, isect_ids_3d, flatten_ids_3d = isect_tiles_3d(
            means3d_v, radii_3d, vol_tile_size, n_tvz, n_tvy, n_tvx
        )
        tile_offsets_3d = isect_offset_encode_3d(
            isect_ids_3d, n_tvz, n_tvy, n_tvx
        )
        render_vol = voxelize_to_voxels_fwd(
            means3d_v, conics_3d, opacity_vol,
            tile_offsets_3d, flatten_ids_3d,
            vol_dim2, vol_dim1, vol_dim0, vol_tile_size
        )
        vol_l2_loss, _ = l2_loss_fwd(render_vol, vol_target)
        vol_loss = lambda_vol * vol_l2_loss

        loss = img_loss + vol_loss

        return ForwardResult(
            outputs={
                'loss': loss,
                'loss_image': img_loss.detach(),
                'loss_vol': vol_loss.detach(),
                'render_image': render_img.detach(),
                'render_volume': render_vol.detach(),
            },
            intermediates={}
        )

    def forward_backward1(self, params, data, config=None):
        from ..cuda._wrapper import (
            parameter_transform_fwd, parameter_transform_bwd,
            world_to_image_projection_fwd, world_to_image_projection_bwd,
            conics_mu_fwd, conics_mu_bwd,
            isect_tiles, isect_offset_encode,
            rasterize_to_pixels_fwd, rasterize_to_pixels_bwd,
            image_loss_fwd, image_loss_bwd,
            world_to_volume_mapping_fwd, world_to_volume_mapping_bwd,
            isect_tiles_3d, isect_offset_encode_3d,
            voxelize_to_voxels_fwd, voxelize_to_voxels_bwd,
            l2_loss_fwd, l2_loss_bwd,
        )

        (means, quats, scale_ori, opacity, opacity_vol,
         viewmats, Ks, targets, vol_ranges, vol_centers, vol_target,
         img_dim1, img_dim0, img_tile_size, eps_blur,
         lambda_l1, lambda_ssim, window_size,
         vol_dim2, vol_dim1, vol_dim0, vol_tile_size,
         eps3d, radius_clip, lambda_vol,
         scale_min, scale_max) = self._extract(params, data, config)

        C = viewmats.shape[0]
        N = means.shape[0]
        grad_one = torch.ones(1, device=means.device, dtype=means.dtype)

        # FORWARD: Shared
        covars, max_scales = parameter_transform_fwd(
            quats, scale_ori, scale_min, scale_max
        )

        # FORWARD: Rasterize path
        means2d, cov2d_3x3, radii_2d = world_to_image_projection_fwd(
            means, covars, viewmats, Ks, img_dim1, img_dim0
        )
        conics_2d, mus = conics_mu_fwd(cov2d_3x3, eps_blur)
        n_th = (img_dim0 + img_tile_size - 1) // img_tile_size
        n_tw = (img_dim1 + img_tile_size - 1) // img_tile_size
        _, isect_ids_2d, flatten_ids_2d = isect_tiles(
            means2d, radii_2d, img_tile_size, n_tw, n_th
        )
        tile_offsets_2d = isect_offset_encode(isect_ids_2d, C, n_tw, n_th)
        render_img = rasterize_to_pixels_fwd(
            means2d, conics_2d, opacity, mus,
            tile_offsets_2d, flatten_ids_2d,
            img_dim1, img_dim0, img_tile_size
        )
        img_loss_result = image_loss_fwd(
            render_img, targets, window_size,
            lambda_l1, lambda_ssim, compute_statistics=True
        )
        img_loss = img_loss_result[0]
        mu_r, mu_m = img_loss_result[3], img_loss_result[4]
        f1, f2, f3, f4 = img_loss_result[5], img_loss_result[6], img_loss_result[7], img_loss_result[8]

        # FORWARD: Voxelize path
        means3d_v, conics_3d, radii_3d = world_to_volume_mapping_fwd(
            means, covars, max_scales, vol_ranges, vol_centers,
            vol_dim2, vol_dim1, vol_dim0, eps3d, radius_clip
        )
        n_tvz = (vol_dim2 + vol_tile_size - 1) // vol_tile_size
        n_tvy = (vol_dim1 + vol_tile_size - 1) // vol_tile_size
        n_tvx = (vol_dim0 + vol_tile_size - 1) // vol_tile_size
        _, isect_ids_3d, flatten_ids_3d = isect_tiles_3d(
            means3d_v, radii_3d, vol_tile_size, n_tvz, n_tvy, n_tvx
        )
        tile_offsets_3d = isect_offset_encode_3d(
            isect_ids_3d, n_tvz, n_tvy, n_tvx
        )
        render_vol = voxelize_to_voxels_fwd(
            means3d_v, conics_3d, opacity_vol,
            tile_offsets_3d, flatten_ids_3d,
            vol_dim2, vol_dim1, vol_dim0, vol_tile_size
        )
        vol_l2_loss, _ = l2_loss_fwd(render_vol, vol_target)
        vol_loss = lambda_vol * vol_l2_loss
        loss = img_loss + vol_loss

        # BACKWARD: Rasterize path
        v_render_img = image_loss_bwd(
            render_img, targets, mu_r, mu_m, f1, f2, f3, f4,
            grad_one, window_size, lambda_l1, lambda_ssim
        )
        v_means2d, v_conics_2d, v_opacity_rast, v_mus = rasterize_to_pixels_bwd(
            means2d, conics_2d, opacity, mus,
            tile_offsets_2d, flatten_ids_2d, v_render_img,
            img_dim1, img_dim0, img_tile_size
        )
        v_cov2d_3x3 = conics_mu_bwd(
            cov2d_3x3, eps_blur, conics_2d, mus, v_conics_2d, v_mus
        )
        v_means_rast, v_covars_rast = world_to_image_projection_bwd(
            means, covars, viewmats, Ks, img_dim1, img_dim0,
            v_means2d, v_cov2d_3x3,
        )

        # BACKWARD: Voxelize path (L2 loss)
        grad_vol = grad_one * lambda_vol
        v_render_vol = l2_loss_bwd(render_vol, vol_target, grad_vol)
        v_means3d_v, v_conics_3d_vol, v_opacity_vol_out = voxelize_to_voxels_bwd(
            means3d_v, conics_3d, opacity_vol,
            tile_offsets_3d, flatten_ids_3d, v_render_vol,
            vol_dim2, vol_dim1, vol_dim0, vol_tile_size
        )
        v_means_vol, v_covars_vol = world_to_volume_mapping_bwd(
            means, covars, vol_ranges, vol_centers,
            vol_dim2, vol_dim1, vol_dim0, eps3d,
            v_means3d_v, v_conics_3d_vol,
        )

        # MERGE + ParameterTransform bwd
        v_covars_total = v_covars_rast + v_covars_vol
        v_means_total = v_means_rast + v_means_vol
        v_quats, v_scale_ori = parameter_transform_bwd(
            quats, scale_ori, scale_min, scale_max, v_covars_total
        )

        forward_result = ForwardResult(
            outputs={'loss': loss, 'loss_image': img_loss.detach(),
                     'loss_vol': vol_loss.detach(),
                     'render_image': render_img.detach(),
                     'render_volume': render_vol.detach()},
            intermediates={'v_means2d': v_means2d}
        )
        grad_result = GradResult(param_grads={
            'means': v_means_total, 'quats': v_quats,
            'scale_ori': v_scale_ori, 'opacity': v_opacity_rast,
            'opacity_vol': v_opacity_vol_out,
        })
        return forward_result, grad_result

    def forward_backward2(self, params, data, config=None):
        from ..cuda._wrapper import (
            parameter_transform_fwd, parameter_transform_bwd2,
            world_to_image_projection_fwd, world_to_image_projection_bwd2,
            conics_mu_fwd, conics_mu_bwd2,
            isect_tiles, isect_offset_encode,
            rasterize_to_pixels_fwd, rasterize_to_pixels_bwd2,
            image_loss_fwd, image_loss_bwd, image_loss_bwd2,
            world_to_volume_mapping_fwd, world_to_volume_mapping_bwd2,
            isect_tiles_3d, isect_offset_encode_3d,
            voxelize_to_voxels_fwd, voxelize_to_voxels_bwd2,
            l2_loss_fwd, l2_loss_bwd, l2_loss_bwd2,
        )

        (means, quats, scale_ori, opacity, opacity_vol,
         viewmats, Ks, targets, vol_ranges, vol_centers, vol_target,
         img_dim1, img_dim0, img_tile_size, eps_blur,
         lambda_l1, lambda_ssim, window_size,
         vol_dim2, vol_dim1, vol_dim0, vol_tile_size,
         eps3d, radius_clip, lambda_vol,
         scale_min, scale_max) = self._extract(params, data, config)

        C = viewmats.shape[0]
        N = means.shape[0]
        dtype = means.dtype
        device = means.device
        grad_one = torch.ones(1, device=device, dtype=dtype)

        # FORWARD: Shared
        covars, max_scales = parameter_transform_fwd(
            quats, scale_ori, scale_min, scale_max
        )

        # FORWARD: Rasterize path
        means2d, cov2d_3x3, radii_2d = world_to_image_projection_fwd(
            means, covars, viewmats, Ks, img_dim1, img_dim0
        )
        conics_2d, mus = conics_mu_fwd(cov2d_3x3, eps_blur)
        n_th = (img_dim0 + img_tile_size - 1) // img_tile_size
        n_tw = (img_dim1 + img_tile_size - 1) // img_tile_size
        _, isect_ids_2d, flatten_ids_2d = isect_tiles(
            means2d, radii_2d, img_tile_size, n_tw, n_th
        )
        tile_offsets_2d = isect_offset_encode(isect_ids_2d, C, n_tw, n_th)
        render_img = rasterize_to_pixels_fwd(
            means2d, conics_2d, opacity, mus,
            tile_offsets_2d, flatten_ids_2d,
            img_dim1, img_dim0, img_tile_size
        )
        img_loss_result = image_loss_fwd(
            render_img, targets, window_size,
            lambda_l1, lambda_ssim, compute_statistics=True
        )
        img_loss = img_loss_result[0]
        mu_r, mu_m = img_loss_result[3], img_loss_result[4]
        f1, f2, f3, f4 = img_loss_result[5], img_loss_result[6], img_loss_result[7], img_loss_result[8]

        # FORWARD: Voxelize path (L2 loss)
        means3d_v, conics_3d, radii_3d = world_to_volume_mapping_fwd(
            means, covars, max_scales, vol_ranges, vol_centers,
            vol_dim2, vol_dim1, vol_dim0, eps3d, radius_clip
        )
        n_tvz = (vol_dim2 + vol_tile_size - 1) // vol_tile_size
        n_tvy = (vol_dim1 + vol_tile_size - 1) // vol_tile_size
        n_tvx = (vol_dim0 + vol_tile_size - 1) // vol_tile_size
        _, isect_ids_3d, flatten_ids_3d = isect_tiles_3d(
            means3d_v, radii_3d, vol_tile_size, n_tvz, n_tvy, n_tvx
        )
        tile_offsets_3d = isect_offset_encode_3d(
            isect_ids_3d, n_tvz, n_tvy, n_tvx
        )
        render_vol = voxelize_to_voxels_fwd(
            means3d_v, conics_3d, opacity_vol,
            tile_offsets_3d, flatten_ids_3d,
            vol_dim2, vol_dim1, vol_dim0, vol_tile_size
        )
        vol_l2_loss, _ = l2_loss_fwd(render_vol, vol_target)
        vol_loss = lambda_vol * vol_l2_loss
        loss = img_loss + vol_loss

        # ========== RASTERIZE PATH: backward2 ==========
        v_render_img = image_loss_bwd(
            render_img, targets, mu_r, mu_m, f1, f2, f3, f4,
            grad_one, window_size, lambda_l1, lambda_ssim
        )
        h_render_img = image_loss_bwd2(
            render_img, targets, mu_r, mu_m, f1, f2, f3, f4,
            grad_one, window_size, lambda_l1, lambda_ssim
        )

        (v_means2d_r, v_conics_2d_r, v_opacity_rast, v_mus_r,
         h_means2d_r, h_conics_2d_r,
         h_mu_mean_r, h_mu_conic_r, h_mean_conic_r,
         h_mu_opacity_r, h_conic_opacity_r, h_mean_opacity_r,
         h_mus_r, h_opacity_rast) = rasterize_to_pixels_bwd2(
            means2d, conics_2d, opacity, mus,
            tile_offsets_2d, flatten_ids_2d,
            v_render_img, h_render_img,
            img_dim1, img_dim0, img_tile_size
        )

        h_means2d_conics_r = h_mean_conic_r.reshape(C, -1, 6)
        v_cov2d_3x3_r, h_cov_cov_r, h_cov_means2d_r, h_cov_opacity_r = conics_mu_bwd2(
            cov2d_3x3, eps_blur, conics_2d, mus,
            v_conics_2d_r, v_mus_r,
            h_conics_2d_r, h_mu_conic_r, h_mus_r,
            h_means2d_conics_r, h_conic_opacity_r, h_mu_mean_r, h_mu_opacity_r,
        )

        h_cov_means2d_mat = h_cov_means2d_r.reshape(C, N, 6, 2)
        h_means2d_cov_r = h_cov_means2d_mat.transpose(-1, -2).reshape(C, N, 12)

        (v_means_rast, v_covars_rast,
         h_means_rast, h_covars_rast, h_means_covars_rast,
         h_means_opacity_rast, h_covars_opacity_rast) = world_to_image_projection_bwd2(
            means, covars, viewmats, Ks, img_dim1, img_dim0,
            v_means2d_r, v_cov2d_3x3_r,
            h_means2d_r, h_cov_cov_r, h_means2d_cov_r,
            h_mean_opacity_r, h_cov_opacity_r,
        )

        # ========== VOXELIZE PATH: backward2 (L2 loss) ==========
        grad_vol = grad_one * lambda_vol
        v_render_vol = l2_loss_bwd(render_vol, vol_target, grad_vol)
        h_render_vol = l2_loss_bwd2(render_vol, grad_vol)

        (v_means3d_v_vol, v_conics_3d_vol, v_opacity_vol_out,
         h_means3d_v_vol, h_conics_3d_vol, h_opacity_vol_out,
         h_op_mean_vol, h_op_conic_vol, h_mean_conic_vol) = voxelize_to_voxels_bwd2(
            means3d_v, conics_3d, opacity_vol,
            tile_offsets_3d, flatten_ids_3d,
            v_render_vol, h_render_vol,
            vol_dim2, vol_dim1, vol_dim0, vol_tile_size
        )

        (v_means_vol, v_covars_vol,
         h_means_vol, h_covars_vol, h_means_covars_vol,
         h_means_opacity_vol, h_covars_opacity_vol) = world_to_volume_mapping_bwd2(
            means, covars, vol_ranges, vol_centers,
            vol_dim2, vol_dim1, vol_dim0, eps3d,
            v_means3d_v_vol, v_conics_3d_vol,
            h_means3d_v_vol, h_conics_3d_vol, h_mean_conic_vol,
            h_op_mean_vol, h_op_conic_vol,
        )

        # ========== MERGE + ParameterTransform bwd2 ==========
        v_means_total = v_means_rast + v_means_vol
        v_covars_total = v_covars_rast + v_covars_vol
        h_means_total = h_means_rast + h_means_vol
        h_covars_total = h_covars_rast + h_covars_vol
        h_means_covars_total = h_means_covars_rast + h_means_covars_vol

        h_covars_opacity_rast_summed = h_covars_opacity_rast.sum(dim=0)
        h_covars_opacity_total = h_covars_opacity_rast_summed + h_covars_opacity_vol

        h_mc_mat = h_means_covars_total.reshape(N, 3, 6)
        h_cm_mat = h_mc_mat.transpose(-1, -2)
        h_covars_means_for_pt = h_cm_mat.reshape(N, 18)

        (v_quats, v_scale_ori,
         h_quats, h_scale_ori, h_quats_scale_ori,
         h_quats_means, h_quats_opacity,
         h_scale_ori_means, h_scale_ori_opacity) = parameter_transform_bwd2(
            quats, scale_ori, scale_min, scale_max,
            v_covars_total, h_covars_total,
            h_covars_means_for_pt, h_covars_opacity_total,
        )

        forward_result = ForwardResult(
            outputs={'loss': loss, 'loss_image': img_loss.detach(),
                     'loss_vol': vol_loss.detach(),
                     'render_image': render_img.detach(),
                     'render_volume': render_vol.detach()},
            intermediates={}
        )
        grad_result = GradResult(param_grads={
            'means': v_means_total, 'quats': v_quats,
            'scale_ori': v_scale_ori, 'opacity': v_opacity_rast,
            'opacity_vol': v_opacity_vol_out,
        })
        hess_result = HessResult(
            param_hess={
                'means': h_means_total,
                'quats': h_quats,
                'scale_ori': h_scale_ori,
                'opacity': h_opacity_rast,
                'opacity_vol': h_opacity_vol_out,
            },
            cross_hess={
                'quats_scale_ori': h_quats_scale_ori,
                'means_opacity_rast': h_means_opacity_rast,
                'means_opacity_vol': h_means_opacity_vol,
                'quats_means': h_quats_means,
                'quats_opacity': h_quats_opacity,
                'scale_ori_means': h_scale_ori_means,
                'scale_ori_opacity': h_scale_ori_opacity,
            }
        )
        return forward_result, grad_result, hess_result


# ============================================================================
# Autograd Implementation (ground-truth reference)
# ============================================================================

class GS3DCTDebugAutograd(SecondForwardBackwardBase):
    """Pure autograd reference: ImageLoss(L1) + L2 volume loss.

    Uses double autograd for exact Hessian computation.
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, params: Params, config: Optional[Config] = None) -> None:
        for name in ['means', 'quats', 'scale_ori', 'opacity', 'opacity_vol']:
            if params.get(name) is None:
                raise ValueError(f"Missing required parameter: {name}")
        self.initialized = True

    def _forward_pipeline(
        self,
        means, quats, scale_ori, opacity, opacity_vol,
        viewmats, Ks, targets, vol_ranges, vol_centers, vol_target,
        img_dim1, img_dim0, eps_blur, lambda_l1,
        vol_dim2, vol_dim1, vol_dim0, eps3d,
        lambda_vol, scale_min, scale_max,
    ):
        """Full differentiable pipeline in pure PyTorch."""
        from .parameter_transform import _compute_covars
        from .world_to_image_projection import _compute_world_to_image_projection
        from .conics_mu import _compute_conics_mu
        from .world_to_volume_mapping import _compute_world_to_volume_mapping

        N = means.shape[0]
        C = viewmats.shape[0]
        device = means.device
        dtype = means.dtype

        # Stage 0: ParameterTransform
        covars = _compute_covars(quats, scale_ori, scale_min, scale_max)
        sig = torch.sigmoid(scale_ori)
        scales = sig * (scale_max - scale_min) + scale_min
        max_scales = scales.max(dim=-1).values

        # ============ Rasterize path ============
        means2d, cov2d_3x3 = _compute_world_to_image_projection(
            means, covars, viewmats, Ks, img_dim1, img_dim0
        )
        conics, mus_2d = _compute_conics_mu(cov2d_3x3, eps_blur)

        y_coords = torch.arange(img_dim0, device=device, dtype=dtype) + 0.5
        x_coords = torch.arange(img_dim1, device=device, dtype=dtype) + 0.5
        py, px = torch.meshgrid(y_coords, x_coords, indexing='ij')
        px = px.unsqueeze(0).unsqueeze(0)
        py = py.unsqueeze(0).unsqueeze(0)

        mean_x = means2d[..., 0].unsqueeze(-1).unsqueeze(-1)
        mean_y = means2d[..., 1].unsqueeze(-1).unsqueeze(-1)
        conic_a = conics[..., 0].unsqueeze(-1).unsqueeze(-1)
        conic_b = conics[..., 1].unsqueeze(-1).unsqueeze(-1)
        conic_c = conics[..., 2].unsqueeze(-1).unsqueeze(-1)
        opa = opacity.unsqueeze(-1).unsqueeze(-1)
        mu = mus_2d.unsqueeze(-1).unsqueeze(-1)

        dx = px - mean_x
        dy = py - mean_y
        sigma = 0.5 * (conic_a * dx * dx + conic_c * dy * dy) + conic_b * dx * dy
        vis = torch.exp(-sigma)
        alpha = opa * mu * vis
        mask = (sigma >= 0) & (alpha.abs() >= 1e-5)
        alpha_masked = torch.where(mask, alpha, torch.zeros_like(alpha))
        render_img = alpha_masked.sum(dim=1)

        img_loss = lambda_l1 * ((render_img - targets) ** 2).mean()

        # ============ Voxelize path ============
        means3d_v, conics_3d, radii = _compute_world_to_volume_mapping(
            means, covars, max_scales, vol_ranges, vol_centers,
            vol_dim2, vol_dim1, vol_dim0, eps3d,
        )

        z_c = torch.arange(vol_dim0, device=device, dtype=dtype) + 0.5
        y_c = torch.arange(vol_dim1, device=device, dtype=dtype) + 0.5
        x_c = torch.arange(vol_dim2, device=device, dtype=dtype) + 0.5
        zz, yy, xx = torch.meshgrid(z_c, y_c, x_c, indexing='ij')
        voxel_coords = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=-1)

        delta = voxel_coords.unsqueeze(1) - means3d_v.unsqueeze(0)
        d_x, d_y, d_z = delta[..., 0], delta[..., 1], delta[..., 2]
        a = conics_3d[:, 0].unsqueeze(0)
        p = conics_3d[:, 1].unsqueeze(0)
        q = conics_3d[:, 2].unsqueeze(0)
        b = conics_3d[:, 3].unsqueeze(0)
        r = conics_3d[:, 4].unsqueeze(0)
        c = conics_3d[:, 5].unsqueeze(0)

        sig3d = 0.5 * (a*d_x*d_x + b*d_y*d_y + c*d_z*d_z) + p*d_x*d_y + q*d_x*d_z + r*d_y*d_z
        G = opacity_vol.unsqueeze(0) * torch.exp(-sig3d)
        mask3d = (sig3d >= 0) & (G.abs() >= 1e-5)
        G_masked = torch.where(mask3d, G, torch.zeros_like(G))
        render_vol = G_masked.sum(dim=1).reshape(vol_dim0, vol_dim1, vol_dim2)

        vol_loss = lambda_vol * ((render_vol - vol_target) ** 2).mean()

        loss = img_loss + vol_loss
        return loss, render_img, render_vol

    def forward_only(self, params, data, config=None):
        e = self._unpack(params, data, config)
        with torch.no_grad():
            loss, ri, rv = self._forward_pipeline(*e)
        return ForwardResult(
            outputs={'loss': loss, 'render_image': ri, 'render_volume': rv},
            intermediates={}
        )

    def forward_backward1(self, params, data, config=None):
        e = self._unpack(params, data, config, requires_grad=True)
        means, quats, scale_ori, opacity, opacity_vol = e[:5]
        loss, ri, rv = self._forward_pipeline(*e)
        loss.backward()
        fwd = ForwardResult(
            outputs={'loss': loss.detach(), 'render_image': ri.detach(), 'render_volume': rv.detach()},
            intermediates={}
        )
        grad = GradResult(param_grads={
            'means': means.grad.detach(), 'quats': quats.grad.detach(),
            'scale_ori': scale_ori.grad.detach(), 'opacity': opacity.grad.detach(),
            'opacity_vol': opacity_vol.grad.detach(),
        })
        return fwd, grad

    def forward_backward2(self, params, data, config=None):
        e = self._unpack(params, data, config, requires_grad=True)
        means, quats, scale_ori, opacity, opacity_vol = e[:5]
        N = means.shape[0]
        C = opacity.shape[0]
        dtype = means.dtype
        device = means.device

        loss, ri, rv = self._forward_pipeline(*e)

        all_params = [means, quats, scale_ori, opacity, opacity_vol]
        v_means, v_quats, v_scale_ori, v_opacity, v_opacity_vol = torch.autograd.grad(
            loss, all_params, create_graph=True, retain_graph=True,
        )

        # Per-Gaussian Hessian blocks (same structure as GS3DProjectionOnlyAutograd)
        h_pp = torch.zeros(N, 3, 3, dtype=dtype, device=device)
        h_qq = torch.zeros(N, 4, 4, dtype=dtype, device=device)
        h_ss = torch.zeros(N, 3, 3, dtype=dtype, device=device)
        h_oo_rast = torch.zeros(C, N, dtype=dtype, device=device)
        h_oo_vol = torch.zeros(N, dtype=dtype, device=device)

        h_pq = torch.zeros(N, 3, 4, dtype=dtype, device=device)
        h_ps = torch.zeros(N, 3, 3, dtype=dtype, device=device)
        h_po_rast = torch.zeros(C, N, 3, dtype=dtype, device=device)
        h_po_vol = torch.zeros(N, 3, dtype=dtype, device=device)
        h_qs = torch.zeros(N, 4, 3, dtype=dtype, device=device)
        h_qo = torch.zeros(N, 4, dtype=dtype, device=device)
        h_so = torch.zeros(N, 3, dtype=dtype, device=device)

        for g in range(N):
            for i in range(3):
                grads = torch.autograd.grad(
                    v_means[g, i], all_params,
                    retain_graph=True, allow_unused=True,
                )
                g_p = grads[0] if grads[0] is not None else torch.zeros_like(means)
                g_q = grads[1] if grads[1] is not None else torch.zeros_like(quats)
                g_s = grads[2] if grads[2] is not None else torch.zeros_like(scale_ori)
                g_o = grads[3] if grads[3] is not None else torch.zeros_like(opacity)
                g_ov = grads[4] if grads[4] is not None else torch.zeros_like(opacity_vol)
                h_pp[g, i, :] = g_p[g, :]
                h_pq[g, i, :] = g_q[g, :]
                h_ps[g, i, :] = g_s[g, :]
                h_po_rast[:, g, i] = g_o[:, g]
                h_po_vol[g, i] = g_ov[g]

            for i in range(4):
                grads = torch.autograd.grad(
                    v_quats[g, i], [quats, scale_ori, opacity, opacity_vol],
                    retain_graph=True, allow_unused=True,
                )
                g_q = grads[0] if grads[0] is not None else torch.zeros_like(quats)
                g_s = grads[1] if grads[1] is not None else torch.zeros_like(scale_ori)
                g_o = grads[2] if grads[2] is not None else torch.zeros_like(opacity)
                g_ov = grads[3] if grads[3] is not None else torch.zeros_like(opacity_vol)
                h_qq[g, i, :] = g_q[g, :]
                h_qs[g, i, :] = g_s[g, :]
                h_qo[g, i] = g_o[:, g].sum() + g_ov[g]

            for i in range(3):
                grads = torch.autograd.grad(
                    v_scale_ori[g, i], [scale_ori, opacity, opacity_vol],
                    retain_graph=True, allow_unused=True,
                )
                g_s = grads[0] if grads[0] is not None else torch.zeros_like(scale_ori)
                g_o = grads[1] if grads[1] is not None else torch.zeros_like(opacity)
                g_ov = grads[2] if grads[2] is not None else torch.zeros_like(opacity_vol)
                h_ss[g, i, :] = g_s[g, :]
                h_so[g, i] = g_o[:, g].sum() + g_ov[g]

            for c_idx in range(C):
                grads = torch.autograd.grad(
                    v_opacity[c_idx, g], [opacity],
                    retain_graph=True, allow_unused=True,
                )
                g_o = grads[0] if grads[0] is not None else torch.zeros_like(opacity)
                h_oo_rast[c_idx, g] = g_o[c_idx, g]

            grads = torch.autograd.grad(
                v_opacity_vol[g], [opacity_vol],
                retain_graph=True, allow_unused=True,
            )
            g_ov = grads[0] if grads[0] is not None else torch.zeros_like(opacity_vol)
            h_oo_vol[g] = g_ov[g]

        # Pack into standard format
        h_means_out = torch.zeros(N, 6, dtype=dtype, device=device)
        idx = 0
        for i in range(3):
            for j in range(i, 3):
                h_means_out[:, idx] = h_pp[:, i, j]
                idx += 1

        h_quats_out = torch.zeros(N, 10, dtype=dtype, device=device)
        idx = 0
        for i in range(4):
            for j in range(i, 4):
                h_quats_out[:, idx] = h_qq[:, i, j]
                idx += 1

        h_scale_ori_out = torch.stack([h_ss[:, i, i] for i in range(3)], dim=-1)
        h_quats_scale_ori_out = h_qs.reshape(N, 12)
        h_quats_means_out = h_pq.transpose(-1, -2).reshape(N, 12)
        h_scale_ori_means_out = h_ps.transpose(-1, -2).reshape(N, 9)

        fwd = ForwardResult(
            outputs={'loss': loss.detach(), 'render_image': ri.detach(), 'render_volume': rv.detach()},
            intermediates={}
        )
        grad_r = GradResult(param_grads={
            'means': v_means.detach(), 'quats': v_quats.detach(),
            'scale_ori': v_scale_ori.detach(), 'opacity': v_opacity.detach(),
            'opacity_vol': v_opacity_vol.detach(),
        })
        hess_r = HessResult(
            param_hess={
                'means': h_means_out.detach(),
                'quats': h_quats_out.detach(),
                'scale_ori': h_scale_ori_out.detach(),
                'opacity': h_oo_rast.detach(),
                'opacity_vol': h_oo_vol.detach(),
            },
            cross_hess={
                'quats_scale_ori': h_quats_scale_ori_out.detach(),
                'means_opacity_rast': h_po_rast.detach(),
                'means_opacity_vol': h_po_vol.detach(),
                'quats_means': h_quats_means_out.detach(),
                'quats_opacity': h_qo.detach(),
                'scale_ori_means': h_scale_ori_means_out.detach(),
                'scale_ori_opacity': h_so.detach(),
            }
        )
        return fwd, grad_r, hess_r

    def _unpack(self, params, data, config, requires_grad=False):
        means = params.get('means').data
        quats = params.get('quats').data
        scale_ori = params.get('scale_ori').data
        opacity = params.get('opacity').data
        opacity_vol = params.get('opacity_vol').data

        if requires_grad:
            means = means.detach().clone().requires_grad_(True)
            quats = quats.detach().clone().requires_grad_(True)
            scale_ori = scale_ori.detach().clone().requires_grad_(True)
            opacity = opacity.detach().clone().requires_grad_(True)
            opacity_vol = opacity_vol.detach().clone().requires_grad_(True)

        viewmats = data.get('viewmats')
        Ks = data.get('Ks')
        targets = data.get('targets')
        vol_ranges = data.get('vol_ranges')
        vol_centers = data.get('vol_centers')
        vol_target = data.get('vol_target')

        img_dim1 = config.get('img_dim1')
        img_dim0 = config.get('img_dim0')
        eps_blur = config.get('eps_blur', 0.3)
        lambda_l1 = config.get('lambda_l1', 1.0)

        vol_dim2 = config.get('vol_dim2')
        vol_dim1 = config.get('vol_dim1')
        vol_dim0 = config.get('vol_dim0')
        eps3d = config.get('eps3d', 1e-4)
        lambda_vol = config.get('lambda_vol', 1.0)

        scale_min = config.get('scale_min', 0.01)
        scale_max = config.get('scale_max', 2.0)

        return (means, quats, scale_ori, opacity, opacity_vol,
                viewmats, Ks, targets, vol_ranges, vol_centers, vol_target,
                img_dim1, img_dim0, eps_blur, lambda_l1,
                vol_dim2, vol_dim1, vol_dim0, eps3d,
                lambda_vol, scale_min, scale_max)


__all__ = ["GS3DCTDebugCUDA", "GS3DCTDebugAutograd"]
