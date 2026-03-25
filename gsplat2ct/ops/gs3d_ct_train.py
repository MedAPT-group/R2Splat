"""
GS3DCTTrainCUDA: Main training operator for 3DGS-based CT reconstruction.

Combines two paths from shared ParameterTransform output (Scheme 3):

  Rasterize path (image fitting):
      φ -> WorldToImageProjection -> ConicsMu -> RasterizeToPixels -> ImageLoss

  Voxelize path (TV regularization):
      φ -> WorldToVolumeMapping -> VoxelizeToVoxels -> VolumeLoss(TV)

  Shared:
      θ = (means, quats, scale_ori, opacity)
          -> [ParameterTransform] -> φ = (covars, max_scales)

For backward2 we use Scheme 3: both sub-pipelines compute gradients and
Hessians back to the ParameterTransform output (means, covars, opacity),
sum them there, then run a single ParameterTransform backward2 pass to
obtain final gradients and Hessians w.r.t. the raw parameters θ.

Expected params:
    - means:     (N, 3)   world-space Gaussian centres (shared across cameras)
    - quats:     (N, 4)   unit quaternions (shared)
    - scale_ori: (N, 3)   raw scale parameters (shared)
    - opacity:   (C, N)   per-Gaussian per-camera opacity for rasterize path
    - opacity_vol: (N,)   per-Gaussian opacity for voxelize path

Expected data:
    - viewmats:  (C, 4, 4)  camera extrinsics
    - Ks:        (C, 3, 3)  camera intrinsics
    - targets:   (C, H, W)  ground-truth projections
    - vol_ranges:  (3,)     physical extent per axis for voxelization
    - vol_centers: (3,)     physical center per axis for voxelization

Expected config (rasterize path):
    - img_dim1, img_dim0:  image size
    - img_tile_size:       (default 16)
    - eps_blur:            ConicsMu / W2I blur epsilon (default 0.3)
    - lambda_l1:           L1 loss weight (default 1.0)
    - lambda_ssim:         SSIM loss weight (default 0.0)
    - window_size:         SSIM window size (default 11)

Expected config (voxelize path):
    - vol_dim2, vol_dim1, vol_dim0:  voxel grid size
    - vol_tile_size:                 (default 4)
    - eps3d:                         W2V matrix inverse eps (default 1e-4)
    - radius_clip:                   (default 3.0)
    - vol_loss_eps:                  TV smooth eps (default 1e-4)
    - lambda_tv:                     TV loss weight (default 1.0)

Expected config (shared):
    - scale_min, scale_max:  ParameterTransform scale bounds

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


class GS3DCTTrainCUDA(SecondForwardBackwardBase):
    """Main training operator combining rasterization + TV voxelization."""

    def __init__(self):
        self.initialized = False

    def initialize(self, params: Params, config: Optional[Config] = None) -> None:
        for name in ['means', 'quats', 'scale_ori', 'opacity', 'opacity_vol']:
            if params.get(name) is None:
                raise ValueError(f"Missing required parameter: {name}")
        means = params.get('means').data
        if not means.is_cuda:
            raise ValueError("Parameters must be CUDA tensors")
        self.initialized = True

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract(params, data, config):
        """Extract all inputs from params/data/config."""
        means = params.get('means').data
        quats = params.get('quats').data
        scale_ori = params.get('scale_ori').data
        opacity = params.get('opacity').data       # (C, N) for rasterize
        opacity_vol = params.get('opacity_vol').data  # (N,) for voxelize

        viewmats = data.get('viewmats')
        Ks = data.get('Ks')
        targets = data.get('targets')
        vol_ranges = data.get('vol_ranges')
        vol_centers = data.get('vol_centers')

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
        vol_loss_eps = config.get('vol_loss_eps', 1e-4)
        lambda_tv = config.get('lambda_tv', 1.0)

        scale_min = config.get('scale_min', 0.01)
        scale_max = config.get('scale_max', 2.0)

        return (means, quats, scale_ori, opacity, opacity_vol,
                viewmats, Ks, targets, vol_ranges, vol_centers,
                img_dim1, img_dim0, img_tile_size, eps_blur,
                lambda_l1, lambda_ssim, window_size,
                vol_dim2, vol_dim1, vol_dim0, vol_tile_size,
                eps3d, radius_clip, vol_loss_eps, lambda_tv,
                scale_min, scale_max)

    # ------------------------------------------------------------------
    # forward_only
    # ------------------------------------------------------------------

    def forward_only(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> ForwardResult:
        from ..cuda._wrapper import (
            parameter_transform_fwd,
            world_to_image_projection_fwd, conics_mu_fwd,
            isect_tiles, isect_offset_encode,
            rasterize_to_pixels_fwd, image_loss_fwd,
            world_to_volume_mapping_fwd,
            isect_tiles_3d, isect_offset_encode_3d,
            voxelize_to_voxels_fwd, volume_loss_fwd,
        )

        (means, quats, scale_ori, opacity, opacity_vol,
         viewmats, Ks, targets, vol_ranges, vol_centers,
         img_dim1, img_dim0, img_tile_size, eps_blur,
         lambda_l1, lambda_ssim, window_size,
         vol_dim2, vol_dim1, vol_dim0, vol_tile_size,
         eps3d, radius_clip, vol_loss_eps, lambda_tv,
         scale_min, scale_max) = self._extract(params, data, config)

        C = viewmats.shape[0]
        N = means.shape[0]

        # === Shared: ParameterTransform ===
        covars, max_scales = parameter_transform_fwd(
            quats, scale_ori, scale_min, scale_max
        )

        # === Rasterize path: forward ===
        means2d, cov2d_3x3, radii_2d = world_to_image_projection_fwd(
            means, covars, viewmats, Ks, img_dim1, img_dim0
        )
        conics_2d, mus = conics_mu_fwd(cov2d_3x3, eps_blur)

        n_tiles_h = (img_dim0 + img_tile_size - 1) // img_tile_size
        n_tiles_w = (img_dim1 + img_tile_size - 1) // img_tile_size
        _, isect_ids_2d, flatten_ids_2d = isect_tiles(
            means2d, radii_2d, img_tile_size, n_tiles_w, n_tiles_h
        )
        tile_offsets_2d = isect_offset_encode(
            isect_ids_2d, C, n_tiles_w, n_tiles_h
        )

        render_img = rasterize_to_pixels_fwd(
            means2d, conics_2d, opacity, mus,
            tile_offsets_2d, flatten_ids_2d,
            img_dim1, img_dim0, img_tile_size
        )

        img_loss_result = image_loss_fwd(
            render_img, targets, window_size,
            lambda_l1, lambda_ssim,
            compute_statistics=False
        )
        img_loss = img_loss_result[0]

        # === Voxelize path: forward ===
        means3d_v, conics_3d, radii_3d = world_to_volume_mapping_fwd(
            means, covars, max_scales, vol_ranges, vol_centers,
            vol_dim2, vol_dim1, vol_dim0, eps3d, radius_clip
        )

        n_tiles_vz = (vol_dim2 + vol_tile_size - 1) // vol_tile_size
        n_tiles_vy = (vol_dim1 + vol_tile_size - 1) // vol_tile_size
        n_tiles_vx = (vol_dim0 + vol_tile_size - 1) // vol_tile_size
        _, isect_ids_3d, flatten_ids_3d = isect_tiles_3d(
            means3d_v, radii_3d, vol_tile_size,
            n_tiles_vz, n_tiles_vy, n_tiles_vx
        )
        tile_offsets_3d = isect_offset_encode_3d(
            isect_ids_3d, n_tiles_vz, n_tiles_vy, n_tiles_vx
        )

        render_vol = voxelize_to_voxels_fwd(
            means3d_v, conics_3d, opacity_vol,
            tile_offsets_3d, flatten_ids_3d,
            vol_dim2, vol_dim1, vol_dim0, vol_tile_size
        )

        render_vol_4d = render_vol.unsqueeze(0)
        vol_loss_total, tv_loss, _tv_map = volume_loss_fwd(
            render_vol_4d, vol_loss_eps, lambda_tv
        )

        loss = img_loss + vol_loss_total

        return ForwardResult(
            outputs={
                'loss': loss,
                'loss_image': img_loss.detach(),
                'loss_tv': vol_loss_total.detach(),
                'render_image': render_img.detach(),
                'render_volume': render_vol.detach(),
            },
            intermediates={}
        )

    # ------------------------------------------------------------------
    # forward_backward1
    # ------------------------------------------------------------------

    def forward_backward1(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> Tuple[ForwardResult, GradResult]:
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
            volume_loss_fwd, volume_loss_bwd,
        )

        (means, quats, scale_ori, opacity, opacity_vol,
         viewmats, Ks, targets, vol_ranges, vol_centers,
         img_dim1, img_dim0, img_tile_size, eps_blur,
         lambda_l1, lambda_ssim, window_size,
         vol_dim2, vol_dim1, vol_dim0, vol_tile_size,
         eps3d, radius_clip, vol_loss_eps, lambda_tv,
         scale_min, scale_max) = self._extract(params, data, config)

        C = viewmats.shape[0]
        N = means.shape[0]

        # ============================================================
        # FORWARD: Shared ParameterTransform
        # ============================================================
        covars, max_scales = parameter_transform_fwd(
            quats, scale_ori, scale_min, scale_max
        )

        # ============================================================
        # FORWARD: Rasterize path
        # ============================================================
        means2d, cov2d_3x3, radii_2d = world_to_image_projection_fwd(
            means, covars, viewmats, Ks, img_dim1, img_dim0
        )
        conics_2d, mus = conics_mu_fwd(cov2d_3x3, eps_blur)

        n_tiles_h = (img_dim0 + img_tile_size - 1) // img_tile_size
        n_tiles_w = (img_dim1 + img_tile_size - 1) // img_tile_size
        _, isect_ids_2d, flatten_ids_2d = isect_tiles(
            means2d, radii_2d, img_tile_size, n_tiles_w, n_tiles_h
        )
        tile_offsets_2d = isect_offset_encode(
            isect_ids_2d, C, n_tiles_w, n_tiles_h
        )

        render_img = rasterize_to_pixels_fwd(
            means2d, conics_2d, opacity, mus,
            tile_offsets_2d, flatten_ids_2d,
            img_dim1, img_dim0, img_tile_size
        )

        img_loss_result = image_loss_fwd(
            render_img, targets, window_size,
            lambda_l1, lambda_ssim,
            compute_statistics=True
        )
        img_loss = img_loss_result[0]
        mu_r, mu_m = img_loss_result[3], img_loss_result[4]
        f1, f2, f3, f4 = img_loss_result[5], img_loss_result[6], img_loss_result[7], img_loss_result[8]

        # ============================================================
        # FORWARD: Voxelize path
        # ============================================================
        means3d_v, conics_3d, radii_3d = world_to_volume_mapping_fwd(
            means, covars, max_scales, vol_ranges, vol_centers,
            vol_dim2, vol_dim1, vol_dim0, eps3d, radius_clip
        )

        n_tiles_vz = (vol_dim2 + vol_tile_size - 1) // vol_tile_size
        n_tiles_vy = (vol_dim1 + vol_tile_size - 1) // vol_tile_size
        n_tiles_vx = (vol_dim0 + vol_tile_size - 1) // vol_tile_size
        _, isect_ids_3d, flatten_ids_3d = isect_tiles_3d(
            means3d_v, radii_3d, vol_tile_size,
            n_tiles_vz, n_tiles_vy, n_tiles_vx
        )
        tile_offsets_3d = isect_offset_encode_3d(
            isect_ids_3d, n_tiles_vz, n_tiles_vy, n_tiles_vx
        )

        render_vol = voxelize_to_voxels_fwd(
            means3d_v, conics_3d, opacity_vol,
            tile_offsets_3d, flatten_ids_3d,
            vol_dim2, vol_dim1, vol_dim0, vol_tile_size
        )

        render_vol_4d = render_vol.unsqueeze(0)
        vol_loss_total, tv_loss, _tv_map = volume_loss_fwd(
            render_vol_4d, vol_loss_eps, lambda_tv
        )

        loss = img_loss + vol_loss_total

        # ============================================================
        # BACKWARD: Rasterize path
        # ============================================================
        grad_one = torch.ones(1, device=means.device, dtype=means.dtype)

        # ImageLoss bwd
        v_render_img = image_loss_bwd(
            render_img, targets,
            mu_r, mu_m, f1, f2, f3, f4,
            grad_one, window_size,
            lambda_l1, lambda_ssim
        )

        # RasterizeToPixels bwd
        v_means2d, v_conics_2d, v_opacity_rast, v_mus = rasterize_to_pixels_bwd(
            means2d, conics_2d, opacity, mus,
            tile_offsets_2d, flatten_ids_2d, v_render_img,
            img_dim1, img_dim0, img_tile_size
        )

        # ConicsMu bwd
        v_cov2d_3x3 = conics_mu_bwd(
            cov2d_3x3, eps_blur, conics_2d, mus, v_conics_2d, v_mus
        )

        # WorldToImageProjection bwd
        v_means_rast, v_covars_rast = world_to_image_projection_bwd(
            means, covars, viewmats, Ks, img_dim1, img_dim0,
            v_means2d, v_cov2d_3x3,
        )

        # ============================================================
        # BACKWARD: Voxelize path
        # ============================================================
        v_render_vol_4d = volume_loss_bwd(
            render_vol_4d, grad_one, vol_loss_eps, lambda_tv
        )
        v_render_vol = v_render_vol_4d.squeeze(0)

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

        # ============================================================
        # MERGE at ParameterTransform output + bwd
        # ============================================================
        v_covars_total = v_covars_rast + v_covars_vol
        v_means_total = v_means_rast + v_means_vol

        v_quats, v_scale_ori = parameter_transform_bwd(
            quats, scale_ori, scale_min, scale_max, v_covars_total
        )

        forward_result = ForwardResult(
            outputs={
                'loss': loss,
                'loss_image': img_loss.detach(),
                'loss_tv': vol_loss_total.detach(),
                'render_image': render_img.detach(),
                'render_volume': render_vol.detach(),
            },
            intermediates={
                'v_means2d': v_means2d,
            }
        )
        grad_result = GradResult(
            param_grads={
                'means': v_means_total,
                'quats': v_quats,
                'scale_ori': v_scale_ori,
                'opacity': v_opacity_rast,
                'opacity_vol': v_opacity_vol_out,
            }
        )
        return forward_result, grad_result

    # ------------------------------------------------------------------
    # forward_backward2
    # ------------------------------------------------------------------

    def forward_backward2(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> Tuple[ForwardResult, GradResult, HessResult]:
        """Forward + second-order backward (Scheme 3).

        Both sub-pipelines backward to the ParameterTransform output
        level (means, covars, opacity). We sum gradients and Hessians
        there, then do a single ParameterTransform backward2 pass.
        """
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
            volume_loss_fwd, volume_loss_bwd, volume_loss_bwd2,
        )

        (means, quats, scale_ori, opacity, opacity_vol,
         viewmats, Ks, targets, vol_ranges, vol_centers,
         img_dim1, img_dim0, img_tile_size, eps_blur,
         lambda_l1, lambda_ssim, window_size,
         vol_dim2, vol_dim1, vol_dim0, vol_tile_size,
         eps3d, radius_clip, vol_loss_eps, lambda_tv,
         scale_min, scale_max) = self._extract(params, data, config)

        C = viewmats.shape[0]
        N = means.shape[0]
        dtype = means.dtype
        device = means.device
        grad_one = torch.ones(1, device=device, dtype=dtype)

        # ============================================================
        # FORWARD: Shared ParameterTransform
        # ============================================================
        covars, max_scales = parameter_transform_fwd(
            quats, scale_ori, scale_min, scale_max
        )

        # ==============================================================
        # RASTERIZE PATH: forward
        # ==============================================================
        means2d, cov2d_3x3, radii_2d = world_to_image_projection_fwd(
            means, covars, viewmats, Ks, img_dim1, img_dim0
        )
        conics_2d, mus = conics_mu_fwd(cov2d_3x3, eps_blur)

        n_tiles_h = (img_dim0 + img_tile_size - 1) // img_tile_size
        n_tiles_w = (img_dim1 + img_tile_size - 1) // img_tile_size
        _, isect_ids_2d, flatten_ids_2d = isect_tiles(
            means2d, radii_2d, img_tile_size, n_tiles_w, n_tiles_h
        )
        tile_offsets_2d = isect_offset_encode(
            isect_ids_2d, C, n_tiles_w, n_tiles_h
        )

        render_img = rasterize_to_pixels_fwd(
            means2d, conics_2d, opacity, mus,
            tile_offsets_2d, flatten_ids_2d,
            img_dim1, img_dim0, img_tile_size
        )

        img_loss_result = image_loss_fwd(
            render_img, targets, window_size,
            lambda_l1, lambda_ssim,
            compute_statistics=True
        )
        img_loss = img_loss_result[0]
        mu_r, mu_m = img_loss_result[3], img_loss_result[4]
        f1, f2, f3, f4 = img_loss_result[5], img_loss_result[6], img_loss_result[7], img_loss_result[8]

        # ==============================================================
        # RASTERIZE PATH: backward2
        # ==============================================================

        # ImageLoss bwd + bwd2
        v_render_img = image_loss_bwd(
            render_img, targets,
            mu_r, mu_m, f1, f2, f3, f4,
            grad_one, window_size, lambda_l1, lambda_ssim
        )
        h_render_img = image_loss_bwd2(
            render_img, targets,
            mu_r, mu_m, f1, f2, f3, f4,
            grad_one, window_size, lambda_l1, lambda_ssim
        )

        # RasterizeToPixels bwd2
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

        # ConicsMu bwd2
        h_means2d_conics_r = h_mean_conic_r.reshape(C, -1, 6)
        v_cov2d_3x3_r, h_cov_cov_r, h_cov_means2d_r, h_cov_opacity_r = conics_mu_bwd2(
            cov2d_3x3, eps_blur, conics_2d, mus,
            v_conics_2d_r, v_mus_r,
            h_conics_2d_r, h_mu_conic_r, h_mus_r,
            h_means2d_conics_r, h_conic_opacity_r, h_mu_mean_r, h_mu_opacity_r,
        )

        # WorldToImageProjection bwd2
        # h_cov_means2d_r is (C,N,12) in 6x2 row-major; need 2x6 for h_means2d_cov2d_3x3
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

        # ==============================================================
        # VOXELIZE PATH: forward
        # ==============================================================
        means3d_v, conics_3d, radii_3d = world_to_volume_mapping_fwd(
            means, covars, max_scales, vol_ranges, vol_centers,
            vol_dim2, vol_dim1, vol_dim0, eps3d, radius_clip
        )

        n_tiles_vz = (vol_dim2 + vol_tile_size - 1) // vol_tile_size
        n_tiles_vy = (vol_dim1 + vol_tile_size - 1) // vol_tile_size
        n_tiles_vx = (vol_dim0 + vol_tile_size - 1) // vol_tile_size
        _, isect_ids_3d, flatten_ids_3d = isect_tiles_3d(
            means3d_v, radii_3d, vol_tile_size,
            n_tiles_vz, n_tiles_vy, n_tiles_vx
        )
        tile_offsets_3d = isect_offset_encode_3d(
            isect_ids_3d, n_tiles_vz, n_tiles_vy, n_tiles_vx
        )

        render_vol = voxelize_to_voxels_fwd(
            means3d_v, conics_3d, opacity_vol,
            tile_offsets_3d, flatten_ids_3d,
            vol_dim2, vol_dim1, vol_dim0, vol_tile_size
        )

        render_vol_4d = render_vol.unsqueeze(0)
        vol_loss_total, tv_loss, _tv_map = volume_loss_fwd(
            render_vol_4d, vol_loss_eps, lambda_tv
        )

        loss = img_loss + vol_loss_total

        # ==============================================================
        # VOXELIZE PATH: backward2
        # ==============================================================

        # VolumeLoss bwd + bwd2
        v_render_vol_4d = volume_loss_bwd(
            render_vol_4d, grad_one, vol_loss_eps, lambda_tv
        )
        v_render_vol = v_render_vol_4d.squeeze(0)

        h_render_vol_4d = volume_loss_bwd2(
            render_vol_4d, grad_one, vol_loss_eps, lambda_tv
        )
        h_render_vol = h_render_vol_4d.squeeze(0)

        # VoxelizeToVoxels bwd2
        (v_means3d_v_vol, v_conics_3d_vol, v_opacity_vol_out,
         h_means3d_v_vol, h_conics_3d_vol, h_opacity_vol_out,
         h_op_mean_vol, h_op_conic_vol, h_mean_conic_vol) = voxelize_to_voxels_bwd2(
            means3d_v, conics_3d, opacity_vol,
            tile_offsets_3d, flatten_ids_3d,
            v_render_vol, h_render_vol,
            vol_dim2, vol_dim1, vol_dim0, vol_tile_size
        )

        # WorldToVolumeMapping bwd2
        (v_means_vol, v_covars_vol,
         h_means_vol, h_covars_vol, h_means_covars_vol,
         h_means_opacity_vol, h_covars_opacity_vol) = world_to_volume_mapping_bwd2(
            means, covars, vol_ranges, vol_centers,
            vol_dim2, vol_dim1, vol_dim0, eps3d,
            v_means3d_v_vol, v_conics_3d_vol,
            h_means3d_v_vol, h_conics_3d_vol, h_mean_conic_vol,
            h_op_mean_vol, h_op_conic_vol,
        )

        # ==============================================================
        # MERGE at ParameterTransform output level
        # ==============================================================

        # First-order gradients
        v_means_total = v_means_rast + v_means_vol
        v_covars_total = v_covars_rast + v_covars_vol

        # Self-Hessians
        h_means_total = h_means_rast + h_means_vol                  # (N, 6)
        h_covars_total = h_covars_rast + h_covars_vol                # (N, 21)

        # Cross-Hessian means-covars
        h_means_covars_total = h_means_covars_rast + h_means_covars_vol  # (N, 18)

        # Cross-Hessians with opacity:
        # Rasterize: opacity is (C, N), hess are (C, N, ...) — sum over C for ParameterTransform
        # Voxelize: opacity_vol is (N,), hess are (N, ...)
        h_covars_opacity_rast_summed = h_covars_opacity_rast.sum(dim=0)  # (N, 6)
        h_covars_opacity_total = h_covars_opacity_rast_summed + h_covars_opacity_vol  # (N, 6)

        # ==============================================================
        # ParameterTransform bwd2 (single pass)
        # ==============================================================

        # Transpose h_means_covars: 3x6 row-major -> 6x3 row-major
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

        # ==============================================================
        # Assemble results
        # ==============================================================

        forward_result = ForwardResult(
            outputs={
                'loss': loss,
                'loss_image': img_loss.detach(),
                'loss_tv': vol_loss_total.detach(),
                'render_image': render_img.detach(),
                'render_volume': render_vol.detach(),
            },
            intermediates={
                'v_means2d': v_means2d_r,
            }
        )

        grad_result = GradResult(
            param_grads={
                'means': v_means_total,
                'quats': v_quats,
                'scale_ori': v_scale_ori,
                'opacity': v_opacity_rast,
                'opacity_vol': v_opacity_vol_out,
            }
        )

        hess_result = HessResult(
            param_hess={
                'means': h_means_total,                     # (N, 6)
                'quats': h_quats,                           # (N, 10)
                'scale_ori': h_scale_ori,                   # (N, 3)
                'opacity': h_opacity_rast,                  # (C, N)
                'opacity_vol': h_opacity_vol_out,           # (N,)
            },
            cross_hess={
                'quats_scale_ori': h_quats_scale_ori,       # (N, 12)
                'means_opacity_rast': h_means_opacity_rast, # (C, N, 3)
                'means_opacity_vol': h_means_opacity_vol,   # (N, 3)
                'quats_means': h_quats_means,               # (N, 12)
                'quats_opacity': h_quats_opacity,           # (N, 4)
                'scale_ori_means': h_scale_ori_means,       # (N, 9)
                'scale_ori_opacity': h_scale_ori_opacity,   # (N, 3)
            }
        )

        return forward_result, grad_result, hess_result


__all__ = ["GS3DCTTrainCUDA"]
