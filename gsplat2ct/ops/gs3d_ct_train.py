"""
GS3DCTLossCUDA: Full CT loss operator (end-to-end forward/backward chain).

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
    - opacity:   (N,)     per-Gaussian opacity (shared across cameras and paths)

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
    - lambda_l2 (or legacy lambda_l1): MSE image term weight (default 1.0)
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
from .data import (
    Params,
    Data,
    Config,
    ForwardResult,
    GradResult,
    HessResult,
    get_lambda_l2,
)
from .loss import l1_loss, l2_loss, ssim, tv_3d_loss, ImageLossTorch, VolumeLossTorch


class _ParamWrapper:
    """Simple wrapper to satisfy ParameterLike protocol."""
    def __init__(self, data: torch.Tensor):
        self.data = data


class GS3DCTLossCUDA(SecondForwardBackwardBase):
    """Full-pipeline loss operator combining rasterization + TV voxelization.

    Image loss mode is configurable:
      - "l2": L2 + SSIM (default)
      - "l1": L1 + SSIM
    """

    def __init__(self, image_loss_type: Optional[str] = None):
        self.initialized = False
        self._default_image_loss_type = image_loss_type

    # ------------------------------------------------------------------
    # Image-loss hooks (override in subclasses if needed)
    # ------------------------------------------------------------------
    def _image_loss_fwd(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        window_size: int,
        lambda_l1: float,
        lambda_ssim: float,
        compute_statistics: bool,
        image_loss_type: str,
    ):
        if image_loss_type == "l1":
            from ..cuda._wrapper import image_loss_l1_fwd as _fwd
        else:
            from ..cuda._wrapper import image_loss_l2_fwd as _fwd
        return _fwd(
            prediction, target, window_size,
            lambda_l1, lambda_ssim, compute_statistics,
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
        lambda_l1: float,
        lambda_ssim: float,
        image_loss_type: str,
    ):
        if image_loss_type == "l1":
            from ..cuda._wrapper import image_loss_l1_bwd as _bwd
        else:
            from ..cuda._wrapper import image_loss_l2_bwd as _bwd
        return _bwd(
            prediction, target,
            mu_r, mu_m, f1, f2, f3, f4,
            grad_output, window_size,
            lambda_l1, lambda_ssim,
        )

    def _image_loss_bwd2(
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
        lambda_l1: float,
        lambda_ssim: float,
        image_loss_type: str,
    ):
        if image_loss_type == "l1":
            from ..cuda._wrapper import image_loss_l1_bwd2 as _bwd2
        else:
            from ..cuda._wrapper import image_loss_l2_bwd2 as _bwd2
        return _bwd2(
            prediction, target,
            mu_r, mu_m, f1, f2, f3, f4,
            grad_output, window_size,
            lambda_l1, lambda_ssim,
        )

    def _resolve_image_loss_type(self, config: Optional[Config]) -> str:
        mode = self._default_image_loss_type
        if mode is None and config is not None:
            mode = config.get("image_loss_type", None)
        if mode is None:
            mode = "l2"
        mode = str(mode).lower()
        if mode not in ("l1", "l2"):
            raise ValueError(f"image_loss_type must be 'l1' or 'l2', got: {mode}")
        return mode

    def initialize(self, params: Params, config: Optional[Config] = None) -> None:
        for name in ['means', 'quats', 'scale_ori', 'opacity']:
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
        opacity = params.get('opacity').data       # (N,)

        viewmats = data.get('viewmats')
        Ks = data.get('Ks')
        targets = data.get('targets')
        vol_ranges = data.get('vol_ranges')
        vol_centers = data.get('vol_centers')

        img_dim1 = config.get('img_dim1')
        img_dim0 = config.get('img_dim0')
        img_tile_size = config.get('img_tile_size', 16)
        eps_blur = config.get('eps_blur', 0.3)
        # Image term weight (MSE): config key ``lambda_l2`` or legacy ``lambda_l1``
        lambda_l1 = get_lambda_l2(config, 1.0)
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

        image_loss_type = config.get('image_loss_type', None)

        return (means, quats, scale_ori, opacity,
                viewmats, Ks, targets, vol_ranges, vol_centers,
                img_dim1, img_dim0, img_tile_size, eps_blur,
                lambda_l1, lambda_ssim, window_size,
                vol_dim2, vol_dim1, vol_dim0, vol_tile_size,
                eps3d, radius_clip, vol_loss_eps, lambda_tv,
                scale_min, scale_max, image_loss_type)

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
            rasterize_to_pixels_fwd,
            world_to_volume_mapping_fwd,
            isect_tiles_3d, isect_offset_encode_3d,
            voxelize_to_voxels_fwd, volume_loss_fwd,
        )

        (means, quats, scale_ori, opacity,
         viewmats, Ks, targets, vol_ranges, vol_centers,
         img_dim1, img_dim0, img_tile_size, eps_blur,
         lambda_l1, lambda_ssim, window_size,
         vol_dim2, vol_dim1, vol_dim0, vol_tile_size,
         eps3d, radius_clip, vol_loss_eps, lambda_tv,
         scale_min, scale_max, image_loss_type_cfg) = self._extract(params, data, config)
        image_loss_type = self._resolve_image_loss_type(config) if image_loss_type_cfg is None else str(image_loss_type_cfg).lower()

        C = viewmats.shape[0]
        N = means.shape[0]

        # Broadcast opacity (N,) -> (C, N) for rasterize path
        opacity_2d = opacity.unsqueeze(0).expand(C, -1).contiguous()

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
            means2d, conics_2d, opacity_2d, mus,
            tile_offsets_2d, flatten_ids_2d,
            img_dim1, img_dim0, img_tile_size
        )

        img_loss_result = self._image_loss_fwd(
            render_img, targets, window_size,
            lambda_l1, lambda_ssim,
            compute_statistics=False,
            image_loss_type=image_loss_type,
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
            means3d_v, conics_3d, opacity,
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
            world_to_volume_mapping_fwd, world_to_volume_mapping_bwd,
            isect_tiles_3d, isect_offset_encode_3d,
            voxelize_to_voxels_fwd, voxelize_to_voxels_bwd,
            volume_loss_fwd, volume_loss_bwd,
        )

        (means, quats, scale_ori, opacity,
         viewmats, Ks, targets, vol_ranges, vol_centers,
         img_dim1, img_dim0, img_tile_size, eps_blur,
         lambda_l1, lambda_ssim, window_size,
         vol_dim2, vol_dim1, vol_dim0, vol_tile_size,
         eps3d, radius_clip, vol_loss_eps, lambda_tv,
         scale_min, scale_max, image_loss_type_cfg) = self._extract(params, data, config)
        image_loss_type = self._resolve_image_loss_type(config) if image_loss_type_cfg is None else str(image_loss_type_cfg).lower()

        C = viewmats.shape[0]
        N = means.shape[0]

        # Broadcast opacity (N,) -> (C, N) for rasterize path
        opacity_2d = opacity.unsqueeze(0).expand(C, -1).contiguous()

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
            means2d, conics_2d, opacity_2d, mus,
            tile_offsets_2d, flatten_ids_2d,
            img_dim1, img_dim0, img_tile_size
        )

        img_loss_result = self._image_loss_fwd(
            render_img, targets, window_size,
            lambda_l1, lambda_ssim,
            compute_statistics=True,
            image_loss_type=image_loss_type,
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
            means3d_v, conics_3d, opacity,
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

        v_render_img = self._image_loss_bwd(
            render_img, targets,
            mu_r, mu_m, f1, f2, f3, f4,
            grad_one, window_size,
            lambda_l1, lambda_ssim,
            image_loss_type=image_loss_type,
        )

        absgrad = bool(config.get("absgrad", False)) if config is not None else False
        v_means2d_abs, v_means2d, v_conics_2d, v_opacity_rast, v_mus = rasterize_to_pixels_bwd(
            means2d, conics_2d, opacity_2d, mus,
            tile_offsets_2d, flatten_ids_2d, v_render_img,
            img_dim1, img_dim0, img_tile_size, absgrad
        )

        v_cov2d_3x3 = conics_mu_bwd(
            cov2d_3x3, eps_blur, conics_2d, mus, v_conics_2d, v_mus
        )

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

        v_means3d_v, v_conics_3d_vol, v_opacity_vol = voxelize_to_voxels_bwd(
            means3d_v, conics_3d, opacity,
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

        # Opacity: sum rasterize (C,N) over cameras, add voxelize (N,)
        v_opacity_total = v_opacity_rast.sum(dim=0) + v_opacity_vol

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
            intermediates={}
        )
        grad_result = GradResult(
            param_grads={
                'means': v_means_total,
                'quats': v_quats,
                'scale_ori': v_scale_ori,
                'opacity': v_opacity_total,
            },
            aux={
                'v_means2d': v_means2d,
                'v_means2d_abs': v_means2d_abs,
                'radii_2d': radii_2d,
            },
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
            world_to_volume_mapping_fwd, world_to_volume_mapping_bwd2,
            isect_tiles_3d, isect_offset_encode_3d,
            voxelize_to_voxels_fwd, voxelize_to_voxels_bwd2,
            volume_loss_fwd, volume_loss_bwd, volume_loss_bwd2,
        )

        (means, quats, scale_ori, opacity,
         viewmats, Ks, targets, vol_ranges, vol_centers,
         img_dim1, img_dim0, img_tile_size, eps_blur,
         lambda_l1, lambda_ssim, window_size,
         vol_dim2, vol_dim1, vol_dim0, vol_tile_size,
         eps3d, radius_clip, vol_loss_eps, lambda_tv,
         scale_min, scale_max, image_loss_type_cfg) = self._extract(params, data, config)
        image_loss_type = self._resolve_image_loss_type(config) if image_loss_type_cfg is None else str(image_loss_type_cfg).lower()

        C = viewmats.shape[0]
        N = means.shape[0]
        dtype = means.dtype
        device = means.device
        grad_one = torch.ones(1, device=device, dtype=dtype)

        # Broadcast opacity (N,) -> (C, N) for rasterize path
        opacity_2d = opacity.unsqueeze(0).expand(C, -1).contiguous()

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
            means2d, conics_2d, opacity_2d, mus,
            tile_offsets_2d, flatten_ids_2d,
            img_dim1, img_dim0, img_tile_size
        )

        img_loss_result = self._image_loss_fwd(
            render_img, targets, window_size,
            lambda_l1, lambda_ssim,
            compute_statistics=True,
            image_loss_type=image_loss_type,
        )
        img_loss = img_loss_result[0]
        mu_r, mu_m = img_loss_result[3], img_loss_result[4]
        f1, f2, f3, f4 = img_loss_result[5], img_loss_result[6], img_loss_result[7], img_loss_result[8]

        # ==============================================================
        # RASTERIZE PATH: backward2
        # ==============================================================

        v_render_img = self._image_loss_bwd(
            render_img, targets,
            mu_r, mu_m, f1, f2, f3, f4,
            grad_one, window_size, lambda_l1, lambda_ssim,
            image_loss_type=image_loss_type,
        )
        h_render_img = self._image_loss_bwd2(
            render_img, targets,
            mu_r, mu_m, f1, f2, f3, f4,
            grad_one, window_size, lambda_l1, lambda_ssim,
            image_loss_type=image_loss_type,
        )

        (v_means2d_r, v_conics_2d_r, v_opacity_rast, v_mus_r,
         h_means2d_r, h_conics_2d_r,
         h_mu_mean_r, h_mu_conic_r, h_mean_conic_r,
         h_mu_opacity_r, h_conic_opacity_r, h_mean_opacity_r,
         h_mus_r, h_opacity_rast) = rasterize_to_pixels_bwd2(
            means2d, conics_2d, opacity_2d, mus,
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
            means3d_v, conics_3d, opacity,
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

        v_render_vol_4d = volume_loss_bwd(
            render_vol_4d, grad_one, vol_loss_eps, lambda_tv
        )
        v_render_vol = v_render_vol_4d.squeeze(0)

        h_render_vol_4d = volume_loss_bwd2(
            render_vol_4d, grad_one, vol_loss_eps, lambda_tv
        )
        h_render_vol = h_render_vol_4d.squeeze(0)

        (v_means3d_v_vol, v_conics_3d_vol, v_opacity_vol,
         h_means3d_v_vol, h_conics_3d_vol, h_opacity_vol,
         h_op_mean_vol, h_op_conic_vol, h_mean_conic_vol) = voxelize_to_voxels_bwd2(
            means3d_v, conics_3d, opacity,
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

        # ==============================================================
        # MERGE at ParameterTransform output level
        # ==============================================================

        # First-order gradients
        v_means_total = v_means_rast + v_means_vol
        v_covars_total = v_covars_rast + v_covars_vol

        # Opacity gradient: sum rasterize (C,N) over cameras, add voxelize (N,)
        v_opacity_total = v_opacity_rast.sum(dim=0) + v_opacity_vol

        # Self-Hessians
        h_means_total = h_means_rast + h_means_vol                  # (N, 6)
        h_covars_total = h_covars_rast + h_covars_vol                # (N, 21)

        # Opacity self-Hessian: sum rasterize (C,N) over cameras, add voxelize (N,)
        h_opacity_total = h_opacity_rast.sum(dim=0) + h_opacity_vol  # (N,)

        # Cross-Hessian means-covars
        h_means_covars_total = h_means_covars_rast + h_means_covars_vol  # (N, 18)

        # Cross-Hessians with opacity: sum rasterize (C,N,...) over cameras, add voxelize (N,...)
        h_means_opacity_total = h_means_opacity_rast.sum(dim=0) + h_means_opacity_vol  # (N, 3)
        h_covars_opacity_total = h_covars_opacity_rast.sum(dim=0) + h_covars_opacity_vol  # (N, 6)

        # ==============================================================
        # ParameterTransform bwd2 (single pass)
        # ==============================================================

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
                'covars': covars.detach(),
                'max_scales': max_scales.detach(),
                'means2d': means2d.detach(),
                'cov2d_3x3': cov2d_3x3.detach(),
                'radii_2d': radii_2d.detach(),
            }
        )

        grad_result = GradResult(
            param_grads={
                'means': v_means_total,
                'quats': v_quats,
                'scale_ori': v_scale_ori,
                'opacity': v_opacity_total,
            },
            aux={
                'v_means2d': v_means2d_r,
                'v_means2d_abs': None,
                'radii_2d': radii_2d,
                'v_covars_total': v_covars_total,
                'h_covars_total': h_covars_total,
                'h_means_covars_total': h_means_covars_total,
                'h_covars_opacity_total': h_covars_opacity_total,
            },
        )

        hess_result = HessResult(
            param_hess={
                'means': h_means_total,                     # (N, 6)
                'quats': h_quats,                           # (N, 10)
                'scale_ori': h_scale_ori,                   # (N, 6)
                'opacity': h_opacity_total,                 # (N,)
            },
            cross_hess={
                'quats_scale_ori': h_quats_scale_ori,       # (N, 12)
                'means_opacity': h_means_opacity_total,     # (N, 3)
                'quats_means': h_quats_means,               # (N, 12)
                'quats_opacity': h_quats_opacity,           # (N, 4)
                'scale_ori_means': h_scale_ori_means,       # (N, 9)
                'scale_ori_opacity': h_scale_ori_opacity,   # (N, 3)
            }
        )

        return forward_result, grad_result, hess_result


class GS3DCTLossAutograd(SecondForwardBackwardBase):
    """Torch autograd reference for the full GS3D CT loss pipeline."""

    def __init__(self, image_loss_type: Optional[str] = None):
        self.initialized = False
        self._default_image_loss_type = image_loss_type

    def initialize(self, params: Params, config: Optional[Config] = None) -> None:
        for name in ["means", "quats", "scale_ori", "opacity"]:
            if params.get(name) is None:
                raise ValueError(f"Missing required parameter: {name}")
        self.initialized = True

    def _resolve_image_loss_type(self, config: Optional[Config]) -> str:
        mode = self._default_image_loss_type
        if mode is None and config is not None:
            mode = config.get("image_loss_type", None)
        if mode is None:
            mode = "l2"
        mode = str(mode).lower()
        if mode not in ("l1", "l2"):
            raise ValueError(f"image_loss_type must be 'l1' or 'l2', got: {mode}")
        return mode

    def _forward_pipeline(
        self,
        means, quats, scale_ori, opacity,
        viewmats, Ks, targets, vol_ranges, vol_centers,
        img_dim1, img_dim0, eps_blur, lambda_img, lambda_ssim, window_size,
        vol_dim2, vol_dim1, vol_dim0, eps3d, radius_clip, vol_loss_eps, lambda_tv,
        scale_min, scale_max, image_loss_type,
    ):
        from .parameter_transform import _compute_covars
        from .world_to_image_projection import _compute_world_to_image_projection
        from .conics_mu import _compute_conics_mu
        from .world_to_volume_mapping import _compute_world_to_volume_mapping

        covars = _compute_covars(quats, scale_ori, scale_min, scale_max)
        sig = torch.sigmoid(scale_ori)
        scales = sig * (scale_max - scale_min) + scale_min
        max_scales = scales.max(dim=-1).values

        # Rasterize path
        means2d, cov2d_3x3 = _compute_world_to_image_projection(
            means, covars, viewmats, Ks, img_dim1, img_dim0
        )
        conics, mus_2d = _compute_conics_mu(cov2d_3x3, eps_blur)

        device = means.device
        dtype = means.dtype
        y_coords = torch.arange(img_dim0, device=device, dtype=dtype) + 0.5
        x_coords = torch.arange(img_dim1, device=device, dtype=dtype) + 0.5
        py, px = torch.meshgrid(y_coords, x_coords, indexing="ij")
        px = px.unsqueeze(0).unsqueeze(0)
        py = py.unsqueeze(0).unsqueeze(0)

        mean_x = means2d[..., 0].unsqueeze(-1).unsqueeze(-1)
        mean_y = means2d[..., 1].unsqueeze(-1).unsqueeze(-1)
        conic_a = conics[..., 0].unsqueeze(-1).unsqueeze(-1)
        conic_b = conics[..., 1].unsqueeze(-1).unsqueeze(-1)
        conic_c = conics[..., 2].unsqueeze(-1).unsqueeze(-1)
        opa = opacity.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        mu = mus_2d.unsqueeze(-1).unsqueeze(-1)
        dx = px - mean_x
        dy = py - mean_y
        sigma = 0.5 * (conic_a * dx * dx + conic_c * dy * dy) + conic_b * dx * dy
        vis = torch.exp(-sigma)
        alpha = opa * mu * vis
        mask = (sigma >= 0) & (alpha.abs() >= 1e-5)
        render_img = torch.where(mask, alpha, torch.zeros_like(alpha)).sum(dim=1)

        if image_loss_type == "l1":
            img_base = l1_loss(render_img, targets)
        else:
            img_base = l2_loss(render_img, targets)
        img_ssim = 1.0 - ssim(render_img, targets, window_size=window_size)
        img_loss = float(lambda_img) * img_base + float(lambda_ssim) * img_ssim

        # Voxelize path
        means3d_v, conics_3d, _radii = _compute_world_to_volume_mapping(
            means, covars, max_scales, vol_ranges, vol_centers,
            vol_dim2, vol_dim1, vol_dim0, eps3d,
        )
        z_c = torch.arange(vol_dim0, device=device, dtype=dtype) + 0.5
        y_c = torch.arange(vol_dim1, device=device, dtype=dtype) + 0.5
        x_c = torch.arange(vol_dim2, device=device, dtype=dtype) + 0.5
        zz, yy, xx = torch.meshgrid(z_c, y_c, x_c, indexing="ij")
        voxel_coords = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=-1)
        delta = voxel_coords.unsqueeze(1) - means3d_v.unsqueeze(0)
        d_x, d_y, d_z = delta[..., 0], delta[..., 1], delta[..., 2]
        a = conics_3d[:, 0].unsqueeze(0)
        p = conics_3d[:, 1].unsqueeze(0)
        q = conics_3d[:, 2].unsqueeze(0)
        b = conics_3d[:, 3].unsqueeze(0)
        r = conics_3d[:, 4].unsqueeze(0)
        c = conics_3d[:, 5].unsqueeze(0)
        sig3d = 0.5 * (a * d_x * d_x + b * d_y * d_y + c * d_z * d_z) + p * d_x * d_y + q * d_x * d_z + r * d_y * d_z
        G = opacity.unsqueeze(0) * torch.exp(-sig3d)
        mask3d = (sig3d >= 0) & (G.abs() >= 1e-5)
        render_vol = torch.where(mask3d, G, torch.zeros_like(G)).sum(dim=1).reshape(vol_dim0, vol_dim1, vol_dim2)
        vol_loss = float(lambda_tv) * tv_3d_loss(render_vol.unsqueeze(0), eps=float(vol_loss_eps))

        loss = img_loss + vol_loss
        return loss, img_loss, vol_loss, render_img, render_vol

    def _unpack(self, params, data, config, requires_grad=False):
        means = params.get("means").data
        quats = params.get("quats").data
        scale_ori = params.get("scale_ori").data
        opacity = params.get("opacity").data
        if requires_grad:
            means = means.detach().clone().requires_grad_(True)
            quats = quats.detach().clone().requires_grad_(True)
            scale_ori = scale_ori.detach().clone().requires_grad_(True)
            opacity = opacity.detach().clone().requires_grad_(True)
        return (
            means, quats, scale_ori, opacity,
            data.get("viewmats"), data.get("Ks"), data.get("targets"),
            data.get("vol_ranges"), data.get("vol_centers"),
            config.get("img_dim1"), config.get("img_dim0"), config.get("eps_blur", 0.3),
            get_lambda_l2(config, 1.0), config.get("lambda_ssim", 0.0), config.get("window_size", 11),
            config.get("vol_dim2"), config.get("vol_dim1"), config.get("vol_dim0"),
            config.get("eps3d", 1e-4), config.get("radius_clip", 3.0),
            config.get("vol_loss_eps", 1e-4), config.get("lambda_tv", 1.0),
            config.get("scale_min", 0.01), config.get("scale_max", 2.0),
            self._resolve_image_loss_type(config),
        )

    def forward_only(self, params, data, config=None):
        with torch.no_grad():
            loss, img_loss, vol_loss, ri, rv = self._forward_pipeline(*self._unpack(params, data, config))
        return ForwardResult(
            outputs={"loss": loss, "loss_image": img_loss, "loss_tv": vol_loss, "render_image": ri, "render_volume": rv},
            intermediates={},
        )

    def forward_backward1(self, params, data, config=None):
        e = self._unpack(params, data, config, requires_grad=True)
        means, quats, scale_ori, opacity = e[:4]
        loss, img_loss, vol_loss, ri, rv = self._forward_pipeline(*e)
        loss.backward()
        return (
            ForwardResult(outputs={"loss": loss.detach(), "loss_image": img_loss.detach(), "loss_tv": vol_loss.detach(), "render_image": ri.detach(), "render_volume": rv.detach()}, intermediates={}),
            GradResult(param_grads={"means": means.grad.detach(), "quats": quats.grad.detach(), "scale_ori": scale_ori.grad.detach(), "opacity": opacity.grad.detach()}, aux={}),
        )

    def forward_backward2(self, params, data, config=None):
        e = self._unpack(params, data, config, requires_grad=True)
        means, quats, scale_ori, opacity = e[:4]
        N = means.shape[0]
        dtype = means.dtype
        device = means.device
        (
            _means, _quats, _scale_ori, _opacity,
            _viewmats, _Ks, targets, _vol_ranges, _vol_centers,
            _img_dim1, _img_dim0, _eps_blur, lambda_img, lambda_ssim, window_size,
            _vol_dim2, _vol_dim1, _vol_dim0, _eps3d, _radius_clip, vol_loss_eps, lambda_tv,
            _scale_min, _scale_max, image_loss_type,
        ) = e
        loss, img_loss, vol_loss, ri, rv = self._forward_pipeline(*e)

        # Build a decoupled surrogate objective around current outputs so that
        # first-order gradients match exactly while second-order terms use
        # output-wise diagonal Hessians.
        target_img = targets
        n_img = float(ri.numel())
        if image_loss_type == "l1":
            grad_base = torch.sign(ri - target_img) / n_img
            hess_base = torch.zeros_like(ri)
        else:
            grad_base = 2.0 * (ri - target_img) / n_img
            hess_base = (2.0 / n_img) * torch.ones_like(ri)
        img_helper = ImageLossTorch(lambda_l2=1.0, lambda_ssim=1.0, window_size=int(window_size))
        grad_ssim = img_helper._compute_ssim_grad_by_window(ri, target_img, int(window_size))
        hess_ssim = img_helper._compute_ssim_hess_by_window(ri, target_img, int(window_size))
        v_img = float(lambda_img) * grad_base + float(lambda_ssim) * grad_ssim
        h_img = float(lambda_img) * hess_base + float(lambda_ssim) * hess_ssim

        vol4d = rv.unsqueeze(0)
        vol_helper = VolumeLossTorch(eps=float(vol_loss_eps))
        v_tv = vol_helper._compute_smooth_tv_grad_manual(vol4d, float(vol_loss_eps))
        h_tv = vol_helper._compute_smooth_tv_hess_manual(vol4d, float(vol_loss_eps))

        ri0 = ri.detach()
        rv0 = vol4d.detach()
        surrogate = (v_img.detach() * (ri - ri0)).sum()
        surrogate = surrogate + 0.5 * (h_img.detach() * (ri - ri0) ** 2).sum()
        surrogate = surrogate + float(lambda_tv) * (v_tv.detach() * (vol4d - rv0)).sum()
        surrogate = surrogate + 0.5 * float(lambda_tv) * (h_tv.detach() * (vol4d - rv0) ** 2).sum()

        all_params = [means, quats, scale_ori, opacity]
        v_means, v_quats, v_scale_ori, v_opacity = torch.autograd.grad(
            surrogate, all_params, create_graph=True, retain_graph=True
        )

        h_pp = torch.zeros(N, 3, 3, dtype=dtype, device=device)
        h_qq = torch.zeros(N, 4, 4, dtype=dtype, device=device)
        h_ss = torch.zeros(N, 3, 3, dtype=dtype, device=device)
        h_oo = torch.zeros(N, dtype=dtype, device=device)
        h_pq = torch.zeros(N, 3, 4, dtype=dtype, device=device)
        h_ps = torch.zeros(N, 3, 3, dtype=dtype, device=device)
        h_po = torch.zeros(N, 3, dtype=dtype, device=device)
        h_qs = torch.zeros(N, 4, 3, dtype=dtype, device=device)
        h_qo = torch.zeros(N, 4, dtype=dtype, device=device)
        h_so = torch.zeros(N, 3, dtype=dtype, device=device)

        for g in range(N):
            for i in range(3):
                grads = torch.autograd.grad(v_means[g, i], all_params, retain_graph=True, allow_unused=True)
                g_p = grads[0] if grads[0] is not None else torch.zeros_like(means)
                g_q = grads[1] if grads[1] is not None else torch.zeros_like(quats)
                g_s = grads[2] if grads[2] is not None else torch.zeros_like(scale_ori)
                g_o = grads[3] if grads[3] is not None else torch.zeros_like(opacity)
                h_pp[g, i, :] = g_p[g, :]
                h_pq[g, i, :] = g_q[g, :]
                h_ps[g, i, :] = g_s[g, :]
                h_po[g, i] = g_o[g]
            for i in range(4):
                grads = torch.autograd.grad(v_quats[g, i], [quats, scale_ori, opacity], retain_graph=True, allow_unused=True)
                g_q = grads[0] if grads[0] is not None else torch.zeros_like(quats)
                g_s = grads[1] if grads[1] is not None else torch.zeros_like(scale_ori)
                g_o = grads[2] if grads[2] is not None else torch.zeros_like(opacity)
                h_qq[g, i, :] = g_q[g, :]
                h_qs[g, i, :] = g_s[g, :]
                h_qo[g, i] = g_o[g]
            for i in range(3):
                grads = torch.autograd.grad(v_scale_ori[g, i], [scale_ori, opacity], retain_graph=True, allow_unused=True)
                g_s = grads[0] if grads[0] is not None else torch.zeros_like(scale_ori)
                g_o = grads[1] if grads[1] is not None else torch.zeros_like(opacity)
                h_ss[g, i, :] = g_s[g, :]
                h_so[g, i] = g_o[g]
            grads = torch.autograd.grad(v_opacity[g], [opacity], retain_graph=True, allow_unused=True)
            g_o = grads[0] if grads[0] is not None else torch.zeros_like(opacity)
            h_oo[g] = g_o[g]

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
        h_scale_ori_out = torch.stack([
            h_ss[:, 0, 0],
            h_ss[:, 0, 1],
            h_ss[:, 0, 2],
            h_ss[:, 1, 1],
            h_ss[:, 1, 2],
            h_ss[:, 2, 2],
        ], dim=-1)
        h_quats_scale_ori_out = h_qs.reshape(N, 12)
        h_quats_means_out = h_pq.transpose(-1, -2).reshape(N, 12)
        h_scale_ori_means_out = h_ps.transpose(-1, -2).reshape(N, 9)

        fwd = ForwardResult(outputs={"loss": loss.detach(), "loss_image": img_loss.detach(), "loss_tv": vol_loss.detach(), "render_image": ri.detach(), "render_volume": rv.detach()}, intermediates={})
        grad_r = GradResult(param_grads={"means": v_means.detach(), "quats": v_quats.detach(), "scale_ori": v_scale_ori.detach(), "opacity": v_opacity.detach()}, aux={})
        hess_r = HessResult(
            param_hess={"means": h_means_out.detach(), "quats": h_quats_out.detach(), "scale_ori": h_scale_ori_out.detach(), "opacity": h_oo.detach()},
            cross_hess={
                "quats_scale_ori": h_quats_scale_ori_out.detach(),
                "means_opacity": h_po.detach(),
                "quats_means": h_quats_means_out.detach(),
                "quats_opacity": h_qo.detach(),
                "scale_ori_means": h_scale_ori_means_out.detach(),
                "scale_ori_opacity": h_so.detach(),
            },
        )
        return fwd, grad_r, hess_r


__all__ = ["GS3DCTLossCUDA", "GS3DCTLossAutograd"]
