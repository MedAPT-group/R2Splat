"""
3DGS Projection-Only Pipeline: End-to-end from raw Gaussian parameters to loss.

This module chains 5 operators into an end-to-end pipeline:
    ParameterTransform → WorldToImageProjection → ConicsMu
        → RasterizeToPixels → ImageLoss

Pipeline:
    (means, quats, scale_ori, opacity)
        → [ParameterTransform]          → covars
        → [WorldToImageProjection]      → (means2d, cov2d_3x3)
        → [ConicsMu]                    → (conics, mus)
        → [RasterizeToPixels]           → render_output
        → [ImageLoss]                   → loss (scalar)

Outer (optimizable) parameters:
    - means:     (N, 3)      world-space Gaussian centres (shared across C cameras)
    - quats:     (N, 4)      unit quaternions (w, x, y, z)
    - scale_ori: (N, 3)      raw scale parameters (before sigmoid)
    - opacity:   (C, N)      per-Gaussian per-camera opacity

Note: means, quats and scale_ori are SHARED across cameras (N-only),
      while opacity is per-camera (C, N).
      ParameterTransform produces covars (N, 6) which is directly
      passed to WorldToImageProjection.

Second-order backward computes:
    - Gradients for all 4 parameter groups
    - Self-Hessians for each parameter group
    - Cross-Hessians between all pairs

Hessian chain rule (backward through the chain):
    H_theta = J^T H_phi J + sum_k v_k Hess_k(theta)

Author: Dy164
Date: 2026-02-20
"""

import math
import torch
from typing import Optional, Tuple
from .base import SecondForwardBackwardBase
from .data import Params, Data, Config, ForwardResult, GradResult, HessResult


# ============================================================================
# Helper: ParamWrapper
# ============================================================================

class _ParamWrapper:
    """Simple wrapper to satisfy ParameterLike protocol."""
    def __init__(self, data: torch.Tensor):
        self.data = data


# ============================================================================
# CUDA Pipeline Implementation
# ============================================================================

class GS3DProjectionOnlyCUDA(SecondForwardBackwardBase):
    """
    CUDA-accelerated end-to-end pipeline:
        ParameterTransform → WorldToImageProjection → ConicsMu
            → RasterizeToPixels → ImageLoss

    Expected params:
        - means:     (N, 3)      world-space Gaussian centres (shared across cameras)
        - quats:     (N, 4)      unit quaternions (shared across cameras)
        - scale_ori: (N, 3)      raw scale params (shared across cameras)
        - opacity:   (C, N)      per-Gaussian per-camera opacity

    Expected data:
        - viewmats: (C, 4, 4)   camera extrinsics
        - Ks:       (C, 3, 3)   camera intrinsics
        - target:   (C, H, W)   ground-truth image for loss

    Expected config:
        - dim1:  int
        - dim0: int
        - tile_size:    int   (default 16)
        - eps_blur:     float (default 0.3)
        - lambda_l2:    float (default 1.0)
        - lambda_ssim:  float (default 0.0)
        - scale_min:    float (default 0.01)
        - scale_max:    float (default 2.0)
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, params: Params, config: Optional[Config] = None) -> None:
        for name in ['means', 'quats', 'scale_ori', 'opacity']:
            if params.get(name) is None:
                raise ValueError(f"Missing required parameter: {name}")
        means = params.get('means').data
        if not means.is_cuda:
            raise ValueError("Parameters must be CUDA tensors")
        self.initialized = True

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _transpose_cross_hess_6x2_to_2x6(
        h_6x2: torch.Tensor, C: int, N: int,
    ) -> torch.Tensor:
        """Transpose (C,N,12) cross-Hessian from 6x2 to 2x6 row-major."""
        mat_6x2 = h_6x2.reshape(C, N, 6, 2)
        mat_2x6 = mat_6x2.transpose(-1, -2)
        return mat_2x6.reshape(C, N, 12).contiguous()

    # ------------------------------------------------------------------
    # forward_only
    # ------------------------------------------------------------------
    def forward_only(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> ForwardResult:
        """Run the full forward pipeline and return loss."""
        from .world_to_image_projection import WorldToImageProjectionCUDA
        from .conics_mu import ConicsMuCUDA
        from .rasterize_to_pixels import RasterizeToPixelsCUDA
        from .loss import ImageLossTorch
        from .parameter_transform import ParameterTransformCUDA

        means = params.get('means').data
        quats = params.get('quats').data
        scale_ori = params.get('scale_ori').data
        opacity = params.get('opacity').data
        N = means.shape[0]

        viewmats = data.get('viewmats')
        Ks = data.get('Ks')
        target = data.get('target')
        C = viewmats.shape[0]
        dim1 = config.get('dim1')
        dim0 = config.get('dim0')
        tile_size = config.get('tile_size', 16)
        eps_blur = config.get('eps_blur', 0.3)
        lambda_l2 = config.get('lambda_l2', 1.0)
        lambda_ssim = config.get('lambda_ssim', 0.0)
        scale_min = config.get('scale_min', 0.01)
        scale_max = config.get('scale_max', 2.0)

        # --- Stage 0: ParameterTransform ---
        pt_params = Params()
        pt_params.add_param('quats', _ParamWrapper(quats))
        pt_params.add_param('scale_ori', _ParamWrapper(scale_ori))
        pt_config = Config()
        pt_config.set('scale_min', scale_min)
        pt_config.set('scale_max', scale_max)

        pt_op = ParameterTransformCUDA()
        pt_op.initialize(pt_params, pt_config)
        pt_fwd = pt_op.forward_only(pt_params, Data(), pt_config)
        covars = pt_fwd.outputs['covars']  # (N, 6)

        # --- Stage 1: WorldToImageProjection ---
        proj_params = Params()
        proj_params.add_param('means', _ParamWrapper(means))
        proj_params.add_param('covars', _ParamWrapper(covars))
        proj_data = Data()
        proj_data.set('viewmats', viewmats)
        proj_data.set('Ks', Ks)
        proj_config = Config()
        proj_config.set('dim1', dim1)
        proj_config.set('dim0', dim0)
        proj_config.set('eps_blur', eps_blur)

        proj_op = WorldToImageProjectionCUDA()
        proj_op.initialize(proj_params, proj_config)
        proj_fwd = proj_op.forward_only(proj_params, proj_data, proj_config)
        means2d = proj_fwd.outputs['means2d']
        cov2d_3x3 = proj_fwd.outputs['cov2d_3x3']
        radii = proj_fwd.outputs['radii']

        # --- Stage 2: ConicsMu ---
        cm_params = Params()
        cm_params.add_param('cov2d_3x3', _ParamWrapper(cov2d_3x3))
        cm_config = Config()
        cm_config.set('eps', eps_blur)

        cm_op = ConicsMuCUDA()
        cm_op.initialize(cm_params, cm_config)
        cm_fwd = cm_op.forward_only(cm_params, Data(), cm_config)
        conics = cm_fwd.outputs['conics']
        mus = cm_fwd.outputs['mus']

        # --- Stage 3: RasterizeToPixels ---
        rast_params = Params()
        rast_params.add_param('means2d', _ParamWrapper(means2d))
        rast_params.add_param('conics', _ParamWrapper(conics))
        rast_params.add_param('opacities', _ParamWrapper(opacity))
        rast_params.add_param('mus', _ParamWrapper(mus))
        rast_data = Data()
        rast_data.set('radii', radii)
        rast_data.set('dim0', dim0)
        rast_data.set('dim1', dim1)
        rast_data.set('tile_size', tile_size)

        rast_op = RasterizeToPixelsCUDA()
        rast_op.initialize(rast_params)
        rast_fwd = rast_op.forward_only(rast_params, rast_data)
        render_output = rast_fwd.outputs['render_output']

        # --- Stage 4: ImageLoss ---
        loss_params = Params()
        loss_params.add_param('prediction', _ParamWrapper(render_output))
        loss_data = Data()
        loss_data.set('target', target)
        loss_config = Config()
        loss_config.set('lambda_l2', lambda_l2)
        loss_config.set('lambda_ssim', lambda_ssim)
        loss_config.set('window_size', 11)

        loss_op = ImageLossTorch(lambda_l2=lambda_l2, lambda_ssim=lambda_ssim)
        loss_op.initialize(loss_params, loss_config)
        loss_fwd = loss_op.forward_only(loss_params, loss_data, loss_config)

        return ForwardResult(
            outputs={
                'loss': loss_fwd.outputs['loss'],
                'render_output': render_output,
                'means2d': means2d,
                'cov2d_3x3': cov2d_3x3,
                'conics': conics,
                'mus': mus,
                'covars': covars,
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
        """Forward + first-order backward through the full 5-stage pipeline."""
        from .world_to_image_projection import WorldToImageProjectionCUDA
        from .conics_mu import ConicsMuCUDA
        from .rasterize_to_pixels import RasterizeToPixelsCUDA
        from .loss import ImageLossTorch
        from .parameter_transform import ParameterTransformCUDA

        means = params.get('means').data
        quats = params.get('quats').data
        scale_ori = params.get('scale_ori').data
        opacity = params.get('opacity').data
        N = means.shape[0]

        viewmats = data.get('viewmats')
        Ks = data.get('Ks')
        target = data.get('target')
        C = viewmats.shape[0]
        dim1 = config.get('dim1')
        dim0 = config.get('dim0')
        tile_size = config.get('tile_size', 16)
        eps_blur = config.get('eps_blur', 0.3)
        lambda_l2 = config.get('lambda_l2', 1.0)
        lambda_ssim = config.get('lambda_ssim', 0.0)
        scale_min = config.get('scale_min', 0.01)
        scale_max = config.get('scale_max', 2.0)

        # ============== FORWARD PASS ==============

        # Stage 0: ParameterTransform
        pt_params = Params()
        pt_params.add_param('quats', _ParamWrapper(quats))
        pt_params.add_param('scale_ori', _ParamWrapper(scale_ori))
        pt_config = Config()
        pt_config.set('scale_min', scale_min)
        pt_config.set('scale_max', scale_max)

        pt_op = ParameterTransformCUDA()
        pt_op.initialize(pt_params, pt_config)
        pt_fwd = pt_op.forward_only(pt_params, Data(), pt_config)
        covars = pt_fwd.outputs['covars']  # (N, 6)

        # Stage 1: WorldToImageProjection
        proj_params = Params()
        proj_params.add_param('means', _ParamWrapper(means))
        proj_params.add_param('covars', _ParamWrapper(covars))
        proj_data = Data()
        proj_data.set('viewmats', viewmats)
        proj_data.set('Ks', Ks)
        proj_config = Config()
        proj_config.set('dim1', dim1)
        proj_config.set('dim0', dim0)
        proj_config.set('eps_blur', eps_blur)

        proj_op = WorldToImageProjectionCUDA()
        proj_op.initialize(proj_params, proj_config)
        proj_fwd = proj_op.forward_only(proj_params, proj_data, proj_config)
        means2d = proj_fwd.outputs['means2d']
        cov2d_3x3 = proj_fwd.outputs['cov2d_3x3']
        radii = proj_fwd.outputs['radii']

        # Stage 2: ConicsMu
        cm_params = Params()
        cm_params.add_param('cov2d_3x3', _ParamWrapper(cov2d_3x3))
        cm_config = Config()
        cm_config.set('eps', eps_blur)

        cm_op = ConicsMuCUDA()
        cm_op.initialize(cm_params, cm_config)
        cm_fwd = cm_op.forward_only(cm_params, Data(), cm_config)
        conics = cm_fwd.outputs['conics']
        mus = cm_fwd.outputs['mus']

        # Stage 3: RasterizeToPixels
        rast_params = Params()
        rast_params.add_param('means2d', _ParamWrapper(means2d))
        rast_params.add_param('conics', _ParamWrapper(conics))
        rast_params.add_param('opacities', _ParamWrapper(opacity))
        rast_params.add_param('mus', _ParamWrapper(mus))
        rast_data = Data()
        rast_data.set('radii', radii)
        rast_data.set('dim0', dim0)
        rast_data.set('dim1', dim1)
        rast_data.set('tile_size', tile_size)

        rast_op = RasterizeToPixelsCUDA()
        rast_op.initialize(rast_params)
        rast_fwd = rast_op.forward_only(rast_params, rast_data)
        render_output = rast_fwd.outputs['render_output']

        # Stage 4: ImageLoss
        loss_params = Params()
        loss_params.add_param('prediction', _ParamWrapper(render_output))
        loss_data = Data()
        loss_data.set('target', target)
        loss_config = Config()
        loss_config.set('lambda_l2', lambda_l2)
        loss_config.set('lambda_ssim', lambda_ssim)
        loss_config.set('window_size', 11)

        loss_op = ImageLossTorch(lambda_l2=lambda_l2, lambda_ssim=lambda_ssim)
        loss_op.initialize(loss_params, loss_config)
        loss_fwd, loss_grad = loss_op.forward_backward1(
            loss_params, loss_data, loss_config
        )
        v_render_output = loss_grad.param_grads['prediction']

        # ============== BACKWARD PASS (first-order) ==============

        # Stage 3 bwd: RasterizeToPixels
        rast_data_bwd = Data()
        rast_data_bwd.set('radii', radii)
        rast_data_bwd.set('dim0', dim0)
        rast_data_bwd.set('dim1', dim1)
        rast_data_bwd.set('tile_size', tile_size)
        rast_data_bwd.set('v_render_output', v_render_output)

        _, rast_grad = rast_op.forward_backward1(rast_params, rast_data_bwd)
        v_means2d = rast_grad.param_grads['means2d']
        v_conics = rast_grad.param_grads['conics']
        v_opacity = rast_grad.param_grads['opacities']
        v_mus = rast_grad.param_grads['mus']

        # Stage 2 bwd: ConicsMu
        cm_data_bwd = Data()
        cm_data_bwd.set('v_conics', v_conics)
        cm_data_bwd.set('v_mus', v_mus)

        _, cm_grad = cm_op.forward_backward1(cm_params, cm_data_bwd, cm_config)
        v_cov2d_3x3 = cm_grad.param_grads['cov2d_3x3']

        # Stage 1 bwd: WorldToImageProjection
        proj_data_bwd = Data()
        proj_data_bwd.set('viewmats', viewmats)
        proj_data_bwd.set('Ks', Ks)
        proj_data_bwd.set('v_means2d', v_means2d)
        proj_data_bwd.set('v_cov2d_3x3', v_cov2d_3x3)

        _, proj_grad = proj_op.forward_backward1(
            proj_params, proj_data_bwd, proj_config
        )
        v_means = proj_grad.param_grads['means']
        v_covars = proj_grad.param_grads['covars']  # (N, 6)

        # Stage 0 bwd: ParameterTransform
        # v_covars is already (N, 6) — no need to sum over cameras
        pt_data_bwd = Data()
        pt_data_bwd.set('v_covars', v_covars)

        _, pt_grad = pt_op.forward_backward1(pt_params, pt_data_bwd, pt_config)
        v_quats = pt_grad.param_grads['quats']
        v_scale_ori = pt_grad.param_grads['scale_ori']

        forward_result = ForwardResult(
            outputs={
                'loss': loss_fwd.outputs['loss'],
                'render_output': render_output,
            },
            intermediates={}
        )
        grad_result = GradResult(
            param_grads={
                'means': v_means,
                'quats': v_quats,
                'scale_ori': v_scale_ori,
                'opacity': v_opacity,
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
        """Forward + second-order backward through the full 5-stage pipeline.

        Chain rule for Hessians (backward through layers):

        The pipeline is:
            (means, quats, scale_ori, opacity)
                → covars → (means2d, cov2d_3x3) → (conics, mus) → render → loss

        Stage 4 (ImageLoss) gives v_render and h_render.
        Stage 3 (Rasterize bwd2) gives all Hessian blocks for
            (means2d, conics, mus, opacity).
        Stage 2 (ConicsMu bwd2) propagates through cov2d_3x3 → (conics, mus).
        Stage 1 (WorldToImageProjection bwd2) propagates through
            (means, covars) → (means2d, cov2d_3x3).
        Stage 0 (ParameterTransform bwd2) propagates through
            (quats, scale_ori) → covars.

        Key subtlety: covars is shared across cameras.
        v_covars must be summed over C before passing to ParameterTransform.
        h_covars (and cross-Hessians with covars) must also be summed over C.
        """
        from .world_to_image_projection import WorldToImageProjectionCUDA
        from .conics_mu import ConicsMuCUDA
        from .rasterize_to_pixels import RasterizeToPixelsCUDA
        from .loss import ImageLossTorch
        from .parameter_transform import ParameterTransformCUDA

        means = params.get('means').data
        quats = params.get('quats').data
        scale_ori = params.get('scale_ori').data
        opacity = params.get('opacity').data
        N = means.shape[0]
        dtype = means.dtype
        device = means.device

        viewmats = data.get('viewmats')
        Ks = data.get('Ks')
        target = data.get('target')
        C = viewmats.shape[0]
        dim1 = config.get('dim1')
        dim0 = config.get('dim0')
        tile_size = config.get('tile_size', 16)
        eps_blur = config.get('eps_blur', 0.3)
        lambda_l2 = config.get('lambda_l2', 1.0)
        lambda_ssim = config.get('lambda_ssim', 0.0)
        scale_min = config.get('scale_min', 0.01)
        scale_max = config.get('scale_max', 2.0)

        # ==================================================================
        # FORWARD PASS: stages 0-4
        # ==================================================================

        # Stage 0: ParameterTransform
        pt_params = Params()
        pt_params.add_param('quats', _ParamWrapper(quats))
        pt_params.add_param('scale_ori', _ParamWrapper(scale_ori))
        pt_config = Config()
        pt_config.set('scale_min', scale_min)
        pt_config.set('scale_max', scale_max)

        pt_op = ParameterTransformCUDA()
        pt_op.initialize(pt_params, pt_config)
        pt_fwd = pt_op.forward_only(pt_params, Data(), pt_config)
        covars = pt_fwd.outputs['covars']  # (N, 6)

        # Stage 1: WorldToImageProjection
        proj_params = Params()
        proj_params.add_param('means', _ParamWrapper(means))
        proj_params.add_param('covars', _ParamWrapper(covars))
        proj_data = Data()
        proj_data.set('viewmats', viewmats)
        proj_data.set('Ks', Ks)
        proj_config = Config()
        proj_config.set('dim1', dim1)
        proj_config.set('dim0', dim0)
        proj_config.set('eps_blur', eps_blur)

        proj_op = WorldToImageProjectionCUDA()
        proj_op.initialize(proj_params, proj_config)
        proj_fwd = proj_op.forward_only(proj_params, proj_data, proj_config)
        means2d = proj_fwd.outputs['means2d']
        cov2d_3x3 = proj_fwd.outputs['cov2d_3x3']
        radii = proj_fwd.outputs['radii']

        # Stage 2: ConicsMu
        cm_params = Params()
        cm_params.add_param('cov2d_3x3', _ParamWrapper(cov2d_3x3))
        cm_config = Config()
        cm_config.set('eps', eps_blur)

        cm_op = ConicsMuCUDA()
        cm_op.initialize(cm_params, cm_config)
        cm_fwd = cm_op.forward_only(cm_params, Data(), cm_config)
        conics = cm_fwd.outputs['conics']
        mus = cm_fwd.outputs['mus']

        # Stage 3: RasterizeToPixels
        rast_params = Params()
        rast_params.add_param('means2d', _ParamWrapper(means2d))
        rast_params.add_param('conics', _ParamWrapper(conics))
        rast_params.add_param('opacities', _ParamWrapper(opacity))
        rast_params.add_param('mus', _ParamWrapper(mus))
        rast_data = Data()
        rast_data.set('radii', radii)
        rast_data.set('dim0', dim0)
        rast_data.set('dim1', dim1)
        rast_data.set('tile_size', tile_size)

        rast_op = RasterizeToPixelsCUDA()
        rast_op.initialize(rast_params)
        rast_fwd = rast_op.forward_only(rast_params, rast_data)
        render_output = rast_fwd.outputs['render_output']

        # Stage 4: ImageLoss
        loss_params = Params()
        loss_params.add_param('prediction', _ParamWrapper(render_output))
        loss_data = Data()
        loss_data.set('target', target)
        loss_config = Config()
        loss_config.set('lambda_l2', lambda_l2)
        loss_config.set('lambda_ssim', lambda_ssim)
        loss_config.set('window_size', 11)

        loss_op = ImageLossTorch(lambda_l2=lambda_l2, lambda_ssim=lambda_ssim)
        loss_op.initialize(loss_params, loss_config)
        loss_fwd, loss_grad, loss_hess = loss_op.forward_backward2(
            loss_params, loss_data, loss_config
        )
        v_render_output = loss_grad.param_grads['prediction']
        h_render_output = loss_hess.param_hess['prediction']

        # ==================================================================
        # BACKWARD PASS stage 3: RasterizeToPixels bwd2
        # ==================================================================
        rast_data_bwd2 = Data()
        rast_data_bwd2.set('radii', radii)
        rast_data_bwd2.set('dim0', dim0)
        rast_data_bwd2.set('dim1', dim1)
        rast_data_bwd2.set('tile_size', tile_size)
        rast_data_bwd2.set('v_render_output', v_render_output)
        rast_data_bwd2.set('h_render_output', h_render_output)

        _, rast_grad, rast_hess = rast_op.forward_backward2(
            rast_params, rast_data_bwd2
        )

        v_means2d_rast = rast_grad.param_grads['means2d']
        v_conics_rast = rast_grad.param_grads['conics']
        v_opacity_rast = rast_grad.param_grads['opacities']
        v_mus_rast = rast_grad.param_grads['mus']

        h_means2d_rast = rast_hess.param_hess['means2d']
        h_conics_rast = rast_hess.param_hess['conics']
        h_mus_rast = rast_hess.param_hess['mus']
        h_opacity_rast = rast_hess.param_hess['opacities']

        h_mu_mean = rast_hess.cross_hess['mu_mean']
        h_mu_conic = rast_hess.cross_hess['mu_conic']
        h_mean_conic = rast_hess.cross_hess['mean_conic']
        h_mu_opacity = rast_hess.cross_hess['mu_opacity']
        h_conic_opacity = rast_hess.cross_hess['conic_opacity']
        h_mean_opacity = rast_hess.cross_hess['mean_opacity']

        # ==================================================================
        # BACKWARD PASS stage 2: ConicsMu bwd2
        # ==================================================================
        cm_data_bwd2 = Data()
        cm_data_bwd2.set('v_conics', v_conics_rast)
        cm_data_bwd2.set('v_mus', v_mus_rast)
        cm_data_bwd2.set('h_conics', h_conics_rast)
        cm_data_bwd2.set('h_conics_mu', h_mu_conic)
        cm_data_bwd2.set('h_mu_mu', h_mus_rast)
        cm_data_bwd2.set('h_means2d_conics', h_mean_conic)
        cm_data_bwd2.set('h_conics_opacity', h_conic_opacity)
        cm_data_bwd2.set('h_mu_means2d', h_mu_mean)
        cm_data_bwd2.set('h_mu_opacity', h_mu_opacity)

        _, cm_grad, cm_hess = cm_op.forward_backward2(
            cm_params, cm_data_bwd2, cm_config
        )

        v_cov2d_3x3 = cm_grad.param_grads['cov2d_3x3']
        h_cov_cov = cm_hess.param_hess['cov_cov']
        h_cov_means2d = cm_hess.param_hess['cov_means2d']
        h_cov_opacity = cm_hess.param_hess['cov_opacity']

        # ==================================================================
        # BACKWARD PASS stage 1: WorldToImageProjection bwd2
        # ==================================================================
        h_means2d = h_means2d_rast
        h_cov2d_3x3 = h_cov_cov
        h_means2d_cov2d_3x3 = self._transpose_cross_hess_6x2_to_2x6(
            h_cov_means2d, C, N
        )
        h_means2d_opacity = h_mean_opacity
        h_cov2d_3x3_opacity = h_cov_opacity

        proj_data_bwd2 = Data()
        proj_data_bwd2.set('viewmats', viewmats)
        proj_data_bwd2.set('Ks', Ks)
        proj_data_bwd2.set('v_means2d', v_means2d_rast)
        proj_data_bwd2.set('v_cov2d_3x3', v_cov2d_3x3)
        proj_data_bwd2.set('h_means2d', h_means2d)
        proj_data_bwd2.set('h_cov2d_3x3', h_cov2d_3x3)
        proj_data_bwd2.set('h_means2d_cov2d_3x3', h_means2d_cov2d_3x3)
        proj_data_bwd2.set('h_means2d_opacity', h_means2d_opacity)
        proj_data_bwd2.set('h_cov2d_3x3_opacity', h_cov2d_3x3_opacity)

        _, proj_grad, proj_hess = proj_op.forward_backward2(
            proj_params, proj_data_bwd2, proj_config
        )

        v_means = proj_grad.param_grads['means']             # (N, 3)
        v_covars = proj_grad.param_grads['covars']            # (N, 6)

        h_means = proj_hess.param_hess['means']               # (N, 6)
        h_covars = proj_hess.param_hess['covars']             # (N, 21)
        h_means_covars = proj_hess.param_hess['means_covars'] # (N, 18)
        h_means_opacity = proj_hess.param_hess['means_opacity'] # (C, N, 3)
        h_covars_opacity = proj_hess.param_hess['covars_opacity'] # (C, N, 6)

        # ==================================================================
        # BACKWARD PASS stage 0: ParameterTransform bwd2
        # ==================================================================
        # covars = f(quats, scale_ori) is SHARED across cameras.
        # h_covars and v_covars are already (N, ...) — no need to sum over C.
        # h_covars_opacity is (C, N, 6) — sum over C since quats/scale_ori are shared.

        pt_data_bwd2 = Data()
        pt_data_bwd2.set('v_covars', v_covars)
        pt_data_bwd2.set('h_covars', h_covars)
        # h_means_covars is H_{means, covars} in 3x6 row-major (N, 18).
        # ParameterTransform expects h_covars_means = H_{covars, means} in 6x3 row-major.
        # Transpose: 3x6 → 6x3
        h_mc_mat = h_means_covars.reshape(N, 3, 6)
        h_cm_mat = h_mc_mat.transpose(-1, -2)  # (N, 6, 3)
        h_covars_means_for_pt = h_cm_mat.reshape(N, 18)
        pt_data_bwd2.set('h_covars_means', h_covars_means_for_pt)
        h_covars_opacity_summed = h_covars_opacity.sum(dim=0)  # (N, 6)
        pt_data_bwd2.set('h_covars_opacity', h_covars_opacity_summed)

        _, pt_grad, pt_hess = pt_op.forward_backward2(
            pt_params, pt_data_bwd2, pt_config
        )

        v_quats = pt_grad.param_grads['quats']           # (N, 4)
        v_scale_ori = pt_grad.param_grads['scale_ori']    # (N, 3)

        h_quats = pt_hess.param_hess['quats']             # (N, 10)
        h_scale_ori = pt_hess.param_hess['scale_ori']     # (N, 3)
        h_quats_scale_ori = pt_hess.cross_hess['quats_scale_ori']   # (N, 12)
        h_quats_means = pt_hess.cross_hess['quats_means']           # (N, 12)
        h_quats_opacity = pt_hess.cross_hess['quats_opacity']       # (N, 4)
        h_scale_ori_means = pt_hess.cross_hess['scale_ori_means']   # (N, 9)
        h_scale_ori_opacity = pt_hess.cross_hess['scale_ori_opacity'] # (N, 3)

        # ==================================================================
        # Assemble results
        # ==================================================================
        forward_result = ForwardResult(
            outputs={
                'loss': loss_fwd.outputs['loss'],
                'render_output': render_output,
            },
            intermediates={}
        )
        grad_result = GradResult(
            param_grads={
                'means': v_means,
                'quats': v_quats,
                'scale_ori': v_scale_ori,
                'opacity': v_opacity_rast,
            }
        )
        hess_result = HessResult(
            param_hess={
                'means': h_means,                   # (N, 6)
                'quats': h_quats,                    # (N, 10)
                'scale_ori': h_scale_ori,            # (N, 3)
                'opacity': h_opacity_rast,           # (C, N)
            },
            cross_hess={
                'quats_scale_ori': h_quats_scale_ori,        # (N, 12)
                'means_opacity': h_means_opacity,            # (C, N, 3)
                'quats_means': h_quats_means,                # (N, 12) -- H_{quats, means}
                'quats_opacity': h_quats_opacity,            # (N, 4)
                'scale_ori_means': h_scale_ori_means,        # (N, 9)
                'scale_ori_opacity': h_scale_ori_opacity,    # (N, 3)
            }
        )
        return forward_result, grad_result, hess_result


# ============================================================================
# Autograd Pipeline Implementation
# ============================================================================

class GS3DProjectionOnlyAutograd(SecondForwardBackwardBase):
    """
    Pure PyTorch autograd-based end-to-end pipeline (ground-truth reference).

    Uses PyTorch automatic differentiation to compute first-order and
    second-order derivatives of the full 5-stage pipeline:
        means, quats, scale_ori, opacity → loss

    Same params/data/config interface as GS3DProjectionOnlyCUDA.
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, params: Params, config: Optional[Config] = None) -> None:
        for name in ['means', 'quats', 'scale_ori', 'opacity']:
            if params.get(name) is None:
                raise ValueError(f"Missing required parameter: {name}")
        self.initialized = True

    def _forward_pipeline(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        scale_ori: torch.Tensor,
        opacity: torch.Tensor,
        viewmats: torch.Tensor,
        Ks: torch.Tensor,
        target: torch.Tensor,
        dim1: int,
        dim0: int,
        eps_blur: float,
        lambda_l2: float,
        scale_min: float,
        scale_max: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the full differentiable pipeline in pure PyTorch.

        Returns:
            loss: scalar
            render_output: (C, H, W)
        """
        from .parameter_transform import _compute_covars
        from .world_to_image_projection import _compute_world_to_image_projection
        from .conics_mu import _compute_conics_mu

        N = means.shape[0]
        C = viewmats.shape[0]
        device = means.device
        dtype = means.dtype

        # Stage 0: ParameterTransform
        covars = _compute_covars(quats, scale_ori, scale_min, scale_max)  # (N, 6)

        # Stage 1: WorldToImageProjection
        means2d, cov2d_3x3 = _compute_world_to_image_projection(
            means, covars, viewmats, Ks, dim1, dim0
        )

        # Stage 2: ConicsMu
        conics, mus = _compute_conics_mu(cov2d_3x3, eps_blur)

        # Stage 3: Rasterize (vectorized pure-torch)
        dim0 = dim0
        dim1 = dim1

        y_coords = torch.arange(dim0, device=device, dtype=dtype) + 0.5
        x_coords = torch.arange(dim1, device=device, dtype=dtype) + 0.5
        py, px = torch.meshgrid(y_coords, x_coords, indexing='ij')

        px = px.unsqueeze(0).unsqueeze(0)
        py = py.unsqueeze(0).unsqueeze(0)

        mean_x = means2d[..., 0].unsqueeze(-1).unsqueeze(-1)
        mean_y = means2d[..., 1].unsqueeze(-1).unsqueeze(-1)
        conic_a = conics[..., 0].unsqueeze(-1).unsqueeze(-1)
        conic_b = conics[..., 1].unsqueeze(-1).unsqueeze(-1)
        conic_c = conics[..., 2].unsqueeze(-1).unsqueeze(-1)
        opa = opacity.unsqueeze(-1).unsqueeze(-1)
        mu = mus.unsqueeze(-1).unsqueeze(-1)

        dx = px - mean_x
        dy = py - mean_y
        sigma = 0.5 * (conic_a * dx * dx + conic_c * dy * dy) + conic_b * dx * dy

        vis = torch.exp(-sigma)
        alpha = opa * mu * vis

        mask = (sigma >= 0) & (alpha.abs() >= 1e-5)
        alpha_masked = torch.where(mask, alpha, torch.zeros_like(alpha))
        render_output = alpha_masked.sum(dim=1)

        # Stage 4: L2 loss
        loss = lambda_l2 * ((render_output - target) ** 2).mean()

        return loss, render_output

    # ------------------------------------------------------------------
    # forward_only
    # ------------------------------------------------------------------
    def forward_only(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> ForwardResult:
        means = params.get('means').data
        quats = params.get('quats').data
        scale_ori = params.get('scale_ori').data
        opacity = params.get('opacity').data
        viewmats = data.get('viewmats')
        Ks = data.get('Ks')
        target = data.get('target')
        dim1 = config.get('dim1')
        dim0 = config.get('dim0')
        eps_blur = config.get('eps_blur', 0.3)
        lambda_l2 = config.get('lambda_l2', 1.0)
        scale_min = config.get('scale_min', 0.01)
        scale_max = config.get('scale_max', 2.0)

        with torch.no_grad():
            loss, render_output = self._forward_pipeline(
                means, quats, scale_ori, opacity,
                viewmats, Ks, target,
                dim1, dim0, eps_blur, lambda_l2,
                scale_min, scale_max,
            )

        return ForwardResult(
            outputs={'loss': loss, 'render_output': render_output},
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
        means_orig = params.get('means').data
        quats_orig = params.get('quats').data
        scale_ori_orig = params.get('scale_ori').data
        opacity_orig = params.get('opacity').data
        viewmats = data.get('viewmats')
        Ks = data.get('Ks')
        target = data.get('target')
        dim1 = config.get('dim1')
        dim0 = config.get('dim0')
        eps_blur = config.get('eps_blur', 0.3)
        lambda_l2 = config.get('lambda_l2', 1.0)
        scale_min = config.get('scale_min', 0.01)
        scale_max = config.get('scale_max', 2.0)

        means = means_orig.detach().clone().requires_grad_(True)
        quats = quats_orig.detach().clone().requires_grad_(True)
        scale_ori = scale_ori_orig.detach().clone().requires_grad_(True)
        opacity = opacity_orig.detach().clone().requires_grad_(True)

        loss, render_output = self._forward_pipeline(
            means, quats, scale_ori, opacity,
            viewmats, Ks, target,
            dim1, dim0, eps_blur, lambda_l2,
            scale_min, scale_max,
        )

        loss.backward()

        forward_result = ForwardResult(
            outputs={'loss': loss.detach(), 'render_output': render_output.detach()},
            intermediates={}
        )
        grad_result = GradResult(
            param_grads={
                'means': means.grad.detach(),
                'quats': quats.grad.detach(),
                'scale_ori': scale_ori.grad.detach(),
                'opacity': opacity.grad.detach(),
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
        """Forward + second-order backward via double autograd.

        Computes full per-Gaussian Hessian blocks by differentiating
        v_theta = dL/d(theta) component by component.

        Note: means (N,3), quats (N,4), scale_ori (N,3) are shared across
        cameras, so their gradients are summed over cameras automatically
        by autograd. The per-Gaussian Hessian is computed by extracting
        diagonal blocks of the full Hessian.
        Only opacity (C,N) is per-camera.
        """
        means_orig = params.get('means').data
        quats_orig = params.get('quats').data
        scale_ori_orig = params.get('scale_ori').data
        opacity_orig = params.get('opacity').data
        viewmats = data.get('viewmats')
        Ks = data.get('Ks')
        target = data.get('target')
        dim1 = config.get('dim1')
        dim0 = config.get('dim0')
        eps_blur = config.get('eps_blur', 0.3)
        lambda_l2 = config.get('lambda_l2', 1.0)
        scale_min = config.get('scale_min', 0.01)
        scale_max = config.get('scale_max', 2.0)
        N = means_orig.shape[0]
        C = viewmats.shape[0]
        dtype = means_orig.dtype
        device = means_orig.device

        # Create leaf tensors for double backward
        means = means_orig.detach().clone().requires_grad_(True)
        quats = quats_orig.detach().clone().requires_grad_(True)
        scale_ori = scale_ori_orig.detach().clone().requires_grad_(True)
        opacity = opacity_orig.detach().clone().requires_grad_(True)

        loss, render_output = self._forward_pipeline(
            means, quats, scale_ori, opacity,
            viewmats, Ks, target,
            dim1, dim0, eps_blur, lambda_l2,
            scale_min, scale_max,
        )

        # First backward with create_graph=True
        v_means_g, v_quats_g, v_scale_ori_g, v_opacity_g = torch.autograd.grad(
            outputs=loss,
            inputs=[means, quats, scale_ori, opacity],
            create_graph=True,
            retain_graph=True,
        )

        # ================================================================
        # Compute per-Gaussian Hessian blocks
        # ================================================================

        # means: (N, 3) — shared across cameras
        h_pp = torch.zeros(N, 3, 3, dtype=dtype, device=device)
        # quats: (N, 4) — shared
        h_qq = torch.zeros(N, 4, 4, dtype=dtype, device=device)
        # scale_ori: (N, 3) — shared
        h_ss = torch.zeros(N, 3, 3, dtype=dtype, device=device)
        # opacity: (C, N) — per-camera
        h_oo = torch.zeros(C, N, dtype=dtype, device=device)

        # Cross-Hessians
        h_pq = torch.zeros(N, 3, 4, dtype=dtype, device=device)     # means × quats
        h_ps = torch.zeros(N, 3, 3, dtype=dtype, device=device)     # means × scale_ori
        h_po = torch.zeros(C, N, 3, dtype=dtype, device=device)     # means × opacity (per-camera)
        h_qs = torch.zeros(N, 4, 3, dtype=dtype, device=device)     # quats × scale_ori
        h_qo = torch.zeros(N, 4, dtype=dtype, device=device)        # quats × opacity (summed over C)
        h_so = torch.zeros(N, 3, dtype=dtype, device=device)        # scale_ori × opacity (summed over C)

        for g in range(N):
            # -- means[g,:] Hessian --
            for i in range(3):
                grads = torch.autograd.grad(
                    outputs=v_means_g[g, i],
                    inputs=[means, quats, scale_ori, opacity],
                    retain_graph=True,
                    allow_unused=True,
                )
                g_p = grads[0] if grads[0] is not None else torch.zeros_like(means)
                g_q = grads[1] if grads[1] is not None else torch.zeros_like(quats)
                g_s = grads[2] if grads[2] is not None else torch.zeros_like(scale_ori)
                g_o = grads[3] if grads[3] is not None else torch.zeros_like(opacity)

                h_pp[g, i, :] = g_p[g, :]
                h_pq[g, i, :] = g_q[g, :]
                h_ps[g, i, :] = g_s[g, :]
                h_po[:, g, i] = g_o[:, g]   # all cameras

            # -- quats[g,:] Hessian --
            for i in range(4):
                grads = torch.autograd.grad(
                    outputs=v_quats_g[g, i],
                    inputs=[quats, scale_ori, opacity],
                    retain_graph=True,
                    allow_unused=True,
                )
                g_q = grads[0] if grads[0] is not None else torch.zeros_like(quats)
                g_s = grads[1] if grads[1] is not None else torch.zeros_like(scale_ori)
                g_o = grads[2] if grads[2] is not None else torch.zeros_like(opacity)

                h_qq[g, i, :] = g_q[g, :]
                h_qs[g, i, :] = g_s[g, :]
                # H_{quats, opacity}: d(v_quats[g,i])/d(opacity) summed over cameras
                h_qo[g, i] = g_o[:, g].sum()

            # -- scale_ori[g,:] Hessian --
            for i in range(3):
                grads = torch.autograd.grad(
                    outputs=v_scale_ori_g[g, i],
                    inputs=[scale_ori, opacity],
                    retain_graph=True,
                    allow_unused=True,
                )
                g_s = grads[0] if grads[0] is not None else torch.zeros_like(scale_ori)
                g_o = grads[1] if grads[1] is not None else torch.zeros_like(opacity)

                h_ss[g, i, :] = g_s[g, :]
                h_so[g, i] = g_o[:, g].sum()

            # -- opacity[:,g] Hessian --
            for c_idx in range(C):
                grads = torch.autograd.grad(
                    outputs=v_opacity_g[c_idx, g],
                    inputs=[opacity],
                    retain_graph=True,
                    allow_unused=True,
                )
                g_o = grads[0] if grads[0] is not None else torch.zeros_like(opacity)
                h_oo[c_idx, g] = g_o[c_idx, g]

        # ================================================================
        # Pack outputs
        # ================================================================

        # h_means: upper-tri 3x3 → 6 elements
        h_means_out = torch.zeros(N, 6, dtype=dtype, device=device)
        for i in range(3):
            for j in range(i, 3):
                flat_idx = i * (5 - i) // 2 + j
                h_means_out[..., flat_idx] = h_pp[..., i, j]

        # h_quats: upper-tri 4x4 → 10 elements
        h_quats_out = torch.zeros(N, 10, dtype=dtype, device=device)
        idx = 0
        for i in range(4):
            for j in range(i, 4):
                h_quats_out[:, idx] = h_qq[:, i, j]
                idx += 1

        # h_scale_ori: upper-tri of 3x3 -> 6 elements
        h_scale_ori_out = torch.zeros(N, 6, dtype=dtype, device=device)
        idx = 0
        for i in range(3):
            for j in range(i, 3):
                h_scale_ori_out[:, idx] = h_ss[:, i, j]
                idx += 1

        # h_quats_scale_ori: 4x3 row-major → 12 elements
        h_quats_scale_ori_out = h_qs.reshape(N, 12)

        # h_quats_means: H_{quats, means} 4x3 row-major → 12 elements
        # h_pq is H_{means, quats} (N, 3, 4), transpose to get H_{quats, means} (N, 4, 3)
        h_quats_means_out = h_pq.transpose(-1, -2).reshape(N, 12)

        # h_scale_ori_means: H_{scale_ori, means} 3x3 row-major → 9 elements
        # h_ps is H_{means, scale_ori} (N, 3, 3), transpose to get H_{scale_ori, means} (N, 3, 3)
        h_scale_ori_means_out = h_ps.transpose(-1, -2).reshape(N, 9)

        # h_means_opacity: (C, N, 3)
        h_means_opacity_out = h_po

        forward_result = ForwardResult(
            outputs={'loss': loss.detach(), 'render_output': render_output.detach()},
            intermediates={}
        )
        grad_result = GradResult(
            param_grads={
                'means': v_means_g.detach(),
                'quats': v_quats_g.detach(),
                'scale_ori': v_scale_ori_g.detach(),
                'opacity': v_opacity_g.detach(),
            }
        )
        hess_result = HessResult(
            param_hess={
                'means': h_means_out.detach(),
                'quats': h_quats_out.detach(),
                'scale_ori': h_scale_ori_out.detach(),
                'opacity': h_oo.detach(),
            },
            cross_hess={
                'quats_scale_ori': h_quats_scale_ori_out.detach(),
                'means_opacity': h_means_opacity_out.detach(),
                'quats_means': h_quats_means_out.detach(),
                'quats_opacity': h_qo.detach(),
                'scale_ori_means': h_scale_ori_means_out.detach(),
                'scale_ori_opacity': h_so.detach(),
            }
        )
        return forward_result, grad_result, hess_result


# ============================================================================
# Factory Function
# ============================================================================

def create_gs3d_projection_only(
    backend: str = "cuda",
) -> SecondForwardBackwardBase:
    """Factory function to create GS3DProjectionOnly operator.

    Args:
        backend: "cuda" for CUDA pipeline, "autograd" for PyTorch reference

    Returns:
        GS3DProjectionOnly instance
    """
    if backend == "cuda":
        return GS3DProjectionOnlyCUDA()
    elif backend == "autograd":
        return GS3DProjectionOnlyAutograd()
    else:
        raise ValueError(
            f"Unknown backend: {backend}. Use 'cuda' or 'autograd'."
        )


__all__ = [
    "GS3DProjectionOnlyCUDA",
    "GS3DProjectionOnlyAutograd",
    "create_gs3d_projection_only",
]
