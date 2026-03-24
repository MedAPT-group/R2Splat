"""
Projection and Rasterization: End-to-end integration pipeline.

This module chains 4 operators into an end-to-end pipeline:
    WorldToImageProjection → ConicsMu → RasterizeToPixels → ImageLoss

Pipeline:
    (means, covars, opacity)
        → [WorldToImageProjection] → (means2d, cov2d_3x3)
        → [ConicsMu]               → (conics, mus)
        → [RasterizeToPixels]      → render_output
        → [ImageLoss]              → loss (scalar)

Outer (optimizable) parameters:
    - means:   (N, 3)     world-space Gaussian centres (shared across C cameras)
    - covars:  (N, 6)     world-space covariance upper-tri (shared across C cameras)
    - opacity: (C, N)     per-Gaussian per-camera opacity

Second-order backward computes:
    - Gradients   v_{means}, v_{covars}, v_{opacity}
    - Self-Hessians  H_{means}(N,6), H_{covars}(N,21), H_{opacity}(C,N)
    - Cross-Hessians H_{means,covars}(N,18), H_{means,opacity}(C,N,3),
                     H_{covars,opacity}(C,N,6)

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

class ProjectionAndRasterizationCUDA(SecondForwardBackwardBase):
    """
    CUDA-accelerated end-to-end pipeline:
        WorldToImageProjection → ConicsMu → RasterizeToPixels → ImageLoss

    Expected params:
        - means:    (N, 3)      world-space Gaussian centres (shared across C cameras)
        - covars:   (N, 6)      world-space covariance upper-tri (shared across C cameras)
        - opacity:  (C, N)      per-Gaussian per-camera opacity

    Expected data:
        - viewmats: (C, 4, 4)   camera extrinsics
        - Ks:       (C, 3, 3)   camera intrinsics
        - target:   (C, H, W)   ground-truth image for loss

    Expected config:
        - dim1:  int
        - dim0: int
        - tile_size:    int  (default 16)
        - eps_blur:     float (default 0.3)
        - lambda_l2:    float (default 1.0)
        - lambda_ssim:  float (default 0.0, set 0 for simplicity)
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, params: Params, config: Optional[Config] = None) -> None:
        for name in ['means', 'covars', 'opacity']:
            if params.get(name) is None:
                raise ValueError(f"Missing required parameter: {name}")
        means = params.get('means').data
        if not means.is_cuda:
            raise ValueError("Parameters must be CUDA tensors")
        self.initialized = True

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

        means = params.get('means').data
        covars = params.get('covars').data
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

        proj_op = WorldToImageProjectionCUDA()
        proj_op.initialize(proj_params, proj_config)
        proj_fwd = proj_op.forward_only(proj_params, proj_data, proj_config)
        means2d = proj_fwd.outputs['means2d']
        cov2d_3x3 = proj_fwd.outputs['cov2d_3x3']

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
        radii = self._compute_radii(means2d, conics, dim1, dim0)
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
        """Forward + first-order backward through the full pipeline."""
        from .world_to_image_projection import WorldToImageProjectionCUDA
        from .conics_mu import ConicsMuCUDA
        from .rasterize_to_pixels import RasterizeToPixelsCUDA
        from .loss import ImageLossTorch

        means = params.get('means').data
        covars = params.get('covars').data
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

        # ============== Forward pass ==============
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

        proj_op = WorldToImageProjectionCUDA()
        proj_op.initialize(proj_params, proj_config)
        proj_fwd = proj_op.forward_only(proj_params, proj_data, proj_config)
        means2d = proj_fwd.outputs['means2d']
        cov2d_3x3 = proj_fwd.outputs['cov2d_3x3']

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
        radii = self._compute_radii(means2d, conics, dim1, dim0)
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

        # Stage 4: ImageLoss → get v_render_output
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
        loss_fwd, loss_grad = loss_op.forward_backward1(loss_params, loss_data, loss_config)
        v_render_output = loss_grad.param_grads['prediction']

        # ============== Backward pass (first-order) ==============
        # Stage 3 bwd: RasterizeToPixels
        rast_data_bwd = Data()
        rast_data_bwd.set('radii', radii)
        rast_data_bwd.set('dim0', dim0)
        rast_data_bwd.set('dim1', dim1)
        rast_data_bwd.set('tile_size', tile_size)
        rast_data_bwd.set('v_render_output', v_render_output)

        _, rast_grad = rast_op.forward_backward1(rast_params, rast_data_bwd)
        v_means2d = rast_grad.param_grads['means2d']       # (C, N, 2)
        v_conics = rast_grad.param_grads['conics']          # (C, N, 3)
        v_opacity = rast_grad.param_grads['opacities']      # (C, N)
        v_mus = rast_grad.param_grads['mus']                # (C, N)

        # Stage 2 bwd: ConicsMu
        cm_data_bwd = Data()
        cm_data_bwd.set('v_conics', v_conics)
        cm_data_bwd.set('v_mus', v_mus)

        _, cm_grad = cm_op.forward_backward1(cm_params, cm_data_bwd, cm_config)
        v_cov2d_3x3 = cm_grad.param_grads['cov2d_3x3']    # (C, N, 6)

        # Stage 1 bwd: WorldToImageProjection
        proj_data_bwd = Data()
        proj_data_bwd.set('viewmats', viewmats)
        proj_data_bwd.set('Ks', Ks)
        proj_data_bwd.set('v_means2d', v_means2d)
        proj_data_bwd.set('v_cov2d_3x3', v_cov2d_3x3)

        _, proj_grad = proj_op.forward_backward1(proj_params, proj_data_bwd, proj_config)
        v_means = proj_grad.param_grads['means']
        v_covars = proj_grad.param_grads['covars']

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
                'covars': v_covars,
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
        """Forward + second-order backward through the full pipeline.

        Chain rule for Hessians (backward through layers):

        Let the pipeline be  theta → phi_1 → phi_2 → phi_3 → L
        where phi_1 = (means2d, cov2d_3x3), phi_2 = (conics, mus),
              phi_3 = render_output, L = loss.

        The Hessian chain rule at each stage is:
            H_{theta} = J^T  H_{phi}  J  +  sum_k v_k Hess_k(theta)

        We propagate backward:
        (a) ImageLoss gives v_render_output and h_render_output (diagonal).
        (b) RasterizeToPixels bwd2: from (v_render, h_render) computes
            v_{means2d, conics, opacity, mus} and all H/cross-H blocks.
        (c) ConicsMu bwd2: from (v_conics, v_mus, H_{conics}, H_{mus}, ...)
            computes v_cov2d_3x3 and H_{cov,cov}, H_{cov,means2d}, H_{cov,opacity}.
        (d) WorldToImageProjection bwd2: from (v_means2d, v_cov2d_3x3,
            H_{means2d}, H_{cov2d_3x3}, H_{means2d,cov2d_3x3}, ...)
            computes v_{means,covars} and H_{means}, H_{covars},
            H_{means,covars}, H_{means,opacity}, H_{covars,opacity}.

        The key challenge is correctly assembling the downstream Hessian
        blocks at each stage boundary, especially cross-Hessians with opacity.
        """
        from .world_to_image_projection import WorldToImageProjectionCUDA
        from .conics_mu import ConicsMuCUDA
        from .rasterize_to_pixels import RasterizeToPixelsCUDA
        from .loss import ImageLossTorch

        means = params.get('means').data
        covars = params.get('covars').data
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

        # ==================================================================
        # FORWARD PASS: stages 1-4
        # ==================================================================

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

        proj_op = WorldToImageProjectionCUDA()
        proj_op.initialize(proj_params, proj_config)
        proj_fwd = proj_op.forward_only(proj_params, proj_data, proj_config)
        means2d = proj_fwd.outputs['means2d']          # (C, N, 2)
        cov2d_3x3 = proj_fwd.outputs['cov2d_3x3']     # (C, N, 6)

        # Stage 2: ConicsMu
        cm_params = Params()
        cm_params.add_param('cov2d_3x3', _ParamWrapper(cov2d_3x3))
        cm_config = Config()
        cm_config.set('eps', eps_blur)

        cm_op = ConicsMuCUDA()
        cm_op.initialize(cm_params, cm_config)
        cm_fwd = cm_op.forward_only(cm_params, Data(), cm_config)
        conics = cm_fwd.outputs['conics']               # (C, N, 3)
        mus = cm_fwd.outputs['mus']                     # (C, N)

        # Stage 3: RasterizeToPixels
        radii = self._compute_radii(means2d, conics, dim1, dim0)
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
        render_output = rast_fwd.outputs['render_output']  # (C, H, W)

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
        v_render_output = loss_grad.param_grads['prediction']   # (C, H, W)
        h_render_output = loss_hess.param_hess['prediction']    # (C, H, W) diag

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

        # First-order gradients from rasterize
        v_means2d_rast = rast_grad.param_grads['means2d']     # (C, N, 2)
        v_conics_rast = rast_grad.param_grads['conics']        # (C, N, 3)
        v_opacity_rast = rast_grad.param_grads['opacities']    # (C, N)
        v_mus_rast = rast_grad.param_grads['mus']              # (C, N)

        # Self-Hessians from rasterize
        h_means2d_rast = rast_hess.param_hess['means2d']      # (C, N, 3) upper-tri 2x2: [H00,H01,H11]
        h_conics_rast = rast_hess.param_hess['conics']         # (C, N, 6) upper-tri 3x3
        h_mus_rast = rast_hess.param_hess['mus']               # (C, N)
        h_opacity_rast = rast_hess.param_hess['opacities']     # (C, N)

        # Cross-Hessians from rasterize
        h_mu_mean = rast_hess.cross_hess['mu_mean']            # (C, N, 2)
        h_mu_conic = rast_hess.cross_hess['mu_conic']          # (C, N, 3)
        h_mean_conic = rast_hess.cross_hess['mean_conic']      # (C, N, 6) row-major 2x3
        h_mu_opacity = rast_hess.cross_hess['mu_opacity']      # (C, N)
        h_conic_opacity = rast_hess.cross_hess['conic_opacity'] # (C, N, 3)
        h_mean_opacity = rast_hess.cross_hess['mean_opacity']  # (C, N, 2)

        # ==================================================================
        # BACKWARD PASS stage 2: ConicsMu bwd2
        # ==================================================================
        # ConicsMu maps:  cov2d_3x3(6) → conics(3), mus(1)
        # Its bwd2 needs downstream Hessian blocks for (conics, mus)
        # and cross-Hessians with external parameters (means2d, opacity).
        #
        # Downstream Hessian layout for ConicsMu:
        #   h_conics:          H_{conics,conics}    (C,N,6) upper-tri 3x3
        #   h_conics_mu:       H_{conics,mu}        (C,N,3)
        #   h_mu_mu:           H_{mu,mu}            (C,N)
        #   h_means2d_conics:  H_{means2d,conics}   (C,N,6) [2x3 row-major]
        #   h_conics_opacity:  H_{conics,opacity}   (C,N,3)
        #   h_mu_means2d:      H_{mu,means2d}       (C,N,2)
        #   h_mu_opacity:      H_{mu,opacity}       (C,N)

        cm_data_bwd2 = Data()
        cm_data_bwd2.set('v_conics', v_conics_rast)
        cm_data_bwd2.set('v_mus', v_mus_rast)
        cm_data_bwd2.set('h_conics', h_conics_rast)
        cm_data_bwd2.set('h_conics_mu', h_mu_conic)       # H_{conics,mu} = H_{mu,conic}^T but both are (C,N,3)
        cm_data_bwd2.set('h_mu_mu', h_mus_rast)
        cm_data_bwd2.set('h_means2d_conics', h_mean_conic) # (C,N,6) [2x3 row-major]
        cm_data_bwd2.set('h_conics_opacity', h_conic_opacity)
        cm_data_bwd2.set('h_mu_means2d', h_mu_mean)        # (C,N,2)
        cm_data_bwd2.set('h_mu_opacity', h_mu_opacity)     # (C,N)

        _, cm_grad, cm_hess = cm_op.forward_backward2(
            cm_params, cm_data_bwd2, cm_config
        )

        v_cov2d_3x3 = cm_grad.param_grads['cov2d_3x3']       # (C, N, 6)

        # ConicsMu Hessian outputs:
        h_cov_cov = cm_hess.param_hess['cov_cov']              # (C, N, 21) H_{cov,cov}
        h_cov_means2d = cm_hess.param_hess['cov_means2d']      # (C, N, 12) H_{cov,means2d} [6x2 row-major]
        h_cov_opacity = cm_hess.param_hess['cov_opacity']       # (C, N, 6)  H_{cov,opacity}

        # ==================================================================
        # BACKWARD PASS stage 1: WorldToImageProjection bwd2
        # ==================================================================
        # WorldToImageProjection maps:
        #   means(3) + covars(6) → means2d(2) + cov2d_3x3(6)
        # Its bwd2 needs downstream Hessian blocks for (means2d, cov2d_3x3)
        # and cross-Hessians with external parameter (opacity).
        #
        # Downstream Hessian layout:
        #   h_means2d:            H_{means2d,means2d}        (C,N,3)  upper-tri 2x2
        #   h_cov2d_3x3:         H_{cov2d_3x3,cov2d_3x3}   (C,N,21) upper-tri 6x6
        #   h_means2d_cov2d_3x3: H_{means2d,cov2d_3x3}      (C,N,12) [2x6 row-major]
        #   h_means2d_opacity:    H_{means2d,opacity}         (C,N,2)
        #   h_cov2d_3x3_opacity:  H_{cov2d_3x3,opacity}      (C,N,6)

        # Build h_means2d (upper-tri 2x2, 3 elements):
        #   From rasterize: h_means2d_rast = (C,N,3) [H_{m0m0}, H_{m0m1}, H_{m1m1}]
        #   This is directly the downstream H_{means2d, means2d}.
        #   But we also need to add contribution from ConicsMu's cross:
        #   The means2d are an intermediate that appears in BOTH rasterize
        #   and WorldToImageProjection.  The h_means2d_rast already captures
        #   the full H_{means2d,means2d} from rasterize (which is the only
        #   module that depends on means2d directly).
        h_means2d = h_means2d_rast  # (C, N, 3)

        # Build h_cov2d_3x3 (upper-tri 6x6, 21 elements):
        #   From ConicsMu bwd2: h_cov_cov already incorporates
        #   the full H_{cov2d_3x3, cov2d_3x3} including the rasterize
        #   contribution propagated through ConicsMu.
        h_cov2d_3x3 = h_cov_cov  # (C, N, 21)

        # Build h_means2d_cov2d_3x3 (2x6 row-major, 12 elements):
        #   From ConicsMu: h_cov_means2d is H_{cov2d_3x3, means2d} in
        #   [6x2 row-major]. We need H_{means2d, cov2d_3x3} in [2x6 row-major],
        #   which is the transpose. Reshape 6x2 → 2x6.
        h_means2d_cov2d_3x3 = self._transpose_cross_hess_6x2_to_2x6(
            h_cov_means2d, C, N
        )  # (C, N, 12)

        # Build h_means2d_opacity (C, N, 2):
        #   Only rasterize produces this: h_mean_opacity
        h_means2d_opacity = h_mean_opacity  # (C, N, 2)

        # Build h_cov2d_3x3_opacity (C, N, 6):
        #   From ConicsMu: h_cov_opacity already has this.
        h_cov2d_3x3_opacity = h_cov_opacity  # (C, N, 6)

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

        v_means = proj_grad.param_grads['means']
        v_covars = proj_grad.param_grads['covars']

        h_means = proj_hess.param_hess['means']                  # (N, 6)
        h_covars = proj_hess.param_hess['covars']                 # (N, 21)
        h_means_covars = proj_hess.param_hess['means_covars']     # (N, 18)
        h_means_opacity = proj_hess.param_hess['means_opacity']   # (C, N, 3)
        h_covars_opacity = proj_hess.param_hess['covars_opacity'] # (C, N, 6)

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
                'covars': v_covars,
                'opacity': v_opacity_rast,
            }
        )
        hess_result = HessResult(
            param_hess={
                'means': h_means,
                'covars': h_covars,
                'opacity': h_opacity_rast,
                'means_covars': h_means_covars,
                'means_opacity': h_means_opacity,
                'covars_opacity': h_covars_opacity,
            }
        )
        return forward_result, grad_result, hess_result

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_radii(
        means2d: torch.Tensor,
        conics: torch.Tensor,
        dim1: int,
        dim0: int,
    ) -> torch.Tensor:
        """Compute radii for tile intersection from conics.

        The covariance matrix is Sigma = conics^{-1}.
        Radius = ceil(3 * sqrt(max_eigenvalue(Sigma))).
        """
        a = conics[..., 0]
        b = conics[..., 1]
        c = conics[..., 2]

        # det(conic) = a*c - b^2
        det = a * c - b * b
        det = det.clamp(min=1e-10)

        # Sigma = (1/det) * [[c, -b], [-b, a]]
        sigma_xx = c / det
        sigma_yy = a / det

        trace = sigma_xx + sigma_yy
        disc = torch.sqrt((trace * trace - 4.0 * (sigma_xx * sigma_yy - (b / det) ** 2)).clamp(min=0.0))
        lambda_max = (trace + disc) / 2.0
        radii = (3.0 * torch.sqrt(lambda_max.clamp(min=1e-10))).ceil().to(torch.int32)
        radii = radii.clamp(min=1)
        return radii

    @staticmethod
    def _transpose_cross_hess_6x2_to_2x6(
        h_6x2: torch.Tensor,
        C: int,
        N: int,
    ) -> torch.Tensor:
        """Transpose a (C,N,12) cross-Hessian from 6x2 row-major to 2x6 row-major.

        Input layout  (6x2): h[i*2+j] = H[cov_i, means2d_j]  i=0..5, j=0..1
        Output layout (2x6): h[j*6+i] = H[means2d_j, cov_i]  j=0..1, i=0..5

        Since H is symmetric (H_{A,B} = H_{B,A}^T), element (i,j) of A×B
        equals element (j,i) of B×A.
        """
        # h_6x2 shape: (C, N, 12), layout: [cov0_m0, cov0_m1, cov1_m0, ...]
        mat_6x2 = h_6x2.reshape(C, N, 6, 2)
        mat_2x6 = mat_6x2.transpose(-1, -2)  # (C, N, 2, 6)
        return mat_2x6.reshape(C, N, 12).contiguous()


# ============================================================================
# Autograd Pipeline Implementation
# ============================================================================

class ProjectionAndRasterizationAutograd(SecondForwardBackwardBase):
    """
    Pure PyTorch autograd-based end-to-end pipeline (ground-truth reference).

    Uses PyTorch automatic differentiation to compute first-order and
    second-order derivatives of the full pipeline:
        means, covars, opacity → loss

    This is used as ground truth for validating the CUDA pipeline.

    Same params/data/config interface as ProjectionAndRasterizationCUDA.
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, params: Params, config: Optional[Config] = None) -> None:
        for name in ['means', 'covars', 'opacity']:
            if params.get(name) is None:
                raise ValueError(f"Missing required parameter: {name}")
        self.initialized = True

    def _forward_pipeline(
        self,
        means: torch.Tensor,
        covars: torch.Tensor,
        opacity: torch.Tensor,
        viewmats: torch.Tensor,
        Ks: torch.Tensor,
        target: torch.Tensor,
        dim1: int,
        dim0: int,
        eps_blur: float,
        lambda_l2: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the full differentiable pipeline in pure PyTorch.

        Returns:
            loss: scalar
            render_output: (C, H, W)
        """
        from .world_to_image_projection import _compute_world_to_image_projection
        from .conics_mu import _compute_conics_mu

        N = means.shape[0]
        C = viewmats.shape[0]
        device = means.device
        dtype = means.dtype

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
        covars = params.get('covars').data
        opacity = params.get('opacity').data
        viewmats = data.get('viewmats')
        Ks = data.get('Ks')
        target = data.get('target')
        dim1 = config.get('dim1')
        dim0 = config.get('dim0')
        eps_blur = config.get('eps_blur', 0.3)
        lambda_l2 = config.get('lambda_l2', 1.0)

        with torch.no_grad():
            loss, render_output = self._forward_pipeline(
                means, covars, opacity, viewmats, Ks, target,
                dim1, dim0, eps_blur, lambda_l2,
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
        covars_orig = params.get('covars').data
        opacity_orig = params.get('opacity').data
        viewmats = data.get('viewmats')
        Ks = data.get('Ks')
        target = data.get('target')
        dim1 = config.get('dim1')
        dim0 = config.get('dim0')
        eps_blur = config.get('eps_blur', 0.3)
        lambda_l2 = config.get('lambda_l2', 1.0)

        means = means_orig.detach().clone().requires_grad_(True)
        covars = covars_orig.detach().clone().requires_grad_(True)
        opacity = opacity_orig.detach().clone().requires_grad_(True)

        loss, render_output = self._forward_pipeline(
            means, covars, opacity, viewmats, Ks, target,
            dim1, dim0, eps_blur, lambda_l2,
        )

        loss.backward()

        forward_result = ForwardResult(
            outputs={'loss': loss.detach(), 'render_output': render_output.detach()},
            intermediates={}
        )
        grad_result = GradResult(
            param_grads={
                'means': means.grad.detach(),
                'covars': covars.grad.detach(),
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

        Computes full Hessian blocks by differentiating v_theta = dL/d(theta)
        component by component w.r.t. (means, covars, opacity).

        Returns:
            ForwardResult, GradResult, HessResult
        """
        means_orig = params.get('means').data
        covars_orig = params.get('covars').data
        opacity_orig = params.get('opacity').data
        viewmats = data.get('viewmats')
        Ks = data.get('Ks')
        target = data.get('target')
        dim1 = config.get('dim1')
        dim0 = config.get('dim0')
        eps_blur = config.get('eps_blur', 0.3)
        lambda_l2 = config.get('lambda_l2', 1.0)
        N = means_orig.shape[0]
        C = viewmats.shape[0]
        dtype = means_orig.dtype
        device = means_orig.device

        # Create leaf tensors for double backward
        means = means_orig.detach().clone().requires_grad_(True)
        covars = covars_orig.detach().clone().requires_grad_(True)
        opacity = opacity_orig.detach().clone().requires_grad_(True)

        loss, render_output = self._forward_pipeline(
            means, covars, opacity, viewmats, Ks, target,
            dim1, dim0, eps_blur, lambda_l2,
        )

        # First backward with create_graph=True
        v_means_graph, v_covars_graph, v_opacity_graph = torch.autograd.grad(
            outputs=loss,
            inputs=[means, covars, opacity],
            create_graph=True,
            retain_graph=True,
        )

        # ================================================================
        # Compute per-Gaussian Hessian by differentiating v_theta
        # component-wise.
        #
        # means (N,3) and covars (N,6) are shared across cameras,
        # so their self-Hessians and cross-Hessian are summed over C
        # automatically by autograd.  Only opacity (C,N) is per-camera,
        # so cross-Hessians involving opacity retain the C dimension.
        # ================================================================

        # Allocate Hessian blocks
        h_pp = torch.zeros(N, 3, 3, dtype=dtype, device=device)      # means self-Hessian
        h_ss = torch.zeros(N, 6, 6, dtype=dtype, device=device)      # covars self-Hessian
        h_ps = torch.zeros(N, 3, 6, dtype=dtype, device=device)      # means x covars
        h_po = torch.zeros(C, N, 3, dtype=dtype, device=device)      # means x opacity (per-camera)
        h_so = torch.zeros(C, N, 6, dtype=dtype, device=device)      # covars x opacity (per-camera)
        h_oo = torch.zeros(C, N, dtype=dtype, device=device)         # opacity self-Hessian (per-camera)

        for g in range(N):
            # -- Differentiate v_means[g,:] w.r.t. (means, covars, opacity) --
            for i in range(3):
                grads = torch.autograd.grad(
                    outputs=v_means_graph[g, i],
                    inputs=[means, covars, opacity],
                    retain_graph=True,
                    allow_unused=True,
                )
                g_p = grads[0] if grads[0] is not None else torch.zeros_like(means)
                g_s = grads[1] if grads[1] is not None else torch.zeros_like(covars)
                g_o = grads[2] if grads[2] is not None else torch.zeros_like(opacity)
                # Only extract the (g) block (per-Gaussian)
                h_pp[g, i, :] = g_p[g, :]
                h_ps[g, i, :] = g_s[g, :]
                h_po[:, g, i] = g_o[:, g]   # all cameras

            # -- Differentiate v_covars[g,:] w.r.t. (covars, opacity) --
            for i in range(6):
                grads = torch.autograd.grad(
                    outputs=v_covars_graph[g, i],
                    inputs=[covars, opacity],
                    retain_graph=True,
                    allow_unused=True,
                )
                g_s = grads[0] if grads[0] is not None else torch.zeros_like(covars)
                g_o = grads[1] if grads[1] is not None else torch.zeros_like(opacity)
                h_ss[g, i, :] = g_s[g, :]
                h_so[:, g, i] = g_o[:, g]   # all cameras

            # -- Differentiate v_opacity[c,g] w.r.t. opacity --
            for c_idx in range(C):
                grads = torch.autograd.grad(
                    outputs=v_opacity_graph[c_idx, g],
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

        # h_covars: upper-tri 6x6 → 21 elements
        h_covars_out = torch.zeros(N, 21, dtype=dtype, device=device)
        for i in range(6):
            for j in range(i, 6):
                flat_idx = i * (11 - i) // 2 + j
                h_covars_out[..., flat_idx] = h_ss[..., i, j]

        # h_means_covars: 3x6 row-major → 18 elements
        h_means_covars_out = h_ps.reshape(N, 18)

        forward_result = ForwardResult(
            outputs={'loss': loss.detach(), 'render_output': render_output.detach()},
            intermediates={}
        )
        grad_result = GradResult(
            param_grads={
                'means': v_means_graph.detach(),
                'covars': v_covars_graph.detach(),
                'opacity': v_opacity_graph.detach(),
            }
        )
        hess_result = HessResult(
            param_hess={
                'means': h_means_out.detach(),
                'covars': h_covars_out.detach(),
                'opacity': h_oo.detach(),
                'means_covars': h_means_covars_out.detach(),
                'means_opacity': h_po.detach(),
                'covars_opacity': h_so.detach(),
            }
        )
        return forward_result, grad_result, hess_result


# ============================================================================
# Factory Function
# ============================================================================

def create_projection_and_rasterization(
    backend: str = "cuda",
) -> SecondForwardBackwardBase:
    """Factory function to create ProjectionAndRasterization operator.

    Args:
        backend: "cuda" for CUDA pipeline, "autograd" for PyTorch reference

    Returns:
        ProjectionAndRasterization instance
    """
    if backend == "cuda":
        return ProjectionAndRasterizationCUDA()
    elif backend == "autograd":
        return ProjectionAndRasterizationAutograd()
    else:
        raise ValueError(
            f"Unknown backend: {backend}. Use 'cuda' or 'autograd'."
        )


__all__ = [
    "ProjectionAndRasterizationCUDA",
    "ProjectionAndRasterizationAutograd",
    "create_projection_and_rasterization",
]
