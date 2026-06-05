"""
GS3DCTRasterizeCUDA: Forward-only rasterization from transformed 3DGS parameters.

Pipeline:
    (means, quats, scale_ori, activated opacity)
        -> [ParameterTransform]          -> covars (N, 6)
        -> [WorldToImageProjection]      -> means2d (C,N,2), cov2d_3x3 (C,N,6)
        -> [ConicsMu]                    -> conics (C,N,3), mus (C,N)
        -> [RasterizeToPixels]           -> render_output (C,H,W)

This operator is used for evaluation and visualization only (no loss, no backward).

Expected params:
    - means:     (N, 3)   world-space Gaussian centres
    - quats:     (N, 4)   unit quaternions (w, x, y, z)
    - scale_ori: (N, 3)   raw scale parameters
    - opacity:   (C, N)   per-Gaussian per-camera activated opacity

Expected data:
    - viewmats:  (C, 4, 4)  camera extrinsics
    - Ks:        (C, 3, 3)  camera intrinsics

Expected config:
    - img_dim1:     int     image width
    - img_dim0:     int     image height
    - tile_size:    int     (default 16)
    - eps_blur:     float   (default 0.3)
    - scale_min:    float   world-space activated scale lower bound
    - scale_max:    float   world-space activated scale upper bound

Author: Dy164
Date: 2026-03-19
"""

from typing import Optional

from ..model.beta import activated_beta_terms
from .base import ForwardBackwardBase
from .data import Config, Data, ForwardResult, Params
from .torch_autograd import activate_uncertainty


class GS3DCTRasterizeCUDA(ForwardBackwardBase):
    """Forward-only rasterization to rendered images."""

    def __init__(self):
        self.initialized = False

    def initialize(self, params: Params, config: Optional[Config] = None) -> None:
        for name in ['means', 'quats', 'scale_ori', 'opacity']:
            if params.get(name) is None:
                raise ValueError(f"Missing required parameter: {name}")
        primitive_type = "gaussian"
        if config is not None:
            primitive_type = str(config.get("primitive_type", primitive_type)).lower()
        if params.get("beta") is not None:
            primitive_type = "beta"
        if primitive_type == "beta" and params.get("beta") is None:
            raise ValueError("primitive_type='beta' requires parameter: beta")
        means = params.get('means').data
        if not means.is_cuda:
            raise ValueError("Parameters must be CUDA tensors")
        self.initialized = True

    def forward_only(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> ForwardResult:
        from ..cuda._wrapper import (
            parameter_transform_fwd,
            world_to_image_projection_fwd,
            conics_mu_fwd,
            isect_tiles, isect_offset_encode,
            beta_rasterize_to_pixels_mean_var_fwd,
            beta_rasterize_to_pixels_fwd,
            rasterize_to_pixels_mean_var_fwd,
            rasterize_to_pixels_fwd,
        )

        means = params.get('means').data
        quats = params.get('quats').data
        scale_ori = params.get('scale_ori').data
        opacity = params.get('opacity').data
        raw_uncertainty_param = params.get('raw_uncertainty')
        raw_uncertainty = (
            raw_uncertainty_param.data if raw_uncertainty_param is not None else None
        )
        beta_param = params.get('beta')
        beta = beta_param.data if beta_param is not None else None

        viewmats = data.get('viewmats')
        Ks = data.get('Ks')
        C = viewmats.shape[0]
        base_device = means.device
        for name, tensor in (
            ("quats", quats),
            ("scale_ori", scale_ori),
            ("opacity", opacity),
            ("viewmats", viewmats),
            ("Ks", Ks),
        ):
            if tensor.device != base_device:
                raise ValueError(
                    f"{name} is on {tensor.device}, but means is on {base_device}"
                )

        img_dim1 = config.get('img_dim1')
        img_dim0 = config.get('img_dim0')
        tile_size = config.get('tile_size', 16)
        eps_blur = config.get('eps_blur', 0.3)
        scale_min = config.get('scale_min', 0.01)
        scale_max = config.get('scale_max', 2.0)
        primitive_type = str(config.get('primitive_type', 'beta' if beta is not None else 'gaussian')).lower()
        beta_gamma_init = config.get('beta_gamma_init', 3.5)
        img_radius_clip = config.get('img_radius_clip', 1.0 if primitive_type == "beta" else 3.0)
        uncertainty_enabled = bool(config.get("uncertainty_enabled", False))
        uncertainty_q_min = float(config.get("uncertainty_q_min", 1.0e-6))
        uncertainty_q_max = config.get("uncertainty_q_max", None)

        # Stage 0: ParameterTransform
        covars, _max_scales = parameter_transform_fwd(
            quats, scale_ori, scale_min, scale_max
        )

        # Stage 1: WorldToImageProjection
        means2d, cov2d_3x3, radii = world_to_image_projection_fwd(
            means, covars, viewmats, Ks, img_dim1, img_dim0, eps_blur, img_radius_clip
        )

        # Stage 2: ConicsMu
        conics, mus = conics_mu_fwd(cov2d_3x3, eps_blur)
        if primitive_type == "beta":
            if beta is None:
                raise ValueError("primitive_type='beta' requires parameter beta")
            gamma, projected_beta, path_ratio, _act_grad = activated_beta_terms(
                beta, gamma_init=beta_gamma_init
            )
            mus_render = mus * path_ratio.unsqueeze(0)
            betas_2d = projected_beta.unsqueeze(0).expand(C, -1).contiguous()
        else:
            mus_render = mus
            betas_2d = None

        # Stage 3: Tile intersections
        n_tiles_dim0 = (img_dim0 + tile_size - 1) // tile_size
        n_tiles_dim1 = (img_dim1 + tile_size - 1) // tile_size

        _, isect_ids, flatten_ids = isect_tiles(
            means2d, radii, tile_size, n_tiles_dim1, n_tiles_dim0
        )
        tile_offsets = isect_offset_encode(
            isect_ids, C, n_tiles_dim1, n_tiles_dim0
        )

        # Stage 4: RasterizeToPixels
        render_var = None
        if uncertainty_enabled:
            if raw_uncertainty is None:
                raise ValueError("uncertainty_enabled=True requires raw_uncertainty")
            q = activate_uncertainty(
                raw_uncertainty,
                q_min=uncertainty_q_min,
                q_max=uncertainty_q_max,
            )
            if primitive_type == "beta":
                render_output, render_var = beta_rasterize_to_pixels_mean_var_fwd(
                    means2d, conics, opacity, mus_render, betas_2d, q,
                    tile_offsets, flatten_ids,
                    img_dim1, img_dim0, tile_size
                )
            else:
                render_output, render_var = rasterize_to_pixels_mean_var_fwd(
                    means2d, conics, opacity, mus_render, q,
                    tile_offsets, flatten_ids,
                    img_dim1, img_dim0, tile_size
                )
        elif primitive_type == "beta":
            render_output = beta_rasterize_to_pixels_fwd(
                means2d, conics, opacity, mus_render, betas_2d,
                tile_offsets, flatten_ids,
                img_dim1, img_dim0, tile_size
            )
        else:
            render_output = rasterize_to_pixels_fwd(
                means2d, conics, opacity, mus_render,
                tile_offsets, flatten_ids,
                img_dim1, img_dim0, tile_size
            )

        outputs = {'render_output': render_output}
        if render_var is not None:
            outputs['render_output_var'] = render_var
            outputs['render_output_std'] = render_var.clamp_min(0).sqrt()
        return ForwardResult(outputs=outputs, intermediates={})

    def forward_backward1(self, params, data, config=None):
        raise NotImplementedError("GS3DCTRasterizeCUDA is forward-only")


__all__ = ["GS3DCTRasterizeCUDA"]
