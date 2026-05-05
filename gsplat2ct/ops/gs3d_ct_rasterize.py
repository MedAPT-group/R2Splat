"""
GS3DCTRasterizeCUDA: Forward-only rasterization from raw 3DGS parameters.

Pipeline:
    (means, quats, scale_ori, opacity)
        -> [ParameterTransform]          -> covars (N, 6)
        -> [WorldToImageProjection]      -> means2d (C,N,2), cov2d_3x3 (C,N,6)
        -> [ConicsMu]                    -> conics (C,N,3), mus (C,N)
        -> [RasterizeToPixels]           -> render_output (C,H,W)

This operator is used for evaluation and visualization only (no loss, no backward).

Expected params:
    - means:     (N, 3)   world-space Gaussian centres
    - quats:     (N, 4)   unit quaternions (w, x, y, z)
    - scale_ori: (N, 3)   raw scale parameters
    - opacity:   (C, N)   per-Gaussian per-camera opacity

Expected data:
    - viewmats:  (C, 4, 4)  camera extrinsics
    - Ks:        (C, 3, 3)  camera intrinsics

Expected config:
    - img_dim1:     int     image width
    - img_dim0:     int     image height
    - tile_size:    int     (default 16)
    - eps_blur:     float   (default 0.3)
    - scale_min:    float   (default 0.01)
    - scale_max:    float   (default 2.0)

Author: Dy164
Date: 2026-03-19
"""

from typing import Optional

from ..model.beta import activated_beta_terms
from .base import ForwardBackwardBase
from .data import Config, Data, ForwardResult, Params


class GS3DCTRasterizeCUDA(ForwardBackwardBase):
    """Forward-only rasterization from raw 3DGS parameters to rendered images."""

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
            beta_rasterize_to_pixels_fwd,
            rasterize_to_pixels_fwd,
        )

        means = params.get('means').data
        quats = params.get('quats').data
        scale_ori = params.get('scale_ori').data
        opacity = params.get('opacity').data
        beta_param = params.get('beta')
        beta = beta_param.data if beta_param is not None else None

        viewmats = data.get('viewmats')
        Ks = data.get('Ks')
        C = viewmats.shape[0]

        img_dim1 = config.get('img_dim1')
        img_dim0 = config.get('img_dim0')
        tile_size = config.get('tile_size', 16)
        eps_blur = config.get('eps_blur', 0.3)
        scale_min = config.get('scale_min', 0.01)
        scale_max = config.get('scale_max', 2.0)
        primitive_type = str(config.get('primitive_type', 'beta' if beta is not None else 'gaussian')).lower()
        beta_gamma_init = config.get('beta_gamma_init', 3.5)

        # Stage 0: ParameterTransform
        covars, _max_scales = parameter_transform_fwd(
            quats, scale_ori, scale_min, scale_max
        )

        # Stage 1: WorldToImageProjection
        means2d, cov2d_3x3, radii = world_to_image_projection_fwd(
            means, covars, viewmats, Ks, img_dim1, img_dim0, eps_blur
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
        if primitive_type == "beta":
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

        return ForwardResult(
            outputs={'render_output': render_output},
            intermediates={}
        )

    def forward_backward1(self, params, data, config=None):
        raise NotImplementedError("GS3DCTRasterizeCUDA is forward-only")


__all__ = ["GS3DCTRasterizeCUDA"]
