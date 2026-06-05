"""
GS3DCTVoxelizeCUDA: Forward-only voxelization from transformed 3DGS parameters.

Pipeline:
    (means, quats, scale_ori, activated opacity)
        -> [ParameterTransform]          -> covars (N, 6), max_scales (N,)
        -> [WorldToVolumeMapping]        -> means3d_v (N,3), conics (N,6), radii (N,3)
        -> [VoxelizeToVoxels]            -> render_output (dim0, dim1, dim2)

This operator is used for evaluation and visualization only (no loss, no backward).

Expected params:
    - means:     (N, 3)   world-space Gaussian centres
    - quats:     (N, 4)   unit quaternions (w, x, y, z)
    - scale_ori: (N, 3)   raw scale parameters
    - opacity:   (N,)     per-Gaussian activated opacity

Expected data:
    - ranges:    (3,)     physical extent per axis
    - centers:   (3,)     physical center per axis

Expected config:
    - vol_dim2:      int     voxel grid x dimension (fastest)
    - vol_dim1:      int     voxel grid y dimension (middle)
    - vol_dim0:      int     voxel grid z dimension (slowest)
    - vol_tile_size: int     (default 4)
    - eps3d:         float   (default 1e-4)
    - radius_clip:   float   (default 3.0 for Gaussian, 1.0 for Beta)
    - scale_min:     float   world-space activated scale lower bound
    - scale_max:     float   world-space activated scale upper bound

Author: Dy164
Date: 2026-03-19
"""

from typing import Optional

from ..model.beta import activated_beta_terms
from .base import ForwardBackwardBase
from .data import Config, Data, ForwardResult, Params
from .torch_autograd import activate_uncertainty


class GS3DCTVoxelizeCUDA(ForwardBackwardBase):
    """Forward-only voxelization to voxel volume."""

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
            world_to_volume_mapping_fwd,
            isect_tiles_3d, isect_offset_encode_3d,
            beta_voxelize_to_voxels_mean_var_fwd,
            beta_voxelize_to_voxels_fwd,
            voxelize_to_voxels_mean_var_fwd,
            voxelize_to_voxels_fwd,
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

        ranges = data.get('ranges')
        centers = data.get('centers')

        vol_dim2 = config.get('vol_dim2')
        vol_dim1 = config.get('vol_dim1')
        vol_dim0 = config.get('vol_dim0')
        vol_tile_size = config.get('vol_tile_size', 4)
        eps3d = config.get('eps3d', 1e-4)
        scale_min = config.get('scale_min', 0.01)
        scale_max = config.get('scale_max', 2.0)
        primitive_type = str(config.get('primitive_type', 'beta' if beta is not None else 'gaussian')).lower()
        radius_clip = config.get('radius_clip', 1.0 if primitive_type == "beta" else 3.0)
        beta_gamma_init = config.get('beta_gamma_init', 3.5)
        uncertainty_enabled = bool(config.get("uncertainty_enabled", False))
        uncertainty_q_min = float(config.get("uncertainty_q_min", 1.0e-6))
        uncertainty_q_max = config.get("uncertainty_q_max", None)

        # Stage 0: ParameterTransform
        covars, max_scales = parameter_transform_fwd(
            quats, scale_ori, scale_min, scale_max
        )
        if primitive_type == "beta":
            if beta is None:
                raise ValueError("primitive_type='beta' requires parameter beta")
            gamma = activated_beta_terms(beta, gamma_init=beta_gamma_init)[0]
        else:
            gamma = None

        # Stage 1: WorldToVolumeMapping
        means3d_v, conics, radii = world_to_volume_mapping_fwd(
            means, covars, max_scales, ranges, centers,
            vol_dim2, vol_dim1, vol_dim0, eps3d, radius_clip
        )

        # Stage 2: Tile intersections (3D)
        n_tiles_dim2 = (vol_dim2 + vol_tile_size - 1) // vol_tile_size
        n_tiles_dim1 = (vol_dim1 + vol_tile_size - 1) // vol_tile_size
        n_tiles_dim0 = (vol_dim0 + vol_tile_size - 1) // vol_tile_size

        _, isect_ids, flatten_ids = isect_tiles_3d(
            means3d_v, radii, vol_tile_size,
            n_tiles_dim2, n_tiles_dim1, n_tiles_dim0
        )
        tile_offsets = isect_offset_encode_3d(
            isect_ids, n_tiles_dim2, n_tiles_dim1, n_tiles_dim0
        )

        # Stage 3: VoxelizeToVoxels
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
                render_output, render_var = beta_voxelize_to_voxels_mean_var_fwd(
                    means3d_v, conics, opacity, gamma, q,
                    tile_offsets, flatten_ids,
                    vol_dim2, vol_dim1, vol_dim0, vol_tile_size
                )
            else:
                render_output, render_var = voxelize_to_voxels_mean_var_fwd(
                    means3d_v, conics, opacity, q,
                    tile_offsets, flatten_ids,
                    vol_dim2, vol_dim1, vol_dim0, vol_tile_size
                )
        elif primitive_type == "beta":
            render_output = beta_voxelize_to_voxels_fwd(
                means3d_v, conics, opacity, gamma,
                tile_offsets, flatten_ids,
                vol_dim2, vol_dim1, vol_dim0, vol_tile_size
            )
        else:
            render_output = voxelize_to_voxels_fwd(
                means3d_v, conics, opacity,
                tile_offsets, flatten_ids,
                vol_dim2, vol_dim1, vol_dim0, vol_tile_size
            )

        outputs = {'render_output': render_output}
        if render_var is not None:
            render_std = render_var.clamp_min(0).sqrt()
            outputs['render_output_var'] = render_var
            outputs['render_output_std'] = render_std
            outputs['render_output_cv'] = render_std / (render_output.abs() + 1.0e-8)
        return ForwardResult(outputs=outputs, intermediates={})

    def forward_backward1(self, params, data, config=None):
        raise NotImplementedError("GS3DCTVoxelizeCUDA is forward-only")


__all__ = ["GS3DCTVoxelizeCUDA"]
