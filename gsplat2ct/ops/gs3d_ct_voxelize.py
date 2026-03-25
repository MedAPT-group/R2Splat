"""
GS3DCTVoxelizeCUDA: Forward-only voxelization from raw 3DGS parameters.

Pipeline:
    (means, quats, scale_ori, opacity)
        -> [ParameterTransform]          -> covars (N, 6), max_scales (N,)
        -> [WorldToVolumeMapping]        -> means3d_v (N,3), conics (N,6), radii (N,3)
        -> [VoxelizeToVoxels]            -> render_output (dim0, dim1, dim2)

This operator is used for evaluation and visualization only (no loss, no backward).

Expected params:
    - means:     (N, 3)   world-space Gaussian centres
    - quats:     (N, 4)   unit quaternions (w, x, y, z)
    - scale_ori: (N, 3)   raw scale parameters
    - opacity:   (N,)     per-Gaussian opacity

Expected data:
    - ranges:    (3,)     physical extent per axis
    - centers:   (3,)     physical center per axis

Expected config:
    - vol_dim2:      int     voxel grid x dimension (fastest)
    - vol_dim1:      int     voxel grid y dimension (middle)
    - vol_dim0:      int     voxel grid z dimension (slowest)
    - vol_tile_size: int     (default 4)
    - eps3d:         float   (default 1e-4)
    - radius_clip:   float   (default 3.0)
    - scale_min:     float   (default 0.01)
    - scale_max:     float   (default 2.0)

Author: Dy164
Date: 2026-03-19
"""

import torch
from typing import Optional, Tuple
from .base import SecondForwardBackwardBase
from .data import Params, Data, Config, ForwardResult, GradResult, HessResult


class GS3DCTVoxelizeCUDA(SecondForwardBackwardBase):
    """Forward-only voxelization from raw 3DGS parameters to voxel volume."""

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
            voxelize_to_voxels_fwd,
        )

        means = params.get('means').data
        quats = params.get('quats').data
        scale_ori = params.get('scale_ori').data
        opacity = params.get('opacity').data

        ranges = data.get('ranges')
        centers = data.get('centers')

        vol_dim2 = config.get('vol_dim2')
        vol_dim1 = config.get('vol_dim1')
        vol_dim0 = config.get('vol_dim0')
        vol_tile_size = config.get('vol_tile_size', 4)
        eps3d = config.get('eps3d', 1e-4)
        radius_clip = config.get('radius_clip', 3.0)
        scale_min = config.get('scale_min', 0.01)
        scale_max = config.get('scale_max', 2.0)

        # Stage 0: ParameterTransform
        covars, max_scales = parameter_transform_fwd(
            quats, scale_ori, scale_min, scale_max
        )

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
        render_output = voxelize_to_voxels_fwd(
            means3d_v, conics, opacity,
            tile_offsets, flatten_ids,
            vol_dim2, vol_dim1, vol_dim0, vol_tile_size
        )

        return ForwardResult(
            outputs={'render_output': render_output},
            intermediates={}
        )

    def forward_backward1(self, params, data, config=None):
        raise NotImplementedError("GS3DCTVoxelizeCUDA is forward-only")

    def forward_backward2(self, params, data, config=None):
        raise NotImplementedError("GS3DCTVoxelizeCUDA is forward-only")


__all__ = ["GS3DCTVoxelizeCUDA"]
