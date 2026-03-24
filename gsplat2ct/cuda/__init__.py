"""
CUDA extensions for gsplat2ct

This module is intentionally minimal. The actual CUDA extension
loading is handled by _backend.py which supports both pre-compiled
and JIT compilation modes.

Following gsplat's pattern: don't directly import _C here to avoid
triggering compilation at import time.
"""

# Note: _C is loaded lazily through _backend.py
# Use: from gsplat2ct.cuda._backend import _C

# Export public API from _wrapper
from ._wrapper import (
    # MLP Classification
    mlp_classification_fwd,
    mlp_classification_bwd,
    mlp_classification_bwd2,
    # Image Loss
    image_loss_fwd,
    image_loss_bwd,
    image_loss_bwd2,
    # L2 Loss
    l2_loss_fwd,
    l2_loss_bwd,
    l2_loss_bwd2,
    # Volume Loss
    volume_loss_fwd,
    volume_loss_bwd,
    volume_loss_bwd2,
    # Isect Tiles (2D)
    isect_tiles,
    isect_offset_encode,
    # Isect Tiles (3D)
    isect_tiles_3d,
    isect_offset_encode_3d,
    # ConicsMu
    conics_mu_fwd,
    conics_mu_bwd,
    conics_mu_bwd2,
    # WorldToImageProjection
    world_to_image_projection_fwd,
    world_to_image_projection_bwd,
    world_to_image_projection_bwd2,
    # Utilities
    is_cuda_extension_available,
)

__all__ = [
    # MLP Classification
    "mlp_classification_fwd",
    "mlp_classification_bwd",
    "mlp_classification_bwd2",
    # Image Loss
    "image_loss_fwd",
    "image_loss_bwd",
    "image_loss_bwd2",
    # L2 Loss
    "l2_loss_fwd",
    "l2_loss_bwd",
    "l2_loss_bwd2",
    # Volume Loss
    "volume_loss_fwd",
    "volume_loss_bwd",
    "volume_loss_bwd2",
    # Isect Tiles (2D)
    "isect_tiles",
    "isect_offset_encode",
    # Isect Tiles (3D)
    "isect_tiles_3d",
    "isect_offset_encode_3d",
    # ConicsMu
    "conics_mu_fwd",
    "conics_mu_bwd",
    "conics_mu_bwd2",
    # WorldToImageProjection
    "world_to_image_projection_fwd",
    "world_to_image_projection_bwd",
    "world_to_image_projection_bwd2",
    # Utilities
    "is_cuda_extension_available",
]
