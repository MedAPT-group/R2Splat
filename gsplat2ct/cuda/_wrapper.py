"""
Python wrapper for CUDA ops with Lazy Loading

This module provides function wrappers that call the compiled CUDA
extension (_C module). Following gsplat's pattern, we use lazy loading
to avoid triggering JIT compilation at import time.

The CUDA extension is only loaded when a function is actually called,
not when this module is imported.

Author: Dy164
Date: 2025-12-02
Updated: 2025-12-16 (Added lazy loading following gsplat pattern)
"""

import torch
from typing import Dict, Callable, Any, Tuple


def _make_lazy_cuda_func(name: str) -> Callable:
    """
    Create a lazy-loading wrapper for a CUDA function.
    
    The CUDA extension (_C) is only imported when the function is
    actually called, not at module import time. This allows:
    - Faster initial import of gsplat2ct
    - JIT compilation only when CUDA functions are needed
    - Graceful handling when CUDA is not available
    
    Args:
        name: Name of the function in the _C module
        
    Returns:
        A wrapper function that loads _C on first call
    """
    def call_cuda(*args, **kwargs) -> Any:
        from ._backend import _C
        if _C is None:
            raise RuntimeError(
                f"CUDA extension not available. Cannot call {name}. "
                "Please ensure CUDA toolkit is installed and run: "
                "pip install -e . or BUILD_NO_CUDA=1 pip install -e ."
            )
        return getattr(_C, name)(*args, **kwargs)
    
    # Preserve function name for debugging
    call_cuda.__name__ = name
    call_cuda.__qualname__ = name
    
    return call_cuda


# ============================================================================
# Lazy-loaded CUDA functions
# ============================================================================

# These functions are NOT loaded at import time.
# The _C module is only imported when these functions are called.

_mlp_classification_fwd = _make_lazy_cuda_func("mlp_classification_fwd")
_mlp_classification_bwd = _make_lazy_cuda_func("mlp_classification_bwd")
_mlp_classification_bwd2 = _make_lazy_cuda_func("mlp_classification_bwd2")

# Image Loss
_image_loss_fwd = _make_lazy_cuda_func("image_loss_fwd")
_image_loss_bwd = _make_lazy_cuda_func("image_loss_bwd")
_image_loss_bwd2 = _make_lazy_cuda_func("image_loss_bwd2")

# L2 Loss
_l2_loss_fwd = _make_lazy_cuda_func("l2_loss_fwd")
_l2_loss_bwd = _make_lazy_cuda_func("l2_loss_bwd")
_l2_loss_bwd2 = _make_lazy_cuda_func("l2_loss_bwd2")

# Volume Loss
_volume_loss_fwd = _make_lazy_cuda_func("volume_loss_fwd")
_volume_loss_bwd = _make_lazy_cuda_func("volume_loss_bwd")
_volume_loss_bwd2 = _make_lazy_cuda_func("volume_loss_bwd2")

# Isect Tiles
_isect_tiles = _make_lazy_cuda_func("isect_tiles")
_isect_offset_encode = _make_lazy_cuda_func("isect_offset_encode")

# Isect Tiles 3D
_isect_tiles_3d = _make_lazy_cuda_func("isect_tiles_3d")
_isect_offset_encode_3d = _make_lazy_cuda_func("isect_offset_encode_3d")

# VoxelizeToVoxels
_voxelize_to_voxels_fwd = _make_lazy_cuda_func("voxelize_to_voxels_fwd")
_voxelize_to_voxels_bwd = _make_lazy_cuda_func("voxelize_to_voxels_bwd")
_voxelize_to_voxels_bwd2 = _make_lazy_cuda_func("voxelize_to_voxels_bwd2")

# Rasterize to Pixels
_rasterize_to_pixels_fwd = _make_lazy_cuda_func("rasterize_to_pixels_fwd")
_rasterize_to_pixels_bwd = _make_lazy_cuda_func("rasterize_to_pixels_bwd")
_rasterize_to_pixels_bwd2 = _make_lazy_cuda_func("rasterize_to_pixels_bwd2")

# ConicsMu
_conics_mu_fwd = _make_lazy_cuda_func("conics_mu_fwd")
_conics_mu_bwd = _make_lazy_cuda_func("conics_mu_bwd")
_conics_mu_bwd2 = _make_lazy_cuda_func("conics_mu_bwd2")

# WorldToImageProjection
_world_to_image_projection_fwd = _make_lazy_cuda_func("world_to_image_projection_fwd")
_world_to_image_projection_bwd = _make_lazy_cuda_func("world_to_image_projection_bwd")
_world_to_image_projection_bwd2 = _make_lazy_cuda_func("world_to_image_projection_bwd2")

# ParameterTransform
_parameter_transform_fwd = _make_lazy_cuda_func("parameter_transform_fwd")
_parameter_transform_bwd = _make_lazy_cuda_func("parameter_transform_bwd")
_parameter_transform_bwd2 = _make_lazy_cuda_func("parameter_transform_bwd2")

# WorldToVolumeMapping
_world_to_volume_mapping_fwd = _make_lazy_cuda_func("world_to_volume_mapping_fwd")
_world_to_volume_mapping_bwd = _make_lazy_cuda_func("world_to_volume_mapping_bwd")
_world_to_volume_mapping_bwd2 = _make_lazy_cuda_func("world_to_volume_mapping_bwd2")


# ============================================================================
# Public API with type hints and documentation
# ============================================================================

def mlp_classification_fwd(
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    save_intermediates: bool = True,
    compute_loss: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Forward pass for MLP classification.
    
    Args:
        w1: First layer weights (m, i)
        b1: First layer bias (m,)
        w2: Second layer weights (o, m)
        b2: Second layer bias (o,)
        x: Input features (i,) or (batch_size, i)
        y: One-hot labels (o,) or (batch_size, o)
        save_intermediates: Whether to save intermediate values
        compute_loss: Whether to compute cross-entropy loss
    
    Returns:
        Dictionary containing:
        - "sigma": Output probabilities
        - "loss": Cross-entropy loss (if compute_loss=True)
        - "a1", "z1", "a2": Intermediate values (if save_intermediates=True)
    """
    return _mlp_classification_fwd(
        w1, b1, w2, b2, x, y,
        save_intermediates, compute_loss
    )


def mlp_classification_bwd(
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    a1: torch.Tensor,
    z1: torch.Tensor,
    a2: torch.Tensor,
    sigma: torch.Tensor,
    accumulate_grad: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Backward pass for MLP classification (gradients only).
    
    Args:
        w1, b1, w2, b2: Network parameters
        x: Input features
        y: One-hot labels
        a1, z1, a2, sigma: Cached forward pass intermediates
        accumulate_grad: Whether to accumulate gradients
    
    Returns:
        Dictionary containing:
        - "grad_w1", "grad_b1", "grad_w2", "grad_b2": Gradients
    """
    return _mlp_classification_bwd(
        w1, b1, w2, b2, x, y,
        a1, z1, a2, sigma,
        accumulate_grad
    )


def mlp_classification_bwd2(
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    a1: torch.Tensor,
    z1: torch.Tensor,
    a2: torch.Tensor,
    sigma: torch.Tensor,
    accumulate_grad: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Backward pass for MLP classification (gradients + Hessians).
    
    Args:
        w1, b1, w2, b2: Network parameters
        x: Input features
        y: One-hot labels
        a1, z1, a2, sigma: Cached forward pass intermediates
        accumulate_grad: Whether to accumulate gradients
    
    Returns:
        Dictionary containing:
        - "grad_w1", "grad_b1", "grad_w2", "grad_b2": Gradients
        - "hess_b2_full", "hess_b1_full": Full Hessian matrices
        - "hess_w2_diag", "hess_w1_diag": Diagonal Hessian approximations
    """
    return _mlp_classification_bwd2(
        w1, b1, w2, b2, x, y,
        a1, z1, a2, sigma,
        accumulate_grad
    )


# ============================================================================
# Utility functions
# ============================================================================

def is_cuda_extension_available() -> bool:
    """
    Check if the CUDA extension is available.
    
    This function triggers lazy loading of the CUDA extension.
    Use this to check availability before calling CUDA functions.
    
    Returns:
        True if CUDA extension is loaded and available
    """
    try:
        from ._backend import _C
        return _C is not None
    except ImportError:
        return False


# ============================================================================
# Image Loss Public API
# ============================================================================

def image_loss_fwd(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    lambda_l2: float = 1.0,
    lambda_ssim: float = 0.2,
    compute_statistics: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Forward pass for image loss (L2 + D-SSIM).
    
    Args:
        prediction: Predicted images (B, H, W)
        target: Target images (B, H, W)
        window_size: SSIM window size (odd number)
        lambda_l2: Weight for L2 loss
        lambda_ssim: Weight for D-SSIM loss
        compute_statistics: Whether to save statistics for backward
    
    Returns:
        Tuple of (loss, l2_loss, ssim_loss, mu_r, mu_m, f1, f2, f3, f4)
    """
    return _image_loss_fwd(
        prediction, target, window_size,
        lambda_l2, lambda_ssim, compute_statistics
    )


def image_loss_bwd(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mu_r: torch.Tensor,
    mu_m: torch.Tensor,
    f1: torch.Tensor,
    f2: torch.Tensor,
    f3: torch.Tensor,
    f4: torch.Tensor,
    grad_output: torch.Tensor,
    window_size: int = 11,
    lambda_l2: float = 1.0,
    lambda_ssim: float = 0.2,
) -> torch.Tensor:
    """
    Backward pass for image loss (L2 + D-SSIM).
    
    Args:
        prediction, target: Input images
        mu_r, mu_m, f1, f2, f3, f4: Statistics from forward pass
        grad_output: Incoming gradient (scalar)
        window_size: SSIM window size
        lambda_l2, lambda_ssim: Loss weights
    
    Returns:
        Gradient w.r.t. prediction (B, H, W)
    """
    return _image_loss_bwd(
        prediction, target,
        mu_r, mu_m, f1, f2, f3, f4,
        grad_output, window_size,
        lambda_l2, lambda_ssim
    )


def image_loss_bwd2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mu_r: torch.Tensor,
    mu_m: torch.Tensor,
    f1: torch.Tensor,
    f2: torch.Tensor,
    f3: torch.Tensor,
    f4: torch.Tensor,
    grad_output: torch.Tensor,
    window_size: int = 11,
    lambda_l2: float = 1.0,
    lambda_ssim: float = 0.2,
) -> torch.Tensor:
    """
    Second-order backward pass for image loss (Hessian diagonal).
    
    Args:
        Same as image_loss_bwd
    
    Returns:
        Diagonal Hessian (B, H, W)
    """
    return _image_loss_bwd2(
        prediction, target,
        mu_r, mu_m, f1, f2, f3, f4,
        grad_output, window_size,
        lambda_l2, lambda_ssim
    )


# ============================================================================
# L2 Loss Public API
# ============================================================================

def l2_loss_fwd(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass for L2 (MSE) loss.
    
    Supports 3D tensors for both batched images (B, H, W)
    and unbatched volumes (D, H, W).
    
    Args:
        prediction: Predicted tensor (dim0, dim1, dim2)
        target: Target tensor (dim0, dim1, dim2)
    
    Returns:
        Tuple of (loss, l2_map)
    """
    return _l2_loss_fwd(prediction, target)


def l2_loss_bwd(
    prediction: torch.Tensor,
    target: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    """
    Backward pass for L2 (MSE) loss.
    
    Args:
        prediction: Predicted tensor (dim0, dim1, dim2)
        target: Target tensor (dim0, dim1, dim2)
        grad_output: Incoming gradient (scalar)
    
    Returns:
        Gradient w.r.t. prediction (dim0, dim1, dim2)
    """
    return _l2_loss_bwd(prediction, target, grad_output)


def l2_loss_bwd2(
    prediction: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    """
    Second-order backward pass for L2 (MSE) loss (Hessian diagonal).
    
    Args:
        prediction: Predicted tensor (for shape/dtype reference)
        grad_output: Incoming gradient (scalar)
    
    Returns:
        Diagonal Hessian (dim0, dim1, dim2)
    """
    return _l2_loss_bwd2(prediction, grad_output)


# ============================================================================
# Volume Loss Public API
# ============================================================================

def volume_loss_fwd(
    volume: torch.Tensor,
    eps: float = 1e-4,
    lambda_tv: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass for volume loss (Smooth Total Variation).
    
    Args:
        volume: Input volume (B, D, H, W)
        eps: Smoothing parameter for smooth TV
        lambda_tv: Weight for TV loss contribution
    
    Returns:
        Tuple of (loss, tv_loss, tv_map) where loss = lambda_tv * tv_loss
    """
    return _volume_loss_fwd(volume, eps, lambda_tv)


def volume_loss_bwd(
    volume: torch.Tensor,
    grad_output: torch.Tensor,
    eps: float = 1e-4,
    lambda_tv: float = 1.0,
) -> torch.Tensor:
    """
    Backward pass for volume loss (smooth TV).
    
    Args:
        volume: Input volume (B, D, H, W)
        grad_output: Incoming gradient (scalar)
        eps: Smoothing parameter for smooth TV
        lambda_tv: Weight for TV loss contribution
    
    Returns:
        Gradient w.r.t. volume (B, D, H, W)
    """
    return _volume_loss_bwd(volume, grad_output, eps, lambda_tv)


def volume_loss_bwd2(
    volume: torch.Tensor,
    grad_output: torch.Tensor,
    eps: float = 1e-4,
    lambda_tv: float = 1.0,
) -> torch.Tensor:
    """
    Second-order backward pass for volume loss (Hessian diagonal).
    
    Args:
        volume: Input volume (B, D, H, W)
        grad_output: Incoming gradient (scalar)
        eps: Smoothing parameter for smooth TV
        lambda_tv: Weight for TV loss contribution
    
    Returns:
        Diagonal Hessian (B, D, H, W)
    """
    return _volume_loss_bwd2(volume, grad_output, eps, lambda_tv)


# ============================================================================
# Isect Tiles Public API
# ============================================================================


def isect_tiles(
    means2d: torch.Tensor,
    radii: torch.Tensor,
    tile_size: int,
    tile_width: int,
    tile_height: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute tile-Gaussian intersections.
    
    Uses a two-pass algorithm:
    1. Count tiles per Gaussian
    2. Generate intersection records
    
    Optimized for CT reconstruction where depth sorting is not needed
    (ray summation instead of alpha blending). Results are always sorted
    by (camera_id, tile_id) for use with rasterization.
    
    Args:
        means2d: 2D projected Gaussian centers (C, N, 2)
        radii: Gaussian radii in pixels (C, N)
        tile_size: Size of each tile in pixels
        tile_width: Number of tiles in width
        tile_height: Number of tiles in height
    
    Returns:
        Tuple of:
        - tiles_per_gauss: Number of tiles per Gaussian (C, N)
        - isect_ids: Sorted encoded (camera_id, tile_id) for each intersection [n_isects]
        - flatten_ids: Sorted original flattened Gaussian index for each intersection [n_isects]
    
    Example:
        >>> means2d = torch.randn(4, 1000, 2, device='cuda')  # 4 cameras, 1000 Gaussians
        >>> radii = torch.randint(1, 10, (4, 1000), device='cuda', dtype=torch.int32)
        >>> tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        ...     means2d, radii, tile_size=16, tile_width=64, tile_height=48
        ... )
    """
    return _isect_tiles(means2d, radii, tile_size, tile_width, tile_height)


def isect_offset_encode(
    isect_ids: torch.Tensor,
    C: int,
    tile_width: int,
    tile_height: int,
) -> torch.Tensor:
    """
    Encode tile offsets from sorted isect_ids.
    
    Creates an offset array for efficient per-tile access to the
    sorted intersection data. offsets[c, h, w] gives the starting
    index in sorted flatten_ids for tile (c, h, w).
    
    Args:
        isect_ids: Sorted encoded (camera_id, tile_id) [n_isects]
        C: Number of cameras
        tile_width: Number of tiles in width
        tile_height: Number of tiles in height
    
    Returns:
        offsets: Per-tile starting offsets (C, tile_height, tile_width)
        
    Example:
        >>> tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(...)
        >>> offsets = isect_offset_encode(isect_ids, C=4, tile_width=64, tile_height=48)
        >>> # Access Gaussians for tile (c=0, y=10, x=20):
        >>> start = offsets[0, 10, 20]
        >>> end = offsets[0, 10, 21]  # or next tile
        >>> tile_gaussians = flatten_ids[start:end]
    """
    return _isect_offset_encode(isect_ids, C, tile_width, tile_height)


# ============================================================================
# Isect Tiles 3D Public API
# ============================================================================


def isect_tiles_3d(
    means3d: torch.Tensor,
    radii: torch.Tensor,
    tile_size: int,
    tile_width: int,
    tile_depth: int,
    tile_height: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute tile-Gaussian intersections in 3D.
    
    Uses a two-pass algorithm:
    1. Count tiles per Gaussian
    2. Generate intersection records
    
    Optimized for CT reconstruction where depth sorting is not needed
    (ray summation instead of alpha blending). Results are always sorted
    by tile_id for use with rasterization.
    
    Args:
        means3d: 3D Gaussian centers (N, 3)
        radii: Gaussian radii in each dimension (N, 3), dtype int32
        tile_size: Size of each tile in voxels
        tile_width: Number of tiles in x dimension
        tile_depth: Number of tiles in y dimension
        tile_height: Number of tiles in z dimension
    
    Returns:
        Tuple of:
        - tiles_per_gauss: Number of tiles per Gaussian (N,)
        - isect_ids: Sorted tile_id for each intersection [n_isects]
        - flatten_ids: Sorted original Gaussian index for each intersection [n_isects]
    
    Example:
        >>> means3d = torch.randn(1000, 3, device='cuda')  # 1000 Gaussians
        >>> radii = torch.randint(1, 10, (1000, 3), device='cuda', dtype=torch.int32)
        >>> tiles_per_gauss, isect_ids, flatten_ids = isect_tiles_3d(
        ...     means3d, radii, tile_size=16, tile_width=64, tile_depth=64, tile_height=48
        ... )
    """
    return _isect_tiles_3d(means3d, radii, tile_size, tile_width, tile_depth, tile_height)


def isect_offset_encode_3d(
    isect_ids: torch.Tensor,
    tile_width: int,
    tile_depth: int,
    tile_height: int,
) -> torch.Tensor:
    """
    Encode tile offsets from sorted isect_ids for 3D.
    
    Creates an offset array for efficient per-tile access to the
    sorted intersection data. offsets[z, y, x] gives the starting
    index in sorted flatten_ids for tile at position (z, y, x).
    
    Args:
        isect_ids: Sorted tile_ids [n_isects]
        tile_width: Number of tiles in x dimension
        tile_depth: Number of tiles in y dimension
        tile_height: Number of tiles in z dimension
    
    Returns:
        offsets: Per-tile starting offsets (tile_height, tile_depth, tile_width)
        
    Example:
        >>> tiles_per_gauss, isect_ids, flatten_ids = isect_tiles_3d(...)
        >>> offsets = isect_offset_encode_3d(isect_ids, tile_width=64, tile_depth=64, tile_height=48)
        >>> # Access Gaussians for tile at (z=10, y=20, x=30):
        >>> tile_id = 10 * 64 * 64 + 20 * 64 + 30
        >>> start = offsets[10, 20, 30]
        >>> # Find end by looking at next tile or using tile_id + 1
        >>> tile_gaussians = flatten_ids[start:end]
    """
    return _isect_offset_encode_3d(isect_ids, tile_width, tile_depth, tile_height)


# ============================================================================
# VoxelizeToVoxels Public API
# ============================================================================


def voxelize_to_voxels_fwd(
    means3d: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    tile_offsets: torch.Tensor,
    flatten_ids: torch.Tensor,
    dim2: int,
    dim1: int,
    dim0: int,
    tile_size: int = 4,
) -> torch.Tensor:
    """
    Voxelize 3D Gaussians to a voxel grid for CT reconstruction.
    
    Tile-based voxelization with shared memory optimization.
    CT formula: voxel_value = Σ (opacity × exp(-sigma))
    where sigma = 0.5 × delta^T × conic × delta
    
    This function performs the forward pass of CT voxelization,
    accumulating contributions from all intersecting Gaussians to each voxel.
    
    Dimension convention:
    - dim2: x-axis (fastest varying, corresponds to means3d[..., 0])
    - dim1: y-axis (middle dimension, corresponds to means3d[..., 1])
    - dim0: z-axis (slowest varying, corresponds to means3d[..., 2])
    - Output shape: (dim0, dim1, dim2)
    
    Args:
        means3d: 3D Gaussian centers (N, 3), format: (x, y, z) = (dim2, dim1, dim0)
        conics: Inverse covariance matrices (N, 6), stored as upper triangular
                elements [c00, c01, c02, c11, c12, c22]
        opacities: Gaussian opacities (N)
        tile_offsets: Per-tile starting offsets (n_tiles_dim0, n_tiles_dim1, n_tiles_dim2)
                      from isect_offset_encode_3d
        flatten_ids: Original Gaussian index for each intersection [n_isects]
                     from isect_tiles_3d
        dim2: Output size in dimension 2 (x-axis, fastest varying)
        dim1: Output size in dimension 1 (y-axis, middle)
        dim0: Output size in dimension 0 (z-axis, slowest varying)
        tile_size: Size of each tile in voxels (default 4)
    
    Returns:
        render_output: Voxelized volume (dim0, dim1, dim2)
    
    Example:
        >>> # First compute tile intersections
        >>> means3d = torch.randn(1000, 3, device='cuda')  # 1000 Gaussians
        >>> radii = torch.randint(1, 10, (1000, 3), device='cuda', dtype=torch.int32)
        >>> tiles_per_gauss, isect_ids, flatten_ids = isect_tiles_3d(
        ...     means3d, radii, tile_size=4,
        ...     tile_width=32, tile_depth=32, tile_height=24
        ... )
        >>> tile_offsets = isect_offset_encode_3d(isect_ids, 
        ...     tile_width=32, tile_depth=32, tile_height=24)
        >>> 
        >>> # Then voxelize
        >>> conics = torch.randn(1000, 6, device='cuda')  # Inverse covariance
        >>> opacities = torch.rand(1000, device='cuda')
        >>> render_output = voxelize_to_voxels_fwd(
        ...     means3d, conics, opacities,
        ...     tile_offsets, flatten_ids,
        ...     dim2=128, dim1=128, dim0=96, tile_size=4
        ... )
    """
    return _voxelize_to_voxels_fwd(
        means3d, conics, opacities,
        tile_offsets, flatten_ids,
        dim2, dim1, dim0, tile_size
    )


def voxelize_to_voxels_bwd(
    means3d: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    tile_offsets: torch.Tensor,
    flatten_ids: torch.Tensor,
    v_render_output: torch.Tensor,
    dim2: int,
    dim1: int,
    dim0: int,
    tile_size: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Backward pass for CT voxelization (first-order gradients).
    
    Computes gradients of the loss w.r.t. Gaussian parameters.
    
    CT formula: G = o × exp(-σ)
    Gradients:
    - ∂G/∂o = exp(-σ)
    - ∂G/∂m = o × exp(-σ) × (Σ^{-1} × (x - m))
    - ∂G/∂Σ^{-1} = -o × exp(-σ) × 0.5 × (x - m)(x - m)^T
    
    Args:
        means3d: 3D Gaussian centers (N, 3)
        conics: Inverse covariance matrices (N, 6)
        opacities: Gaussian opacities (N)
        tile_offsets: Per-tile starting offsets (n_tiles_dim0, n_tiles_dim1, n_tiles_dim2)
        flatten_ids: Gaussian indices for each intersection [n_isects]
        v_render_output: Upstream gradient (dim0, dim1, dim2)
        dim2, dim1, dim0: Output dimensions
        tile_size: Size of each tile in voxels
    
    Returns:
        v_means3d: Gradient w.r.t. means3d (N, 3)
        v_conics: Gradient w.r.t. conics (N, 6)
        v_opacities: Gradient w.r.t. opacities (N)
    """
    return _voxelize_to_voxels_bwd(
        means3d, conics, opacities,
        tile_offsets, flatten_ids,
        v_render_output,
        dim2, dim1, dim0, tile_size
    )


def voxelize_to_voxels_bwd2(
    means3d: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    tile_offsets: torch.Tensor,
    flatten_ids: torch.Tensor,
    v_render_output: torch.Tensor,
    h_render_output: torch.Tensor,
    dim2: int,
    dim1: int,
    dim0: int,
    tile_size: int = 4,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,  # First-order gradients
    torch.Tensor, torch.Tensor, torch.Tensor,  # Hessian diagonals
    torch.Tensor, torch.Tensor, torch.Tensor,  # Cross-terms
]:
    """
    Second-order backward pass for CT voxelization (gradients + Hessians).
    
    Computes both first-order gradients and second-order Hessians using chain rule:
    v_θ = v_O × ∂G/∂θ
    H_{θi,θj} = h_O × (∂G/∂θi)(∂G/∂θj) + v_O × ∂²G/∂θi∂θj
    
    Hessian storage uses upper triangular elements:
    - h_means3d: (N, 6) upper triangle of 3×3 matrix [H00, H01, H02, H11, H12, H22]
    - h_conics: (N, 21) upper triangle of 6×6 matrix
    - h_opacities: (N) scalar
    - h_op_mean: (N, 3) cross term opacity-means3d
    - h_op_conic: (N, 6) cross term opacity-conics
    - h_mean_conic: (N, 18) = (N, 3, 6) cross term means3d-conics
    
    Args:
        means3d: 3D Gaussian centers (N, 3)
        conics: Inverse covariance matrices (N, 6)
        opacities: Gaussian opacities (N)
        tile_offsets: Per-tile starting offsets
        flatten_ids: Gaussian indices for each intersection
        v_render_output: First-order upstream gradient (dim0, dim1, dim2)
        h_render_output: Second-order upstream gradient (dim0, dim1, dim2)
        dim2, dim1, dim0: Output dimensions
        tile_size: Size of each tile in voxels
    
    Returns:
        v_means3d: First-order gradient w.r.t. means3d (N, 3)
        v_conics: First-order gradient w.r.t. conics (N, 6)
        v_opacities: First-order gradient w.r.t. opacities (N)
        h_means3d: Hessian upper triangle w.r.t. means3d (N, 6)
        h_conics: Hessian upper triangle w.r.t. conics (N, 21)
        h_opacities: Hessian w.r.t. opacities (N)
        h_op_mean: Cross-term Hessian opacity-means3d (N, 3)
        h_op_conic: Cross-term Hessian opacity-conics (N, 6)
        h_mean_conic: Cross-term Hessian means3d-conics (N, 18)
    """
    return _voxelize_to_voxels_bwd2(
        means3d, conics, opacities,
        tile_offsets, flatten_ids,
        v_render_output, h_render_output,
        dim2, dim1, dim0, tile_size
    )


def rasterize_to_pixels_fwd(
    means2d: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    mus: torch.Tensor,
    tile_offsets: torch.Tensor,
    flatten_ids: torch.Tensor,
    dim1: int,
    dim0: int,
    tile_size: int = 16,
) -> torch.Tensor:
    """
    Rasterize Gaussians to pixels for CT reconstruction.
    
    Tile-based rasterization with shared memory optimization.
    CT formula: pixel_value = Σ (opacity × mu × exp(-sigma))
    
    This function performs the forward pass of CT rasterization,
    accumulating contributions from all visible Gaussians to each pixel.
    
    Dimension convention:
    - dim0: Number of rows (y-axis), corresponds to means2d[..., 1]
    - dim1: Number of columns (x-axis), corresponds to means2d[..., 0]
    - Output shape: (C, dim0, dim1)
    
    Args:
        means2d: 2D projected Gaussian centers (C, N, 2), format: (x, y) = (col, row)
        conics: Inverse covariance matrices (C, N, 3), stored as [a, b, c]
                representing [[a, b], [b, c]]
        opacities: Gaussian opacities (C, N), values in [-1, 1]
        mus: Gaussian mu values (CT attenuation coefficients) (C, N)
        tile_offsets: Per-tile starting offsets (C, n_tiles_dim0, n_tiles_dim1)
                      from isect_offset_encode
        flatten_ids: Original flattened Gaussian index [n_isects]
                     from isect_tiles
        dim1: Output size in dimension 1 (columns, x-axis)
        dim0: Output size in dimension 0 (rows, y-axis)
        tile_size: Size of each tile in pixels (default 16)
    
    Returns:
        render_output: Rendered CT image (C, dim0, dim1)
    
    Example:
        >>> # First compute tile intersections
        >>> tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        ...     means2d, radii, tile_size=16, n_tiles_dim1=64, n_tiles_dim0=48
        ... )
        >>> tile_offsets = isect_offset_encode(isect_ids, C=4, tile_width=64, tile_height=48)
        >>> 
        >>> # Then rasterize
        >>> render_output = rasterize_to_pixels_fwd(
        ...     means2d, conics, opacities, mus,
        ...     tile_offsets, flatten_ids,
        ...     dim1=1024, dim0=768, tile_size=16
        ... )
    """
    return _rasterize_to_pixels_fwd(
        means2d, conics, opacities, mus,
        tile_offsets, flatten_ids,
        dim1, dim0, tile_size
    )


def rasterize_to_pixels_bwd(
    means2d: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    mus: torch.Tensor,
    tile_offsets: torch.Tensor,
    flatten_ids: torch.Tensor,
    v_render_output: torch.Tensor,
    dim1: int,
    dim0: int,
    tile_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Backward pass for CT rasterization.
    
    Computes gradients of the loss w.r.t. Gaussian parameters:
    means2d, conics, opacities, and mus.
    
    CT Formula: O = Σ (opacity × mu × exp(-sigma))
    where sigma = 0.5 × (a×δx² + c×δy²) + b×δx×δy
    
    Unlike 3DGS which requires back-to-front processing due to alpha blending,
    CT uses simple summation so gradient computation is order-independent.
    
    Dimension convention:
    - dim0: Number of rows (y-axis), corresponds to means2d[..., 1]
    - dim1: Number of columns (x-axis), corresponds to means2d[..., 0]
    
    Args:
        means2d: 2D projected Gaussian centers (C, N, 2), format: (x, y) = (col, row)
        conics: Inverse covariance matrices (C, N, 3), stored as [a, b, c]
                representing [[a, b], [b, c]]
        opacities: Gaussian opacities (C, N)
        mus: Gaussian mu values (CT attenuation coefficients) (C, N)
        tile_offsets: Per-tile starting offsets (C, n_tiles_dim0, n_tiles_dim1)
        flatten_ids: Original flattened Gaussian index [n_isects]
        v_render_output: Upstream gradient ∂L/∂O (C, dim0, dim1)
        dim1: Output size in dimension 1 (columns, x-axis)
        dim0: Output size in dimension 0 (rows, y-axis)
        tile_size: Size of each tile in pixels (default 16)
    
    Returns:
        Tuple of gradients:
        - v_means2d: Gradient w.r.t. means2d (C, N, 2)
        - v_conics: Gradient w.r.t. conics (C, N, 3)
        - v_opacities: Gradient w.r.t. opacities (C, N)
        - v_mus: Gradient w.r.t. mus (C, N)
    
    Example:
        >>> # Forward pass
        >>> render_output = rasterize_to_pixels_fwd(
        ...     means2d, conics, opacities, mus,
        ...     tile_offsets, flatten_ids, dim1, dim0
        ... )
        >>> 
        >>> # Compute loss and upstream gradient
        >>> loss = (render_output - target).pow(2).sum()
        >>> v_render_output = 2.0 * (render_output - target)
        >>> 
        >>> # Backward pass
        >>> v_means2d, v_conics, v_opacities, v_mus = rasterize_to_pixels_bwd(
        ...     means2d, conics, opacities, mus,
        ...     tile_offsets, flatten_ids, v_render_output,
        ...     dim1, dim0
        ... )
    """
    return _rasterize_to_pixels_bwd(
        means2d, conics, opacities, mus,
        tile_offsets, flatten_ids, v_render_output,
        dim1, dim0, tile_size
    )


def rasterize_to_pixels_bwd2(
    means2d: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    mus: torch.Tensor,
    tile_offsets: torch.Tensor,
    flatten_ids: torch.Tensor,
    v_render_output: torch.Tensor,
    h_render_output: torch.Tensor,
    dim1: int,
    dim0: int,
    tile_size: int = 16,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,  # First-order
    torch.Tensor, torch.Tensor,  # Hessian upper triangles
    torch.Tensor, torch.Tensor, torch.Tensor,  # Cross-terms (mu-mean, mu-conic, mean-conic)
    torch.Tensor, torch.Tensor, torch.Tensor,  # Opacity cross-terms (mu-opacity, conic-opacity, mean-opacity)
    torch.Tensor, torch.Tensor  # Scalar Hessians (h_mus, h_opacities)
]:
    """
    CT Rasterization backward pass with second-order derivatives (Hessian).
    
    Computes both first-order gradients and second-order derivatives.
    
    For CT reconstruction: O = Σ_i G_i where G_i = μ_i × o_i × exp(-σ_i)
    
    Given upstream:
        v_O = ∂L/∂O (first derivative)
        h_O = ∂²L/∂O² (second derivative / Hessian diagonal)
    
    We compute using chain rule:
        v_θ = v_O × ∂G/∂θ
        h_{θi,θj} = h_O × (∂G/∂θi)(∂G/∂θj) + v_O × ∂²G/∂θi∂θj
    
    Hessian storage uses upper triangular elements only (consistent with conics):
        - means2d: [H00, H01, H11] (3 elements per Gaussian)
        - conics: [H00, H01, H02, H11, H12, H22] (6 elements per Gaussian)
    
    Args:
        means2d: 2D projected Gaussian centers (C, N, 2)
        conics: Inverse covariance matrices (C, N, 3)
        opacities: Gaussian opacities (C, N)
        mus: Gaussian mu values (C, N)
        tile_offsets: Per-tile starting offsets (C, n_tiles_dim0, n_tiles_dim1)
        flatten_ids: Original flattened Gaussian index [n_isects]
        v_render_output: First-order upstream gradient ∂L/∂O (C, dim0, dim1)
        h_render_output: Second-order upstream gradient ∂²L/∂O² (C, dim0, dim1)
        dim1: Output size in dimension 1 (columns)
        dim0: Output size in dimension 0 (rows)
        tile_size: Size of each tile in pixels (default 16)
    
    Returns:
        Tuple of:
        - v_means2d: First-order gradient w.r.t. means2d (C, N, 2)
        - v_conics: First-order gradient w.r.t. conics (C, N, 3)
        - v_opacities: First-order gradient w.r.t. opacities (C, N)
        - v_mus: First-order gradient w.r.t. mus (C, N)
        - h_means2d: Hessian upper triangle w.r.t. means2d (C, N, 3)
        - h_conics: Hessian upper triangle w.r.t. conics (C, N, 6)
        - h_mu_mean: Cross-term Hessian μ-mean (C, N, 2)
        - h_mu_conic: Cross-term Hessian μ-conic (C, N, 3)
        - h_mean_conic: Cross-term Hessian mean-conic (C, N, 2, 3)
        - h_mu_opacity: Cross-term Hessian μ-opacity (C, N)
        - h_conic_opacity: Cross-term Hessian conic-opacity (C, N, 3)
        - h_mean_opacity: Cross-term Hessian mean-opacity (C, N, 2)
        - h_mus: Scalar Hessian w.r.t. mus (C, N)
        - h_opacities: Scalar Hessian w.r.t. opacities (C, N)
    """
    return _rasterize_to_pixels_bwd2(
        means2d, conics, opacities, mus,
        tile_offsets, flatten_ids, v_render_output, h_render_output,
        dim1, dim0, tile_size
    )


# ============================================================================
# ConicsMu Operations
# ============================================================================

def conics_mu_fwd(
    cov2d_3x3: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute conics and mu from projected 3x3 covariance matrices with blur.
    
    Mathematical notation:
        hat{Sigma}_b = hat{Sigma} + eps * I_2   (blurred upper-left 2x2)
        conics = hat{Sigma}_b^{-1}  stored as (u11, u12, u22)
        mu = sqrt(2*pi * det(tilde{Sigma}_b) / det(hat{Sigma}_b))
    
    Args:
        cov2d_3x3: Projected 3x3 covariance matrices (C, N, 6)
                    stored as upper triangular (a, b, c, d, e, f)
        eps: Blur epsilon added to diagonal before inversion
    
    Returns:
        Tuple of:
        - conics: Inverse of blurred 2x2 (C, N, 3) as (u11, u12, u22)
        - mus: Compensation factor (C, N)
    """
    return _conics_mu_fwd(cov2d_3x3, eps)


def conics_mu_bwd(
    cov2d_3x3: torch.Tensor,
    eps: float,
    conics: torch.Tensor,
    mus: torch.Tensor,
    v_conics: torch.Tensor,
    v_mus: torch.Tensor,
) -> torch.Tensor:
    """
    Backward pass for ConicsMu: compute gradient w.r.t. cov2d_3x3.
    
    Recomputes cov3d_inv = (tilde{Sigma} + eps*I_3)^{-1} locally.
    
    Key formulas:
        d(conics)/d(Sigma): dP_ij/dS_kl = -P_ik * P_lj
        d(mu)/d(Sigma) = (mu/2) * (cov3d_inv - pad(conics))
    
    Args:
        cov2d_3x3: Forward input: projected covariance (C, N, 6)
        eps: Blur epsilon (same value used in forward)
        conics: Forward output: hat{Sigma}_b^{-1} (C, N, 3)
        mus: Forward output: compensation factor (C, N)
        v_conics: Upstream gradient dL/d(conics) (C, N, 3)
        v_mus: Upstream gradient dL/d(mus) (C, N)
    
    Returns:
        v_cov2d_3x3: Gradient dL/d(cov2d_3x3) (C, N, 6)
    """
    return _conics_mu_bwd(cov2d_3x3, eps, conics, mus, v_conics, v_mus)


def conics_mu_bwd2(
    cov2d_3x3: torch.Tensor,
    eps: float,
    conics: torch.Tensor,
    mus: torch.Tensor,
    v_conics: torch.Tensor,
    v_mus: torch.Tensor,
    h_conics: torch.Tensor,
    h_conics_mu: torch.Tensor,
    h_mu_mu: torch.Tensor,
    h_means2d_conics: torch.Tensor,
    h_conics_opacity: torch.Tensor,
    h_mu_means2d: torch.Tensor,
    h_mu_opacity: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Second-order backward pass for ConicsMu.
    
    Computes upstream gradients and Hessians w.r.t. cov2d_3x3,
    plus cross-Hessians with means2d and opacity.
    
    Args:
        cov2d_3x3: Forward input (C, N, 6)
        eps: Blur epsilon
        conics: Forward output conics (C, N, 3)
        mus: Forward output mu (C, N)
        v_conics: Downstream dL/d(conics) (C, N, 3)
        v_mus: Downstream dL/d(mu) (C, N)
        h_conics: H_{conics,conics} upper-tri (C, N, 6)
        h_conics_mu: H_{conics,mu} (C, N, 3)
        h_mu_mu: H_{mu,mu} (C, N)
        h_means2d_conics: H_{means2d,conics} (C, N, 6) [2x3 row-major]
            Layout: [H_{mx,u11}, H_{mx,u12}, H_{mx,u22},
                     H_{my,u11}, H_{my,u12}, H_{my,u22}]
            Matches RasterizeToPixels bwd2 h_mean_conic output directly.
        h_conics_opacity: H_{conics,opacity} (C, N, 3)
        h_mu_means2d: H_{mu,means2d} (C, N, 2)
        h_mu_opacity: H_{mu,opacity} (C, N)
    
    Returns:
        Tuple of:
        - v_cov2d_3x3: Gradient (C, N, 6)
        - h_cov_cov: H_{sigma,sigma} upper-tri (C, N, 21)
        - h_cov_means2d: H_{sigma,means2d} (C, N, 12)
        - h_cov_opacity: H_{sigma,opacity} (C, N, 6)
    """
    return _conics_mu_bwd2(
        cov2d_3x3, eps, conics, mus,
        v_conics, v_mus,
        h_conics, h_conics_mu, h_mu_mu,
        h_means2d_conics, h_conics_opacity, h_mu_means2d, h_mu_opacity,
    )


# ============================================================================
# WorldToImageProjection Operations
# ============================================================================

def world_to_image_projection_fwd(
    means: torch.Tensor,
    covars: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    dim1: int,
    dim0: int,
    eps_blur: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """World-to-image projection forward pass.
    
    Fused world-to-camera transform + perspective projection.
    Computes projected 2D means, 3x3 projected covariance, and radii.
    
    Pipeline: p_c = R*p+t → perspective proj → FOV clamp → J → cov2d_3x3 → radii

    Args:
        means: World-space Gaussian centers (N, 3), shared across cameras
        covars: World-space covariance upper-tri (N, 6), shared across cameras
        viewmats: Camera extrinsic matrices (C, 4, 4) row-major
        Ks: Camera intrinsic matrices (C, 3, 3) row-major
        dim1: Image width in pixels
        dim0: Image height in pixels
        eps_blur: Blur epsilon for radii computation (default 0.3)
    
    Returns:
        Tuple of:
        - means2d: Projected 2D centers (C, N, 2)
        - cov2d_3x3: Projected 3x3 covariance upper-tri (C, N, 6)
        - radii: Integer radii for tile intersection (C, N), int32
    """
    return _world_to_image_projection_fwd(
        means, covars, viewmats, Ks, dim1, dim0, eps_blur
    )


def world_to_image_projection_bwd(
    means: torch.Tensor,
    covars: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    dim1: int,
    dim0: int,
    v_means2d: torch.Tensor,
    v_cov2d_3x3: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """World-to-image projection backward pass (first-order gradients).
    
    Computes dL/d(means) and dL/d(covars) from downstream gradients.
    Gradients are accumulated over C cameras.

    Args:
        means: Forward input (N, 3), shared across cameras
        covars: Forward input (N, 6), shared across cameras
        viewmats: Camera extrinsic (C, 4, 4)
        Ks: Camera intrinsic (C, 3, 3)
        dim1: Image width
        dim0: Image height
        v_means2d: Upstream dL/d(means2d) (C, N, 2)
        v_cov2d_3x3: Upstream dL/d(cov2d_3x3) (C, N, 6)
    
    Returns:
        Tuple of:
        - v_means: Gradient w.r.t. means (N, 3), summed over C
        - v_covars: Gradient w.r.t. covars (N, 6), summed over C
    """
    return _world_to_image_projection_bwd(
        means, covars, viewmats, Ks, dim1, dim0,
        v_means2d, v_cov2d_3x3,
    )


def world_to_image_projection_bwd2(
    means: torch.Tensor,
    covars: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    dim1: int,
    dim0: int,
    v_means2d: torch.Tensor,
    v_cov2d_3x3: torch.Tensor,
    h_means2d: torch.Tensor,
    h_cov2d_3x3: torch.Tensor,
    h_means2d_cov2d_3x3: torch.Tensor,
    h_means2d_opacity: torch.Tensor,
    h_cov2d_3x3_opacity: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    """World-to-image projection second-order backward pass (Hessians).
    
    Uses Gauss-Newton approximation for the position Hessian.
    Covars Hessian is exact (linear map).
    Gradients and Hessians for means/covars are accumulated over C cameras.
    Cross-Hessians with opacity remain per-camera.

    Args:
        means: Forward input (N, 3), shared across cameras
        covars: Forward input (N, 6), shared across cameras
        viewmats: Camera extrinsic (C, 4, 4)
        Ks: Camera intrinsic (C, 3, 3)
        dim1: Image width
        dim0: Image height
        v_means2d: Downstream gradient (C, N, 2)
        v_cov2d_3x3: Downstream gradient (C, N, 6)
        h_means2d: Downstream Hessian upper-tri 2x2 (C, N, 3)
        h_cov2d_3x3: Downstream Hessian upper-tri 6x6 (C, N, 21)
        h_means2d_cov2d_3x3: Downstream cross-Hessian (C, N, 12)
        h_means2d_opacity: Downstream cross-Hessian (C, N, 2)
        h_cov2d_3x3_opacity: Downstream cross-Hessian (C, N, 6)
    
    Returns:
        Tuple of:
        - v_means: Gradient (N, 3), summed over C
        - v_covars: Gradient (N, 6), summed over C
        - h_means: Hessian H_{p,p} upper-tri (N, 6), summed over C
        - h_covars: Hessian H_{s,s} upper-tri (N, 21), summed over C
        - h_means_covars: Cross-Hessian H_{p,s} (N, 18), summed over C
        - h_means_opacity: Cross-Hessian (C, N, 3), per-camera
        - h_covars_opacity: Cross-Hessian (C, N, 6), per-camera
    """
    return _world_to_image_projection_bwd2(
        means, covars, viewmats, Ks, dim1, dim0,
        v_means2d, v_cov2d_3x3,
        h_means2d, h_cov2d_3x3, h_means2d_cov2d_3x3,
        h_means2d_opacity, h_cov2d_3x3_opacity,
    )


# ============================================================================
# ParameterTransform Operations
# ============================================================================

def parameter_transform_fwd(
    quats: torch.Tensor,
    scale_ori: torch.Tensor,
    scale_min: float,
    scale_max: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Parameter transform forward pass: (quats, scale_ori) -> (covar, max_scales).
    
    Transforms unit quaternions and raw scale parameters into world-space
    3D covariance matrices, and also outputs max activated scale per Gaussian.

    Pipeline:
        s = sigmoid(scale_ori) * (scale_max - scale_min) + scale_min
        R = quat_to_rotmat(quats)   (unit quaternion, no normalization)
        M = R * diag(s)
        covars = M * M^T  (stored as upper-tri 6 elements)
        max_scales = max(s[0], s[1], s[2])

    Args:
        quats: Unit quaternions (N, 4) as (w, x, y, z), ||q||=1
        scale_ori: Raw scale parameters (N, 3) before sigmoid activation
        scale_min: Minimum scale after activation
        scale_max: Maximum scale after activation
    
    Returns:
        Tuple of:
        - covars: World-space covariance upper-tri (N, 6)
        - max_scales: Max activated scale per Gaussian (N)
    """
    return _parameter_transform_fwd(quats, scale_ori, scale_min, scale_max)


def parameter_transform_bwd(
    quats: torch.Tensor,
    scale_ori: torch.Tensor,
    scale_min: float,
    scale_max: float,
    v_covars: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Parameter transform backward pass (first-order gradients).
    
    Computes dL/d(quats) and dL/d(scale_ori) from downstream dL/d(covars).

    Args:
        quats: Forward input (N, 4)
        scale_ori: Forward input (N, 3)
        scale_min: Min scale
        scale_max: Max scale
        v_covars: Downstream dL/d(covars) (N, 6)
    
    Returns:
        Tuple of:
        - v_quats: Gradient w.r.t. quats (N, 4)
        - v_scale_ori: Gradient w.r.t. scale_ori (N, 3)
    """
    return _parameter_transform_bwd(
        quats, scale_ori, scale_min, scale_max, v_covars
    )


def parameter_transform_bwd2(
    quats: torch.Tensor,
    scale_ori: torch.Tensor,
    scale_min: float,
    scale_max: float,
    v_covars: torch.Tensor,
    h_covars: torch.Tensor,
    h_covars_means: torch.Tensor,
    h_covars_opacity: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    """Parameter transform second-order backward pass (Hessians).
    
    Given downstream first-order gradients and second-order Hessians w.r.t.
    covars, plus cross-Hessians with mean and opacity, compute upstream
    gradients and Hessians w.r.t. (quats, scale_ori), plus cross-Hessians.

    Args:
        quats: Forward input (N, 4)
        scale_ori: Forward input (N, 3)
        scale_min: Min scale
        scale_max: Max scale
        v_covars: Downstream dL/d(covars) (N, 6)
        h_covars: Downstream H_{c,c} upper-tri (N, 21)
        h_covars_means: Downstream H_{c,mean} (N, 18) 6x3 row-major
        h_covars_opacity: Downstream H_{c,opacity} (N, 6)
    
    Returns:
        Tuple of:
        - v_quats (N, 4): First-order gradient
        - v_scale_ori (N, 3): First-order gradient
        - h_quats (N, 10): H_{q,q} upper-tri 4x4
        - h_scale_ori (N, 3): H_{s,s} diagonal
        - h_quats_scale_ori (N, 12): H_{q,s} 4x3 row-major
        - h_quats_means (N, 12): H_{q,mean} 4x3 row-major
        - h_quats_opacity (N, 4): H_{q,opacity}
        - h_scale_ori_means (N, 9): H_{s,mean} 3x3 row-major
        - h_scale_ori_opacity (N, 3): H_{s,opacity}
    """
    return _parameter_transform_bwd2(
        quats, scale_ori, scale_min, scale_max,
        v_covars, h_covars, h_covars_means, h_covars_opacity,
    )


# ============================================================================
# WorldToVolumeMapping Operations
# ============================================================================

def world_to_volume_mapping_fwd(
    means: torch.Tensor,
    covars: torch.Tensor,
    max_scales: torch.Tensor,
    ranges: torch.Tensor,
    centers: torch.Tensor,
    dim2: int,
    dim1: int,
    dim0: int,
    eps3d: float = 1e-4,
    radius_clip: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """World-to-volume mapping forward pass.

    Transforms world-space Gaussian parameters into voxel-space.
    Computes voxel-space means, conics (inverse covariance), and radii.

    Pipeline:
        r[k] = dim[k] / range[k]
        means3d_v[k] = (means[k] - center[k]) * r[k] + dim[k]/2
        Sigma_v[i][j] = r[i]*r[j]*covars[i][j] + eps3d*delta_{ij}
        conics = Sigma_v^{-1}
        radii[k] = ceil(radius_clip * max_scales * r[k])

    Args:
        means: World-space Gaussian centers (N, 3)
        covars: World-space covariance upper-tri (N, 6)
        max_scales: Max activated scale per Gaussian (N)
        ranges: Physical extent per axis (3,) as (rx, ry, rz)
        centers: Physical center per axis (3,) as (cx, cy, cz)
        dim2: Voxel grid x dimension (fastest)
        dim1: Voxel grid y dimension (middle)
        dim0: Voxel grid z dimension (slowest)
        eps3d: Blur epsilon added to diagonal (default 1e-4)
        radius_clip: Radius multiplier for radii computation (default 3.0)

    Returns:
        Tuple of:
        - means3d_v: Voxel-space centers (N, 3)
        - conics: Inverse covariance upper-tri (N, 6)
        - radii: Integer radii per axis (N, 3) as int32
    """
    return _world_to_volume_mapping_fwd(
        means, covars, max_scales, ranges, centers,
        dim2, dim1, dim0, eps3d, radius_clip
    )


def world_to_volume_mapping_bwd(
    means: torch.Tensor,
    covars: torch.Tensor,
    ranges: torch.Tensor,
    centers: torch.Tensor,
    dim2: int,
    dim1: int,
    dim0: int,
    eps3d: float,
    v_means3d_v: torch.Tensor,
    v_conics: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """World-to-volume mapping backward pass (first-order gradients).

    Computes dL/d(means) and dL/d(covars) from downstream gradients.

    Args:
        means: Forward input (N, 3)
        covars: Forward input (N, 6)
        ranges: Physical extent (3,)
        centers: Physical center (3,)
        dim2, dim1, dim0: Voxel dimensions
        eps3d: Blur epsilon
        v_means3d_v: Upstream dL/d(means3d_v) (N, 3)
        v_conics: Upstream dL/d(conics) (N, 6)

    Returns:
        Tuple of:
        - v_means: Gradient w.r.t. means (N, 3)
        - v_covars: Gradient w.r.t. covars (N, 6)
    """
    return _world_to_volume_mapping_bwd(
        means, covars, ranges, centers,
        dim2, dim1, dim0, eps3d,
        v_means3d_v, v_conics,
    )


def world_to_volume_mapping_bwd2(
    means: torch.Tensor,
    covars: torch.Tensor,
    ranges: torch.Tensor,
    centers: torch.Tensor,
    dim2: int,
    dim1: int,
    dim0: int,
    eps3d: float,
    v_means3d_v: torch.Tensor,
    v_conics: torch.Tensor,
    h_means3d_v: torch.Tensor,
    h_conics: torch.Tensor,
    h_means3d_v_conics: torch.Tensor,
    h_means3d_v_opacity: torch.Tensor,
    h_conics_opacity: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    """World-to-volume mapping second-order backward pass (Hessians).

    Computes upstream gradients and Hessians through the world-to-volume
    mapping from downstream gradients and Hessians.

    The means transform is affine (linear), so its Hessian block has
    zero curvature term. The covars→conics path goes through a matrix
    inverse which contributes curvature terms.

    Args:
        means: Forward input (N, 3)
        covars: Forward input (N, 6)
        ranges: Physical extent (3,)
        centers: Physical center (3,)
        dim2, dim1, dim0: Voxel dimensions
        eps3d: Blur epsilon
        v_means3d_v: Downstream gradient (N, 3)
        v_conics: Downstream gradient (N, 6)
        h_means3d_v: Downstream Hessian upper-tri 3x3 (N, 6)
        h_conics: Downstream Hessian upper-tri 6x6 (N, 21)
        h_means3d_v_conics: Downstream cross-Hessian (N, 18)
        h_means3d_v_opacity: Downstream cross-Hessian (N, 3)
        h_conics_opacity: Downstream cross-Hessian (N, 6)

    Returns:
        Tuple of:
        - v_means: Gradient (N, 3)
        - v_covars: Gradient (N, 6)
        - h_means: Hessian H_{p,p} upper-tri (N, 6)
        - h_covars: Hessian H_{s,s} upper-tri (N, 21)
        - h_means_covars: Cross-Hessian H_{p,s} (N, 18)
        - h_means_opacity: Cross-Hessian (N, 3)
        - h_covars_opacity: Cross-Hessian (N, 6)
    """
    return _world_to_volume_mapping_bwd2(
        means, covars, ranges, centers,
        dim2, dim1, dim0, eps3d,
        v_means3d_v, v_conics,
        h_means3d_v, h_conics, h_means3d_v_conics,
        h_means3d_v_opacity, h_conics_opacity,
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
    # VoxelizeToVoxels
    "voxelize_to_voxels_fwd",
    "voxelize_to_voxels_bwd",
    "voxelize_to_voxels_bwd2",
    # Rasterize to Pixels
    "rasterize_to_pixels_fwd",
    "rasterize_to_pixels_bwd",
    "rasterize_to_pixels_bwd2",
    # ConicsMu
    "conics_mu_fwd",
    "conics_mu_bwd",
    "conics_mu_bwd2",
    # WorldToImageProjection
    "world_to_image_projection_fwd",
    "world_to_image_projection_bwd",
    "world_to_image_projection_bwd2",
    # ParameterTransform
    "parameter_transform_fwd",
    "parameter_transform_bwd",
    "parameter_transform_bwd2",
    # WorldToVolumeMapping
    "world_to_volume_mapping_fwd",
    "world_to_volume_mapping_bwd",
    "world_to_volume_mapping_bwd2",
    # Utilities
    "is_cuda_extension_available",
]