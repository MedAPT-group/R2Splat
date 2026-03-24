"""
Rasterize to Pixels operators for CT reconstruction.

This module provides CUDA and PyTorch implementations of tile-based
Gaussian rasterization for CT reconstruction.

CT formula: pixel_value = Σ (opacity × mu × exp(-sigma))
where sigma = 0.5 × (x - mean)^T × conic × (x - mean)

Supports:
- RasterizeToPixelsCUDA: CUDA-accelerated implementation
- RasterizeToPixelsTorch: Pure PyTorch reference implementation

Author: Dy164
Date: 2025-01-25
"""

import torch
from typing import Optional, Tuple
from .base import SecondForwardBackwardBase
from .data import Params, Data, Config, ForwardResult, GradResult, HessResult


# ============================================================================
# CUDA Availability Check (Lazy Loading)
# ============================================================================

def _is_cuda_available() -> bool:
    """Lazy check for CUDA availability without triggering JIT compilation."""
    try:
        from ..cuda._backend import is_cuda_available
        return is_cuda_available()
    except ImportError:
        return False


# ============================================================================
# CUDA Implementation
# ============================================================================

class RasterizeToPixelsCUDA(SecondForwardBackwardBase):
    """
    CUDA-accelerated rasterize to pixels implementation.
    
    Uses tile-based rasterization with shared memory optimization.
    Requires isect_tiles and isect_offset_encode for preprocessing.
    
    CT formula: pixel_value = Σ (opacity × mu × exp(-sigma))
    where sigma = 0.5 × (x - mean)^T × conic × (x - mean)
    
    Supports only: (C, H, W) - multi-camera 2D images without channel dimension
    
    This class computes:
    - Forward: render output (C, H, W)
    - Backward1: gradient w.r.t. Gaussian parameters (not yet implemented)
    - Backward2: diagonal Hessian (not yet implemented)
    """
    
    def __init__(self):
        if not _is_cuda_available():
            raise RuntimeError(
                "CUDA extension not available. "
                "Please compile the CUDA extension first with 'pip install -e .'"
            )
        self.initialized = False
    
    def initialize(
        self,
        params: Params,
        config: Optional[Config] = None,
    ) -> None:
        """Initialize and validate parameters."""
        required_params = ['means2d', 'conics', 'opacities', 'mus']
        for name in required_params:
            if params.get(name) is None:
                raise ValueError(f"Missing required parameter: {name}")
        
        means2d = params.get('means2d').data
        
        # Must be 3D: (C, N, 2)
        if means2d.dim() != 3 or means2d.size(-1) != 2:
            raise ValueError(
                f"means2d must be 3D with shape (C, N, 2), got {means2d.shape}"
            )
        
        # Ensure CUDA tensors
        if not means2d.is_cuda:
            raise ValueError("All parameters must be CUDA tensors")
        
        self.initialized = True
    
    def forward_only(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> ForwardResult:
        """Forward computation only (no gradients).
        
        Expected params:
            - means2d: (C, N, 2) 2D Gaussian centers, format: (x, y) = (col, row)
            - conics: (C, N, 3) Inverse covariance matrices [a, b, c]
            - opacities: (C, N) Gaussian opacities in [-1, 1]
            - mus: (C, N) Gaussian mu values
        
        Expected data:
            - radii: (C, N) Gaussian radii for tile intersection
            - dim1: int - output size in dimension 1 (columns, x-axis)
            - dim0: int - output size in dimension 0 (rows, y-axis)
            - tile_size: int (default 16)
        
        Returns:
            ForwardResult with render_output (C, dim0, dim1)
        """
        # Lazy import to avoid triggering JIT at module load time
        from ..cuda._wrapper import (
            isect_tiles,
            isect_offset_encode,
            rasterize_to_pixels_fwd,
        )
        
        # Extract parameters
        means2d = params.get('means2d').data
        conics = params.get('conics').data
        opacities = params.get('opacities').data
        mus = params.get('mus').data
        
        # Extract data
        radii = data.get('radii')
        dim1 = data.get('dim1')
        dim0 = data.get('dim0')
        tile_size = data.get('tile_size', 16)
        
        C = means2d.size(0)
        n_tiles_dim1 = (dim1 + tile_size - 1) // tile_size
        n_tiles_dim0 = (dim0 + tile_size - 1) // tile_size
        
        # Compute tile intersections (always sorts internally)
        _, isect_ids, flatten_ids = isect_tiles(
            means2d, radii, tile_size, n_tiles_dim1, n_tiles_dim0
        )
        
        # Encode tile offsets
        tile_offsets = isect_offset_encode(isect_ids, C, n_tiles_dim1, n_tiles_dim0)
        
        # Rasterize
        render_output = rasterize_to_pixels_fwd(
            means2d, conics, opacities, mus,
            tile_offsets, flatten_ids,
            dim1, dim0, tile_size
        )
        
        return ForwardResult(
            outputs={'render_output': render_output},
            intermediates={
                'tile_offsets': tile_offsets,
                'flatten_ids': flatten_ids,
            }
        )
    
    def forward_backward1(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> Tuple[ForwardResult, GradResult]:
        """Forward + first-order backward using CUDA kernels.
        
        Computes gradients of render_output w.r.t. all Gaussian parameters.
        
        Expected params: same as forward_only
        Expected data: same as forward_only, plus:
            - v_render_output: (C, dim0, dim1) upstream gradient (optional)
              If not provided, uses ones (equivalent to sum of all pixels)
        
        Returns:
            ForwardResult with render_output (C, dim0, dim1)
            GradResult with gradients for means2d, conics, opacities, mus
        """
        # Lazy import to avoid triggering JIT at module load time
        from ..cuda._wrapper import (
            isect_tiles,
            isect_offset_encode,
            rasterize_to_pixels_fwd,
            rasterize_to_pixels_bwd,
        )
        
        # Extract parameters
        means2d = params.get('means2d').data
        conics = params.get('conics').data
        opacities = params.get('opacities').data
        mus = params.get('mus').data
        
        # Extract data
        radii = data.get('radii')
        dim1 = data.get('dim1')
        dim0 = data.get('dim0')
        tile_size = data.get('tile_size', 16)
        
        C = means2d.size(0)
        n_tiles_dim1 = (dim1 + tile_size - 1) // tile_size
        n_tiles_dim0 = (dim0 + tile_size - 1) // tile_size
        
        # Compute tile intersections
        _, isect_ids, flatten_ids = isect_tiles(
            means2d, radii, tile_size, n_tiles_dim1, n_tiles_dim0
        )
        
        # Encode tile offsets
        tile_offsets = isect_offset_encode(isect_ids, C, n_tiles_dim1, n_tiles_dim0)
        
        # Forward pass
        render_output = rasterize_to_pixels_fwd(
            means2d, conics, opacities, mus,
            tile_offsets, flatten_ids,
            dim1, dim0, tile_size
        )
        
        # Get upstream gradient (default to ones if not provided)
        v_render_output = data.get('v_render_output', None)
        if v_render_output is None:
            v_render_output = torch.ones_like(render_output)
        
        # Backward pass
        v_means2d, v_conics, v_opacities, v_mus = rasterize_to_pixels_bwd(
            means2d, conics, opacities, mus,
            tile_offsets, flatten_ids, v_render_output,
            dim1, dim0, tile_size
        )
        
        forward_result = ForwardResult(
            outputs={'render_output': render_output},
            intermediates={
                'tile_offsets': tile_offsets,
                'flatten_ids': flatten_ids,
            }
        )
        
        grad_result = GradResult(
            param_grads={
                'means2d': v_means2d,
                'conics': v_conics,
                'opacities': v_opacities,
                'mus': v_mus,
            }
        )
        
        return forward_result, grad_result
    
    def forward_backward2(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> Tuple[ForwardResult, GradResult, HessResult]:
        """Forward + second-order backward using CUDA kernels.
        
        Computes both first-order gradients and second-order Hessian.
        
        For CT reconstruction: O = Σ_i G_i where G_i = μ_i × o_i × exp(-σ_i)
        
        Expected params: same as forward_only
        Expected data: same as forward_only, plus:
            - v_render_output: (C, dim0, dim1) first-order upstream gradient
            - h_render_output: (C, dim0, dim1) second-order upstream gradient
              If not provided, uses ones
        
        Hessian storage uses upper triangular elements only (consistent with conics):
            - means2d: [H00, H01, H11] (3 elements)
            - conics: [H00, H01, H02, H11, H12, H22] (6 elements)
        
        Returns:
            ForwardResult with render_output (C, dim0, dim1)
            GradResult with first-order gradients
            HessResult with:
                - param_hess: Hessian upper triangles for each parameter group
                    - means2d: (C, N, 3)  [H00, H01, H11]
                    - conics: (C, N, 6)   [H00, H01, H02, H11, H12, H22]
                    - mus: (C, N)
                    - opacities: (C, N)
                - cross_hess: Cross-term Hessians between parameter groups
                    - mu_mean: (C, N, 2)
                    - mu_conic: (C, N, 3)
                    - mean_conic: (C, N, 2, 3)
                    - mu_opacity: (C, N)
                    - conic_opacity: (C, N, 3)
                    - mean_opacity: (C, N, 2)
        """
        # Lazy import to avoid triggering JIT at module load time
        from ..cuda._wrapper import (
            isect_tiles,
            isect_offset_encode,
            rasterize_to_pixels_fwd,
            rasterize_to_pixels_bwd2,
        )
        
        # Extract parameters
        means2d = params.get('means2d').data
        conics = params.get('conics').data
        opacities = params.get('opacities').data
        mus = params.get('mus').data
        
        # Extract data
        radii = data.get('radii')
        dim1 = data.get('dim1')
        dim0 = data.get('dim0')
        tile_size = data.get('tile_size', 16)
        
        C = means2d.size(0)
        n_tiles_dim1 = (dim1 + tile_size - 1) // tile_size
        n_tiles_dim0 = (dim0 + tile_size - 1) // tile_size
        
        # Compute tile intersections
        _, isect_ids, flatten_ids = isect_tiles(
            means2d, radii, tile_size, n_tiles_dim1, n_tiles_dim0
        )
        
        # Encode tile offsets
        tile_offsets = isect_offset_encode(isect_ids, C, n_tiles_dim1, n_tiles_dim0)
        
        # Forward pass
        render_output = rasterize_to_pixels_fwd(
            means2d, conics, opacities, mus,
            tile_offsets, flatten_ids,
            dim1, dim0, tile_size
        )
        
        # Get upstream gradients (default to ones if not provided)
        v_render_output = data.get('v_render_output', None)
        if v_render_output is None:
            v_render_output = torch.ones_like(render_output)
        
        h_render_output = data.get('h_render_output', None)
        if h_render_output is None:
            h_render_output = torch.ones_like(render_output)
        
        # Second-order backward pass
        (v_means2d, v_conics, v_opacities, v_mus,
         h_means2d, h_conics,
         h_mu_mean, h_mu_conic, h_mean_conic,
         h_mu_opacity, h_conic_opacity, h_mean_opacity,
         h_mus, h_opacities) = rasterize_to_pixels_bwd2(
            means2d, conics, opacities, mus,
            tile_offsets, flatten_ids,
            v_render_output, h_render_output,
            dim1, dim0, tile_size
        )
        
        forward_result = ForwardResult(
            outputs={'render_output': render_output},
            intermediates={
                'tile_offsets': tile_offsets,
                'flatten_ids': flatten_ids,
            }
        )
        
        grad_result = GradResult(
            param_grads={
                'means2d': v_means2d,
                'conics': v_conics,
                'opacities': v_opacities,
                'mus': v_mus,
            }
        )
        
        hess_result = HessResult(
            param_hess={
                'means2d': h_means2d,
                'conics': h_conics,
                'mus': h_mus,
                'opacities': h_opacities,
            },
            cross_hess={
                'mu_mean': h_mu_mean,
                'mu_conic': h_mu_conic,
                'mean_conic': h_mean_conic,
                'mu_opacity': h_mu_opacity,
                'conic_opacity': h_conic_opacity,
                'mean_opacity': h_mean_opacity,
            }
        )
        
        return forward_result, grad_result, hess_result


# ============================================================================
# PyTorch Reference Implementation
# ============================================================================

class RasterizeToPixelsTorch(SecondForwardBackwardBase):
    """
    Pure PyTorch reference implementation of rasterize to pixels.
    
    This is a brute-force implementation that iterates over all Gaussians
    for each pixel. It's slower than CUDA but useful for validation.
    
    CT formula: pixel_value = Σ (opacity × mu × exp(-sigma))
    where sigma = 0.5 × (x - mean)^T × conic × (x - mean)
    
    Note: This implementation does NOT use tiles - it directly computes
    contributions from ALL Gaussians to ALL pixels.
    
    Supports only: (C, H, W) - multi-camera 2D images without channel dimension
    """
    
    def __init__(self):
        self.initialized = False
    
    def initialize(
        self,
        params: Params,
        config: Optional[Config] = None,
    ) -> None:
        """Initialize and validate parameters."""
        required_params = ['means2d', 'conics', 'opacities', 'mus']
        for name in required_params:
            if params.get(name) is None:
                raise ValueError(f"Missing required parameter: {name}")
        
        means2d = params.get('means2d').data
        
        # Must be 3D: (C, N, 2)
        if means2d.dim() != 3 or means2d.size(-1) != 2:
            raise ValueError(
                f"means2d must be 3D with shape (C, N, 2), got {means2d.shape}"
            )
        
        self.initialized = True
    
    def forward_only(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> ForwardResult:
        """Forward computation only (no gradients).
        
        Expected params:
            - means2d: (C, N, 2) 2D Gaussian centers, format: (x, y) = (col, row)
            - conics: (C, N, 3) Inverse covariance matrices [a, b, c]
            - opacities: (C, N) Gaussian opacities in [-1, 1]
            - mus: (C, N) Gaussian mu values
        
        Expected data:
            - radii: (C, N) Gaussian radii for tile intersection (not used here)
            - dim1: int - output size in dimension 1 (columns, x-axis)
            - dim0: int - output size in dimension 0 (rows, y-axis)
            - tile_size: int (not used here)
        
        Returns:
            ForwardResult with render_output (C, dim0, dim1)
        """
        # Extract parameters
        means2d = params.get('means2d').data
        conics = params.get('conics').data
        opacities = params.get('opacities').data
        mus = params.get('mus').data
        
        # Extract data
        dim1 = data.get('dim1')
        dim0 = data.get('dim0')
        
        C, N, _ = means2d.shape
        device = means2d.device
        dtype = means2d.dtype
        
        # Create pixel grid with 0.5 offset (pixel center coordinates)
        # y corresponds to row index (dim0), x corresponds to column index (dim1)
        y_coords = torch.arange(dim0, device=device, dtype=dtype) + 0.5
        x_coords = torch.arange(dim1, device=device, dtype=dtype) + 0.5
        py, px = torch.meshgrid(y_coords, x_coords, indexing='ij')  # [dim0, dim1]
        
        # Initialize output
        render_output = torch.zeros(C, dim0, dim1, device=device, dtype=dtype)
        
        # Brute-force: iterate over all cameras and Gaussians
        # means2d: [C, N, 2] -> means2d[c, g, 0] = x (column), means2d[c, g, 1] = y (row)
        # conics: [C, N, 3] -> [a, b, c] representing [[a, b], [b, c]]
        
        for c in range(C):
            for g in range(N):
                # Get Gaussian parameters
                mean_x = means2d[c, g, 0]  # column (dim1)
                mean_y = means2d[c, g, 1]  # row (dim0)
                conic_a = conics[c, g, 0]
                conic_b = conics[c, g, 1]
                conic_c = conics[c, g, 2]
                opacity = opacities[c, g]
                mu = mus[c, g]
                
                # Compute delta = pixel - mean
                # px is x coordinate (column), py is y coordinate (row)
                dx = px - mean_x  # [H, W]
                dy = py - mean_y  # [H, W]
                
                # Compute sigma = 0.5 * delta^T * conic * delta
                # sigma = 0.5 * (a * dx^2 + c * dy^2) + b * dx * dy
                sigma = 0.5 * (conic_a * dx * dx + conic_c * dy * dy) + conic_b * dx * dy
                
                # Compute alpha = opacity * mu * exp(-sigma)
                alpha = opacity * mu * torch.exp(-sigma)
                
                # Apply same numerical checks as CUDA
                # Skip if sigma < 0 or |alpha| < 1e-5
                mask = (sigma >= 0) & (alpha.abs() >= 1e-5)
                
                # Accumulate contribution
                render_output[c] += torch.where(mask, alpha, torch.zeros_like(alpha))
        
        return ForwardResult(
            outputs={'render_output': render_output},
            intermediates={}
        )
    
    def forward_backward1(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> Tuple[ForwardResult, GradResult]:
        """Forward + first-order backward using PyTorch autograd.
        
        This is the reference implementation for validating CUDA backward.
        Uses torch.autograd to compute gradients via automatic differentiation.
        
        Expected params: same as forward_only
        Expected data: same as forward_only, plus:
            - v_render_output: (C, dim0, dim1) upstream gradient (optional)
              If not provided, uses ones (equivalent to sum of all pixels)
        
        Returns:
            ForwardResult with render_output (C, dim0, dim1)
            GradResult with gradients for means2d, conics, opacities, mus
        """
        # Extract parameters and enable gradients
        means2d = params.get('means2d').data.requires_grad_(True)
        conics = params.get('conics').data.requires_grad_(True)
        opacities = params.get('opacities').data.requires_grad_(True)
        mus = params.get('mus').data.requires_grad_(True)
        
        # Extract data
        dim1 = data.get('dim1')
        dim0 = data.get('dim0')
        
        C, N, _ = means2d.shape
        device = means2d.device
        dtype = means2d.dtype
        
        # Create pixel grid with 0.5 offset (pixel center coordinates)
        y_coords = torch.arange(dim0, device=device, dtype=dtype) + 0.5
        x_coords = torch.arange(dim1, device=device, dtype=dtype) + 0.5
        py, px = torch.meshgrid(y_coords, x_coords, indexing='ij')  # [dim0, dim1]
        
        # Vectorized computation for all Gaussians at once
        # Expand pixel grid for broadcasting: [1, 1, dim0, dim1]
        px = px.unsqueeze(0).unsqueeze(0)  # [1, 1, dim0, dim1]
        py = py.unsqueeze(0).unsqueeze(0)  # [1, 1, dim0, dim1]
        
        # Expand means2d: [C, N, 2] -> [C, N, 1, 1]
        mean_x = means2d[..., 0].unsqueeze(-1).unsqueeze(-1)  # [C, N, 1, 1]
        mean_y = means2d[..., 1].unsqueeze(-1).unsqueeze(-1)  # [C, N, 1, 1]
        
        # Expand conics: [C, N, 3] -> [C, N, 1, 1]
        conic_a = conics[..., 0].unsqueeze(-1).unsqueeze(-1)  # [C, N, 1, 1]
        conic_b = conics[..., 1].unsqueeze(-1).unsqueeze(-1)  # [C, N, 1, 1]
        conic_c = conics[..., 2].unsqueeze(-1).unsqueeze(-1)  # [C, N, 1, 1]
        
        # Expand opacities and mus: [C, N] -> [C, N, 1, 1]
        opacity = opacities.unsqueeze(-1).unsqueeze(-1)  # [C, N, 1, 1]
        mu = mus.unsqueeze(-1).unsqueeze(-1)  # [C, N, 1, 1]
        
        # Compute delta = pixel - mean: [C, N, dim0, dim1]
        dx = px - mean_x
        dy = py - mean_y
        
        # Compute sigma = 0.5 * (a * dx^2 + c * dy^2) + b * dx * dy
        sigma = 0.5 * (conic_a * dx * dx + conic_c * dy * dy) + conic_b * dx * dy
        
        # Compute alpha = opacity * mu * exp(-sigma)
        alpha = opacity * mu * torch.exp(-sigma)
        
        # Apply numerical mask (same as CUDA)
        mask = (sigma >= 0) & (alpha.abs() >= 1e-5)
        alpha = torch.where(mask, alpha, torch.zeros_like(alpha))
        
        # Sum over Gaussians: [C, N, dim0, dim1] -> [C, dim0, dim1]
        render_output = alpha.sum(dim=1)
        
        # Get upstream gradient
        v_render_output = data.get('v_render_output', None)
        if v_render_output is None:
            v_render_output = torch.ones_like(render_output)
        
        # Backward pass using autograd
        render_output.backward(v_render_output)
        
        forward_result = ForwardResult(
            outputs={'render_output': render_output.detach()},
            intermediates={}
        )
        
        grad_result = GradResult(
            param_grads={
                'means2d': means2d.grad.detach(),
                'conics': conics.grad.detach(),
                'opacities': opacities.grad.detach(),
                'mus': mus.grad.detach(),
            }
        )
        
        return forward_result, grad_result
    
    def forward_backward2(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> Tuple[ForwardResult, GradResult, HessResult]:
        """Forward + second-order backward using PyTorch.
        
        Computes both first-order gradients and second-order Hessian elements
        using manual chain rule (matching CUDA implementation).
        
        For CT reconstruction: O = Σ_i G_i where G_i = μ_i × o_i × exp(-σ_i)
        
        Given upstream:
            v_O = ∂L/∂O (first derivative)
            h_O = ∂²L/∂O² (second derivative / Hessian diagonal)
        
        We compute using chain rule:
            v_θ = v_O × ∂G/∂θ
            h_θ = h_O × (∂G/∂θ)² + v_O × ∂²G/∂θ²
        
        Expected params: same as forward_only
        Expected data: same as forward_only, plus:
            - v_render_output: (C, dim0, dim1) first-order upstream gradient
            - h_render_output: (C, dim0, dim1) second-order upstream gradient
        
        Returns:
            ForwardResult with render_output (C, dim0, dim1)
            GradResult with first-order gradients
            HessResult with Hessian diagonals and cross-terms
        """
        # Extract parameters
        means2d = params.get('means2d').data
        conics = params.get('conics').data
        opacities = params.get('opacities').data
        mus = params.get('mus').data
        
        # Extract data
        dim1 = data.get('dim1')
        dim0 = data.get('dim0')
        
        C, N, _ = means2d.shape
        device = means2d.device
        dtype = means2d.dtype
        
        # Create pixel grid with 0.5 offset (pixel center coordinates)
        y_coords = torch.arange(dim0, device=device, dtype=dtype) + 0.5
        x_coords = torch.arange(dim1, device=device, dtype=dtype) + 0.5
        py, px = torch.meshgrid(y_coords, x_coords, indexing='ij')  # [dim0, dim1]
        
        # Expand pixel grid for broadcasting: [1, 1, dim0, dim1]
        px = px.unsqueeze(0).unsqueeze(0)  # [1, 1, dim0, dim1]
        py = py.unsqueeze(0).unsqueeze(0)  # [1, 1, dim0, dim1]
        
        # Expand means2d: [C, N, 2] -> [C, N, 1, 1]
        mean_x = means2d[..., 0].unsqueeze(-1).unsqueeze(-1)  # [C, N, 1, 1]
        mean_y = means2d[..., 1].unsqueeze(-1).unsqueeze(-1)  # [C, N, 1, 1]
        
        # Expand conics: [C, N, 3] -> [C, N, 1, 1]
        conic_a = conics[..., 0].unsqueeze(-1).unsqueeze(-1)  # [C, N, 1, 1]
        conic_b = conics[..., 1].unsqueeze(-1).unsqueeze(-1)  # [C, N, 1, 1]
        conic_c = conics[..., 2].unsqueeze(-1).unsqueeze(-1)  # [C, N, 1, 1]
        
        # Expand opacities and mus: [C, N] -> [C, N, 1, 1]
        opacity = opacities.unsqueeze(-1).unsqueeze(-1)  # [C, N, 1, 1]
        mu = mus.unsqueeze(-1).unsqueeze(-1)  # [C, N, 1, 1]
        
        # Compute delta = pixel - mean: [C, N, dim0, dim1]
        dx = px - mean_x  # delta.x = px - mean_x
        dy = py - mean_y  # delta.y = py - mean_y
        
        # Compute sigma = 0.5 * (a * dx^2 + c * dy^2) + b * dx * dy
        sigma = 0.5 * (conic_a * dx * dx + conic_c * dy * dy) + conic_b * dx * dy
        
        # Compute vis = exp(-sigma) and alpha = opacity * mu * vis
        vis = torch.exp(-sigma)
        alpha = opacity * mu * vis
        
        # Apply numerical mask (same as CUDA)
        mask = (sigma >= 0) & (alpha.abs() >= 1e-5)
        alpha_masked = torch.where(mask, alpha, torch.zeros_like(alpha))
        
        # Sum over Gaussians: [C, N, dim0, dim1] -> [C, dim0, dim1]
        render_output = alpha_masked.sum(dim=1)
        
        # Get upstream gradients
        v_render_output = data.get('v_render_output', None)
        if v_render_output is None:
            v_render_output = torch.ones_like(render_output)
        
        h_render_output = data.get('h_render_output', None)
        if h_render_output is None:
            h_render_output = torch.ones_like(render_output)
        
        # Expand upstream gradients for broadcasting: [C, 1, dim0, dim1]
        v_O = v_render_output.unsqueeze(1)  # [C, 1, dim0, dim1]
        h_O = h_render_output.unsqueeze(1)  # [C, 1, dim0, dim1]
        
        # =====================================================================
        # First-order derivatives (same as bwd)
        # =====================================================================
        
        # s = [a*δ1 + b*δ2, c*δ2 + b*δ1]  (∂σ/∂δ)
        s_x = conic_a * dx + conic_b * dy  # [C, N, dim0, dim1]
        s_y = conic_c * dy + conic_b * dx  # [C, N, dim0, dim1]
        
        # t = [0.5*δ1², δ1*δ2, 0.5*δ2²]  (∂σ/∂c)
        t_a = 0.5 * dx * dx  # [C, N, dim0, dim1]
        t_b = dx * dy        # [C, N, dim0, dim1]
        t_c = 0.5 * dy * dy  # [C, N, dim0, dim1]
        
        # Apply mask to alpha for consistent computation
        alpha = torch.where(mask, alpha, torch.zeros_like(alpha))
        oI = torch.where(mask, opacity * vis, torch.zeros_like(vis))  # o * I = o * exp(-sigma)
        
        # v_sigma = -v_O * alpha
        v_sigma = -v_O * alpha  # [C, 1, dim0, dim1] * [C, N, dim0, dim1] -> [C, N, dim0, dim1]
        
        # First-order gradients (summed over all pixels)
        v_opacity = (v_O * mu.unsqueeze(-1).unsqueeze(-1) * vis).sum(dim=(2, 3))  # Wrong shape, recalculate
        
        # Recalculate properly:
        # v_opacity_local = v_O * mu * vis  -> sum over pixels
        # v_mu_local = v_O * opacity * vis  -> sum over pixels
        v_opacity_g = (v_O * mu * vis * mask.float()).sum(dim=(2, 3))  # [C, N]
        v_mu_g = (v_O * opacity * vis * mask.float()).sum(dim=(2, 3))  # [C, N]
        
        # v_means2d: v_xy = -v_sigma * s  -> ∂δ/∂m = -1
        v_mean_x = (-v_sigma * s_x * mask.float()).sum(dim=(2, 3))  # [C, N]
        v_mean_y = (-v_sigma * s_y * mask.float()).sum(dim=(2, 3))  # [C, N]
        
        # v_conics: v_c = v_sigma * t
        v_conic_a = (v_sigma * t_a * mask.float()).sum(dim=(2, 3))  # [C, N]
        v_conic_b = (v_sigma * t_b * mask.float()).sum(dim=(2, 3))  # [C, N]
        v_conic_c = (v_sigma * t_c * mask.float()).sum(dim=(2, 3))  # [C, N]
        
        # Stack gradients
        v_means2d_out = torch.stack([v_mean_x, v_mean_y], dim=-1)  # [C, N, 2]
        v_conics_out = torch.stack([v_conic_a, v_conic_b, v_conic_c], dim=-1)  # [C, N, 3]
        
        # =====================================================================
        # Second-order derivatives
        # =====================================================================
        
        # Common factor: factor1 = h_O * α + v_O
        factor1 = h_O * alpha + v_O  # [C, N, dim0, dim1] (broadcasting)
        
        # h_means2d diagonal: h_m[i] = factor1 * α * s[i]² + v_O * α * (-Σ⁻¹[i,i])
        # Note: Σ⁻¹ = [[a,b],[b,c]], so -Σ⁻¹[0,0] = -a, -Σ⁻¹[1,1] = -c
        h_mean_x_diag = ((factor1 * alpha * s_x * s_x + v_O * alpha * (-conic_a)) * mask.float()).sum(dim=(2, 3))
        h_mean_y_diag = ((factor1 * alpha * s_y * s_y + v_O * alpha * (-conic_c)) * mask.float()).sum(dim=(2, 3))
        
        # h_conics diagonal: h_c[i] = factor1 * α * t[i]²
        h_conic_a_diag = ((factor1 * alpha * t_a * t_a) * mask.float()).sum(dim=(2, 3))
        h_conic_b_diag = ((factor1 * alpha * t_b * t_b) * mask.float()).sum(dim=(2, 3))
        h_conic_c_diag = ((factor1 * alpha * t_c * t_c) * mask.float()).sum(dim=(2, 3))
        
        # Cross term h_mu_mean: h_{μ,m} = factor1 * oI * s
        h_mu_mean_x = ((factor1 * oI * s_x) * mask.float()).sum(dim=(2, 3))
        h_mu_mean_y = ((factor1 * oI * s_y) * mask.float()).sum(dim=(2, 3))
        
        # Cross term h_mu_conic: h_{μ,c} = -factor1 * oI * t
        h_mu_conic_a = ((-factor1 * oI * t_a) * mask.float()).sum(dim=(2, 3))
        h_mu_conic_b = ((-factor1 * oI * t_b) * mask.float()).sum(dim=(2, 3))
        h_mu_conic_c = ((-factor1 * oI * t_c) * mask.float()).sum(dim=(2, 3))
        
        # Cross term h_mean_conic: 2x3 matrix
        # h_{m,c} = -h_O * α² * s⊗t - v_O * α * s⊗t + v_O * α * ∂s/∂c
        # where ∂s/∂c = [[δ1, δ2, 0], [0, δ1, δ2]]
        common_mc = -h_O * alpha * alpha - v_O * alpha
        h_mc_00 = ((common_mc * s_x * t_a + v_O * alpha * dx) * mask.float()).sum(dim=(2, 3))  # dm1/da
        h_mc_01 = ((common_mc * s_x * t_b + v_O * alpha * dy) * mask.float()).sum(dim=(2, 3))  # dm1/db
        h_mc_02 = ((common_mc * s_x * t_c) * mask.float()).sum(dim=(2, 3))                     # dm1/dc
        h_mc_10 = ((common_mc * s_y * t_a) * mask.float()).sum(dim=(2, 3))                     # dm2/da
        h_mc_11 = ((common_mc * s_y * t_b + v_O * alpha * dx) * mask.float()).sum(dim=(2, 3))  # dm2/db
        h_mc_12 = ((common_mc * s_y * t_c + v_O * alpha * dy) * mask.float()).sum(dim=(2, 3))  # dm2/dc
        
        # =====================================================================
        # Opacity cross-term Hessians
        # =====================================================================
        # muI = mu * vis = mu * exp(-sigma)  (like oI but with mu instead of opacity)
        muI = torch.where(mask, mu * vis, torch.zeros_like(vis))  # [C, N, dim0, dim1]
        
        # h_{mu, opacity} = h_O * oI * muI + v_O * vis
        # where oI = opacity * vis, muI = mu * vis
        h_mu_opacity_out = ((h_O * oI * muI + v_O * vis) * mask.float()).sum(dim=(2, 3))  # [C, N]
        
        # factor_opac = (h_O * alpha + v_O) * muI
        factor_opac = (h_O * alpha + v_O) * muI  # [C, N, dim0, dim1]
        
        # h_{conic, opacity} = -factor_opac * t
        h_co_a = ((-factor_opac * t_a) * mask.float()).sum(dim=(2, 3))  # [C, N]
        h_co_b = ((-factor_opac * t_b) * mask.float()).sum(dim=(2, 3))  # [C, N]
        h_co_c = ((-factor_opac * t_c) * mask.float()).sum(dim=(2, 3))  # [C, N]
        
        # h_{mean, opacity} = factor_opac * s
        h_mo_x = ((factor_opac * s_x) * mask.float()).sum(dim=(2, 3))  # [C, N]
        h_mo_y = ((factor_opac * s_y) * mask.float()).sum(dim=(2, 3))  # [C, N]
        
        # =====================================================================
        # Scalar Hessians
        # =====================================================================
        # h_mus: ∂²L/∂μ² = h_O * (oI)²  (no second-order term since ∂²G/∂μ² = 0)
        h_mus_out = ((h_O * oI * oI) * mask.float()).sum(dim=(2, 3))  # [C, N]
        
        # h_opacities: ∂²L/∂o² = h_O * (muI)²  (no second-order term since ∂²G/∂o² = 0)
        h_opacities_out = ((h_O * muI * muI) * mask.float()).sum(dim=(2, 3))  # [C, N]
        
        # Stack outputs
        h_means2d_out = torch.stack([h_mean_x_diag, h_mean_y_diag], dim=-1)  # [C, N, 2]
        h_conics_out = torch.stack([h_conic_a_diag, h_conic_b_diag, h_conic_c_diag], dim=-1)  # [C, N, 3]
        h_mu_mean_out = torch.stack([h_mu_mean_x, h_mu_mean_y], dim=-1)  # [C, N, 2]
        h_mu_conic_out = torch.stack([h_mu_conic_a, h_mu_conic_b, h_mu_conic_c], dim=-1)  # [C, N, 3]
        h_mean_conic_out = torch.stack([h_mc_00, h_mc_01, h_mc_02, h_mc_10, h_mc_11, h_mc_12], dim=-1)  # [C, N, 6]
        h_conic_opacity_out = torch.stack([h_co_a, h_co_b, h_co_c], dim=-1)  # [C, N, 3]
        h_mean_opacity_out = torch.stack([h_mo_x, h_mo_y], dim=-1)  # [C, N, 2]
        
        forward_result = ForwardResult(
            outputs={'render_output': render_output},
            intermediates={}
        )
        
        grad_result = GradResult(
            param_grads={
                'means2d': v_means2d_out,
                'conics': v_conics_out,
                'opacities': v_opacity_g,
                'mus': v_mu_g,
            }
        )
        
        hess_result = HessResult(
            param_hess={
                'means2d': h_means2d_out,
                'conics': h_conics_out,
                'mus': h_mus_out,
                'opacities': h_opacities_out,
            },
            cross_hess={
                'mu_mean': h_mu_mean_out,
                'mu_conic': h_mu_conic_out,
                'mean_conic': h_mean_conic_out,
                'mu_opacity': h_mu_opacity_out,
                'conic_opacity': h_conic_opacity_out,
                'mean_opacity': h_mean_opacity_out,
            }
        )
        
        return forward_result, grad_result, hess_result


# ============================================================================
# Factory Function
# ============================================================================

def create_rasterize_to_pixels(backend: str = "cuda") -> SecondForwardBackwardBase:
    """Factory function to create rasterize to pixels operator.
    
    Args:
        backend: "cuda" for CUDA implementation, "torch" for PyTorch reference
    
    Returns:
        RasterizeToPixelsBase instance
    """
    if backend == "cuda":
        return RasterizeToPixelsCUDA()
    elif backend == "torch":
        return RasterizeToPixelsTorch()
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'cuda' or 'torch'.")


__all__ = [
    "RasterizeToPixelsCUDA",
    "RasterizeToPixelsTorch",
    "create_rasterize_to_pixels",
]
