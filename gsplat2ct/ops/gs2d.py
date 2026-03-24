"""
2D Gaussian Splatting with Image Loss

This module provides end-to-end implementations for 2D Gaussian Splatting:
- Forward: Render 2D Gaussians to image, compute image loss
- Backward1: Compute first-order gradients w.r.t. Gaussian parameters
- Backward2: Compute second-order Hessians w.r.t. Gaussian parameters

The pipeline is:
    2D Gaussian Parameters (means2d, conics, opacities, mus)
        -> Rasterization -> Rendered Image
        -> Image Loss (L1 + SSIM) -> Scalar Loss

For second-order optimization:
    Loss -> v_image (first derivative) + h_image (second derivative)
         -> Chain rule through rasterization
         -> Gradients + Hessians for Gaussian parameters

Note on SSIM:
    - SSIM introduces cross-pixel dependencies in the Hessian
    - Our CUDA implementation uses diagonal approximation (ignoring cross-pixel terms)
    - When lambda_ssim=0 (L1 only), CUDA and torch autograd should match exactly

Author: Dy164
Date: 2026-01-30
"""

import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple

from .base import SecondForwardBackwardBase
from .data import Params, Data, Config, ForwardResult, GradResult, HessResult


# ============================================================================
# Data Structures for 2D Gaussian Splatting
# ============================================================================

@dataclass
class GS2DParams(Params):
    """Parameters for 2D Gaussian Splatting.
    
    Attributes:
        means2d: 2D projected Gaussian centers (C, N, 2), format: (x, y) = (col, row)
        conics: Inverse covariance matrices (C, N, 3), stored as [a, b, c]
        opacities: Gaussian opacities (C, N)
        mus: Gaussian mu values (CT attenuation coefficients) (C, N)
    """
    means2d: Optional[torch.Tensor] = None
    conics: Optional[torch.Tensor] = None
    opacities: Optional[torch.Tensor] = None
    mus: Optional[torch.Tensor] = None
    
    def __post_init__(self):
        """Sync named fields to param_map."""
        if self.means2d is not None:
            self.param_map['means2d'] = type('Param', (), {'data': self.means2d})()
        if self.conics is not None:
            self.param_map['conics'] = type('Param', (), {'data': self.conics})()
        if self.opacities is not None:
            self.param_map['opacities'] = type('Param', (), {'data': self.opacities})()
        if self.mus is not None:
            self.param_map['mus'] = type('Param', (), {'data': self.mus})()


@dataclass
class GS2DData(Data):
    """Data for 2D Gaussian Splatting.
    
    Attributes:
        target: Target image (C, H, W)
        radii: Gaussian radii for tile intersection (C, N)
    """
    target: Optional[torch.Tensor] = None
    radii: Optional[torch.Tensor] = None
    
    def __post_init__(self):
        """Sync named fields to data_map."""
        if self.target is not None:
            self.data_map['target'] = self.target
        if self.radii is not None:
            self.data_map['radii'] = self.radii


@dataclass
class GS2DConfig(Config):
    """Configuration for 2D Gaussian Splatting.
    
    Attributes:
        tile_size: Size of each tile in pixels (default 16)
        lambda_l1: Weight for L1 loss (default 1.0)
        lambda_ssim: Weight for SSIM loss (default 0.0 for testing)
        window_size: SSIM window size (default 11)
    """
    tile_size: int = 16
    lambda_l1: float = 1.0
    lambda_ssim: float = 0.0  # Default 0 for exact comparison with torch
    window_size: int = 11
    
    def __post_init__(self):
        """Sync named fields to options."""
        self.options['tile_size'] = self.tile_size
        self.options['lambda_l1'] = self.lambda_l1
        self.options['lambda_ssim'] = self.lambda_ssim
        self.options['window_size'] = self.window_size


# ============================================================================
# Autograd Implementation (for testing/validation)
# ============================================================================

class GS2DAutograd(SecondForwardBackwardBase):
    """
    2D Gaussian Splatting with Image Loss using torch autograd.
    
    This implementation uses pure PyTorch operations and autograd for:
    - Forward rasterization
    - First-order gradients (autograd)
    - Second-order Hessians (autograd)
    
    Use this for testing and validation against the CUDA implementation.
    Note: Only accurate when lambda_ssim=0 (L1 loss only).
    """
    
    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_ssim: float = 0.0,
        window_size: int = 11,
        tile_size: int = 16,
    ):
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.window_size = window_size
        self.tile_size = tile_size
        self.initialized = False
    
    def initialize(
        self,
        params: Params,
        config: Optional[Config] = None,
    ) -> None:
        """Initialize and validate parameters."""
        self.initialized = True
    
    def _rasterize_torch(
        self,
        means2d: torch.Tensor,
        conics: torch.Tensor,
        opacities: torch.Tensor,
        mus: torch.Tensor,
        dim0: int,
        dim1: int,
    ) -> torch.Tensor:
        """
        Pure PyTorch rasterization for 2D Gaussians.
        
        CT Formula: O[i,j] = Σ_g (opacity_g × mu_g × exp(-sigma_g))
        where sigma_g = 0.5 × (a×δx² + c×δy²) + b×δx×δy
        
        Args:
            means2d: (C, N, 2) Gaussian centers
            conics: (C, N, 3) Inverse covariance [a, b, c]
            opacities: (C, N) Gaussian opacities
            mus: (C, N) Gaussian mu values
            dim0: Output height (rows)
            dim1: Output width (columns)
            
        Returns:
            render_output: (C, dim0, dim1) Rendered image
        """
        C, N, _ = means2d.shape
        device = means2d.device
        dtype = means2d.dtype
        
        # Create pixel grid (pixel centers at +0.5)
        y_coords = torch.arange(dim0, device=device, dtype=dtype) + 0.5
        x_coords = torch.arange(dim1, device=device, dtype=dtype) + 0.5
        py, px = torch.meshgrid(y_coords, x_coords, indexing='ij')  # [dim0, dim1]
        
        # Initialize output
        render_output = torch.zeros(C, dim0, dim1, device=device, dtype=dtype)
        
        # Render each camera and Gaussian
        for c in range(C):
            for g in range(N):
                mean = means2d[c, g]  # [2]: (x, y) = (col, row)
                conic = conics[c, g]  # [3]: (a, b, c)
                opacity = opacities[c, g]
                mu = mus[c, g]
                
                # Compute delta (pixel - center)
                dx = px - mean[0]  # x = column
                dy = py - mean[1]  # y = row
                
                # Compute sigma = 0.5 * (a*dx^2 + c*dy^2) + b*dx*dy
                sigma = 0.5 * (conic[0] * dx * dx + conic[2] * dy * dy) + conic[1] * dx * dy
                
                # Compute contribution
                alpha = opacity * mu * torch.exp(-sigma)
                
                # Apply numerical mask (same as CUDA)
                mask = (sigma >= 0) & (alpha.abs() >= 1e-5)
                alpha = torch.where(mask, alpha, torch.zeros_like(alpha))
                
                render_output[c] += alpha
        
        return render_output
    
    def _compute_l1_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute L2 (MSE) loss — matches the CUDA image_loss_fwd kernel which
        uses compute_l2_map_kernel: l2_map = (pred - target)^2, loss = l2_map.mean()."""
        diff = prediction - target
        return (diff * diff).mean()
    
    def forward_only(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> ForwardResult:
        """Forward computation only."""
        # Extract parameters
        if isinstance(params, GS2DParams):
            means2d = params.means2d
            conics = params.conics
            opacities = params.opacities
            mus = params.mus
        else:
            means2d = params.get('means2d').data
            conics = params.get('conics').data
            opacities = params.get('opacities').data
            mus = params.get('mus').data
        
        # Extract data
        if isinstance(data, GS2DData):
            target = data.target
        else:
            target = data.get('target')
        
        # Get config
        lambda_l1 = config.get('lambda_l1', self.lambda_l1) if config else self.lambda_l1
        lambda_ssim = config.get('lambda_ssim', self.lambda_ssim) if config else self.lambda_ssim
        
        C, dim0, dim1 = target.shape
        
        # Rasterize
        render_output = self._rasterize_torch(means2d, conics, opacities, mus, dim0, dim1)
        
        # Compute loss
        l1_loss = self._compute_l1_loss(render_output, target)
        loss = lambda_l1 * l1_loss
        
        if lambda_ssim > 0:
            from .loss import ssim
            ssim_val = ssim(render_output, target, window_size=self.window_size)
            loss = loss + lambda_ssim * (1.0 - ssim_val)
        
        return ForwardResult(
            outputs={
                'loss': loss,
                'render_output': render_output.detach(),
            },
            intermediates={}
        )
    
    def forward_backward1(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> Tuple[ForwardResult, GradResult]:
        """Forward + first-order backward."""
        # Extract parameters with gradients
        if isinstance(params, GS2DParams):
            means2d = params.means2d.detach().clone().requires_grad_(True)
            conics = params.conics.detach().clone().requires_grad_(True)
            opacities = params.opacities.detach().clone().requires_grad_(True)
            mus = params.mus.detach().clone().requires_grad_(True)
        else:
            means2d = params.get('means2d').data.detach().clone().requires_grad_(True)
            conics = params.get('conics').data.detach().clone().requires_grad_(True)
            opacities = params.get('opacities').data.detach().clone().requires_grad_(True)
            mus = params.get('mus').data.detach().clone().requires_grad_(True)
        
        # Extract data
        if isinstance(data, GS2DData):
            target = data.target
        else:
            target = data.get('target')
        
        # Get config
        lambda_l1 = config.get('lambda_l1', self.lambda_l1) if config else self.lambda_l1
        lambda_ssim = config.get('lambda_ssim', self.lambda_ssim) if config else self.lambda_ssim
        
        C, dim0, dim1 = target.shape
        
        # Forward pass
        render_output = self._rasterize_torch(means2d, conics, opacities, mus, dim0, dim1)
        
        # Compute loss
        l1_loss = self._compute_l1_loss(render_output, target)
        loss = lambda_l1 * l1_loss
        
        if lambda_ssim > 0:
            from .loss import ssim
            ssim_val = ssim(render_output, target, window_size=self.window_size)
            loss = loss + lambda_ssim * (1.0 - ssim_val)
        
        # Backward pass
        loss.backward()
        
        forward_result = ForwardResult(
            outputs={
                'loss': loss.detach(),
                'render_output': render_output.detach(),
            },
            intermediates={}
        )
        
        grad_result = GradResult()
        grad_result.set('means2d', means2d.grad.detach())
        grad_result.set('conics', conics.grad.detach())
        grad_result.set('opacities', opacities.grad.detach())
        grad_result.set('mus', mus.grad.detach())
        
        return forward_result, grad_result
    
    def forward_backward2(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> Tuple[ForwardResult, GradResult, HessResult]:
        """Forward + first-order + second-order backward using torch autograd."""
        # Extract parameters with gradients
        if isinstance(params, GS2DParams):
            means2d = params.means2d.detach().clone().requires_grad_(True)
            conics = params.conics.detach().clone().requires_grad_(True)
            opacities = params.opacities.detach().clone().requires_grad_(True)
            mus = params.mus.detach().clone().requires_grad_(True)
        else:
            means2d = params.get('means2d').data.detach().clone().requires_grad_(True)
            conics = params.get('conics').data.detach().clone().requires_grad_(True)
            opacities = params.get('opacities').data.detach().clone().requires_grad_(True)
            mus = params.get('mus').data.detach().clone().requires_grad_(True)
        
        # Extract data
        if isinstance(data, GS2DData):
            target = data.target
        else:
            target = data.get('target')
        
        # Get config
        lambda_l1 = config.get('lambda_l1', self.lambda_l1) if config else self.lambda_l1
        lambda_ssim = config.get('lambda_ssim', self.lambda_ssim) if config else self.lambda_ssim
        
        C, dim0, dim1 = target.shape
        N = means2d.shape[1]
        device = means2d.device
        dtype = means2d.dtype
        
        # Forward pass
        render_output = self._rasterize_torch(means2d, conics, opacities, mus, dim0, dim1)
        
        # Compute loss
        l1_loss = self._compute_l1_loss(render_output, target)
        loss = lambda_l1 * l1_loss
        
        if lambda_ssim > 0:
            from .loss import ssim
            ssim_val = ssim(render_output, target, window_size=self.window_size)
            loss = loss + lambda_ssim * (1.0 - ssim_val)
        
        # First-order gradients
        v_means2d, v_conics, v_opacities, v_mus = torch.autograd.grad(
            loss, [means2d, conics, opacities, mus],
            create_graph=True
        )
        
        # Second-order Hessians (diagonal only, per-Gaussian)
        # h_means2d: [C, N, 3] upper triangle [H00, H01, H11]
        # h_conics: [C, N, 6] upper triangle [H00, H01, H02, H11, H12, H22]
        # h_mus: [C, N] scalar
        # h_opacities: [C, N] scalar
        
        h_means2d = torch.zeros(C, N, 3, device=device, dtype=dtype)
        h_conics = torch.zeros(C, N, 6, device=device, dtype=dtype)
        h_mus = torch.zeros(C, N, device=device, dtype=dtype)
        h_opacities = torch.zeros(C, N, device=device, dtype=dtype)
        
        # Cross-terms
        h_mu_mean = torch.zeros(C, N, 2, device=device, dtype=dtype)
        h_mu_conic = torch.zeros(C, N, 3, device=device, dtype=dtype)
        h_mean_conic = torch.zeros(C, N, 2, 3, device=device, dtype=dtype)
        h_mu_opacity = torch.zeros(C, N, device=device, dtype=dtype)
        h_conic_opacity = torch.zeros(C, N, 3, device=device, dtype=dtype)
        h_mean_opacity = torch.zeros(C, N, 2, device=device, dtype=dtype)
        
        # Compute Hessians for each Gaussian
        for c in range(C):
            for g in range(N):
                # h_means2d
                for i in range(2):
                    grad_i = torch.autograd.grad(
                        v_means2d[c, g, i], means2d,
                        retain_graph=True, allow_unused=True
                    )[0]
                    if grad_i is not None:
                        if i == 0:
                            h_means2d[c, g, 0] = grad_i[c, g, 0]  # H00
                            h_means2d[c, g, 1] = grad_i[c, g, 1]  # H01
                        else:
                            h_means2d[c, g, 2] = grad_i[c, g, 1]  # H11
                
                # h_conics
                for i in range(3):
                    grad_i = torch.autograd.grad(
                        v_conics[c, g, i], conics,
                        retain_graph=True, allow_unused=True
                    )[0]
                    if grad_i is not None:
                        if i == 0:
                            h_conics[c, g, 0] = grad_i[c, g, 0]  # H00
                            h_conics[c, g, 1] = grad_i[c, g, 1]  # H01
                            h_conics[c, g, 2] = grad_i[c, g, 2]  # H02
                        elif i == 1:
                            h_conics[c, g, 3] = grad_i[c, g, 1]  # H11
                            h_conics[c, g, 4] = grad_i[c, g, 2]  # H12
                        else:
                            h_conics[c, g, 5] = grad_i[c, g, 2]  # H22
                
                # h_mus
                try:
                    grad_mu = torch.autograd.grad(
                        v_mus[c, g], mus,
                        retain_graph=True, allow_unused=True
                    )[0]
                    if grad_mu is not None:
                        h_mus[c, g] = grad_mu[c, g]
                except RuntimeError:
                    pass
                
                # h_opacities
                try:
                    grad_o = torch.autograd.grad(
                        v_opacities[c, g], opacities,
                        retain_graph=True, allow_unused=True
                    )[0]
                    if grad_o is not None:
                        h_opacities[c, g] = grad_o[c, g]
                except RuntimeError:
                    pass
                
                # Cross-term h_mu_mean
                try:
                    grad_mu_mean = torch.autograd.grad(
                        v_mus[c, g], means2d,
                        retain_graph=True, allow_unused=True
                    )[0]
                    if grad_mu_mean is not None:
                        h_mu_mean[c, g] = grad_mu_mean[c, g]
                except RuntimeError:
                    pass
                
                # Cross-term h_mu_conic
                try:
                    grad_mu_conic = torch.autograd.grad(
                        v_mus[c, g], conics,
                        retain_graph=True, allow_unused=True
                    )[0]
                    if grad_mu_conic is not None:
                        h_mu_conic[c, g] = grad_mu_conic[c, g]
                except RuntimeError:
                    pass
                
                # Cross-term h_mean_conic
                for i in range(2):
                    try:
                        grad_mc = torch.autograd.grad(
                            v_means2d[c, g, i], conics,
                            retain_graph=True, allow_unused=True
                        )[0]
                        if grad_mc is not None:
                            h_mean_conic[c, g, i] = grad_mc[c, g]
                    except RuntimeError:
                        pass
                
                # Cross-term h_mu_opacity
                try:
                    grad_mu_o = torch.autograd.grad(
                        v_mus[c, g], opacities,
                        retain_graph=True, allow_unused=True
                    )[0]
                    if grad_mu_o is not None:
                        h_mu_opacity[c, g] = grad_mu_o[c, g]
                except RuntimeError:
                    pass
                
                # Cross-term h_conic_opacity
                for i in range(3):
                    try:
                        grad_co = torch.autograd.grad(
                            v_conics[c, g, i], opacities,
                            retain_graph=True, allow_unused=True
                        )[0]
                        if grad_co is not None:
                            h_conic_opacity[c, g, i] = grad_co[c, g]
                    except RuntimeError:
                        pass
                
                # Cross-term h_mean_opacity
                for i in range(2):
                    try:
                        grad_mo = torch.autograd.grad(
                            v_means2d[c, g, i], opacities,
                            retain_graph=True, allow_unused=True
                        )[0]
                        if grad_mo is not None:
                            h_mean_opacity[c, g, i] = grad_mo[c, g]
                    except RuntimeError:
                        pass
        
        forward_result = ForwardResult(
            outputs={
                'loss': loss.detach(),
                'render_output': render_output.detach(),
            },
            intermediates={}
        )
        
        grad_result = GradResult()
        grad_result.set('means2d', v_means2d.detach())
        grad_result.set('conics', v_conics.detach())
        grad_result.set('opacities', v_opacities.detach())
        grad_result.set('mus', v_mus.detach())
        
        # For optimizer: extract diagonal Hessians (matching gradient shapes)
        # h_means2d: (C, N, 3) [H00, H01, H11] -> (C, N, 2) [H00, H11]
        # h_conics: (C, N, 6) [H00, H01, H02, H11, H12, H22] -> (C, N, 3) [H00, H11, H22]
        # h_mus: (C, N) already scalar
        # h_opacities: (C, N) already scalar
        h_means2d_diag = torch.stack([h_means2d[..., 0], h_means2d[..., 2]], dim=-1)  # (C, N, 2)
        h_conics_diag = torch.stack([h_conics[..., 0], h_conics[..., 3], h_conics[..., 5]], dim=-1)  # (C, N, 3)
        
        hess_result = HessResult()
        hess_result.set('means2d', h_means2d_diag)
        hess_result.set('conics', h_conics_diag)
        hess_result.set('mus', h_mus)
        hess_result.set('opacities', h_opacities)
        hess_result.set_cross(
            mu_mean=h_mu_mean,
            mu_conic=h_mu_conic,
            mean_conic=h_mean_conic,
            mu_opacity=h_mu_opacity,
            conic_opacity=h_conic_opacity,
            mean_opacity=h_mean_opacity,
        )
        
        return forward_result, grad_result, hess_result


# ============================================================================
# CUDA Implementation
# ============================================================================

class GS2DCUDA(SecondForwardBackwardBase):
    """
    2D Gaussian Splatting with Image Loss using CUDA kernels.
    
    This implementation uses custom CUDA kernels for:
    - Forward rasterization (rasterize_to_pixels_fwd)
    - First-order backward (rasterize_to_pixels_bwd + image_loss_bwd)
    - Second-order backward (rasterize_to_pixels_bwd2 + image_loss_bwd2)
    
    The pipeline chains:
    1. Rasterization: Gaussian params -> Rendered image
    2. Image loss: Rendered image vs Target -> Loss
    
    For backward passes, we use chain rule:
    - v_params = v_image @ (∂image/∂params)
    - h_params = (∂image/∂params)^T @ h_image @ (∂image/∂params) + ...
    """
    
    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_ssim: float = 0.0,
        window_size: int = 11,
        tile_size: int = 16,
    ):
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.window_size = window_size
        self.tile_size = tile_size
        self.initialized = False
        
        # Cached intermediate values
        self._tile_offsets = None
        self._flatten_ids = None
        self._render_output = None
        
        # Image loss statistics
        self._mu_r = None
        self._mu_m = None
        self._f1 = None
        self._f2 = None
        self._f3 = None
        self._f4 = None
    
    def initialize(
        self,
        params: Params,
        config: Optional[Config] = None,
    ) -> None:
        """Initialize and validate parameters."""
        self.initialized = True
    
    def _compute_tile_intersections(
        self,
        means2d: torch.Tensor,
        radii: torch.Tensor,
        dim0: int,
        dim1: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute tile intersections for rasterization."""
        from ..cuda._wrapper import isect_tiles, isect_offset_encode
        
        C = means2d.shape[0]
        n_tiles_dim0 = (dim0 + self.tile_size - 1) // self.tile_size
        n_tiles_dim1 = (dim1 + self.tile_size - 1) // self.tile_size
        
        _, isect_ids, flatten_ids = isect_tiles(
            means2d, radii, self.tile_size, n_tiles_dim1, n_tiles_dim0
        )
        tile_offsets = isect_offset_encode(isect_ids, C, n_tiles_dim1, n_tiles_dim0)
        
        return tile_offsets, flatten_ids
    
    def forward_only(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> ForwardResult:
        """Forward computation only."""
        from ..cuda._wrapper import rasterize_to_pixels_fwd, image_loss_fwd
        
        # Extract parameters
        if isinstance(params, GS2DParams):
            means2d = params.means2d
            conics = params.conics
            opacities = params.opacities
            mus = params.mus
        else:
            means2d = params.get('means2d').data
            conics = params.get('conics').data
            opacities = params.get('opacities').data
            mus = params.get('mus').data
        
        # Extract data
        if isinstance(data, GS2DData):
            target = data.target
            radii = data.radii
        else:
            target = data.get('target')
            radii = data.get('radii')
        
        # Get config
        lambda_l1 = config.get('lambda_l1', self.lambda_l1) if config else self.lambda_l1
        lambda_ssim = config.get('lambda_ssim', self.lambda_ssim) if config else self.lambda_ssim
        window_size = config.get('window_size', self.window_size) if config else self.window_size
        
        C, dim0, dim1 = target.shape
        
        # Compute tile intersections
        tile_offsets, flatten_ids = self._compute_tile_intersections(
            means2d, radii, dim0, dim1
        )
        self._tile_offsets = tile_offsets
        self._flatten_ids = flatten_ids
        
        # Rasterize
        render_output = rasterize_to_pixels_fwd(
            means2d, conics, opacities, mus,
            tile_offsets, flatten_ids,
            dim1, dim0, self.tile_size
        )
        self._render_output = render_output
        
        # Compute image loss
        result = image_loss_fwd(
            render_output, target, window_size,
            lambda_l1, lambda_ssim,
            compute_statistics=False
        )
        loss = result[0]
        
        return ForwardResult(
            outputs={
                'loss': loss,
                'render_output': render_output.detach(),
            },
            intermediates={}
        )
    
    def forward_backward1(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> Tuple[ForwardResult, GradResult]:
        """Forward + first-order backward."""
        from ..cuda._wrapper import (
            rasterize_to_pixels_fwd, rasterize_to_pixels_bwd,
            image_loss_fwd, image_loss_bwd
        )
        
        # Extract parameters
        if isinstance(params, GS2DParams):
            means2d = params.means2d
            conics = params.conics
            opacities = params.opacities
            mus = params.mus
        else:
            means2d = params.get('means2d').data
            conics = params.get('conics').data
            opacities = params.get('opacities').data
            mus = params.get('mus').data
        
        # Extract data
        if isinstance(data, GS2DData):
            target = data.target
            radii = data.radii
        else:
            target = data.get('target')
            radii = data.get('radii')
        
        # Get config
        lambda_l1 = config.get('lambda_l1', self.lambda_l1) if config else self.lambda_l1
        lambda_ssim = config.get('lambda_ssim', self.lambda_ssim) if config else self.lambda_ssim
        window_size = config.get('window_size', self.window_size) if config else self.window_size
        
        C, dim0, dim1 = target.shape
        
        # Compute tile intersections
        tile_offsets, flatten_ids = self._compute_tile_intersections(
            means2d, radii, dim0, dim1
        )
        self._tile_offsets = tile_offsets
        self._flatten_ids = flatten_ids
        
        # Forward: Rasterize
        render_output = rasterize_to_pixels_fwd(
            means2d, conics, opacities, mus,
            tile_offsets, flatten_ids,
            dim1, dim0, self.tile_size
        )
        self._render_output = render_output
        
        # Forward: Image loss with statistics
        result = image_loss_fwd(
            render_output, target, window_size,
            lambda_l1, lambda_ssim,
            compute_statistics=True
        )
        loss = result[0]
        self._mu_r, self._mu_m = result[3], result[4]
        self._f1, self._f2, self._f3, self._f4 = result[5], result[6], result[7], result[8]
        
        # Backward: Image loss
        grad_output = torch.ones(1, device=render_output.device, dtype=render_output.dtype)
        v_render_output = image_loss_bwd(
            render_output, target,
            self._mu_r, self._mu_m,
            self._f1, self._f2, self._f3, self._f4,
            grad_output, window_size,
            lambda_l1, lambda_ssim
        )
        
        # Backward: Rasterize
        v_means2d, v_conics, v_opacities, v_mus = rasterize_to_pixels_bwd(
            means2d, conics, opacities, mus,
            tile_offsets, flatten_ids, v_render_output,
            dim1, dim0, self.tile_size
        )
        
        forward_result = ForwardResult(
            outputs={
                'loss': loss,
                'render_output': render_output.detach(),
            },
            intermediates={
                'v_render_output': v_render_output,
            }
        )
        
        grad_result = GradResult()
        grad_result.set('means2d', v_means2d)
        grad_result.set('conics', v_conics)
        grad_result.set('opacities', v_opacities)
        grad_result.set('mus', v_mus)
        
        return forward_result, grad_result
    
    def forward_backward2(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> Tuple[ForwardResult, GradResult, HessResult]:
        """Forward + first-order + second-order backward."""
        from ..cuda._wrapper import (
            rasterize_to_pixels_fwd, rasterize_to_pixels_bwd2,
            image_loss_fwd, image_loss_bwd, image_loss_bwd2
        )
        
        # Extract parameters
        if isinstance(params, GS2DParams):
            means2d = params.means2d
            conics = params.conics
            opacities = params.opacities
            mus = params.mus
        else:
            means2d = params.get('means2d').data
            conics = params.get('conics').data
            opacities = params.get('opacities').data
            mus = params.get('mus').data
        
        # Extract data
        if isinstance(data, GS2DData):
            target = data.target
            radii = data.radii
        else:
            target = data.get('target')
            radii = data.get('radii')
        
        # Get config
        lambda_l1 = config.get('lambda_l1', self.lambda_l1) if config else self.lambda_l1
        lambda_ssim = config.get('lambda_ssim', self.lambda_ssim) if config else self.lambda_ssim
        window_size = config.get('window_size', self.window_size) if config else self.window_size
        
        C, dim0, dim1 = target.shape
        
        # Compute tile intersections
        tile_offsets, flatten_ids = self._compute_tile_intersections(
            means2d, radii, dim0, dim1
        )
        self._tile_offsets = tile_offsets
        self._flatten_ids = flatten_ids
        
        # Forward: Rasterize
        render_output = rasterize_to_pixels_fwd(
            means2d, conics, opacities, mus,
            tile_offsets, flatten_ids,
            dim1, dim0, self.tile_size
        )
        self._render_output = render_output
        
        # Forward: Image loss with statistics
        result = image_loss_fwd(
            render_output, target, window_size,
            lambda_l1, lambda_ssim,
            compute_statistics=True
        )
        loss = result[0]
        self._mu_r, self._mu_m = result[3], result[4]
        self._f1, self._f2, self._f3, self._f4 = result[5], result[6], result[7], result[8]
        
        # Backward1: Image loss -> v_render_output
        grad_output = torch.ones(1, device=render_output.device, dtype=render_output.dtype)
        v_render_output = image_loss_bwd(
            render_output, target,
            self._mu_r, self._mu_m,
            self._f1, self._f2, self._f3, self._f4,
            grad_output, window_size,
            lambda_l1, lambda_ssim
        )
        
        # Backward2: Image loss -> h_render_output (diagonal Hessian)
        h_render_output = image_loss_bwd2(
            render_output, target,
            self._mu_r, self._mu_m,
            self._f1, self._f2, self._f3, self._f4,
            grad_output, window_size,
            lambda_l1, lambda_ssim
        )
        
        # Backward2: Rasterize with both v and h
        (v_means2d, v_conics, v_opacities, v_mus,
         h_means2d, h_conics,
         h_mu_mean, h_mu_conic, h_mean_conic,
         h_mu_opacity, h_conic_opacity, h_mean_opacity,
         h_mus, h_opacities) = rasterize_to_pixels_bwd2(
            means2d, conics, opacities, mus,
            tile_offsets, flatten_ids,
            v_render_output, h_render_output,
            dim1, dim0, self.tile_size
        )
        
        forward_result = ForwardResult(
            outputs={
                'loss': loss,
                'render_output': render_output.detach(),
            },
            intermediates={
                'v_render_output': v_render_output,
                'h_render_output': h_render_output,
            }
        )
        
        grad_result = GradResult()
        grad_result.set('means2d', v_means2d)
        grad_result.set('conics', v_conics)
        grad_result.set('opacities', v_opacities)
        grad_result.set('mus', v_mus)
        
        # For optimizer: extract diagonal Hessians (matching gradient shapes)
        # h_means2d: (C, N, 3) [H00, H01, H11] -> (C, N, 2) [H00, H11]
        # h_conics: (C, N, 6) [H00, H01, H02, H11, H12, H22] -> (C, N, 3) [H00, H11, H22]
        # h_mus: (C, N) already scalar
        # h_opacities: (C, N) already scalar
        h_means2d_diag = torch.stack([h_means2d[..., 0], h_means2d[..., 2]], dim=-1)  # (C, N, 2)
        h_conics_diag = torch.stack([h_conics[..., 0], h_conics[..., 3], h_conics[..., 5]], dim=-1)  # (C, N, 3)
        
        hess_result = HessResult()
        hess_result.set('means2d', h_means2d_diag)
        hess_result.set('conics', h_conics_diag)
        hess_result.set('mus', h_mus)
        hess_result.set('opacities', h_opacities)
        hess_result.set_cross(
            mu_mean=h_mu_mean,
            mu_conic=h_mu_conic,
            mean_conic=h_mean_conic,
            mu_opacity=h_mu_opacity,
            conic_opacity=h_conic_opacity,
            mean_opacity=h_mean_opacity,
        )
        
        return forward_result, grad_result, hess_result


# ============================================================================
# Factory Functions
# ============================================================================

def create_gs2d_autograd(
    lambda_l1: float = 1.0,
    lambda_ssim: float = 0.0,
    window_size: int = 11,
    tile_size: int = 16,
) -> GS2DAutograd:
    """Create 2D Gaussian Splatting operator (autograd version)."""
    return GS2DAutograd(
        lambda_l1=lambda_l1,
        lambda_ssim=lambda_ssim,
        window_size=window_size,
        tile_size=tile_size,
    )


def create_gs2d_cuda(
    lambda_l1: float = 1.0,
    lambda_ssim: float = 0.0,
    window_size: int = 11,
    tile_size: int = 16,
) -> GS2DCUDA:
    """Create 2D Gaussian Splatting operator (CUDA version)."""
    return GS2DCUDA(
        lambda_l1=lambda_l1,
        lambda_ssim=lambda_ssim,
        window_size=window_size,
        tile_size=tile_size,
    )


__all__ = [
    # Data structures
    'GS2DParams',
    'GS2DData',
    'GS2DConfig',
    # Operators
    'GS2DAutograd',
    'GS2DCUDA',
    # Factory functions
    'create_gs2d_autograd',
    'create_gs2d_cuda',
]
