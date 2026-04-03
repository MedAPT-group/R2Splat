"""
2D Gaussian Splatting with ConicsMu Rectification + Image Loss

This module provides end-to-end implementations for 2D Gaussian Splatting
where the pipeline starts from projected 3x3 covariance matrices instead of
pre-computed conics:

    cov2d_3x3 -> ConicsMu -> (conics, mu) -> Rasterization -> Image -> Loss

Pipeline:
    Input: (cov2d_3x3[C,N,6], means2d[C,N,2], opacities[C,N])
        -> ConicsMu Forward -> conics[C,N,3], mus[C,N]
        -> Tile intersection
        -> RasterizeToPixels Forward -> render_output[C,H,W]
        -> ImageLoss Forward -> loss (scalar)

For second-order optimization, backward2 chains:
    ImageLoss bwd/bwd2 -> R2P bwd2 (14 outputs) -> ConicsMu bwd2 (4 outputs)

The h_mean_conic output from R2P (2x3 row-major) is passed directly to
ConicsMu bwd2 as h_means2d_conics (same layout), avoiding any transpose.

Author: Dy164
Date: 2026-02-19
"""

import math
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple

from .base import SecondForwardBackwardBase
from .data import Params, Data, Config, ForwardResult, GradResult, HessResult, get_lambda_l2


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class GS2DRectifyingParams(Params):
    """Parameters for 2D Gaussian Splatting with ConicsMu rectification.
    
    Attributes:
        cov2d_3x3: Projected 3x3 covariance matrices (C, N, 6),
                    upper-triangular storage (a, b, c, d, e, f)
        means2d: 2D projected Gaussian centers (C, N, 2), format: (x, y) = (col, row)
        opacities: Gaussian opacities (C, N)
    """
    cov2d_3x3: Optional[torch.Tensor] = None
    means2d: Optional[torch.Tensor] = None
    opacities: Optional[torch.Tensor] = None
    
    def __post_init__(self):
        """Sync named fields to param_map."""
        if self.cov2d_3x3 is not None:
            self.param_map['cov2d_3x3'] = type('Param', (), {'data': self.cov2d_3x3})()
        if self.means2d is not None:
            self.param_map['means2d'] = type('Param', (), {'data': self.means2d})()
        if self.opacities is not None:
            self.param_map['opacities'] = type('Param', (), {'data': self.opacities})()


@dataclass
class GS2DRectifyingData(Data):
    """Data for 2D Gaussian Splatting with ConicsMu rectification.
    
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
class GS2DRectifyingConfig(Config):
    """Configuration for 2D Gaussian Splatting with ConicsMu rectification.
    
    Attributes:
        eps: ConicsMu blur epsilon (default 0.3)
        tile_size: Size of each tile in pixels (default 16)
        lambda_l2: Weight for L2 loss (default 1.0)
        lambda_ssim: Weight for SSIM loss (default 0.0 for testing)
        window_size: SSIM window size (default 11)
    """
    eps: float = 0.3
    tile_size: int = 16
    lambda_l2: float = 1.0
    lambda_ssim: float = 0.0
    window_size: int = 11
    
    def __post_init__(self):
        """Sync named fields to options."""
        self.options['eps'] = self.eps
        self.options['tile_size'] = self.tile_size
        self.options['lambda_l2'] = self.lambda_l2
        self.options['lambda_ssim'] = self.lambda_ssim
        self.options['window_size'] = self.window_size


# ============================================================================
# Autograd Implementation (for testing/validation)
# ============================================================================

class GS2DRectifyingAutograd(SecondForwardBackwardBase):
    """
    2D Gaussian Splatting with ConicsMu + ImageLoss using torch autograd.
    
    Pipeline: cov2d_3x3 -> ConicsMu -> Rasterize -> Loss
    
    Uses pure PyTorch operations and autograd for:
    - Forward: ConicsMu + rasterization + loss
    - Backward1: First-order gradients (autograd)
    - Backward2: Second-order Hessians (autograd double backward)
    
    Use this for testing and validation against the CUDA implementation.
    Note: Only accurate when lambda_ssim=0 (L2 loss only).
    """
    
    def __init__(
        self,
        eps: float = 0.3,
        lambda_l2: float = 1.0,
        lambda_ssim: float = 0.0,
        window_size: int = 11,
        tile_size: int = 16,
    ):
        self.eps = eps
        self.lambda_l2 = lambda_l2
        self.lambda_ssim = lambda_ssim
        self.window_size = window_size
        self.tile_size = tile_size
        self.initialized = False
    
    def initialize(
        self,
        params: Params,
        config: Optional[Config] = None,
    ) -> None:
        self.initialized = True
    
    @staticmethod
    def _compute_conics_mu(cov2d_3x3: torch.Tensor, eps: float):
        """Pure PyTorch differentiable ConicsMu computation."""
        a = cov2d_3x3[..., 0]
        b = cov2d_3x3[..., 1]
        c = cov2d_3x3[..., 2]
        d = cov2d_3x3[..., 3]
        e = cov2d_3x3[..., 4]
        f = cov2d_3x3[..., 5]
        
        ab = a + eps
        db = d + eps
        fb = f + eps
        
        D2b = ab * db - b * b
        inv_D2b = 1.0 / D2b
        u11 = db * inv_D2b
        u12 = -b * inv_D2b
        u22 = ab * inv_D2b
        conics = torch.stack([u11, u12, u22], dim=-1)
        
        D3b = (ab * (db * fb - e * e)
               - b * (b * fb - e * c)
               + c * (b * e - db * c))
        
        ratio = D3b / D2b
        ratio_clamped = torch.clamp(ratio, min=0.0)
        mus = torch.sqrt(2.0 * math.pi * ratio_clamped)
        
        return conics, mus
    
    @staticmethod
    def _rasterize_torch(
        means2d: torch.Tensor,
        conics: torch.Tensor,
        opacities: torch.Tensor,
        mus: torch.Tensor,
        dim0: int,
        dim1: int,
    ) -> torch.Tensor:
        """Pure PyTorch rasterization for 2D Gaussians.
        
        CT Formula: O[i,j] = Σ_g (opacity_g × mu_g × exp(-sigma_g))
        """
        C, N, _ = means2d.shape
        device = means2d.device
        dtype = means2d.dtype
        
        y_coords = torch.arange(dim0, device=device, dtype=dtype) + 0.5
        x_coords = torch.arange(dim1, device=device, dtype=dtype) + 0.5
        py, px = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        render_output = torch.zeros(C, dim0, dim1, device=device, dtype=dtype)
        
        for c_idx in range(C):
            for g in range(N):
                mean = means2d[c_idx, g]
                conic = conics[c_idx, g]
                opacity = opacities[c_idx, g]
                mu = mus[c_idx, g]
                
                dx = px - mean[0]
                dy = py - mean[1]
                
                sigma = 0.5 * (conic[0] * dx * dx + conic[2] * dy * dy) + conic[1] * dx * dy
                alpha = opacity * mu * torch.exp(-sigma)
                mask = (sigma >= 0) & (alpha.abs() >= 1e-5)
                alpha = torch.where(mask, alpha, torch.zeros_like(alpha))
                
                render_output[c_idx] += alpha
        
        return render_output
    
    @staticmethod
    def _compute_l2_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute L2 (MSE) loss — matches the CUDA image_loss_fwd kernel."""
        diff = prediction - target
        return (diff * diff).mean()
    
    def _extract_params(self, params, data, config):
        """Extract parameters, data, and config values."""
        if isinstance(params, GS2DRectifyingParams):
            cov2d_3x3 = params.cov2d_3x3
            means2d = params.means2d
            opacities = params.opacities
        else:
            cov2d_3x3 = params.get('cov2d_3x3').data
            means2d = params.get('means2d').data
            opacities = params.get('opacities').data
        
        if isinstance(data, GS2DRectifyingData):
            target = data.target
        else:
            target = data.get('target')
        
        eps = config.get('eps', self.eps) if config else self.eps
        lambda_l2 = get_lambda_l2(config, self.lambda_l2)
        lambda_ssim = config.get('lambda_ssim', self.lambda_ssim) if config else self.lambda_ssim
        
        return cov2d_3x3, means2d, opacities, target, eps, lambda_l2, lambda_ssim
    
    def forward_only(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> ForwardResult:
        """Forward computation only."""
        cov2d_3x3, means2d, opacities, target, eps, lambda_l2, lambda_ssim = \
            self._extract_params(params, data, config)
        
        C, dim0, dim1 = target.shape
        
        # ConicsMu forward
        conics, mus = self._compute_conics_mu(cov2d_3x3, eps)
        
        # Rasterize
        render_output = self._rasterize_torch(means2d, conics, opacities, mus, dim0, dim1)
        
        # Loss
        l2_loss = self._compute_l2_loss(render_output, target)
        loss = lambda_l2 * l2_loss
        
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
        cov2d_3x3_orig, means2d_orig, opacities_orig, target, eps, lambda_l2, lambda_ssim = \
            self._extract_params(params, data, config)
        
        # Create leaf tensors with gradients
        cov2d_3x3 = cov2d_3x3_orig.detach().clone().requires_grad_(True)
        means2d = means2d_orig.detach().clone().requires_grad_(True)
        opacities = opacities_orig.detach().clone().requires_grad_(True)
        
        C, dim0, dim1 = target.shape
        
        # Forward
        conics, mus = self._compute_conics_mu(cov2d_3x3, eps)
        render_output = self._rasterize_torch(means2d, conics, opacities, mus, dim0, dim1)
        l2_loss = self._compute_l2_loss(render_output, target)
        loss = lambda_l2 * l2_loss
        
        if lambda_ssim > 0:
            from .loss import ssim
            ssim_val = ssim(render_output, target, window_size=self.window_size)
            loss = loss + lambda_ssim * (1.0 - ssim_val)
        
        # Backward
        loss.backward()
        
        forward_result = ForwardResult(
            outputs={
                'loss': loss.detach(),
                'render_output': render_output.detach(),
            },
            intermediates={}
        )
        
        grad_result = GradResult()
        grad_result.set('cov2d_3x3', cov2d_3x3.grad.detach())
        grad_result.set('means2d', means2d.grad.detach())
        grad_result.set('opacities', opacities.grad.detach())
        
        return forward_result, grad_result
    
    def forward_backward2(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> Tuple[ForwardResult, GradResult, HessResult]:
        """Forward + first-order + second-order backward using torch autograd."""
        cov2d_3x3_orig, means2d_orig, opacities_orig, target, eps, lambda_l2, lambda_ssim = \
            self._extract_params(params, data, config)
        
        # Create leaf tensors with gradients
        cov2d_3x3 = cov2d_3x3_orig.detach().clone().requires_grad_(True)
        means2d = means2d_orig.detach().clone().requires_grad_(True)
        opacities = opacities_orig.detach().clone().requires_grad_(True)
        
        C, dim0, dim1 = target.shape
        N = means2d.shape[1]
        device = means2d.device
        dtype = means2d.dtype
        
        # Forward
        conics, mus = self._compute_conics_mu(cov2d_3x3, eps)
        render_output = self._rasterize_torch(means2d, conics, opacities, mus, dim0, dim1)
        l2_loss = self._compute_l2_loss(render_output, target)
        loss = lambda_l2 * l2_loss
        
        if lambda_ssim > 0:
            from .loss import ssim
            ssim_val = ssim(render_output, target, window_size=self.window_size)
            loss = loss + lambda_ssim * (1.0 - ssim_val)
        
        # First-order gradients
        v_cov2d_3x3, v_means2d, v_opacities = torch.autograd.grad(
            loss, [cov2d_3x3, means2d, opacities],
            create_graph=True
        )
        
        # Second-order Hessians
        # Diagonal Hessians:
        #   h_cov_cov: [C, N, 21] upper-tri of 6x6
        #   h_means2d: [C, N, 3] upper-tri of 2x2
        #   h_opacities: [C, N] scalar
        # Cross-term Hessians:
        #   h_cov_means2d: [C, N, 12] 6x2 row-major
        #   h_cov_opacity: [C, N, 6]
        #   h_mean_opacity: [C, N, 2]
        
        h_cov_cov = torch.zeros(C, N, 21, device=device, dtype=dtype)
        h_means2d_diag = torch.zeros(C, N, 3, device=device, dtype=dtype)
        h_opacities = torch.zeros(C, N, device=device, dtype=dtype)
        h_cov_means2d = torch.zeros(C, N, 12, device=device, dtype=dtype)
        h_cov_opacity = torch.zeros(C, N, 6, device=device, dtype=dtype)
        h_mean_opacity = torch.zeros(C, N, 2, device=device, dtype=dtype)
        
        for c_idx in range(C):
            for g in range(N):
                # h_cov_cov: 6x6 upper-tri = 21 elements
                for i in range(6):
                    grad_i = torch.autograd.grad(
                        v_cov2d_3x3[c_idx, g, i], cov2d_3x3,
                        retain_graph=True, allow_unused=True
                    )[0]
                    if grad_i is not None:
                        for j in range(i, 6):
                            flat_idx = i * (11 - i) // 2 + j
                            h_cov_cov[c_idx, g, flat_idx] = grad_i[c_idx, g, j]
                
                # h_means2d: 2x2 upper-tri = 3 elements [H00, H01, H11]
                for i in range(2):
                    grad_i = torch.autograd.grad(
                        v_means2d[c_idx, g, i], means2d,
                        retain_graph=True, allow_unused=True
                    )[0]
                    if grad_i is not None:
                        if i == 0:
                            h_means2d_diag[c_idx, g, 0] = grad_i[c_idx, g, 0]  # H00
                            h_means2d_diag[c_idx, g, 1] = grad_i[c_idx, g, 1]  # H01
                        else:
                            h_means2d_diag[c_idx, g, 2] = grad_i[c_idx, g, 1]  # H11
                
                # h_opacities: scalar
                try:
                    grad_o = torch.autograd.grad(
                        v_opacities[c_idx, g], opacities,
                        retain_graph=True, allow_unused=True
                    )[0]
                    if grad_o is not None:
                        h_opacities[c_idx, g] = grad_o[c_idx, g]
                except RuntimeError:
                    pass
                
                # Cross-term h_cov_means2d: 6x2 row-major
                for i in range(6):
                    try:
                        grad_cm = torch.autograd.grad(
                            v_cov2d_3x3[c_idx, g, i], means2d,
                            retain_graph=True, allow_unused=True
                        )[0]
                        if grad_cm is not None:
                            for j in range(2):
                                h_cov_means2d[c_idx, g, i * 2 + j] = grad_cm[c_idx, g, j]
                    except RuntimeError:
                        pass
                
                # Cross-term h_cov_opacity: 6 elements
                for i in range(6):
                    try:
                        grad_co = torch.autograd.grad(
                            v_cov2d_3x3[c_idx, g, i], opacities,
                            retain_graph=True, allow_unused=True
                        )[0]
                        if grad_co is not None:
                            h_cov_opacity[c_idx, g, i] = grad_co[c_idx, g]
                    except RuntimeError:
                        pass
                
                # Cross-term h_mean_opacity: 2 elements
                for i in range(2):
                    try:
                        grad_mo = torch.autograd.grad(
                            v_means2d[c_idx, g, i], opacities,
                            retain_graph=True, allow_unused=True
                        )[0]
                        if grad_mo is not None:
                            h_mean_opacity[c_idx, g, i] = grad_mo[c_idx, g]
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
        grad_result.set('cov2d_3x3', v_cov2d_3x3.detach())
        grad_result.set('means2d', v_means2d.detach())
        grad_result.set('opacities', v_opacities.detach())
        
        # For optimizer: extract diagonal Hessians
        # h_means2d: (C, N, 3) [H00, H01, H11] -> (C, N, 2) [H00, H11]
        h_means2d_for_optim = torch.stack(
            [h_means2d_diag[..., 0], h_means2d_diag[..., 2]], dim=-1
        )
        
        hess_result = HessResult()
        hess_result.set('cov2d_3x3', h_cov_cov)
        hess_result.set('means2d', h_means2d_for_optim)
        hess_result.set('opacities', h_opacities)
        hess_result.set_cross(
            cov_means2d=h_cov_means2d,
            cov_opacity=h_cov_opacity,
            mean_opacity=h_mean_opacity,
        )
        
        return forward_result, grad_result, hess_result


# ============================================================================
# CUDA Implementation
# ============================================================================

class GS2DRectifyingCUDA(SecondForwardBackwardBase):
    """
    2D Gaussian Splatting with ConicsMu + ImageLoss using CUDA kernels.
    
    Pipeline: cov2d_3x3 -> ConicsMu -> Rasterize -> Loss
    
    This implementation chains three CUDA modules:
    1. ConicsMu: cov2d_3x3 -> (conics, mus)
    2. RasterizeToPixels: (means2d, conics, opacities, mus) -> render_output
    3. ImageLoss: (render_output, target) -> loss
    
    For backward2, the chain runs in reverse:
    - ImageLoss bwd/bwd2 -> v_render + h_render
    - R2P bwd2 -> 14 outputs (gradients + Hessians for means2d, conics, opacities, mus)
    - ConicsMu bwd2 -> gradients + Hessians for cov2d_3x3
    
    The h_mean_conic output from R2P bwd2 [C,N,2,3] is reshaped to [C,N,6]
    and passed directly as h_means2d_conics to ConicsMu bwd2 (same 2x3 row-major
    layout), avoiding any transpose.
    """
    
    def __init__(
        self,
        eps: float = 0.3,
        lambda_l2: float = 1.0,
        lambda_ssim: float = 0.0,
        window_size: int = 11,
        tile_size: int = 16,
    ):
        self.eps = eps
        self.lambda_l2 = lambda_l2
        self.lambda_ssim = lambda_ssim
        self.window_size = window_size
        self.tile_size = tile_size
        self.initialized = False
        
        # Cached intermediate values
        self._tile_offsets = None
        self._flatten_ids = None
        self._render_output = None
        self._conics = None
        self._mus = None
        
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
        self.initialized = True
    
    def _extract_params(self, params, data, config):
        """Extract parameters, data, and config values."""
        if isinstance(params, GS2DRectifyingParams):
            cov2d_3x3 = params.cov2d_3x3
            means2d = params.means2d
            opacities = params.opacities
        else:
            cov2d_3x3 = params.get('cov2d_3x3').data
            means2d = params.get('means2d').data
            opacities = params.get('opacities').data
        
        if isinstance(data, GS2DRectifyingData):
            target = data.target
            radii = data.radii
        else:
            target = data.get('target')
            radii = data.get('radii')
        
        eps = config.get('eps', self.eps) if config else self.eps
        lambda_l2 = get_lambda_l2(config, self.lambda_l2)
        lambda_ssim = config.get('lambda_ssim', self.lambda_ssim) if config else self.lambda_ssim
        window_size = config.get('window_size', self.window_size) if config else self.window_size
        
        return cov2d_3x3, means2d, opacities, target, radii, eps, lambda_l2, lambda_ssim, window_size
    
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
        from ..cuda._wrapper import conics_mu_fwd, rasterize_to_pixels_fwd, image_loss_fwd
        
        cov2d_3x3, means2d, opacities, target, radii, eps, lambda_l2, lambda_ssim, window_size = \
            self._extract_params(params, data, config)
        
        C, dim0, dim1 = target.shape
        
        # Step 1: ConicsMu forward
        conics, mus = conics_mu_fwd(cov2d_3x3, eps)
        self._conics = conics
        self._mus = mus
        
        # Step 2: Tile intersections
        tile_offsets, flatten_ids = self._compute_tile_intersections(means2d, radii, dim0, dim1)
        self._tile_offsets = tile_offsets
        self._flatten_ids = flatten_ids
        
        # Step 3: Rasterize
        render_output = rasterize_to_pixels_fwd(
            means2d, conics, opacities, mus,
            tile_offsets, flatten_ids,
            dim1, dim0, self.tile_size
        )
        self._render_output = render_output
        
        # Step 4: Image loss
        result = image_loss_fwd(
            render_output, target, window_size,
            lambda_l2, lambda_ssim,
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
            conics_mu_fwd, conics_mu_bwd,
            rasterize_to_pixels_fwd, rasterize_to_pixels_bwd,
            image_loss_fwd, image_loss_bwd
        )
        
        cov2d_3x3, means2d, opacities, target, radii, eps, lambda_l2, lambda_ssim, window_size = \
            self._extract_params(params, data, config)
        
        C, dim0, dim1 = target.shape
        
        # Forward: ConicsMu
        conics, mus = conics_mu_fwd(cov2d_3x3, eps)
        
        # Forward: Tile intersections
        tile_offsets, flatten_ids = self._compute_tile_intersections(means2d, radii, dim0, dim1)
        
        # Forward: Rasterize
        render_output = rasterize_to_pixels_fwd(
            means2d, conics, opacities, mus,
            tile_offsets, flatten_ids,
            dim1, dim0, self.tile_size
        )
        
        # Forward: Image loss with statistics
        result = image_loss_fwd(
            render_output, target, window_size,
            lambda_l2, lambda_ssim,
            compute_statistics=True
        )
        loss = result[0]
        mu_r, mu_m = result[3], result[4]
        f1, f2, f3, f4 = result[5], result[6], result[7], result[8]
        
        # Backward: Image loss
        grad_output = torch.ones(1, device=render_output.device, dtype=render_output.dtype)
        v_render_output = image_loss_bwd(
            render_output, target,
            mu_r, mu_m, f1, f2, f3, f4,
            grad_output, window_size,
            lambda_l2, lambda_ssim
        )
        
        # Backward: Rasterize
        _v_means2d_abs, v_means2d, v_conics, v_opacities, v_mus = rasterize_to_pixels_bwd(
            means2d, conics, opacities, mus,
            tile_offsets, flatten_ids, v_render_output,
            dim1, dim0, self.tile_size, False
        )
        
        # Backward: ConicsMu
        v_cov2d_3x3 = conics_mu_bwd(
            cov2d_3x3, eps, conics, mus, v_conics, v_mus
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
        grad_result.set('cov2d_3x3', v_cov2d_3x3)
        grad_result.set('means2d', v_means2d)
        grad_result.set('opacities', v_opacities)
        
        return forward_result, grad_result
    
    def forward_backward2(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> Tuple[ForwardResult, GradResult, HessResult]:
        """Forward + first-order + second-order backward.
        
        Chain:
        1. ImageLoss bwd + bwd2 -> v_render_output + h_render_output
        2. R2P bwd2 -> 14 outputs (v_means2d, v_conics, v_opacities, v_mus,
                                    h_means2d, h_conics, h_mu_mean, h_mu_conic,
                                    h_mean_conic, h_mu_opacity, h_conic_opacity,
                                    h_mean_opacity, h_mus, h_opacities)
        3. ConicsMu bwd2 -> v_cov2d_3x3, h_cov_cov, h_cov_means2d, h_cov_opacity
        
        Key: h_mean_conic from R2P [C,N,2,3] is reshaped to [C,N,6] and passed
        directly as h_means2d_conics to ConicsMu bwd2 (same 2x3 row-major layout).
        """
        from ..cuda._wrapper import (
            conics_mu_fwd, conics_mu_bwd2,
            rasterize_to_pixels_fwd, rasterize_to_pixels_bwd2,
            image_loss_fwd, image_loss_bwd, image_loss_bwd2
        )
        
        cov2d_3x3, means2d, opacities, target, radii, eps, lambda_l2, lambda_ssim, window_size = \
            self._extract_params(params, data, config)
        
        C, dim0, dim1 = target.shape
        
        # Forward: ConicsMu
        conics, mus = conics_mu_fwd(cov2d_3x3, eps)
        
        # Forward: Tile intersections
        tile_offsets, flatten_ids = self._compute_tile_intersections(means2d, radii, dim0, dim1)
        
        # Forward: Rasterize
        render_output = rasterize_to_pixels_fwd(
            means2d, conics, opacities, mus,
            tile_offsets, flatten_ids,
            dim1, dim0, self.tile_size
        )
        
        # Forward: Image loss with statistics
        result = image_loss_fwd(
            render_output, target, window_size,
            lambda_l2, lambda_ssim,
            compute_statistics=True
        )
        loss = result[0]
        mu_r, mu_m = result[3], result[4]
        f1, f2, f3, f4 = result[5], result[6], result[7], result[8]
        
        # Backward1: Image loss
        grad_output = torch.ones(1, device=render_output.device, dtype=render_output.dtype)
        v_render_output = image_loss_bwd(
            render_output, target,
            mu_r, mu_m, f1, f2, f3, f4,
            grad_output, window_size,
            lambda_l2, lambda_ssim
        )
        
        # Backward2: Image loss -> h_render_output (diagonal Hessian)
        h_render_output = image_loss_bwd2(
            render_output, target,
            mu_r, mu_m, f1, f2, f3, f4,
            grad_output, window_size,
            lambda_l2, lambda_ssim
        )
        
        # Backward2: RasterizeToPixels -> 14 outputs
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
        
        # ====================================================================
        # Backward2: ConicsMu
        # ====================================================================
        # R2P bwd2 outputs -> ConicsMu bwd2 inputs mapping:
        #   v_conics [C,N,3]          -> v_conics [C,N,3]           (direct)
        #   v_mus [C,N]               -> v_mus [C,N]                (direct)
        #   h_conics [C,N,6]          -> h_conics [C,N,6]           (direct)
        #   h_mu_conic [C,N,3]        -> h_conics_mu [C,N,3]        (symmetric)
        #   h_mus [C,N]               -> h_mu_mu [C,N]              (direct)
        #   h_mean_conic [C,N,2,3]    -> h_means2d_conics [C,N,6]   (reshape, same layout!)
        #   h_conic_opacity [C,N,3]   -> h_conics_opacity [C,N,3]   (direct)
        #   h_mu_mean [C,N,2]         -> h_mu_means2d [C,N,2]       (direct)
        #   h_mu_opacity [C,N]        -> h_mu_opacity [C,N]         (direct)
        
        # h_mean_conic is [C, N, 2, 3] in row-major = [C, N, 6] 2x3 row-major
        # This exactly matches h_means2d_conics expected by ConicsMu bwd2!
        h_means2d_conics = h_mean_conic.reshape(C, -1, 6)  # [C, N, 6]
        
        v_cov2d_3x3, h_cov_cov, h_cov_means2d, h_cov_opacity = conics_mu_bwd2(
            cov2d_3x3, eps, conics, mus,
            v_conics, v_mus,
            h_conics, h_mu_conic, h_mus,  # h_conics_mu=h_mu_conic, h_mu_mu=h_mus
            h_means2d_conics, h_conic_opacity, h_mu_mean, h_mu_opacity,
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
        grad_result.set('cov2d_3x3', v_cov2d_3x3)
        grad_result.set('means2d', v_means2d)
        grad_result.set('opacities', v_opacities)
        
        # For optimizer: extract diagonal Hessians
        # h_means2d: (C, N, 3) [H00, H01, H11] -> (C, N, 2) [H00, H11]
        h_means2d_for_optim = torch.stack(
            [h_means2d[..., 0], h_means2d[..., 2]], dim=-1
        )
        
        hess_result = HessResult()
        hess_result.set('cov2d_3x3', h_cov_cov)
        hess_result.set('means2d', h_means2d_for_optim)
        hess_result.set('opacities', h_opacities)
        hess_result.set_cross(
            cov_means2d=h_cov_means2d,
            cov_opacity=h_cov_opacity,
            mean_opacity=h_mean_opacity,
        )
        
        return forward_result, grad_result, hess_result


# ============================================================================
# Factory Functions
# ============================================================================

def create_gs2d_rectifying_autograd(
    eps: float = 0.3,
    lambda_l2: float = 1.0,
    lambda_ssim: float = 0.0,
    window_size: int = 11,
    tile_size: int = 16,
) -> GS2DRectifyingAutograd:
    """Create 2D Gaussian Splatting with ConicsMu operator (autograd version)."""
    return GS2DRectifyingAutograd(
        eps=eps,
        lambda_l2=lambda_l2,
        lambda_ssim=lambda_ssim,
        window_size=window_size,
        tile_size=tile_size,
    )


def create_gs2d_rectifying_cuda(
    eps: float = 0.3,
    lambda_l2: float = 1.0,
    lambda_ssim: float = 0.0,
    window_size: int = 11,
    tile_size: int = 16,
) -> GS2DRectifyingCUDA:
    """Create 2D Gaussian Splatting with ConicsMu operator (CUDA version)."""
    return GS2DRectifyingCUDA(
        eps=eps,
        lambda_l2=lambda_l2,
        lambda_ssim=lambda_ssim,
        window_size=window_size,
        tile_size=tile_size,
    )


__all__ = [
    # Data structures
    'GS2DRectifyingParams',
    'GS2DRectifyingData',
    'GS2DRectifyingConfig',
    # Operators
    'GS2DRectifyingAutograd',
    'GS2DRectifyingCUDA',
    # Factory functions
    'create_gs2d_rectifying_autograd',
    'create_gs2d_rectifying_cuda',
]
