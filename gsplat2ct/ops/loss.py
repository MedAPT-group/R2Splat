"""
Loss Functions with Second-Order Derivatives

This module provides loss function implementations for image and volume reconstruction:
1. ImageLossAutograd: L1 + SSIM weighted loss for image reconstruction (autograd)
2. ImageLossTorch: L1 + SSIM weighted loss (manual torch implementation)
3. ImageLossCUDA: L1 + SSIM weighted loss (CUDA-accelerated)
4. VolumeLossAutograd: Total Variation (TV) loss for volume regularization

Simplified to support:
- Images: (B, H, W) - multi-batch 2D images without channel dimension
- Volumes: (B, D, H, W) - multi-batch 3D volumes without channel dimension

All implementations inherit from SecondForwardBackwardBase and support:
- Forward-only computation
- Forward + first-order gradients
- Forward + first-order + second-order (diagonal Hessian only)

Author: Dy164
Date: 2025-12-17
"""

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from dataclasses import dataclass
from typing import Optional

from .base import SecondForwardBackwardBase
from .data import (
    Params, Data, Config,
    ForwardResult, GradResult, HessResult,
)


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
# Loss Utility Functions
# ============================================================================

def l1_loss(network_output: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """L1 loss (mean absolute error)."""
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """L2 loss (mean squared error)."""
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    """Create 1D Gaussian kernel."""
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size: int) -> torch.Tensor:
    """Create 2D Gaussian window for SSIM (single channel only)."""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    return _2D_window


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, size_average: bool = True) -> torch.Tensor:
    """
    Structural Similarity Index (SSIM) for images.
    
    Args:
        img1, img2: Input images of shape (B, H, W)
        window_size: Size of the Gaussian window
        size_average: If True, return mean SSIM
    
    Returns:
        SSIM value(s)
    """
    assert img1.dim() == 3, f"SSIM expects 3D input (B, H, W), got shape {img1.shape}"
    
    # Add channel dimension for conv2d: (B, H, W) -> (B, 1, H, W)
    img1_4d = img1.unsqueeze(1)
    img2_4d = img2.unsqueeze(1)
    
    window = create_window(window_size).to(img1.device).type_as(img1)
    
    return _ssim(img1_4d, img2_4d, window, window_size, size_average)


def _ssim(img1: torch.Tensor, img2: torch.Tensor, window: torch.Tensor, 
          window_size: int, size_average: bool = True) -> torch.Tensor:
    """Internal SSIM computation for images with shape (B, 1, H, W)."""
    pad = window_size // 2
    
    mu1 = F.conv2d(img1, window, padding=pad)
    mu2 = F.conv2d(img2, window, padding=pad)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(dim=(1, 2, 3))  # Per-batch mean


def tv_3d_loss(vol: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Smooth Total Variation loss for 3D volumes (mean reduction).
    
    Uses smooth approximation: sqrt(x^2 + eps) instead of |x|
    This gives well-defined second-order derivatives everywhere.
    
    Args:
        vol: Volume tensor of shape (B, D, H, W)
        eps: Smoothing parameter (default: 1e-4)
    
    Returns:
        TV loss scalar
    """
    assert vol.dim() == 4, f"Expected 4D volume (B, D, H, W), got shape {vol.shape}"
    
    B, D, H, W = vol.shape
    
    dx = torch.diff(vol, dim=1)  # Along D
    dy = torch.diff(vol, dim=2)  # Along H
    dz = torch.diff(vol, dim=3)  # Along W

    tv = torch.sum(torch.sqrt(dx ** 2 + eps)) + torch.sum(torch.sqrt(dy ** 2 + eps)) + torch.sum(torch.sqrt(dz ** 2 + eps))

    total_elements = B * (
        (D - 1) * H * W
        + D * (H - 1) * W
        + D * H * (W - 1)
    )
    return tv / total_elements


# ============================================================================
# Image Loss Implementation
# ============================================================================

class ImageLossAutograd(SecondForwardBackwardBase):
    """
    Image reconstruction loss: weighted combination of L2 and SSIM.
    
    Loss = lambda_l2 * L2(pred, target) + lambda_ssim * (1 - SSIM(pred, target))
    
    Supports only: (B, H, W) - multi-batch 2D images without channel dimension
    
    This class computes:
    - Forward: loss value
    - Backward1: gradient w.r.t. prediction (same shape as prediction)
    - Backward2: diagonal Hessian w.r.t. prediction (same shape as prediction)
    
    Args:
        lambda_l2: Weight for L2 loss (default: 1.0)
        lambda_ssim: Weight for SSIM loss (default: 1.0)
        window_size: Window size for SSIM (default: 11)
    """
    
    def __init__(self, lambda_l2: float = 1.0, lambda_ssim: float = 1.0, window_size: int = 11):
        self.lambda_l2 = lambda_l2
        self.lambda_ssim = lambda_ssim
        self.window_size = window_size
        self.initialized = False
    
    def initialize(
        self,
        params: Params,
        config: Optional[Config] = None,
    ) -> None:
        """Initialize and validate parameters."""
        if params.get('prediction') is None:
            raise ValueError("Missing required parameter: 'prediction'")
        
        prediction = params.get('prediction').data
        
        # Must be 3D: (B, H, W)
        if prediction.dim() != 3:
            raise ValueError(
                f"Prediction must be 3D (B, H, W), got shape {prediction.shape}"
            )
        
        B, H, W = prediction.shape
        
        # Validate minimum size for SSIM
        if H < self.window_size or W < self.window_size:
            raise ValueError(
                f"Image size ({H}×{W}) too small for SSIM with window_size={self.window_size}. "
                f"Need at least {self.window_size}×{self.window_size}"
            )
        
        self.initialized = True
    
    def forward_only(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> ForwardResult:
        """Forward computation only (no gradients)."""
        prediction = params.get('prediction').data
        target = data.get('target')
        
        lambda_l2 = config.get('lambda_l2', self.lambda_l2) if config else self.lambda_l2
        lambda_ssim = config.get('lambda_ssim', self.lambda_ssim) if config else self.lambda_ssim
        window_size = config.get('window_size', self.window_size) if config else self.window_size
        
        if prediction.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: prediction {prediction.shape} vs target {target.shape}"
            )
        
        l2 = l2_loss(prediction, target)
        ssim_val = ssim(prediction, target, window_size=window_size, size_average=True)
        loss = lambda_l2 * l2 + lambda_ssim * (1 - ssim_val)
        
        return ForwardResult(
            outputs={'loss': loss, 'l2': l2, 'ssim': ssim_val},
            intermediates={}
        )
    
    def forward_backward1(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> tuple[ForwardResult, GradResult]:
        """Forward + first-order backward computation."""
        prediction = params.get('prediction').data.clone().detach().requires_grad_(True)
        target = data.get('target')
        
        lambda_l2 = config.get('lambda_l2', self.lambda_l2) if config else self.lambda_l2
        lambda_ssim = config.get('lambda_ssim', self.lambda_ssim) if config else self.lambda_ssim
        window_size = config.get('window_size', self.window_size) if config else self.window_size
        
        l2 = l2_loss(prediction, target)
        ssim_val = ssim(prediction, target, window_size=window_size, size_average=True)
        loss = lambda_l2 * l2 + lambda_ssim * (1 - ssim_val)
        
        loss.backward()
        grad_prediction = prediction.grad.clone()
        
        forward_result = ForwardResult(
            outputs={'loss': loss.detach(), 'l2': l2.detach(), 'ssim': ssim_val.detach()},
            intermediates={}
        )
        
        grad_result = GradResult()
        grad_result.set('prediction', grad_prediction)
        
        return forward_result, grad_result
    
    def forward_backward2(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> tuple[ForwardResult, GradResult, HessResult]:
        """Forward + first-order + second-order backward computation."""
        forward_result, grad_result = self.forward_backward1(params, data, config)
        
        prediction = params.get('prediction').data.clone().detach().requires_grad_(True)
        target = data.get('target')
        
        lambda_l2 = config.get('lambda_l2', self.lambda_l2) if config else self.lambda_l2
        lambda_ssim = config.get('lambda_ssim', self.lambda_ssim) if config else self.lambda_ssim
        window_size = config.get('window_size', self.window_size) if config else self.window_size
        
        l2 = l2_loss(prediction, target)
        ssim_val = ssim(prediction, target, window_size=window_size, size_average=True)
        loss = lambda_l2 * l2 + lambda_ssim * (1 - ssim_val)
        
        grad1 = torch.autograd.grad(loss, prediction, create_graph=True)[0]
        
        diag_hessian = torch.zeros_like(prediction)
        flat_grad = grad1.reshape(-1)
        
        for i in range(flat_grad.numel()):
            if flat_grad[i].requires_grad:
                grad_of_grad = torch.autograd.grad(
                    flat_grad[i],
                    prediction,
                    retain_graph=(i < flat_grad.numel() - 1)
                )[0]
                diag_hessian.reshape(-1)[i] = grad_of_grad.reshape(-1)[i]
        
        hess_result = HessResult()
        hess_result.set('prediction', diag_hessian.detach())
        
        return forward_result, grad_result, hess_result


# ============================================================================
# Volume Loss Implementation
# ============================================================================

class VolumeLossAutograd(SecondForwardBackwardBase):
    """
    Smooth Total Variation (TV) regularization loss for 3D volumes.
    
    Loss = SmoothTV(volume) = mean(Σ sqrt(|∇volume|² + eps))
    
    Supports only: (B, D, H, W) - multi-batch 3D volumes without channel dimension
    
    This class computes:
    - Forward: loss value (always averaged)
    - Backward1: gradient w.r.t. volume (same shape as volume)
    - Backward2: diagonal Hessian w.r.t. volume (same shape as volume)
    
    Args:
        eps: Smoothing parameter for smooth TV (default: 1e-4)
    """
    
    def __init__(self, eps: float = 1e-4):
        self.eps = eps
        self.initialized = False
    
    def initialize(
        self,
        params: Params,
        config: Optional[Config] = None,
    ) -> None:
        """Initialize and validate parameters."""
        if params.get('volume') is None:
            raise ValueError("Missing required parameter: 'volume'")
        
        volume = params.get('volume').data
        
        if volume.dim() != 4:
            raise ValueError(
                f"Volume must be 4D (B, D, H, W), got shape {volume.shape}"
            )
        
        self.initialized = True
    
    def forward_only(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> ForwardResult:
        """Forward computation only (no gradients)."""
        volume = params.get('volume').data
        eps = config.get('eps', self.eps) if config else self.eps
        loss = tv_3d_loss(volume, eps=eps)
        
        return ForwardResult(
            outputs={'loss': loss},
            intermediates={}
        )
    
    def forward_backward1(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> tuple[ForwardResult, GradResult]:
        """Forward + first-order backward computation."""
        volume = params.get('volume').data.clone().detach().requires_grad_(True)
        eps = config.get('eps', self.eps) if config else self.eps
        
        loss = tv_3d_loss(volume, eps=eps)
        loss.backward()
        grad_volume = volume.grad.clone()
        
        forward_result = ForwardResult(
            outputs={'loss': loss.detach()},
            intermediates={}
        )
        
        grad_result = GradResult()
        grad_result.set('volume', grad_volume)
        
        return forward_result, grad_result
    
    def forward_backward2(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> tuple[ForwardResult, GradResult, HessResult]:
        """Forward + first-order + second-order backward computation."""
        forward_result, grad_result = self.forward_backward1(params, data, config)
        
        # Compute Hessian diagonal via autograd
        volume = params.get('volume').data.clone().detach().requires_grad_(True)
        eps = config.get('eps', self.eps) if config else self.eps
        
        loss = tv_3d_loss(volume, eps=eps)
        grad1 = torch.autograd.grad(loss, volume, create_graph=True)[0]
        
        diag_hessian = torch.zeros_like(volume)
        flat_grad = grad1.reshape(-1)
        
        for i in range(flat_grad.numel()):
            if flat_grad[i].requires_grad:
                grad_of_grad = torch.autograd.grad(
                    flat_grad[i],
                    volume,
                    retain_graph=(i < flat_grad.numel() - 1)
                )[0]
                diag_hessian.reshape(-1)[i] = grad_of_grad.reshape(-1)[i]
        
        hess_result = HessResult()
        hess_result.set('volume', diag_hessian.detach())
        
        return forward_result, grad_result, hess_result


# ============================================================================
# Manual Torch Implementation - Image Loss
# ============================================================================

class ImageLossTorch(SecondForwardBackwardBase):
    """
    Manual torch implementation of image loss: L2 + D-SSIM.
    
    Supports only: (B, H, W) - multi-batch 2D images without channel dimension
    
    This class manually computes gradients and Hessians using the window-iteration approach.
    
    Loss = lambda_l2 * L2(pred, target) + lambda_ssim * (1 - SSIM(pred, target))
    
    Args:
        lambda_l2: Weight for L2 loss (default: 1.0)
        lambda_ssim: Weight for SSIM loss (default: 1.0)
        window_size: Window size for SSIM (default: 11)
    """
    
    def __init__(self, lambda_l2: float = 1.0, lambda_ssim: float = 1.0, window_size: int = 11):
        self.lambda_l2 = lambda_l2
        self.lambda_ssim = lambda_ssim
        self.window_size = window_size
        self.initialized = False
    
    def initialize(
        self,
        params: Params,
        config: Optional[Config] = None,
    ) -> None:
        """Initialize and validate parameters."""
        if params.get('prediction') is None:
            raise ValueError("Missing required parameter: 'prediction'")
        
        prediction = params.get('prediction').data
        
        if prediction.dim() != 3:
            raise ValueError(
                f"Prediction must be 3D (B, H, W), got shape {prediction.shape}"
            )
        
        B, H, W = prediction.shape
        
        window_size = config.get('window_size', self.window_size) if config else self.window_size
        if window_size <= 0:
            raise ValueError(
                f"Window size must be positive, got {window_size}"
            )
        if window_size <= 0:
            raise ValueError(
                f"Window size must be positive, got {window_size}"
            )
        if window_size > H or window_size > W:
            raise ValueError(
                f"Window size {window_size} exceeds image dimensions H={H}, W={W}"
            )
        
        self.initialized = True
    
    def forward_only(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> ForwardResult:
        """Forward computation only (no gradients)."""
        prediction = params.get('prediction').data
        target = data.get('target')
        
        lambda_l2 = config.get('lambda_l2', self.lambda_l2) if config else self.lambda_l2
        lambda_ssim = config.get('lambda_ssim', self.lambda_ssim) if config else self.lambda_ssim
        window_size = config.get('window_size', self.window_size) if config else self.window_size
        
        l2 = ((prediction - target) ** 2).mean()
        ssim_val = ssim(prediction, target, window_size=window_size, size_average=True)
        loss = lambda_l2 * l2 + lambda_ssim * (1 - ssim_val)
        
        return ForwardResult(
            outputs={'loss': loss, 'l2': l2, 'ssim': ssim_val},
            intermediates={}
        )
    
    def forward_backward1(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> tuple[ForwardResult, GradResult]:
        """Forward + first-order backward computation (manual)."""
        prediction = params.get('prediction').data
        target = data.get('target')
        
        lambda_l2 = config.get('lambda_l2', self.lambda_l2) if config else self.lambda_l2
        lambda_ssim = config.get('lambda_ssim', self.lambda_ssim) if config else self.lambda_ssim
        window_size = config.get('window_size', self.window_size) if config else self.window_size
        
        l2 = ((prediction - target) ** 2).mean()
        ssim_val = ssim(prediction, target, window_size=window_size, size_average=True)
        loss = lambda_l2 * l2 + lambda_ssim * (1 - ssim_val)
        
        N = prediction.numel()
        grad_l2 = 2.0 * (prediction - target) / N
        grad_ssim = self._compute_ssim_grad_by_window(prediction, target, window_size)
        grad_total = lambda_l2 * grad_l2 + lambda_ssim * grad_ssim
        
        forward_result = ForwardResult(
            outputs={'loss': loss, 'l2': l2, 'ssim': ssim_val},
            intermediates={}
        )
        
        grad_result = GradResult()
        grad_result.set('prediction', grad_total)
        
        return forward_result, grad_result
    
    def forward_backward2(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> tuple[ForwardResult, GradResult, HessResult]:
        """Forward + first-order + second-order backward computation (manual)."""
        forward_result, grad_result = self.forward_backward1(params, data, config)
        
        prediction = params.get('prediction').data
        target = data.get('target')
        
        lambda_l2 = config.get('lambda_l2', self.lambda_l2) if config else self.lambda_l2
        lambda_ssim = config.get('lambda_ssim', self.lambda_ssim) if config else self.lambda_ssim
        window_size = config.get('window_size', self.window_size) if config else self.window_size
        
        N = prediction.numel()
        hess_l2 = 2.0 / N * torch.ones_like(prediction)
        hess_ssim = self._compute_ssim_hess_by_window(prediction, target, window_size)
        hess_total = lambda_l2 * hess_l2 + lambda_ssim * hess_ssim
        
        hess_result = HessResult()
        hess_result.set('prediction', hess_total)
        
        return forward_result, grad_result, hess_result
    
    def _compute_ssim_grad_by_window(
        self, 
        prediction: torch.Tensor, 
        target: torch.Tensor, 
        window_size: int
    ) -> torch.Tensor:
        """Compute D-SSIM gradient by iterating over windows.
        
        Args:
            prediction: (B, H, W) tensor
            target: (B, H, W) tensor
            window_size: SSIM window size
        
        Returns:
            Gradient tensor of shape (B, H, W)
        """
        B, H, W = prediction.shape
        device = prediction.device
        
        window = create_window(window_size).to(device).type_as(prediction)
        gaussian_kernel = window[0, 0, :, :].clone()
        
        grad_accum = torch.zeros_like(prediction)
        
        pad_size = window_size // 2
        pred_padded = F.pad(prediction.unsqueeze(1), (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0).squeeze(1)
        target_padded = F.pad(target.unsqueeze(1), (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0).squeeze(1)
        
        out_H = H + 2 * pad_size - window_size + 1
        out_W = W + 2 * pad_size - window_size + 1
        
        for out_i in range(out_H):
            for out_j in range(out_W):
                for b in range(B):
                    pred_win = pred_padded[b, out_i:out_i+window_size, out_j:out_j+window_size]
                    tgt_win = target_padded[b, out_i:out_i+window_size, out_j:out_j+window_size]
                    
                    mu_r = (pred_win * gaussian_kernel).sum()
                    mu_m = (tgt_win * gaussian_kernel).sum()
                    sigma_r_sq = ((pred_win ** 2) * gaussian_kernel).sum() - mu_r ** 2
                    sigma_m_sq = ((tgt_win ** 2) * gaussian_kernel).sum() - mu_m ** 2
                    sigma_rm = ((pred_win * tgt_win) * gaussian_kernel).sum() - mu_r * mu_m
                    
                    C1, C2 = 0.01 ** 2, 0.03 ** 2
                    f1 = 2 * mu_r * mu_m + C1
                    f2 = 2 * sigma_rm + C2
                    f3 = mu_r ** 2 + mu_m ** 2 + C1
                    f4 = sigma_r_sq + sigma_m_sq + C2
                    
                    ssim_val = (f1 * f2) / (f3 * f4)
                    
                    g1 = 2 * gaussian_kernel * mu_m
                    g2 = 2 * gaussian_kernel * (tgt_win - mu_m)
                    g3 = 2 * gaussian_kernel * mu_r
                    g4 = 2 * gaussian_kernel * (pred_win - mu_r)
                    
                    log_deriv = g1/f1 + g2/f2 - g3/f3 - g4/f4
                    grad_win = -ssim_val * log_deriv
                    
                    for wi in range(window_size):
                        for wj in range(window_size):
                            img_i = out_i + wi - pad_size
                            img_j = out_j + wj - pad_size
                            if 0 <= img_i < H and 0 <= img_j < W:
                                grad_accum[b, img_i, img_j] += grad_win[wi, wj]
        
        return grad_accum / (B * out_H * out_W)
    
    def _compute_ssim_hess_by_window(
        self, 
        prediction: torch.Tensor, 
        target: torch.Tensor, 
        window_size: int
    ) -> torch.Tensor:
        """Compute D-SSIM diagonal Hessian by iterating over windows.
        
        Args:
            prediction: (B, H, W) tensor
            target: (B, H, W) tensor
            window_size: SSIM window size
        
        Returns:
            Diagonal Hessian tensor of shape (B, H, W)
        """
        B, H, W = prediction.shape
        device = prediction.device
        
        window = create_window(window_size).to(device).type_as(prediction)
        gaussian_kernel = window[0, 0, :, :].clone()
        
        hess_accum = torch.zeros_like(prediction)
        
        pad_size = window_size // 2
        pred_padded = F.pad(prediction.unsqueeze(1), (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0).squeeze(1)
        target_padded = F.pad(target.unsqueeze(1), (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0).squeeze(1)
        
        out_H = H + 2 * pad_size - window_size + 1
        out_W = W + 2 * pad_size - window_size + 1
        
        for out_i in range(out_H):
            for out_j in range(out_W):
                for b in range(B):
                    pred_win = pred_padded[b, out_i:out_i+window_size, out_j:out_j+window_size]
                    tgt_win = target_padded[b, out_i:out_i+window_size, out_j:out_j+window_size]
                    
                    mu_r = (pred_win * gaussian_kernel).sum()
                    mu_m = (tgt_win * gaussian_kernel).sum()
                    sigma_r_sq = ((pred_win ** 2) * gaussian_kernel).sum() - mu_r ** 2
                    sigma_m_sq = ((tgt_win ** 2) * gaussian_kernel).sum() - mu_m ** 2
                    sigma_rm = ((pred_win * tgt_win) * gaussian_kernel).sum() - mu_r * mu_m
                    
                    C1, C2 = 0.01 ** 2, 0.03 ** 2
                    f1 = 2 * mu_r * mu_m + C1
                    f2 = 2 * sigma_rm + C2
                    f3 = mu_r ** 2 + mu_m ** 2 + C1
                    f4 = sigma_r_sq + sigma_m_sq + C2
                    
                    ssim_val = (f1 * f2) / (f3 * f4)
                    
                    g1 = 2 * gaussian_kernel * mu_m
                    g2 = 2 * gaussian_kernel * (tgt_win - mu_m)
                    g3 = 2 * gaussian_kernel * mu_r
                    g4 = 2 * gaussian_kernel * (pred_win - mu_r)
                    
                    h1 = torch.zeros_like(gaussian_kernel)
                    h2 = torch.zeros_like(gaussian_kernel)
                    h3 = 2 * gaussian_kernel ** 2
                    h4 = 2 * gaussian_kernel * (1 - gaussian_kernel)
                    
                    log_deriv = g1/f1 + g2/f2 - g3/f3 - g4/f4
                    
                    term1 = log_deriv ** 2
                    term2 = (
                        (h1 * f1 - g1**2) / f1**2 +
                        (h2 * f2 - g2**2) / f2**2 -
                        (h3 * f3 - g3**2) / f3**2 -
                        (h4 * f4 - g4**2) / f4**2
                    )
                    
                    hess_win = -ssim_val * (term1 + term2)
                    
                    for wi in range(window_size):
                        for wj in range(window_size):
                            img_i = out_i + wi - pad_size
                            img_j = out_j + wj - pad_size
                            if 0 <= img_i < H and 0 <= img_j < W:
                                hess_accum[b, img_i, img_j] += hess_win[wi, wj]
        
        return hess_accum / (B * out_H * out_W)


# ============================================================================
# Manual Torch Implementation - Image Loss (Pixel Parallel)
# ============================================================================

class ImageLossTorchPixelParallel(SecondForwardBackwardBase):
    """
    Pixel-parallel manual torch implementation of image loss: L2 + D-SSIM.
    
    Supports only: (B, H, W) - multi-batch 2D images without channel dimension
    
    This approach simulates CUDA pixel-parallel execution:
    - Pre-computes window statistics using convolution
    - Iterates over pixels (each pixel = one CUDA thread)
    - No write conflicts
    
    Args:
        lambda_l2: Weight for L2 loss (default: 1.0)
        lambda_ssim: Weight for SSIM loss (default: 1.0)
        window_size: Window size for SSIM (default: 11)
    """
    
    def __init__(self, lambda_l2: float = 1.0, lambda_ssim: float = 1.0, window_size: int = 11):
        self.lambda_l2 = lambda_l2
        self.lambda_ssim = lambda_ssim
        self.window_size = window_size
        self.initialized = False
    
    def initialize(
        self,
        params: Params,
        config: Optional[Config] = None,
    ) -> None:
        """Initialize and validate parameters."""
        if params.get('prediction') is None:
            raise ValueError("Missing required parameter: 'prediction'")
        
        prediction = params.get('prediction').data
        
        if prediction.dim() != 3:
            raise ValueError(
                f"Prediction must be 3D (B, H, W), got shape {prediction.shape}"
            )
        
        B, H, W = prediction.shape
        
        window_size = config.get('window_size', self.window_size) if config else self.window_size
        if window_size <= 0:
            raise ValueError(
                f"Window size must be positive, got {window_size}"
            )
        if window_size <= 0:
            raise ValueError(
                f"Window size must be positive, got {window_size}"
            )
        if window_size > H or window_size > W:
            raise ValueError(
                f"Window size {window_size} exceeds image dimensions H={H}, W={W}"
            )
        
        self.initialized = True
    
    def forward_only(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> ForwardResult:
        """Forward computation only (no gradients)."""
        prediction = params.get('prediction').data
        target = data.get('target')
        
        lambda_l2 = config.get('lambda_l2', self.lambda_l2) if config else self.lambda_l2
        lambda_ssim = config.get('lambda_ssim', self.lambda_ssim) if config else self.lambda_ssim
        window_size = config.get('window_size', self.window_size) if config else self.window_size
        
        l2 = ((prediction - target) ** 2).mean()
        ssim_val = ssim(prediction, target, window_size=window_size, size_average=True)
        loss = lambda_l2 * l2 + lambda_ssim * (1 - ssim_val)
        
        return ForwardResult(
            outputs={'loss': loss, 'l2': l2, 'ssim': ssim_val},
            intermediates={}
        )
    
    def forward_backward1(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> tuple[ForwardResult, GradResult]:
        """Forward + first-order backward computation (pixel-parallel approach)."""
        prediction = params.get('prediction').data
        target = data.get('target')
        
        lambda_l2 = config.get('lambda_l2', self.lambda_l2) if config else self.lambda_l2
        lambda_ssim = config.get('lambda_ssim', self.lambda_ssim) if config else self.lambda_ssim
        window_size = config.get('window_size', self.window_size) if config else self.window_size
        
        l2 = ((prediction - target) ** 2).mean()
        ssim_val = ssim(prediction, target, window_size=window_size, size_average=True)
        loss = lambda_l2 * l2 + lambda_ssim * (1 - ssim_val)
        
        N = prediction.numel()
        grad_l2 = 2.0 * (prediction - target) / N
        grad_ssim = self._compute_ssim_grad_pixel_parallel(prediction, target, window_size)
        grad_total = lambda_l2 * grad_l2 + lambda_ssim * grad_ssim
        
        forward_result = ForwardResult(
            outputs={'loss': loss, 'l2': l2, 'ssim': ssim_val},
            intermediates={}
        )
        
        grad_result = GradResult()
        grad_result.set('prediction', grad_total)
        
        return forward_result, grad_result
    
    def forward_backward2(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> tuple[ForwardResult, GradResult, HessResult]:
        """Forward + first-order + second-order backward computation (pixel-parallel approach)."""
        forward_result, grad_result = self.forward_backward1(params, data, config)
        
        prediction = params.get('prediction').data
        target = data.get('target')
        
        lambda_l2 = config.get('lambda_l2', self.lambda_l2) if config else self.lambda_l2
        lambda_ssim = config.get('lambda_ssim', self.lambda_ssim) if config else self.lambda_ssim
        window_size = config.get('window_size', self.window_size) if config else self.window_size
        
        N = prediction.numel()
        hess_l2 = 2.0 / N * torch.ones_like(prediction)
        hess_ssim = self._compute_ssim_hess_pixel_parallel(prediction, target, window_size)
        hess_total = lambda_l2 * hess_l2 + lambda_ssim * hess_ssim
        
        hess_result = HessResult()
        hess_result.set('prediction', hess_total)
        
        return forward_result, grad_result, hess_result
    
    def _precompute_window_statistics(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        window_size: int
    ) -> dict:
        """Pre-compute all window statistics using convolution.
        
        Args:
            prediction: (B, H, W) tensor
            target: (B, H, W) tensor
            window_size: SSIM window size
        
        Returns:
            Dictionary with pre-computed statistics
        """
        B, H, W = prediction.shape
        device = prediction.device
        
        # Add channel dimension for conv2d
        pred_4d = prediction.unsqueeze(1)  # (B, 1, H, W)
        tgt_4d = target.unsqueeze(1)  # (B, 1, H, W)
        
        window = create_window(window_size).to(device).type_as(prediction)
        pad_size = window_size // 2
        
        mu_r = F.conv2d(pred_4d, window, padding=pad_size)
        mu_m = F.conv2d(tgt_4d, window, padding=pad_size)
        
        sigma_r_sq = F.conv2d(pred_4d ** 2, window, padding=pad_size) - mu_r ** 2
        sigma_m_sq = F.conv2d(tgt_4d ** 2, window, padding=pad_size) - mu_m ** 2
        sigma_rm = F.conv2d(pred_4d * tgt_4d, window, padding=pad_size) - mu_r * mu_m
        
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        f1 = 2 * mu_r * mu_m + C1
        f2 = 2 * sigma_rm + C2
        f3 = mu_r ** 2 + mu_m ** 2 + C1
        f4 = sigma_r_sq + sigma_m_sq + C2
        ssim_map = (f1 * f2) / (f3 * f4)
        
        # Squeeze channel dimension for output: (B, 1, H, W) -> (B, H, W)
        return {
            'mu_r': mu_r.squeeze(1), 'mu_m': mu_m.squeeze(1),
            'ssim_map': ssim_map.squeeze(1),
            'f1': f1.squeeze(1), 'f2': f2.squeeze(1), 
            'f3': f3.squeeze(1), 'f4': f4.squeeze(1),
        }
    
    def _compute_ssim_grad_pixel_parallel(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        window_size: int
    ) -> torch.Tensor:
        """Compute D-SSIM gradient using pixel-parallel approach.
        
        Args:
            prediction: (B, H, W) tensor
            target: (B, H, W) tensor
            window_size: SSIM window size
        
        Returns:
            Gradient tensor of shape (B, H, W)
        """
        B, H, W = prediction.shape
        device = prediction.device
        
        stats = self._precompute_window_statistics(prediction, target, window_size)
        
        window = create_window(window_size).to(device).type_as(prediction)
        gaussian_kernel = window[0, 0, :, :].clone()
        
        pad_size = window_size // 2
        pred_padded = F.pad(prediction.unsqueeze(1), (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0).squeeze(1)
        target_padded = F.pad(target.unsqueeze(1), (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0).squeeze(1)
        
        out_H = H + 2 * pad_size - window_size + 1
        out_W = W + 2 * pad_size - window_size + 1
        
        grad = torch.zeros_like(prediction)
        
        for b in range(B):
            for img_i in range(H):
                for img_j in range(W):
                    pixel_grad = 0.0
                    
                    for out_i in range(out_H):
                        for out_j in range(out_W):
                            win_top = out_i - pad_size
                            win_left = out_j - pad_size
                            wi = img_i - win_top
                            wj = img_j - win_left
                            
                            if 0 <= wi < window_size and 0 <= wj < window_size:
                                ssim_val = stats['ssim_map'][b, out_i, out_j]
                                f1 = stats['f1'][b, out_i, out_j]
                                f2 = stats['f2'][b, out_i, out_j]
                                f3 = stats['f3'][b, out_i, out_j]
                                f4 = stats['f4'][b, out_i, out_j]
                                mu_r = stats['mu_r'][b, out_i, out_j]
                                mu_m = stats['mu_m'][b, out_i, out_j]
                                
                                w = gaussian_kernel[wi, wj]
                                pred_val = pred_padded[b, out_i + wi, out_j + wj]
                                tgt_val = target_padded[b, out_i + wi, out_j + wj]
                                
                                g1 = 2 * w * mu_m
                                g2 = 2 * w * (tgt_val - mu_m)
                                g3 = 2 * w * mu_r
                                g4 = 2 * w * (pred_val - mu_r)
                                
                                log_deriv = g1/f1 + g2/f2 - g3/f3 - g4/f4
                                pixel_grad += -ssim_val * log_deriv
                    
                    grad[b, img_i, img_j] = pixel_grad
        
        return grad / (B * out_H * out_W)
    
    def _compute_ssim_hess_pixel_parallel(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        window_size: int
    ) -> torch.Tensor:
        """Compute D-SSIM diagonal Hessian using pixel-parallel approach.
        
        Args:
            prediction: (B, H, W) tensor
            target: (B, H, W) tensor
            window_size: SSIM window size
        
        Returns:
            Diagonal Hessian tensor of shape (B, H, W)
        """
        B, H, W = prediction.shape
        device = prediction.device
        
        stats = self._precompute_window_statistics(prediction, target, window_size)
        
        window = create_window(window_size).to(device).type_as(prediction)
        gaussian_kernel = window[0, 0, :, :].clone()
        
        pad_size = window_size // 2
        pred_padded = F.pad(prediction.unsqueeze(1), (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0).squeeze(1)
        target_padded = F.pad(target.unsqueeze(1), (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0).squeeze(1)
        
        out_H = H + 2 * pad_size - window_size + 1
        out_W = W + 2 * pad_size - window_size + 1
        
        hess = torch.zeros_like(prediction)
        
        for b in range(B):
            for img_i in range(H):
                for img_j in range(W):
                    pixel_hess = 0.0
                    
                    for out_i in range(out_H):
                        for out_j in range(out_W):
                            win_top = out_i - pad_size
                            win_left = out_j - pad_size
                            wi = img_i - win_top
                            wj = img_j - win_left
                            
                            if 0 <= wi < window_size and 0 <= wj < window_size:
                                ssim_val = stats['ssim_map'][b, out_i, out_j]
                                f1 = stats['f1'][b, out_i, out_j]
                                f2 = stats['f2'][b, out_i, out_j]
                                f3 = stats['f3'][b, out_i, out_j]
                                f4 = stats['f4'][b, out_i, out_j]
                                mu_r = stats['mu_r'][b, out_i, out_j]
                                mu_m = stats['mu_m'][b, out_i, out_j]
                                
                                w = gaussian_kernel[wi, wj]
                                pred_val = pred_padded[b, out_i + wi, out_j + wj]
                                tgt_val = target_padded[b, out_i + wi, out_j + wj]
                                
                                g1 = 2 * w * mu_m
                                g2 = 2 * w * (tgt_val - mu_m)
                                g3 = 2 * w * mu_r
                                g4 = 2 * w * (pred_val - mu_r)
                                
                                h1, h2 = 0.0, 0.0
                                h3 = 2 * w ** 2
                                h4 = 2 * w * (1 - w)
                                
                                log_deriv = g1/f1 + g2/f2 - g3/f3 - g4/f4
                                
                                term1 = log_deriv ** 2
                                term2 = (
                                    (h1 * f1 - g1**2) / f1**2 +
                                    (h2 * f2 - g2**2) / f2**2 -
                                    (h3 * f3 - g3**2) / f3**2 -
                                    (h4 * f4 - g4**2) / f4**2
                                )
                                
                                pixel_hess += -ssim_val * (term1 + term2)
                    
                    hess[b, img_i, img_j] = pixel_hess
        
        return hess / (B * out_H * out_W)


# ============================================================================
# Manual Torch Implementation - Volume Loss
# ============================================================================

class VolumeLossTorch(SecondForwardBackwardBase):
    """
    Manual torch implementation of smooth TV loss for 3D volumes.
    
    Supports only: (B, D, H, W) - multi-batch 3D volumes without channel dimension
    
    Loss = SmoothTV(volume) = mean(Σ sqrt(|∇volume|² + eps))
    
    Args:
        eps: Smoothing parameter for smooth TV (default: 1e-4)
    """
    
    def __init__(self, eps: float = 1e-4):
        self.eps = eps
        self.initialized = False
    
    def initialize(
        self,
        params: Params,
        config: Optional[Config] = None,
    ) -> None:
        """Initialize and validate parameters."""
        if params.get('volume') is None:
            raise ValueError("Missing required parameter: 'volume'")
        
        volume = params.get('volume').data
        
        if volume.dim() != 4:
            raise ValueError(
                f"Volume must be 4D (B, D, H, W), got shape {volume.shape}"
            )
        
        self.initialized = True
    
    def forward_only(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> ForwardResult:
        """Forward computation only (no gradients)."""
        volume = params.get('volume').data
        eps = config.get('eps', self.eps) if config else self.eps
        loss = tv_3d_loss(volume, eps=eps)
        
        return ForwardResult(
            outputs={'loss': loss},
            intermediates={}
        )
    
    def forward_backward1(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> tuple[ForwardResult, GradResult]:
        """Forward + first-order backward computation (manual)."""
        volume = params.get('volume').data
        eps = config.get('eps', self.eps) if config else self.eps
        loss = tv_3d_loss(volume, eps=eps)
        grad = self._compute_smooth_tv_grad_manual(volume, eps)
        
        forward_result = ForwardResult(
            outputs={'loss': loss},
            intermediates={}
        )
        
        grad_result = GradResult()
        grad_result.set('volume', grad)
        
        return forward_result, grad_result
    
    def forward_backward2(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> tuple[ForwardResult, GradResult, HessResult]:
        """Forward + first-order + second-order backward computation."""
        forward_result, grad_result = self.forward_backward1(params, data, config)
        
        volume = params.get('volume').data
        eps = config.get('eps', self.eps) if config else self.eps
        hess = self._compute_smooth_tv_hess_manual(volume, eps)
        
        hess_result = HessResult()
        hess_result.set('volume', hess)
        
        return forward_result, grad_result, hess_result
    
    def _compute_smooth_tv_grad_manual(self, volume: torch.Tensor, eps: float) -> torch.Tensor:
        """Manually compute smooth TV gradient.
        
        For smooth TV: sqrt(x^2 + eps), gradient is x / sqrt(x^2 + eps)
        """
        B, D, H, W = volume.shape
        N = B * ((D - 1) * H * W + D * (H - 1) * W + D * H * (W - 1))
        
        grad = torch.zeros_like(volume)
        
        # D direction
        diff_d = volume[:, 1:, :, :] - volume[:, :-1, :, :]
        smooth_sign_d = diff_d / torch.sqrt(diff_d ** 2 + eps)
        grad[:, 1:, :, :] += smooth_sign_d
        grad[:, :-1, :, :] -= smooth_sign_d
        
        # H direction
        diff_h = volume[:, :, 1:, :] - volume[:, :, :-1, :]
        smooth_sign_h = diff_h / torch.sqrt(diff_h ** 2 + eps)
        grad[:, :, 1:, :] += smooth_sign_h
        grad[:, :, :-1, :] -= smooth_sign_h
        
        # W direction
        diff_w = volume[:, :, :, 1:] - volume[:, :, :, :-1]
        smooth_sign_w = diff_w / torch.sqrt(diff_w ** 2 + eps)
        grad[:, :, :, 1:] += smooth_sign_w
        grad[:, :, :, :-1] -= smooth_sign_w
        
        return grad / N
    
    def _compute_smooth_tv_hess_manual(self, volume: torch.Tensor, eps: float) -> torch.Tensor:
        """Manually compute smooth TV diagonal Hessian.
        
        For smooth TV: sqrt(x^2 + eps), second derivative is eps / (x^2 + eps)^(3/2)
        Each voxel participates in up to 6 differences (3 as start, 3 as end).
        """
        B, D, H, W = volume.shape
        N = B * ((D - 1) * H * W + D * (H - 1) * W + D * H * (W - 1))
        
        hess = torch.zeros_like(volume)
        
        def smooth_hess_fn(diff):
            denom = diff ** 2 + eps
            return eps / (denom * torch.sqrt(denom))
        
        # D direction
        diff_d = volume[:, 1:, :, :] - volume[:, :-1, :, :]
        sh_d = smooth_hess_fn(diff_d)
        hess[:, 1:, :, :] += sh_d   # contribution as "end"
        hess[:, :-1, :, :] += sh_d   # contribution as "start"
        
        # H direction
        diff_h = volume[:, :, 1:, :] - volume[:, :, :-1, :]
        sh_h = smooth_hess_fn(diff_h)
        hess[:, :, 1:, :] += sh_h
        hess[:, :, :-1, :] += sh_h
        
        # W direction
        diff_w = volume[:, :, :, 1:] - volume[:, :, :, :-1]
        sh_w = smooth_hess_fn(diff_w)
        hess[:, :, :, 1:] += sh_w
        hess[:, :, :, :-1] += sh_w
        
        return hess / N


# ============================================================================
# CUDA Implementation - Image Loss
# ============================================================================

class ImageLossCUDA(SecondForwardBackwardBase):
    """
    CUDA-accelerated image loss implementation: L2 + D-SSIM.
    
    This class wraps the compiled CUDA kernels for forward and backward passes.
    It provides high-performance computation on GPU with pixel-parallel execution.
    
    Supports only: (B, H, W) - multi-batch 2D images without channel dimension
    
    Loss = lambda_l2 * L2(pred, target) + lambda_ssim * (1 - SSIM(pred, target))
    
    Args:
        lambda_l2: Weight for L2 loss (default: 1.0)
        lambda_ssim: Weight for SSIM loss (default: 0.2)
        window_size: Window size for SSIM (default: 11)
    """
    
    def __init__(self, lambda_l2: float = 1.0, lambda_ssim: float = 0.2, window_size: int = 11):
        if not _is_cuda_available():
            raise RuntimeError(
                "CUDA extension not available. Use ImageLossAutograd or ImageLossTorch instead. "
                "To enable CUDA, install with: pip install -e ."
            )
        
        self.lambda_l2 = lambda_l2
        self.lambda_ssim = lambda_ssim
        self.window_size = window_size
        self.initialized = False
        
        # Cached intermediate statistics from forward pass (for backward reuse)
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
        if params.get('prediction') is None:
            raise ValueError("Missing required parameter: 'prediction'")
        
        prediction = params.get('prediction').data
        
        # Must be 3D: (B, H, W)
        if prediction.dim() != 3:
            raise ValueError(
                f"Prediction must be 3D (B, H, W), got shape {prediction.shape}"
            )
        
        # Must be on CUDA
        if not prediction.is_cuda:
            raise ValueError("Prediction must be a CUDA tensor for ImageLossCUDA")
        
        B, H, W = prediction.shape
        
        # Validate minimum size for SSIM
        if H < self.window_size or W < self.window_size:
            raise ValueError(
                f"Image size ({H}×{W}) too small for SSIM with window_size={self.window_size}. "
                f"Need at least {self.window_size}×{self.window_size}"
            )
        
        self.initialized = True
    
    def forward_only(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> ForwardResult:
        """Forward computation only (no gradients)."""
        prediction = params.get('prediction').data
        target = data.get('target')
        
        lambda_l2 = config.get('lambda_l2', self.lambda_l2) if config else self.lambda_l2
        lambda_ssim = config.get('lambda_ssim', self.lambda_ssim) if config else self.lambda_ssim
        window_size = config.get('window_size', self.window_size) if config else self.window_size
        
        # Validate inputs
        if prediction.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: prediction {prediction.shape} vs target {target.shape}"
            )
        
        if not prediction.is_cuda:
            raise ValueError("Prediction must be a CUDA tensor")
        if not target.is_cuda:
            raise ValueError("Target must be a CUDA tensor")
        
        # Ensure contiguous
        prediction = prediction.contiguous()
        target = target.contiguous()
        
        # Lazy import to avoid triggering JIT at module load time
        from ..cuda import _wrapper
        
        # Call CUDA forward function
        # Returns: (loss, l2_loss, ssim_loss, mu_r, mu_m, f1, f2, f3, f4)
        result = _wrapper.image_loss_fwd(
            prediction, target, window_size,
            lambda_l2, lambda_ssim,
            compute_statistics=False  # Don't need statistics for forward-only
        )
        
        loss, l2_loss_val, ssim_loss_val = result[0], result[1], result[2]
        
        return ForwardResult(
            outputs={
                'loss': loss,
                'l2': l2_loss_val,
                'ssim': 1.0 - ssim_loss_val  # Convert D-SSIM back to SSIM
            },
            intermediates={}
        )
    
    def forward_backward1(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> tuple[ForwardResult, GradResult]:
        """Forward + first-order backward computation."""
        prediction = params.get('prediction').data
        target = data.get('target')
        
        lambda_l2 = config.get('lambda_l2', self.lambda_l2) if config else self.lambda_l2
        lambda_ssim = config.get('lambda_ssim', self.lambda_ssim) if config else self.lambda_ssim
        window_size = config.get('window_size', self.window_size) if config else self.window_size
        
        # Validate inputs
        if prediction.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: prediction {prediction.shape} vs target {target.shape}"
            )
        
        if not prediction.is_cuda:
            raise ValueError("Prediction must be a CUDA tensor")
        if not target.is_cuda:
            raise ValueError("Target must be a CUDA tensor")
        
        # Ensure contiguous
        prediction = prediction.contiguous()
        target = target.contiguous()
        
        # Lazy import to avoid triggering JIT at module load time
        from ..cuda import _wrapper
        
        # Call CUDA forward function with statistics
        # Returns: (loss, l2_loss, ssim_loss, mu_r, mu_m, f1, f2, f3, f4)
        result = _wrapper.image_loss_fwd(
            prediction, target, window_size,
            lambda_l2, lambda_ssim,
            compute_statistics=True
        )
        
        loss, l2_loss_val, ssim_loss_val = result[0], result[1], result[2]
        self._mu_r, self._mu_m = result[3], result[4]
        self._f1, self._f2, self._f3, self._f4 = result[5], result[6], result[7], result[8]
        
        # Call CUDA backward function
        grad_output = torch.ones(1, device=prediction.device, dtype=prediction.dtype)
        grad_pred = _wrapper.image_loss_bwd(
            prediction, target,
            self._mu_r, self._mu_m,
            self._f1, self._f2, self._f3, self._f4,
            grad_output, window_size,
            lambda_l2, lambda_ssim
        )
        
        forward_result = ForwardResult(
            outputs={
                'loss': loss,
                'l2': l2_loss_val,
                'ssim': 1.0 - ssim_loss_val
            },
            intermediates={
                'mu_r': self._mu_r,
                'mu_m': self._mu_m,
                'f1': self._f1,
                'f2': self._f2,
                'f3': self._f3,
                'f4': self._f4
            }
        )
        
        grad_result = GradResult()
        grad_result.set('prediction', grad_pred)
        
        return forward_result, grad_result
    
    def forward_backward2(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> tuple[ForwardResult, GradResult, HessResult]:
        """Forward + first-order + second-order backward computation."""
        # First compute forward and first-order backward
        forward_result, grad_result = self.forward_backward1(params, data, config)
        
        prediction = params.get('prediction').data
        target = data.get('target')
        
        lambda_l2 = config.get('lambda_l2', self.lambda_l2) if config else self.lambda_l2
        lambda_ssim = config.get('lambda_ssim', self.lambda_ssim) if config else self.lambda_ssim
        window_size = config.get('window_size', self.window_size) if config else self.window_size
        
        # Ensure contiguous
        prediction = prediction.contiguous()
        target = target.contiguous()
        
        # Lazy import to avoid triggering JIT at module load time
        from ..cuda import _wrapper
        
        # Call CUDA second-order backward function
        grad_output = torch.ones(1, device=prediction.device, dtype=prediction.dtype)
        hessian_diag = _wrapper.image_loss_bwd2(
            prediction, target,
            self._mu_r, self._mu_m,
            self._f1, self._f2, self._f3, self._f4,
            grad_output, window_size,
            lambda_l2, lambda_ssim
        )
        
        hess_result = HessResult()
        hess_result.set('prediction', hessian_diag)
        
        return forward_result, grad_result, hess_result


# ============================================================================
# L2 Loss Implementation (Autograd)
# ============================================================================

class L2LossAutograd(SecondForwardBackwardBase):
    """
    L2 (Mean Squared Error) loss with autograd-based derivatives.
    
    Loss = (1/N) * Σ (prediction - target)^2
    
    Supports 3D tensors for both:
    - Batched images: (B, H, W) - multi-batch 2D images
    - Unbatched volumes: (D, H, W) - 3D volumes
    
    This class computes:
    - Forward: loss value
    - Backward1: gradient w.r.t. prediction = 2 * (prediction - target) / N
    - Backward2: diagonal Hessian w.r.t. prediction = 2 / N (constant)
    """
    
    def __init__(self):
        self.initialized = False
    
    def initialize(
        self,
        params: Params,
        config: Optional[Config] = None,
    ) -> None:
        """Initialize and validate parameters."""
        if params.get('prediction') is None:
            raise ValueError("Missing required parameter: 'prediction'")
        
        prediction = params.get('prediction').data
        
        # Must be 3D: (B, H, W) for images or (D, H, W) for volumes
        if prediction.dim() != 3:
            raise ValueError(
                f"Prediction must be 3D (B, H, W) or (D, H, W), got shape {prediction.shape}"
            )
        
        self.initialized = True
    
    def forward_only(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> ForwardResult:
        """Forward computation only (no gradients)."""
        prediction = params.get('prediction').data
        target = data.get('target')
        
        if prediction.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: prediction {prediction.shape} vs target {target.shape}"
            )
        
        l2_map = (prediction - target) ** 2
        loss = l2_map.mean()
        
        return ForwardResult(
            outputs={'loss': loss, 'l2_map': l2_map},
            intermediates={}
        )
    
    def forward_backward1(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> tuple[ForwardResult, GradResult]:
        """Forward + first-order backward computation (using autograd)."""
        prediction = params.get('prediction').data.clone().detach().requires_grad_(True)
        target = data.get('target')
        
        if prediction.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: prediction {prediction.shape} vs target {target.shape}"
            )
        
        l2_map = (prediction - target) ** 2
        loss = l2_map.mean()
        
        # Use autograd to compute gradient
        loss.backward()
        grad_prediction = prediction.grad.clone()
        
        forward_result = ForwardResult(
            outputs={'loss': loss.detach(), 'l2_map': l2_map.detach()},
            intermediates={}
        )
        
        grad_result = GradResult()
        grad_result.set('prediction', grad_prediction)
        
        return forward_result, grad_result
    
    def forward_backward2(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> tuple[ForwardResult, GradResult, HessResult]:
        """Forward + first-order + second-order backward computation (using autograd)."""
        # First get forward and gradient
        forward_result, grad_result = self.forward_backward1(params, data, config)
        
        # Recompute for Hessian calculation
        prediction = params.get('prediction').data.clone().detach().requires_grad_(True)
        target = data.get('target')
        
        l2_map = (prediction - target) ** 2
        loss = l2_map.mean()
        
        # Compute first-order gradient with create_graph=True for Hessian
        grad1 = torch.autograd.grad(loss, prediction, create_graph=True)[0]
        
        # Compute diagonal Hessian by computing d(grad1[i])/d(prediction[i])
        diag_hessian = torch.zeros_like(prediction)
        flat_grad = grad1.reshape(-1)
        
        for i in range(flat_grad.numel()):
            if flat_grad[i].requires_grad:
                hess_i = torch.autograd.grad(
                    flat_grad[i], prediction,
                    retain_graph=(i < flat_grad.numel() - 1),
                    create_graph=False
                )[0]
                diag_hessian.reshape(-1)[i] = hess_i.reshape(-1)[i]
        
        hess_result = HessResult()
        hess_result.set('prediction', diag_hessian.detach())
        
        return forward_result, grad_result, hess_result


# ============================================================================
# L2 Loss Implementation (CUDA)
# ============================================================================

class L2LossCUDA(SecondForwardBackwardBase):
    """
    CUDA-accelerated L2 (Mean Squared Error) loss.
    
    This class wraps the compiled CUDA kernels for forward and backward passes.
    
    Supports 3D tensors for both:
    - Batched images: (B, H, W) - multi-batch 2D images
    - Unbatched volumes: (D, H, W) - 3D volumes
    
    Loss = (1/N) * Σ (prediction - target)^2
    """
    
    def __init__(self):
        if not _is_cuda_available():
            raise RuntimeError(
                "CUDA extension not available. Use L2LossAutograd instead. "
                "To enable CUDA, install with: pip install -e ."
            )
        
        self.initialized = False
    
    def initialize(
        self,
        params: Params,
        config: Optional[Config] = None,
    ) -> None:
        """Initialize and validate parameters."""
        if params.get('prediction') is None:
            raise ValueError("Missing required parameter: 'prediction'")
        
        prediction = params.get('prediction').data
        
        # Must be 3D: (B, H, W) for images or (D, H, W) for volumes
        if prediction.dim() != 3:
            raise ValueError(
                f"Prediction must be 3D (B, H, W) or (D, H, W), got shape {prediction.shape}"
            )
        
        # Must be on CUDA
        if not prediction.is_cuda:
            raise ValueError("Prediction must be a CUDA tensor for L2LossCUDA")
        
        self.initialized = True
    
    def forward_only(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> ForwardResult:
        """Forward computation only (no gradients)."""
        prediction = params.get('prediction').data
        target = data.get('target')
        
        if prediction.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: prediction {prediction.shape} vs target {target.shape}"
            )
        
        if not prediction.is_cuda:
            raise ValueError("Prediction must be a CUDA tensor")
        if not target.is_cuda:
            raise ValueError("Target must be a CUDA tensor")
        
        # Ensure contiguous
        prediction = prediction.contiguous()
        target = target.contiguous()
        
        # Lazy import to avoid triggering JIT at module load time
        from ..cuda import _wrapper
        
        # Call CUDA forward function
        loss, l2_map = _wrapper.l2_loss_fwd(prediction, target)
        
        return ForwardResult(
            outputs={'loss': loss, 'l2_map': l2_map},
            intermediates={}
        )
    
    def forward_backward1(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> tuple[ForwardResult, GradResult]:
        """Forward + first-order backward computation."""
        prediction = params.get('prediction').data
        target = data.get('target')
        
        if prediction.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: prediction {prediction.shape} vs target {target.shape}"
            )
        
        if not prediction.is_cuda:
            raise ValueError("Prediction must be a CUDA tensor")
        if not target.is_cuda:
            raise ValueError("Target must be a CUDA tensor")
        
        # Ensure contiguous
        prediction = prediction.contiguous()
        target = target.contiguous()
        
        # Lazy import
        from ..cuda import _wrapper
        
        # Forward
        loss, l2_map = _wrapper.l2_loss_fwd(prediction, target)
        
        # Backward
        grad_output = torch.ones(1, device=prediction.device, dtype=prediction.dtype)
        grad_pred = _wrapper.l2_loss_bwd(prediction, target, grad_output)
        
        forward_result = ForwardResult(
            outputs={'loss': loss, 'l2_map': l2_map},
            intermediates={}
        )
        
        grad_result = GradResult()
        grad_result.set('prediction', grad_pred)
        
        return forward_result, grad_result
    
    def forward_backward2(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> tuple[ForwardResult, GradResult, HessResult]:
        """Forward + first-order + second-order backward computation."""
        forward_result, grad_result = self.forward_backward1(params, data, config)
        
        prediction = params.get('prediction').data
        
        # Ensure contiguous
        prediction = prediction.contiguous()
        
        # Lazy import
        from ..cuda import _wrapper
        
        # Second-order backward
        grad_output = torch.ones(1, device=prediction.device, dtype=prediction.dtype)
        hessian_diag = _wrapper.l2_loss_bwd2(prediction, grad_output)
        
        hess_result = HessResult()
        hess_result.set('prediction', hessian_diag)
        
        return forward_result, grad_result, hess_result


# ============================================================================
# Factory Functions
# ============================================================================

def create_image_loss(
    lambda_l2: float = 1.0,
    lambda_ssim: float = 1.0,
    window_size: int = 11
) -> ImageLossAutograd:
    """Factory function to create image loss operator (autograd version)."""
    return ImageLossAutograd(lambda_l2=lambda_l2, lambda_ssim=lambda_ssim, window_size=window_size)


def create_image_loss_torch(
    lambda_l2: float = 1.0,
    lambda_ssim: float = 1.0,
    window_size: int = 11
) -> ImageLossTorch:
    """Factory function to create image loss operator (manual torch version)."""
    return ImageLossTorch(lambda_l2=lambda_l2, lambda_ssim=lambda_ssim, window_size=window_size)


def create_image_loss_torch_pixel_parallel(
    lambda_l2: float = 1.0,
    lambda_ssim: float = 1.0,
    window_size: int = 11
) -> ImageLossTorchPixelParallel:
    """Factory function to create image loss operator (pixel-parallel torch version)."""
    return ImageLossTorchPixelParallel(lambda_l2=lambda_l2, lambda_ssim=lambda_ssim, window_size=window_size)


def create_image_loss_cuda(
    lambda_l2: float = 1.0,
    lambda_ssim: float = 0.2,
    window_size: int = 11
) -> ImageLossCUDA:
    """Factory function to create image loss operator (CUDA version)."""
    return ImageLossCUDA(lambda_l2=lambda_l2, lambda_ssim=lambda_ssim, window_size=window_size)


def create_volume_loss(eps: float = 1e-4) -> VolumeLossAutograd:
    """Factory function to create volume loss operator (autograd version)."""
    return VolumeLossAutograd(eps=eps)


def create_volume_loss_torch(eps: float = 1e-4) -> VolumeLossTorch:
    """Factory function to create volume loss operator (manual torch version)."""
    return VolumeLossTorch(eps=eps)


def create_l2_loss() -> L2LossAutograd:
    """Factory function to create L2 loss operator (autograd version)."""
    return L2LossAutograd()


def create_l2_loss_cuda() -> L2LossCUDA:
    """Factory function to create L2 loss operator (CUDA version)."""
    return L2LossCUDA()


__all__ = [
    # Loss functions (autograd)
    'ImageLossAutograd',
    'VolumeLossAutograd',
    'L2LossAutograd',
    
    # Loss functions (manual torch)
    'ImageLossTorch',
    'ImageLossTorchPixelParallel',
    'VolumeLossTorch',
    
    # Loss functions (CUDA)
    'ImageLossCUDA',
    'L2LossCUDA',
    
    # Factory functions
    'create_image_loss',
    'create_image_loss_torch',
    'create_image_loss_torch_pixel_parallel',
    'create_image_loss_cuda',
    'create_volume_loss',
    'create_volume_loss_torch',
    'create_l2_loss',
    'create_l2_loss_cuda',
    
    # Utility functions
    'l1_loss',
    'l2_loss',
    'ssim',
    'tv_3d_loss',
]
