"""
2D Gaussian Splatting (2DGS) second-order optimizer with damping.

This module provides specialized implementations for optimizing 2D Gaussian parameters
using damped Newton's method with diagonal Hessian approximation.

Key Classes:
- GS2DOptimizerConfig: Configuration for 2DGS optimization
- GS2DSecondOrderOptimizer: Optimizer for 2D Gaussian Splatting parameters

2DGS Parameters:
- means2d: (C, N, 2) 2D projected Gaussian centers
- conics: (C, N, 3) Inverse covariance matrices [a, b, c]
- opacities: (C, N) Gaussian opacities
- mus: (C, N) Gaussian mu values (CT attenuation)

Update Formula (diagonal Hessian per element):
    delta[i] = -grad[i] / (damping + |hess[i]|)
    param_new[i] = param_old[i] + lr * delta[i]

Author: Dy164
Date: 2026-01-30
"""

from dataclasses import dataclass
from typing import Optional, List, Dict

import torch

from gsplat2ct.ops.data import Params
from .base import OptimizerConfig, SecondOrderOptimizerBase


# ============================================================================
# 2DGS Optimizer Configuration
# ============================================================================

@dataclass
class GS2DOptimizerConfig(OptimizerConfig):
    """Optimizer configuration specialized for 2D Gaussian Splatting.
    
    Provides convenient initialization for 2DGS parameters.
    Supports unified settings or per-parameter overrides.
    
    Default parameter names: ['means2d', 'conics', 'opacities', 'mus']
    
    Args:
        lr: Default learning rate for all parameters
        damping: Default damping coefficient for all parameters
        param_names: Optional list of parameter names to initialize
    """
    
    def __init__(
        self,
        lr: float = 1.0,
        damping: float = 1e-3,
        param_names: Optional[List[str]] = None,
    ):
        super().__init__()
        
        # Default parameter names for 2DGS
        if param_names is None:
            param_names = ['means2d', 'conics', 'opacities', 'mus']
        
        # Initialize all parameters with default values
        for name in param_names:
            self.lr_map[name] = lr
            self.damping_map[name] = damping
    
    def set_means2d_lr(self, lr: float) -> None:
        """Set learning rate for means2d parameter."""
        self.lr_map['means2d'] = lr
    
    def set_means2d_damping(self, damping: float) -> None:
        """Set damping for means2d parameter."""
        self.damping_map['means2d'] = damping
    
    def set_conics_lr(self, lr: float) -> None:
        """Set learning rate for conics parameter."""
        self.lr_map['conics'] = lr
    
    def set_conics_damping(self, damping: float) -> None:
        """Set damping for conics parameter."""
        self.damping_map['conics'] = damping
    
    def set_opacities_lr(self, lr: float) -> None:
        """Set learning rate for opacities parameter."""
        self.lr_map['opacities'] = lr
    
    def set_opacities_damping(self, damping: float) -> None:
        """Set damping for opacities parameter."""
        self.damping_map['opacities'] = damping
    
    def set_mus_lr(self, lr: float) -> None:
        """Set learning rate for mus parameter."""
        self.lr_map['mus'] = lr
    
    def set_mus_damping(self, damping: float) -> None:
        """Set damping for mus parameter."""
        self.damping_map['mus'] = damping
    
    def set_per_param(
        self,
        means2d_lr: Optional[float] = None,
        means2d_damping: Optional[float] = None,
        conics_lr: Optional[float] = None,
        conics_damping: Optional[float] = None,
        opacities_lr: Optional[float] = None,
        opacities_damping: Optional[float] = None,
        mus_lr: Optional[float] = None,
        mus_damping: Optional[float] = None,
    ) -> None:
        """Set per-parameter learning rates and damping in one call.
        
        Only updates parameters where values are provided (not None).
        """
        if means2d_lr is not None:
            self.lr_map['means2d'] = means2d_lr
        if means2d_damping is not None:
            self.damping_map['means2d'] = means2d_damping
        if conics_lr is not None:
            self.lr_map['conics'] = conics_lr
        if conics_damping is not None:
            self.damping_map['conics'] = conics_damping
        if opacities_lr is not None:
            self.lr_map['opacities'] = opacities_lr
        if opacities_damping is not None:
            self.damping_map['opacities'] = opacities_damping
        if mus_lr is not None:
            self.lr_map['mus'] = mus_lr
        if mus_damping is not None:
            self.damping_map['mus'] = mus_damping


# ============================================================================
# 2DGS Optimizer Implementation
# ============================================================================

class GS2DSecondOrderOptimizer(SecondOrderOptimizerBase):
    """Second-order optimizer for 2D Gaussian Splatting.
    
    Specialized implementation for optimizing 2DGS parameters using
    diagonal Hessian approximation (per-element second derivatives).
    
    Key Features:
    - Handles mixed parameter shapes (means2d: (C,N,2), conics: (C,N,3), etc.)
    - Uses diagonal Hessian (same shape as gradient and parameter)
    - Element-wise update: delta = -grad / (damping + |hess|)
    - Optional constraints: opacity clipping, mu bounds, etc.
    
    Constraints:
    - opacities: Clipped to [0, 1] after update
    - mus: Can be clipped to [mu_min, mu_max] if specified
    - conics: Must remain positive semi-definite (a > 0, c > 0, ac - b^2 > 0)
    
    Note: The abs(hess) handles potential negative curvature regions.
    """
    
    def __init__(
        self,
        params: Params,
        config: GS2DOptimizerConfig,
        opacity_clip: bool = True,
        mu_min: Optional[float] = None,
        mu_max: Optional[float] = None,
        conic_constraint: bool = False,
    ):
        """Initialize 2DGS optimizer.
        
        Args:
            params: Parameter container
            config: Optimizer configuration
            opacity_clip: Whether to clip opacities to [0, 1] (default: True)
            mu_min: Minimum mu value (optional)
            mu_max: Maximum mu value (optional)
            conic_constraint: Whether to enforce positive semi-definite conics (default: False)
        """
        super().__init__(params, config)
        self.opacity_clip = opacity_clip
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.conic_constraint = conic_constraint
    
    def step(self) -> None:
        """Perform optimization step for all 2DGS parameters.
        
        Applies element-wise damped Newton update to each parameter.
        
        Raises:
            RuntimeError: If parameter is missing gradient or Hessian
        """
        for name in self.params.keys():
            param = self.params.get(name)
            grad = self.params.get_grad(name)
            hess = self.params.get_hess(name)
            
            if grad is None or hess is None:
                raise RuntimeError(
                    f"Parameter '{name}' missing gradient or Hessian. "
                    f"Did you forget to call add_grad_hess()?"
                )
            
            lr = self.config.get_lr(name)
            damping = self.config.get_damping(name)
            
            # Apply element-wise damped Newton update
            self._update_with_diagonal_hessian(param, grad, hess, lr, damping)
            
            # Apply constraints
            self._apply_constraints(name, param)
    
    def _update_with_diagonal_hessian(
        self,
        param,
        grad: torch.Tensor,
        hess: torch.Tensor,
        lr: float,
        damping: float,
    ) -> None:
        """Update parameter using diagonal Hessian (element-wise update).
        
        For diagonal Hessian, the Newton update simplifies to:
            delta = -grad / (damping + |hess|)
            param_new = param_old + lr * delta
        
        Args:
            param: Parameter object with .data attribute
            grad: Gradient tensor (same shape as param.data)
            hess: Diagonal Hessian tensor (same shape as param.data)
            lr: Learning rate
            damping: Damping coefficient
        """
        # Reshape hess to match grad if necessary
        if hess.shape != grad.shape:
            hess = hess.view(grad.shape)
        
        # Element-wise damped Newton step
        # Use abs(hess) to handle negative curvature
        hess_damped = damping + torch.abs(hess)
        
        # Compute Newton step direction
        delta = -grad / hess_damped
        
        # Update parameter: param_new = param_old + lr * delta
        param.data = param.data + lr * delta
    
    def _apply_constraints(self, name: str, param) -> None:
        """Apply constraints to specific parameters.
        
        Args:
            name: Parameter name
            param: Parameter object with .data attribute
        """
        if name == 'opacities' and self.opacity_clip:
            # Clip opacities to [0, 1]
            param.data = param.data.clamp(0.0, 1.0)
        
        elif name == 'mus':
            # Clip mus to [mu_min, mu_max] if specified
            if self.mu_min is not None:
                param.data = param.data.clamp(min=self.mu_min)
            if self.mu_max is not None:
                param.data = param.data.clamp(max=self.mu_max)
        
        elif name == 'conics' and self.conic_constraint:
            # Enforce positive semi-definite: a > 0, c > 0, ac - b^2 > 0
            # conics: (C, N, 3) with [a, b, c]
            eps = 1e-6
            a = param.data[..., 0]
            b = param.data[..., 1]
            c = param.data[..., 2]
            
            # Enforce a > 0, c > 0
            a = torch.clamp(a, min=eps)
            c = torch.clamp(c, min=eps)
            
            # Enforce ac - b^2 > 0: |b| < sqrt(ac)
            max_b = torch.sqrt(a * c) - eps
            b = torch.clamp(b, min=-max_b, max=max_b)
            
            param.data[..., 0] = a
            param.data[..., 1] = b
            param.data[..., 2] = c
    
    def get_param_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all parameters.
        
        Returns:
            Dictionary with stats for each parameter:
            {param_name: {min, max, mean, std}}
        """
        stats = {}
        for name in self.params.keys():
            param = self.params.get(name)
            data = param.data
            stats[name] = {
                'min': data.min().item(),
                'max': data.max().item(),
                'mean': data.mean().item(),
                'std': data.std().item(),
            }
        return stats


__all__ = [
    "GS2DOptimizerConfig",
    "GS2DSecondOrderOptimizer",
]
