"""
3D Gaussian Splatting with TV Smoothing second-order optimizer.

This module provides specialized implementations for optimizing 3D Gaussian parameters
with Total Variation regularization using damped Newton's method with diagonal Hessian.

Key Classes:
- GS3DSmoothingOptimizerConfig: Configuration for 3D GS optimization
- GS3DSmoothingSecondOrderOptimizer: Optimizer for 3D Gaussian Splatting parameters

3DGS Parameters:
- means3d: (N, 3) 3D Gaussian centers [x, y, z]
- conics: (N, 6) Inverse covariance matrices [a, p, q, b, r, c]
- opacities: (N,) Gaussian opacities

Update Formula (diagonal Hessian per element):
    delta[i] = -grad[i] / (damping + |hess[i]|)
    param_new[i] = param_old[i] + lr * delta[i]

Author: Dy164
Date: 2026-02-03
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import torch

from gsplat2ct.ops.data import Params
from .base import OptimizerConfig, SecondOrderOptimizerBase


# ============================================================================
# 3DGS Smoothing Optimizer Configuration
# ============================================================================

@dataclass
class GS3DSmoothingOptimizerConfig(OptimizerConfig):
    """Optimizer configuration specialized for 3D Gaussian Splatting with Smoothing.
    
    Provides convenient initialization for 3DGS parameters.
    Supports unified settings or per-parameter overrides.
    
    Default parameter names: ['means3d', 'conics', 'opacities']
    
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
        
        # Default parameter names for 3DGS
        if param_names is None:
            param_names = ['means3d', 'conics', 'opacities']
        
        # Initialize all parameters with default values
        for name in param_names:
            self.lr_map[name] = lr
            self.damping_map[name] = damping
    
    def set_means3d_lr(self, lr: float) -> None:
        """Set learning rate for means3d parameter."""
        self.lr_map['means3d'] = lr
    
    def set_means3d_damping(self, damping: float) -> None:
        """Set damping for means3d parameter."""
        self.damping_map['means3d'] = damping
    
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
    
    def set_per_param(
        self,
        means3d_lr: Optional[float] = None,
        means3d_damping: Optional[float] = None,
        conics_lr: Optional[float] = None,
        conics_damping: Optional[float] = None,
        opacities_lr: Optional[float] = None,
        opacities_damping: Optional[float] = None,
    ) -> None:
        """Set per-parameter learning rates and damping in one call.
        
        Only updates parameters where values are provided (not None).
        """
        if means3d_lr is not None:
            self.lr_map['means3d'] = means3d_lr
        if means3d_damping is not None:
            self.damping_map['means3d'] = means3d_damping
        if conics_lr is not None:
            self.lr_map['conics'] = conics_lr
        if conics_damping is not None:
            self.damping_map['conics'] = conics_damping
        if opacities_lr is not None:
            self.lr_map['opacities'] = opacities_lr
        if opacities_damping is not None:
            self.damping_map['opacities'] = opacities_damping


# ============================================================================
# 3DGS Smoothing Optimizer Implementation
# ============================================================================

class GS3DSmoothingSecondOrderOptimizer(SecondOrderOptimizerBase):
    """Second-order optimizer for 3D Gaussian Splatting with TV Smoothing.
    
    Specialized implementation for optimizing 3DGS parameters using
    diagonal Hessian approximation (per-element second derivatives).
    
    Key Features:
    - Handles 3D Gaussian parameters: means3d (N,3), conics (N,6), opacities (N,)
    - Uses diagonal Hessian (per-Gaussian block diagonal elements)
    - Element-wise update: delta = -grad / (damping + |hess|)
    - Optional constraints: opacity clipping, position bounds, conic validity
    
    Constraints:
    - opacities: Clipped to [opacity_min, opacity_max] after update
    - means3d: Can be bounded to volume limits
    - conics: Must remain positive definite (diagonal elements > 0)
    
    Note: The abs(hess) handles potential negative curvature regions.
    """
    
    def __init__(
        self,
        params: Params,
        config: GS3DSmoothingOptimizerConfig,
        opacity_clip: bool = True,
        opacity_min: float = 0.0,
        opacity_max: float = 1.0,
        position_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        conic_constraint: bool = False,
    ):
        """Initialize 3DGS Smoothing optimizer.
        
        Args:
            params: Parameter container
            config: Optimizer configuration
            opacity_clip: Whether to clip opacities (default: True)
            opacity_min: Minimum opacity value (default: 0.0)
            opacity_max: Maximum opacity value (default: 1.0)
            position_bounds: Optional (min_bounds, max_bounds) for means3d
                             Each is (3,) tensor with [x_min/max, y_min/max, z_min/max]
            conic_constraint: Whether to enforce positive definite conics (default: False)
        """
        super().__init__(params, config)
        self.opacity_clip = opacity_clip
        self.opacity_min = opacity_min
        self.opacity_max = opacity_max
        self.position_bounds = position_bounds
        self.conic_constraint = conic_constraint
    
    def step(self) -> None:
        """Perform optimization step for all 3DGS parameters.
        
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
            self._update_with_diagonal_hessian(name, param, grad, hess, lr, damping)
            
            # Apply constraints
            self._apply_constraints(name, param)
    
    def _update_with_diagonal_hessian(
        self,
        name: str,
        param,
        grad: torch.Tensor,
        hess: torch.Tensor,
        lr: float,
        damping: float,
    ) -> None:
        """Update parameter using diagonal Hessian (element-wise update).
        
        The Hessian is already in diagonal form (extracted in forward_backward2):
        - means3d: (N, 3) diagonal elements [H00, H11, H22]
        - conics: (N, 6) diagonal elements of 6x6 matrix
        - opacities: (N,) scalar per Gaussian
        
        Args:
            name: Parameter name
            param: Parameter object with .data attribute
            grad: Gradient tensor
            hess: Hessian tensor (diagonal elements, same shape as grad)
            lr: Learning rate
            damping: Damping coefficient
        """
        # Element-wise damped Newton step (Hessian already diagonal)
        hess_damped = damping + torch.abs(hess)
        delta = -grad / hess_damped
        param.data = param.data + lr * delta
    
    def _apply_constraints(self, name: str, param) -> None:
        """Apply constraints to specific parameters.
        
        Args:
            name: Parameter name
            param: Parameter object with .data attribute
        """
        if name == 'opacities' and self.opacity_clip:
            # Clip opacities to [opacity_min, opacity_max]
            param.data = param.data.clamp(self.opacity_min, self.opacity_max)
        
        elif name == 'means3d' and self.position_bounds is not None:
            # Clip positions to bounds
            min_bounds, max_bounds = self.position_bounds
            param.data = torch.max(param.data, min_bounds.to(param.data.device))
            param.data = torch.min(param.data, max_bounds.to(param.data.device))
        
        elif name == 'conics' and self.conic_constraint:
            # Enforce positive diagonal elements for positive definiteness
            # conics: (N, 6) with [a, p, q, b, r, c]
            # a, b, c must be positive (diagonal of 3x3 inverse covariance)
            eps = 1e-6
            param.data[:, 0] = torch.clamp(param.data[:, 0], min=eps)  # a
            param.data[:, 3] = torch.clamp(param.data[:, 3], min=eps)  # b
            param.data[:, 5] = torch.clamp(param.data[:, 5], min=eps)  # c
    
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


# ============================================================================
# Factory Functions
# ============================================================================

def create_gs3d_smoothing_optimizer(
    params: Params,
    lr: float = 1.0,
    damping: float = 1e-3,
    opacity_clip: bool = True,
    opacity_min: float = 0.0,
    opacity_max: float = 1.0,
    position_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    conic_constraint: bool = False,
) -> GS3DSmoothingSecondOrderOptimizer:
    """Create a 3D Gaussian Splatting Smoothing optimizer.
    
    Convenience function to create an optimizer with common settings.
    
    Args:
        params: Parameter container
        lr: Learning rate for all parameters
        damping: Damping coefficient for all parameters
        opacity_clip: Whether to clip opacities
        opacity_min: Minimum opacity value
        opacity_max: Maximum opacity value
        position_bounds: Optional bounds for means3d positions
        conic_constraint: Whether to enforce conic constraints
        
    Returns:
        Configured optimizer instance
    """
    config = GS3DSmoothingOptimizerConfig(lr=lr, damping=damping)
    return GS3DSmoothingSecondOrderOptimizer(
        params=params,
        config=config,
        opacity_clip=opacity_clip,
        opacity_min=opacity_min,
        opacity_max=opacity_max,
        position_bounds=position_bounds,
        conic_constraint=conic_constraint,
    )


__all__ = [
    "GS3DSmoothingOptimizerConfig",
    "GS3DSmoothingSecondOrderOptimizer",
    "create_gs3d_smoothing_optimizer",
]
