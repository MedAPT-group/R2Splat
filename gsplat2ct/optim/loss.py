"""
ImageLoss-specific second-order optimizer and configuration.

This module provides specialized implementations for optimizing image predictions
using damped Newton's method with diagonal Hessian approximation.

Key Classes:
- ImageLossOptimizerConfig: Configuration for image loss optimization
- ImageLossSecondOrderOptimizer: Optimizer for pixel-level image reconstruction

Key Difference from MLP Optimizer:
- MLP optimizer uses full Hessian matrix and solves linear systems
- ImageLoss optimizer uses diagonal Hessian (per-pixel second derivatives)
  so the update simplifies to element-wise division:
  
  delta = -grad / (damping + hess_diag)
  param_new = param_old + lr * delta

Author: Dy164
Date: 2025-12-19
"""

from dataclasses import dataclass
from typing import Optional, List

import torch

from gsplat2ct.ops.data import Params
from .base import OptimizerConfig, SecondOrderOptimizerBase


# ============================================================================
# ImageLoss Optimizer Configuration
# ============================================================================

@dataclass
class ImageLossOptimizerConfig(OptimizerConfig):
    """Optimizer configuration specialized for image loss optimization.
    
    Provides convenient initialization for image prediction parameters.
    Supports unified settings or per-parameter overrides.
    
    For image loss, the typical parameter is 'prediction' which represents
    the predicted image tensor of shape (B, H, W).
    
    Args:
        lr: Default learning rate for all parameters
        damping: Default damping coefficient for all parameters
        param_names: Optional list of parameter names to initialize
                    (default: ['prediction'])
    """
    
    def __init__(
        self,
        lr: float = 1.0,
        damping: float = 1e-3,
        param_names: Optional[List[str]] = None,
    ):
        super().__init__()
        
        # Default parameter names for image loss optimization
        if param_names is None:
            param_names = ['prediction']
        
        # Initialize all parameters with default values
        for name in param_names:
            self.lr_map[name] = lr
            self.damping_map[name] = damping
    
    def set_prediction_lr(self, lr: float) -> None:
        """Convenience method to set learning rate for prediction parameter.
        
        Args:
            lr: Learning rate for the prediction parameter
        """
        self.lr_map['prediction'] = lr
    
    def set_prediction_damping(self, damping: float) -> None:
        """Convenience method to set damping for prediction parameter.
        
        Args:
            damping: Damping coefficient for the prediction parameter
        """
        self.damping_map['prediction'] = damping


# ============================================================================
# ImageLoss Optimizer Implementation
# ============================================================================

class ImageLossSecondOrderOptimizer(SecondOrderOptimizerBase):
    """Second-order optimizer for image loss minimization.
    
    Specialized implementation for optimizing image predictions using
    diagonal Hessian approximation (per-pixel second derivatives).
    
    Key Features:
    - Uses diagonal Hessian (same shape as gradient and parameter)
    - Element-wise update instead of matrix inversion
    - Efficient for large images (no O(n^3) matrix operations)
    
    Update Formula (per-pixel):
        delta[i] = -grad[i] / (damping + hess[i])
        param_new[i] = param_old[i] + lr * delta[i]
    
    This is equivalent to damped Newton's method with diagonal Hessian:
        param_new = param_old - lr * diag(H + λI)^(-1) * g
    
    where:
    - H: diagonal Hessian (per-pixel second derivatives)
    - λ: damping coefficient
    - g: gradient
    - lr: learning rate
    
    Note: The damping term prevents division by zero and provides
    regularization when the Hessian is close to zero or negative.
    """
    
    def step(self) -> None:
        """Perform optimization step for all image parameters.
        
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
            delta = -grad / (damping + hess)
            param_new = param_old + lr * delta
        
        This avoids expensive matrix inversion and is suitable for
        large-scale image optimization where the Hessian is diagonal.
        
        Args:
            param: Parameter object with .data attribute
            grad: Gradient tensor (same shape as param.data)
            hess: Diagonal Hessian tensor (same shape as param.data)
            lr: Learning rate
            damping: Damping coefficient (added to Hessian for stability)
        
        Note:
            - The damping ensures numerical stability when hess is near zero
            - For positive-definite Hessian, damping can be small (1e-6 to 1e-3)
            - For non-convex regions, larger damping provides regularization
        """
        # Ensure shapes match
        assert grad.shape == param.data.shape, (
            f"Gradient shape {grad.shape} != parameter shape {param.data.shape}"
        )
        assert hess.shape == param.data.shape, (
            f"Hessian shape {hess.shape} != parameter shape {param.data.shape}"
        )
        
        # Element-wise damped Newton step
        # delta = -grad / (damping + hess)
        # Note: We use abs(hess) to handle potential negative curvature
        # This is a common practice in quasi-Newton methods
        hess_damped = damping + torch.abs(hess)
        
        # Compute Newton step direction
        delta = -grad / hess_damped
        
        # Update parameter: param_new = param_old + lr * delta
        param.data = param.data + lr * delta


__all__ = [
    "ImageLossOptimizerConfig",
    "ImageLossSecondOrderOptimizer",
]
