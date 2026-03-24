"""
MLP-specific second-order optimizer and configuration.

This module provides specialized implementations for optimizing MLP parameters
using damped Newton's method with support for weight matrices and biases.

Key Classes:
- MLPOptimizerConfig: Configuration for MLP parameters (w1, b1, w2, b2)
- MLPSecondOrderOptimizer: Optimizer specialized for MLP architecture

Author: Dy164
Date: 2025-12-18
"""

from dataclasses import dataclass
from typing import Optional

import torch

from gsplat2ct.ops.data import Params
from .base import OptimizerConfig, SecondOrderOptimizerBase


# ============================================================================
# MLP Optimizer Configuration
# ============================================================================

@dataclass
class MLPOptimizerConfig(OptimizerConfig):
    """Optimizer configuration specialized for MLP parameters.
    
    Provides convenient initialization for w1, b1, w2, b2.
    Supports unified settings or per-layer/per-parameter overrides.
    
    Args:
        lr: Default learning rate for all parameters
        damping: Default damping coefficient for all parameters
        lr_w: Optional learning rate specifically for weight matrices (w1, w2)
        lr_b: Optional learning rate specifically for biases (b1, b2)
        damping_w: Optional damping specifically for weight matrices
        damping_b: Optional damping specifically for biases
    """
    
    def __init__(
        self,
        lr: float = 1.0,
        damping: float = 1e-3,
        lr_w: Optional[float] = None,
        lr_b: Optional[float] = None,
        damping_w: Optional[float] = None,
        damping_b: Optional[float] = None,
    ):
        super().__init__()
        
        # Initialize all parameters with default values
        for name in ['w1', 'b1', 'w2', 'b2']:
            self.lr_map[name] = lr
            self.damping_map[name] = damping
        
        # Override with specific values if provided
        if lr_w is not None:
            self.lr_map['w1'] = lr_w
            self.lr_map['w2'] = lr_w
        
        if lr_b is not None:
            self.lr_map['b1'] = lr_b
            self.lr_map['b2'] = lr_b
        
        if damping_w is not None:
            self.damping_map['w1'] = damping_w
            self.damping_map['w2'] = damping_w
        
        if damping_b is not None:
            self.damping_map['b1'] = damping_b
            self.damping_map['b2'] = damping_b


# ============================================================================
# MLP Optimizer Implementation
# ============================================================================

class MLPSecondOrderOptimizer(SecondOrderOptimizerBase):
    """Second-order optimizer for MLP parameters.
    
    Specialized implementation for MLP with parameters: w1, b1, w2, b2.
    
    Updates using damped Newton's method:
    - For bias parameters (b1, b2): Direct solution of small linear system
    - For weight matrices (w1, w2): Solution via flattening (Stage 1 implementation)
    
    Newton step formula:
        Δθ = (H + λI)^(-1) * (-g)
        θ_new = θ_old + lr * Δθ
    
    where:
    - H: Hessian matrix (full rank decomposition)
    - λ: Damping coefficient
    - g: Gradient
    - lr: Learning rate
    
    Note: This is Stage 1 implementation using direct matrix inversion.
    Future improvements can leverage Kronecker structure for efficiency.
    """
    
    def step(self) -> None:
        """Perform optimization step for all MLP parameters.
        
        Routes each parameter to the appropriate update method based on its type.
        
        Raises:
            RuntimeError: If parameter is missing gradient or Hessian
            ValueError: If parameter name is not recognized
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
            
            # Route to appropriate update method
            if name in ['w1', 'w2']:
                self._update_weight_matrix(name, param, grad, hess, lr, damping)
            elif name in ['b1', 'b2']:
                self._update_bias(name, param, grad, hess, lr, damping)
            else:
                raise ValueError(f"Unknown parameter name: {name}")
    
    def _update_weight_matrix(
        self,
        name: str,
        param,
        grad: torch.Tensor,
        hess: torch.Tensor,
        lr: float,
        damping: float,
    ) -> None:
        """Update weight matrix parameter (w1 or w2).
        
        Solves the damped Newton system for a weight matrix:
            (H + λI) * Δθ = -g
        
        Then updates: θ_new = θ_old + lr * Δθ
        
        Args:
            name: Parameter name ('w1' or 'w2')
            param: Parameter object with .data attribute
            grad: Gradient tensor (same shape as param)
            hess: Hessian tensor (full matrix, 2D)
            lr: Learning rate
            damping: Damping coefficient
        
        Raises:
            ValueError: If Hessian is None
            torch.linalg.LinAlgError: If matrix is singular (uses fallback)
        """
        # 1. Get original shape and flatten (row-major)
        shape = param.data.shape  # (m, i) or (o, m)
        param_flat = param.data.reshape(-1)
        grad_flat = grad.reshape(-1)
        
        # 2. Get full Hessian matrix
        if hess is None:
            raise ValueError(f"Parameter '{name}' requires full Hessian")
        H_full = hess  # (m*i, m*i) or (o*m, o*m)
        
        # 3. Add damping: H_damped = H + λI
        device = param.data.device
        n = H_full.size(0)
        H_damped = H_full + damping * torch.eye(n, device=device, dtype=H_full.dtype)
        
        # 4. Solve Newton system: (H + λI) * Δθ = -g
        try:
            delta_flat = torch.linalg.solve(H_damped, -grad_flat)
        except torch.linalg.LinAlgError:
            # Fallback: use least-squares solution if matrix is singular
            print(f"Warning: Singular Hessian for '{name}', using pseudo-inverse")
            delta_flat = -torch.linalg.lstsq(H_damped, grad_flat).solution
        
        # 5. Update parameter: θ_new = θ_old + lr * Δθ
        param_flat_new = param_flat + lr * delta_flat
        
        # 6. Reshape back to original shape
        param.data = param_flat_new.reshape(shape)
    
    def _update_bias(
        self,
        name: str,
        param,
        grad: torch.Tensor,
        hess: torch.Tensor,
        lr: float,
        damping: float,
    ) -> None:
        """Update bias parameter (b1 or b2).
        
        Solves the damped Newton system for a bias vector:
            (H + λI) * Δθ = -g
        
        Then updates: θ_new = θ_old + lr * Δθ
        
        Args:
            name: Parameter name ('b1' or 'b2')
            param: Parameter object with .data attribute
            grad: Gradient tensor (same shape as param)
            hess: Hessian tensor (full matrix, 2D)
            lr: Learning rate
            damping: Damping coefficient
        
        Raises:
            ValueError: If Hessian is None
            torch.linalg.LinAlgError: If matrix is singular (uses fallback)
        """
        # 1. Flatten (bias is already essentially a vector, but be consistent)
        shape = param.data.shape  # (m, 1) or (o, 1)
        param_flat = param.data.reshape(-1)
        grad_flat = grad.reshape(-1)
        
        # 2. Get full Hessian matrix
        if hess is None:
            raise ValueError(f"Parameter '{name}' requires full Hessian")
        H = hess  # (m, m) or (o, o)
        
        # 3. Add damping: H_damped = H + λI
        device = param.data.device
        n = H.size(0)
        H_damped = H + damping * torch.eye(n, device=device, dtype=H.dtype)
        
        # 4. Solve Newton system: (H + λI) * Δθ = -g
        try:
            delta_flat = torch.linalg.solve(H_damped, -grad_flat)
        except torch.linalg.LinAlgError:
            # Fallback: use least-squares solution if matrix is singular
            print(f"Warning: Singular Hessian for '{name}', using pseudo-inverse")
            delta_flat = -torch.linalg.lstsq(H_damped, grad_flat).solution
        
        # 5. Update parameter: θ_new = θ_old + lr * Δθ
        param_flat_new = param_flat + lr * delta_flat
        
        # 6. Reshape back to original shape
        param.data = param_flat_new.reshape(shape)


__all__ = [
    "MLPOptimizerConfig",
    "MLPSecondOrderOptimizer",
]
