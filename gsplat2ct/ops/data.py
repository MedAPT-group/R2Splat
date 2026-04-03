"""
Generic data structures for operators.

This module defines common input/output data structures that can be used
across different operator implementations.

Author: Dy164
Date: 2025-12-02
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Protocol, Iterable


# ============================================================================
# 1. Input Data Structures
# ============================================================================

class ParameterLike(Protocol):
    """Protocol for optimizable parameters.
    
    This defines the minimal interface for a parameter that can be optimized.
    Your custom NewtonParameter can implement this protocol by having:
    - .data: Tensor
    - Optionally .grad1, .hess fields for gradients and Hessians
    
    Note:
    - Computed gradients and Hessians can be stored in .grad1, .hess fields
    - The optimizer can register methods to manage these fields
    - This enables in-place Newton method optimization with damping
    - The optimizer also manages the lifecycle of gradients and Hessians
    - E.g., a clear_grad_hess() method to clear all parameter gradients/Hessians
    """
    data: torch.Tensor


@dataclass
class Params:
    """Container for optimizable parameters.
    
    Generic design: stores parameters as name -> ParameterLike dictionary.
    For specific modules (like MLP), you can define more specific fields
    in a subclass and sync them to param_map in __post_init__.
    
    Gradient and Hessian Management:
    - grad_map: stores gradients for each parameter
    - hess_map: stores Hessian tensors for each parameter (typically diagonal)
    - These are managed by the optimizer during training
    """
    param_map: Dict[str, ParameterLike] = field(default_factory=dict)
    grad_map: Dict[str, torch.Tensor] = field(default_factory=dict)
    hess_map: Dict[str, torch.Tensor] = field(default_factory=dict)

    def items(self):
        return self.param_map.items()

    def values(self):
        return self.param_map.values()

    def keys(self):
        return self.param_map.keys()

    def get(self, name: str) -> Optional[ParameterLike]:
        return self.param_map.get(name)

    def add_param(self, name: str, param: ParameterLike) -> None:
        self.param_map[name] = param

    def __iter__(self) -> Iterable[ParameterLike]:
        return iter(self.param_map.values())
    
    def add_grad(self, name: str, grad: torch.Tensor) -> None:
        """Add gradient for a parameter."""
        self.grad_map[name] = grad
    
    def add_hess(self, name: str, hess: torch.Tensor) -> None:
        """Add Hessian tensor for a parameter."""
        self.hess_map[name] = hess
    
    def get_grad(self, name: str) -> Optional[torch.Tensor]:
        """Get gradient for a parameter."""
        return self.grad_map.get(name)
    
    def get_hess(self, name: str) -> Optional[torch.Tensor]:
        """Get Hessian tensor for a parameter."""
        return self.hess_map.get(name)
    
    def clear_grads_hess(self) -> None:
        """Clear all gradients and Hessians."""
        self.grad_map.clear()
        self.hess_map.clear()


@dataclass
class Data:
    """Container for non-optimizable data/labels.
    
    Generic design: data_map is name -> Any (usually Tensor or more complex structures).
    For MLP + loss, typical contents:
      - inputs: torch.Tensor
      - targets: torch.Tensor
    """
    data_map: Dict[str, Any] = field(default_factory=dict)

    def get(self, name: str, default: Any = None) -> Any:
        """Get a value with optional default."""
        return self.data_map.get(name, default)

    def get_optional(self, name: str) -> Any:
        """Get a value that may be None (for optional inputs like test labels)."""
        return self.data_map.get(name, None)

    def set(self, name: str = None, value: Any = None, **kwargs) -> None:
        """Set data values. Supports both single and batch setting.
        
        Usage:
            data.set('key', value)  # Single setting
            data.set(key1=value1, key2=value2)  # Batch setting with **kwargs
        """
        if name is not None:
            self.data_map[name] = value
        for k, v in kwargs.items():
            self.data_map[k] = v


@dataclass
class Config:
    """Configuration options for computation.
    
    Example fields (optional):
      - hessian_type: 'none' | 'diag' | 'full' | 'gauss_newton'
      - damping: float
      - device: str
      - dtype: torch.dtype
    
    Left empty here, specific implementations can add their own fields.
    """
    options: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.options.get(key, default)

    def set(self, key: str = None, value: Any = None, **kwargs) -> None:
        """Set config options. Supports both single and batch setting.
        
        Usage:
            config.set('key', value)  # Single setting
            config.set(key1=value1, key2=value2)  # Batch setting with **kwargs
        """
        if key is not None:
            self.options[key] = value
        for k, v in kwargs.items():
            self.options[k] = v


def get_lambda_l2(config: Optional[Config], default: float = 1.0) -> float:
    """Weight for the L2 (MSE) image term: ``lambda_l2 * mean((pred - tgt)^2)``.

    The legacy key ``lambda_l1`` is still read if ``lambda_l2`` is absent.
    """
    if config is None:
        return default
    v = config.get("lambda_l2")
    if v is not None:
        return float(v)
    v = config.get("lambda_l1")
    if v is not None:
        return float(v)
    return default


# ============================================================================
# 2. Output Data Structures
# ============================================================================

@dataclass
class ForwardResult:
    """Forward computation result.
    
    outputs: Main output dictionary, typically contains 'loss' and other key results
    intermediates: Intermediate values dictionary, for debugging or backward pass
    """
    outputs: Dict[str, Any] = field(default_factory=dict)
    intermediates: Dict[str, Any] = field(default_factory=dict)
    
    def get_loss(self) -> torch.Tensor:
        """Convenience method to get the loss value."""
        return self.outputs.get('loss')


@dataclass
class GradResult:
    """First-order gradient result."""
    param_grads: Dict[str, torch.Tensor] = field(default_factory=dict)
    aux: Dict[str, Any] = field(default_factory=dict)

    def get(self, name: str) -> torch.Tensor:
        return self.param_grads[name]

    def set(self, name: str = None, grad: torch.Tensor = None, **kwargs) -> None:
        """Set gradients. Supports both single and batch setting.
        
        Usage:
            result.set('param', grad_tensor)  # Single setting
            result.set(param1=grad1, param2=grad2)  # Batch setting with **kwargs
        """
        if name is not None:
            self.param_grads[name] = grad
        for k, v in kwargs.items():
            self.param_grads[k] = v

    def get_aux(self, name: str, default: Any = None) -> Any:
        return self.aux.get(name, default)

    def set_aux(self, name: str = None, value: Any = None, **kwargs) -> None:
        if name is not None:
            self.aux[name] = value
        for k, v in kwargs.items():
            self.aux[k] = v


@dataclass
class HessResult:
    """Second-order Hessian result.
    
    Stores Hessian tensors (typically diagonal) for each parameter,
    and cross-term Hessians between different parameters.
    """
    param_hess: Dict[str, torch.Tensor] = field(default_factory=dict)
    cross_hess: Dict[str, torch.Tensor] = field(default_factory=dict)

    def get(self, name: str) -> Optional[torch.Tensor]:
        return self.param_hess.get(name)

    def get_cross(self, name: str) -> Optional[torch.Tensor]:
        """Get cross-term Hessian by name."""
        return self.cross_hess.get(name)

    def set(self, name: str = None, hess: torch.Tensor = None, **kwargs) -> None:
        """Set Hessian tensors. Supports both single and batch setting.
        
        Usage:
            result.set('param', hess_tensor)  # Single setting
            result.set(param1=hess1, param2=hess2)  # Batch setting with **kwargs
        """
        if name is not None:
            self.param_hess[name] = hess
        for k, v in kwargs.items():
            self.param_hess[k] = v

    def set_cross(self, name: str = None, hess: torch.Tensor = None, **kwargs) -> None:
        """Set cross-term Hessian tensors. Supports both single and batch setting.
        
        Usage:
            result.set_cross('mu_opacity', hess_tensor)
            result.set_cross(mu_mean=h1, mu_conic=h2)
        """
        if name is not None:
            self.cross_hess[name] = hess
        for k, v in kwargs.items():
            self.cross_hess[k] = v


__all__ = [
    "ParameterLike",
    "Params",
    "Data",
    "Config",
    "ForwardResult",
    "GradResult",
    "HessResult",
]
