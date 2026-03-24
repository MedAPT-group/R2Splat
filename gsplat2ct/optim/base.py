"""
Base classes for second-order optimizers.

This module provides abstract base classes and configuration interfaces
for implementing second-order optimization methods based on damped Newton's method.

Key Classes:
- OptimizerConfig: Base configuration class for hyperparameter management
- SecondOrderOptimizerBase: Abstract base class defining the optimizer interface

Author: Dy164
Date: 2025-12-18
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict

import torch

from gsplat2ct.ops.data import Params, GradResult, HessResult


# ============================================================================
# Optimizer Configuration
# ============================================================================

@dataclass
class OptimizerConfig:
    """Base configuration class for second-order optimizer.
    
    Stores hyperparameters (learning rate, damping) for each parameter.
    Supports both unified and per-parameter settings.
    
    This is a base class that can be subclassed for specific optimizer types.
    """
    lr_map: Dict[str, float] = field(default_factory=dict)
    damping_map: Dict[str, float] = field(default_factory=dict)
    
    def set_lr(self, name: str, lr: float) -> None:
        """Set learning rate for a specific parameter.
        
        Args:
            name: Parameter name
            lr: Learning rate value
        """
        self.lr_map[name] = lr
    
    def set_damping(self, name: str, damping: float) -> None:
        """Set damping coefficient for a specific parameter.
        
        Args:
            name: Parameter name
            damping: Damping coefficient value
        """
        self.damping_map[name] = damping
    
    def get_lr(self, name: str, default: float = 1.0) -> float:
        """Get learning rate for a parameter (with default fallback).
        
        Args:
            name: Parameter name
            default: Default learning rate if not found
        
        Returns:
            Learning rate for the parameter
        """
        return self.lr_map.get(name, default)
    
    def get_damping(self, name: str, default: float = 1e-3) -> float:
        """Get damping coefficient for a parameter (with default fallback).
        
        Args:
            name: Parameter name
            default: Default damping coefficient if not found
        
        Returns:
            Damping coefficient for the parameter
        """
        return self.damping_map.get(name, default)


# ============================================================================
# Base Optimizer Class
# ============================================================================

class SecondOrderOptimizerBase(ABC):
    """Abstract base class for second-order optimizers.
    
    Implements damped Newton's method: θ_new = θ_old - lr * (H + λI)^(-1) * g
    
    where:
    - θ: parameters
    - g: gradient
    - H: Hessian matrix
    - λ: damping coefficient (regularization)
    - lr: learning rate (step size)
    
    Life cycle:
    1. __init__(params, config): Initialize optimizer
    2. add_grad_hess(grad_result, hess_result): Add gradients and Hessians to params
    3. step(): Perform optimization step
    4. zero_grad(): Clear gradients and Hessians
    5. Repeat from step 2
    
    Subclasses must implement the step() method to define parameter update logic.
    """
    
    def __init__(
        self,
        params: Params,
        config: OptimizerConfig,
    ):
        """Initialize optimizer.
        
        Args:
            params: Parameter container (will be modified in-place)
            config: Optimizer configuration (learning rates, damping)
        """
        self.params = params
        self.config = config
        self._step_count = 0
    
    def add_grad_hess(
        self,
        grad_result: GradResult,
        hess_result: HessResult,
    ) -> None:
        """Add gradient and Hessian information to parameters.
        
        This method transfers gradient and Hessian from computation results
        to the parameter container for use in optimization step.
        
        Note: Only extracts Hessians matching gradient parameter names.
        Cross-term Hessians (e.g., h_op_mean, h_mean_conic) in hess_result
        are ignored by the optimizer as they're not needed for diagonal updates.
        
        Diagonal Hessian Priority:
        For each parameter, if a '{name}_diag' key exists in hess_result,
        it will be used instead of '{name}'. This allows operators to return
        both full upper triangular Hessians (for debugging/testing) and
        diagonal versions (for optimization).
        
        Args:
            grad_result: Gradients for all parameters
            hess_result: Hessians for all parameters (may include cross-terms and _diag versions)
        
        Raises:
            AssertionError: If validation fails (missing Hessians or dimension mismatch)
        """
        self._validate(grad_result, hess_result)
        
        # Transfer gradients and Hessians to params (only for gradient parameters)
        for name in grad_result.param_grads.keys():
            self.params.add_grad(name, grad_result.get(name))
            
            # Prefer diagonal version if available (for optimization efficiency)
            diag_name = f"{name}_diag"
            if hess_result.get(diag_name) is not None:
                self.params.add_hess(name, hess_result.get(diag_name))
            else:
                self.params.add_hess(name, hess_result.get(name))
    
    @abstractmethod
    def step(self) -> None:
        """Perform one optimization step.
        
        Updates parameters in-place using stored gradients and Hessians.
        Must be implemented by subclasses.
        
        Raises:
            NotImplementedError: This is an abstract method
        """
        raise NotImplementedError("Subclass must implement step()")
    
    def zero_grad(self) -> None:
        """Clear gradients and Hessians from parameters.
        
        Should be called after each optimization step to prepare for next iteration.
        """
        self.params.clear_grads_hess()
        self._step_count += 1
    
    def state_dict(self) -> Dict:
        """Return optimizer state as dictionary.
        
        Can be used for checkpointing and resuming training.
        
        Returns:
            Dictionary containing optimizer state
        """
        return {
            'step_count': self._step_count,
            'lr_map': self.config.lr_map.copy(),
            'damping_map': self.config.damping_map.copy(),
        }
    
    def load_state_dict(self, state: Dict) -> None:
        """Load optimizer state from dictionary.
        
        Args:
            state: State dictionary from state_dict()
        """
        self._step_count = state['step_count']
        self.config.lr_map = state['lr_map'].copy()
        self.config.damping_map = state['damping_map'].copy()
    
    def _validate(
        self,
        grad_result: GradResult,
        hess_result: HessResult,
    ) -> None:
        """Validate gradient and Hessian consistency.
        
        Checks:
        1. All parameters with gradients must have corresponding Hessians
           (either as '{name}' or '{name}_diag' key)
        2. Gradient dimensions match parameter dimensions
        3. Hessian dimensions match expected diagonal dimensions (using _diag if available)
        
        Args:
            grad_result: Gradient computation result
            hess_result: Hessian computation result
        
        Raises:
            AssertionError: If any validation check fails
        """
        # Check 1: All gradient params must have Hessians (but not vice versa)
        grad_names = set(grad_result.param_grads.keys())
        hess_names = set(hess_result.param_hess.keys())
        
        # Hessian can have extra entries (e.g., cross-terms like h_op_mean, or _diag versions)
        # Check that each gradient param has either '{name}' or '{name}_diag' in Hessian
        missing_hess = set()
        for name in grad_names:
            if name not in hess_names and f"{name}_diag" not in hess_names:
                missing_hess.add(name)
        
        assert not missing_hess, (
            f"Missing Hessians for parameters with gradients:\n"
            f"  Parameters with gradients: {grad_names}\n"
            f"  Parameters with Hessians: {hess_names}\n"
            f"  Missing: {missing_hess}\n"
            f"  Note: Hessian may be provided as '{name}' or '{name}_diag'"
        )
        
        # Check 2 & 3: Dimensions match
        for name in grad_names:
            param = self.params.get(name)
            grad = grad_result.get(name)
            
            # Check gradient shape
            assert param.data.shape == grad.shape, (
                f"Parameter '{name}' shape {param.data.shape} != "
                f"gradient shape {grad.shape}"
            )
            
            # Get Hessian (prefer _diag version if available)
            diag_name = f"{name}_diag"
            hess = hess_result.get(diag_name) if diag_name in hess_names else hess_result.get(name)
            
            # Check Hessian dimensions
            # Supports three formats:
            # 1. Same shape as gradient (element-wise diagonal Hessian)
            # 2. Flattened 1D diagonal Hessian with size == numel
            # 3. Full upper triangular (if no _diag version exists)
            if hess is not None:
                grad_numel = grad.numel()
                hess_numel = hess.numel()
                same_shape = hess.shape == grad.shape
                same_numel = hess_numel == grad_numel
                
                # For full upper triangular Hessians, we skip the shape check
                # because they will be converted to diagonal in add_grad_hess
                is_diag_version = diag_name in hess_names
                
                if is_diag_version or (name in hess_names and hess_result.get(diag_name) is None):
                    # Only validate if we're using diagonal version or no _diag exists
                    if not (same_shape or same_numel):
                        # Check if this looks like a full upper triangular (skip validation)
                        # means3d: grad (N,3) -> full hess (N,6), conics: grad (N,6) -> full hess (N,21)
                        is_likely_full_hess = (
                            (grad.shape[-1] == 3 and hess.shape[-1] == 6) or   # means3d
                            (grad.shape[-1] == 6 and hess.shape[-1] == 21)     # conics
                        )
                        if not is_likely_full_hess:
                            assert False, (
                                f"Parameter '{name}': Hessian shape {hess.shape} "
                                f"must either match gradient shape {grad.shape} or "
                                f"have numel {grad_numel} (got {hess_numel})"
                            )


__all__ = [
    "OptimizerConfig",
    "SecondOrderOptimizerBase",
]
