"""
Abstract base class for operators with second-order derivatives.

This module defines the unified interface for forward/backward computation
with support for first and second-order derivatives (gradients and Hessians).

Author: Dy164
Date: 2025-12-02
"""

from abc import ABC, abstractmethod
from typing import Optional

# Import will be from .data after it's created
# For now, use TYPE_CHECKING to avoid circular import
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .data import Params, Data, Config, ForwardResult, GradResult, HessResult


class SecondForwardBackwardBase(ABC):
    """Unified interface for forward/backward computation with second-order derivatives.
    
    This abstract base class defines the standard interface for algorithms that support:
    - Forward-only computation (for validation/testing)
    - Forward + first-order backward (for standard first-order training)
    - Forward + first-order + second-order backward (for second-order training)
    
    All custom operators should be wrapped under this interface by implementing
    these four abstract methods.
    
    Note:
    - This is our custom interface, different from torch.autograd.Function
    - We need explicit control over first and second-order derivative computation
    - This design supports flexible computation modes (forward-only, first-order, second-order)
    """

    @abstractmethod
    def initialize(
        self,
        params: "Params",
        config: Optional["Config"] = None,
    ) -> None:
        """Initialize the operator.
        
        This method is called once before any forward/backward computation.
        It should validate parameters and set up any necessary state.
        
        Args:
            params: Container of optimizable parameters
            config: Optional configuration
        """
        raise NotImplementedError

    @abstractmethod
    def forward_only(
        self,
        params: "Params",
        data: "Data",
        config: Optional["Config"] = None,
    ) -> "ForwardResult":
        """Forward computation only (no gradients).
        
        This mode is used for validation or testing when gradients are not needed.
        
        Args:
            params: Container of optimizable parameters
            data: Container of input data and targets
            config: Optional configuration
            
        Returns:
            ForwardResult containing outputs (e.g., loss) and optional intermediates
        """
        raise NotImplementedError

    @abstractmethod
    def forward_backward1(
        self,
        params: "Params",
        data: "Data",
        config: Optional["Config"] = None,
    ) -> tuple["ForwardResult", "GradResult"]:
        """Forward + first-order backward computation.
        
        This mode is used for standard first-order optimization (e.g., SGD, Adam).
        
        Args:
            params: Container of optimizable parameters
            data: Container of input data and targets
            config: Optional configuration
            
        Returns:
            Tuple of (ForwardResult, GradResult)
            - ForwardResult: outputs and intermediates from forward pass
            - GradResult: first-order gradients for all parameters
        """
        raise NotImplementedError

    @abstractmethod
    def forward_backward2(
        self,
        params: "Params",
        data: "Data",
        config: Optional["Config"] = None,
    ) -> tuple["ForwardResult", "GradResult", "HessResult"]:
        """Forward + first-order + second-order backward computation.
        
        This mode is used for second-order optimization (e.g., Newton's method, L-BFGS).
        
        Args:
            params: Container of optimizable parameters
            data: Container of input data and targets
            config: Optional configuration
            
        Returns:
            Tuple of (ForwardResult, GradResult, HessResult)
            - ForwardResult: outputs and intermediates from forward pass
            - GradResult: first-order gradients for all parameters
            - HessResult: second-order Hessians (diagonal and/or full) for all parameters
        """
        raise NotImplementedError


__all__ = ["SecondForwardBackwardBase"]
