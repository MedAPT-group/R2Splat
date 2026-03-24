"""
MLP Classification Operators with Second-Order Derivatives

This module provides three implementations of MLP classification:
1. MLPClassificationCUDA: CUDA-accelerated implementation
2. MLPClassificationTorch: Pure PyTorch manual implementation
3. MLPClassificationAutograd: PyTorch autograd-based implementation

All three inherit from SecondForwardBackwardBase and can be used
interchangeably for validation and benchmarking.

Author: Dy164
Date: 2025-12-02
Updated: 2025-12-16 (Lazy loading for CUDA extension)
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union

from .base import SecondForwardBackwardBase
from .data import (
    Params, Data, Config,
    ForwardResult, GradResult, HessResult,
)

# Note: We import _wrapper lazily in MLPClassificationCUDA to avoid
# triggering JIT compilation at module import time.
# from ..cuda._backend import is_cuda_available  # Moved to lazy check
# from ..cuda import _wrapper  # Moved to lazy import


def _is_cuda_available() -> bool:
    """Lazy check for CUDA availability without triggering JIT compilation."""
    try:
        from ..cuda._backend import is_cuda_available
        return is_cuda_available()
    except ImportError:
        return False


# ============================================================================
# MLP-Specific Data Structures
# ============================================================================

@dataclass
class MLPParams(Params):
    """MLP-specific parameter container.
    
    Args:
        w1: First layer weights (m, i)
        b1: First layer bias (m, 1)
        w2: Second layer weights (o, m)
        b2: Second layer bias (o, 1)
    """
    w1: Optional[torch.Tensor] = None
    b1: Optional[torch.Tensor] = None
    w2: Optional[torch.Tensor] = None
    b2: Optional[torch.Tensor] = None

    def __post_init__(self):
        # Sync to param_map
        if self.w1 is not None:
            self.param_map['w1'] = type('Param', (), {'data': self.w1})()
        if self.b1 is not None:
            self.param_map['b1'] = type('Param', (), {'data': self.b1})()
        if self.w2 is not None:
            self.param_map['w2'] = type('Param', (), {'data': self.w2})()
        if self.b2 is not None:
            self.param_map['b2'] = type('Param', (), {'data': self.b2})()


@dataclass
class MLPData(Data):
    """MLP-specific data container.
    
    Args:
        inputs: Input features (batch_size, i) where batch_size >= 1
        targets: Target labels (batch_size, o) where batch_size >= 1
        task_type: 'classification' or 'regression'
    
    Note: The first dimension is ALWAYS the batch dimension, even for single samples (batch_size=1).
    """
    inputs: Optional[torch.Tensor] = None
    targets: Optional[torch.Tensor] = None
    task_type: str = 'classification'

    def __post_init__(self):
        # Sync to data_map
        if self.inputs is not None:
            self.data_map['inputs'] = self.inputs
        if self.targets is not None:
            self.data_map['targets'] = self.targets
        self.data_map['task_type'] = self.task_type


@dataclass
class SecondOrderConfig(Config):
    """Configuration for second-order optimization.
    
    Args:
        hessian_type: 'diag' / 'full' / 'none' / 'gauss_newton'
        damping: Damping factor for Newton's method
    """
    hessian_type: str = "full"
    damping: float = 0.0

    def __post_init__(self):
        self.options['hessian_type'] = self.hessian_type
        self.options['damping'] = self.damping


# ============================================================================
# 1. CUDA Implementation
# ============================================================================

class MLPClassificationCUDA(SecondForwardBackwardBase):
    """
    CUDA-accelerated MLP classification implementation.
    
    This class wraps the compiled CUDA kernels for forward and backward passes.
    It provides high-performance computation on GPU.
    
    Args:
        i: Input dimension
        m: Hidden layer dimension
        o: Output dimension (number of classes)
    """
    
    def __init__(self, i: int, m: int, o: int):
        if not _is_cuda_available():
            raise RuntimeError(
                "CUDA extension is not available. "
                "Please compile the extension or use MLPClassificationTorch instead."
            )
        
        self.i = i
        self.m = m
        self.o = o
        self.initialized = False
        
        # Cached parameters and data
        self.w1 = None
        self.b1 = None
        self.w2 = None
        self.b2 = None
        self.x = None
        self.y = None
        
        # Cached intermediates from forward pass
        self.a1 = None
        self.z1 = None
        self.a2 = None
        self.sigma = None
        self.loss = None
    
    def initialize(
        self,
        params: Params,
        config: Optional[Config] = None,
    ) -> None:
        """Initialize the operator and validate parameters."""
        required_params = ['w1', 'b1', 'w2', 'b2']
        for name in required_params:
            if params.get(name) is None:
                raise ValueError(f"Missing required parameter: {name}")
        
        # Validate shapes
        w1 = params.get('w1').data
        b1 = params.get('b1').data
        w2 = params.get('w2').data
        b2 = params.get('b2').data
        
        if w1.shape != (self.m, self.i):
            raise ValueError(f"w1 shape mismatch: expected ({self.m}, {self.i}), got {w1.shape}")
        if b1.shape[0] != self.m:
            raise ValueError(f"b1 shape mismatch: expected ({self.m},) or ({self.m}, 1), got {b1.shape}")
        if w2.shape != (self.o, self.m):
            raise ValueError(f"w2 shape mismatch: expected ({self.o}, {self.m}), got {w2.shape}")
        if b2.shape[0] != self.o:
            raise ValueError(f"b2 shape mismatch: expected ({self.o},) or ({self.o}, 1), got {b2.shape}")
        
        # Ensure all parameters are on CUDA
        if not all([w1.is_cuda, b1.is_cuda, w2.is_cuda, b2.is_cuda]):
            raise ValueError("All parameters must be CUDA tensors")
        
        self.initialized = True
    
    def forward_only(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> ForwardResult:
        """Forward computation only (no gradients).
        
        If targets are None in data, only computes predictions without loss.
        """
        # Extract parameters
        w1 = params.get('w1').data
        b1 = params.get('b1').data
        w2 = params.get('w2').data
        b2 = params.get('b2').data
        x = data.get('inputs')  # (batch_size, i)
        y = data.get_optional('targets')  # (batch_size, o) or None
        
        # Ensure all are CUDA tensors
        if not x.is_cuda:
            x = x.cuda()
        
        # Check if we should compute loss
        compute_loss = y is not None
        
        # Create dummy labels if not computing loss
        if not compute_loss:
            y = torch.zeros((x.shape[0], w2.shape[0]), dtype=torch.float32, device=x.device)
        else:
            if not y.is_cuda:
                y = y.cuda()
        
        # Flatten biases to 1D if needed
        if b1.dim() == 2:
            b1 = b1.squeeze(1)
        if b2.dim() == 2:
            b2 = b2.squeeze(1)
        
        # Validate input dimensions (batch_size, features)
        assert x.dim() == 2, f"Input x must be 2D (batch_size, i), got shape {x.shape}"
        assert y.dim() == 2, f"Target y must be 2D (batch_size, o), got shape {y.shape}"
        
        # Lazy import to avoid triggering JIT at module load time
        from ..cuda import _wrapper
        
        # Call CUDA forward function (returns tuple) - C++ handles batch dimension
        sigma, loss, _, _, _ = _wrapper.mlp_classification_fwd(
            w1, b1, w2, b2, x, y,
            save_intermediates=False,
            compute_loss=compute_loss
        )
        
        return ForwardResult(
            outputs={
                'loss': loss if compute_loss else None,
                'predictions': sigma,  # (batch_size, o)
                'logits': None  # Not returned by CUDA forward
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
        # Extract parameters
        self.w1 = params.get('w1').data
        self.b1 = params.get('b1').data
        self.w2 = params.get('w2').data
        self.b2 = params.get('b2').data
        self.x = data.get('inputs')  # (batch_size, i)
        self.y = data.get('targets')  # (batch_size, o)
        
        # Ensure all are CUDA tensors
        if not self.x.is_cuda:
            self.x = self.x.cuda()
        if not self.y.is_cuda:
            self.y = self.y.cuda()
        
        # Flatten biases to 1D if needed
        if self.b1.dim() == 2:
            self.b1 = self.b1.squeeze(1)
        if self.b2.dim() == 2:
            self.b2 = self.b2.squeeze(1)
        
        # Validate input dimensions (batch_size, features)
        assert self.x.dim() == 2, f"Input x must be 2D (batch_size, i), got shape {self.x.shape}"
        assert self.y.dim() == 2, f"Target y must be 2D (batch_size, o), got shape {self.y.shape}"
        
        # Lazy import to avoid triggering JIT at module load time
        from ..cuda import _wrapper
        
        # Call CUDA forward function (returns tuple) - C++ handles batch dimension
        self.sigma, self.loss, self.a1, self.z1, self.a2 = _wrapper.mlp_classification_fwd(
            self.w1, self.b1, self.w2, self.b2,
            self.x, self.y,
            save_intermediates=True,
            compute_loss=True
        )
        
        # Call CUDA backward function (returns tuple)
        grad_w1, grad_b1, grad_w2, grad_b2 = _wrapper.mlp_classification_bwd(
            self.w1, self.b1, self.w2, self.b2,
            self.x, self.y,
            self.a1, self.z1, self.a2, self.sigma,
            accumulate_grad=False
        )
        
        # Build forward result (sigma/a2 are already (batch_size, o/m) from C++)
        forward_result = ForwardResult(
            outputs={
                'loss': self.loss,
                'predictions': self.sigma,  # (batch_size, o)
                'logits': self.a2  # (batch_size, o)
            },
            intermediates={
                'a1': self.a1,  # (batch_size, m)
                'z1': self.z1,  # (batch_size, m)
                'a2': self.a2,  # (batch_size, o)
                'sigma': self.sigma  # (batch_size, o)
            }
        )
        
        # Build gradient result (reshape to 2D if needed)
        grad_result = GradResult()
        grad_result.set('w1', grad_w1)
        grad_result.set('b1', grad_b1.unsqueeze(1))
        grad_result.set('w2', grad_w2)
        grad_result.set('b2', grad_b2.unsqueeze(1))
        
        return forward_result, grad_result
    
    def forward_backward2(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> tuple[ForwardResult, GradResult, HessResult]:
        """Forward + first-order + second-order backward computation."""
        # Extract parameters
        self.w1 = params.get('w1').data
        self.b1 = params.get('b1').data
        self.w2 = params.get('w2').data
        self.b2 = params.get('b2').data
        self.x = data.get('inputs')  # (batch_size, i)
        self.y = data.get('targets')  # (batch_size, o)
        
        # Ensure all are CUDA tensors
        if not self.x.is_cuda:
            self.x = self.x.cuda()
        if not self.y.is_cuda:
            self.y = self.y.cuda()
        
        # Flatten biases to 1D if needed
        if self.b1.dim() == 2:
            self.b1 = self.b1.squeeze(1)
        if self.b2.dim() == 2:
            self.b2 = self.b2.squeeze(1)
        
        # Validate input dimensions (batch_size, features)
        assert self.x.dim() == 2, f"Input x must be 2D (batch_size, i), got shape {self.x.shape}"
        assert self.y.dim() == 2, f"Target y must be 2D (batch_size, o), got shape {self.y.shape}"
        
        # Lazy import to avoid triggering JIT at module load time
        from ..cuda import _wrapper
        
        # Call CUDA forward function (returns tuple) - C++ handles batch dimension
        self.sigma, self.loss, self.a1, self.z1, self.a2 = _wrapper.mlp_classification_fwd(
            self.w1, self.b1, self.w2, self.b2,
            self.x, self.y,
            save_intermediates=True,
            compute_loss=True
        )
        
        # Call CUDA backward function with Hessian (returns tuple)
        grad_w1, grad_b1, grad_w2, grad_b2, hess_b2_full, hess_b1_full, hess_w2_full, hess_w1_full = _wrapper.mlp_classification_bwd2(
            self.w1, self.b1, self.w2, self.b2,
            self.x, self.y,
            self.a1, self.z1, self.a2, self.sigma,
            accumulate_grad=False
        )
        
        # Build forward result (sigma/a2 are already (batch_size, o/m) from C++)
        forward_result = ForwardResult(
            outputs={
                'loss': self.loss,
                'predictions': self.sigma,  # (batch_size, o)
                'logits': self.a2  # (batch_size, o)
            },
            intermediates={
                'a1': self.a1,  # (batch_size, m)
                'z1': self.z1,  # (batch_size, m)
                'a2': self.a2,  # (batch_size, o)
                'sigma': self.sigma  # (batch_size, o)
            }
        )
        
        # Build results
        grad_result = GradResult()
        grad_result.set(w1=grad_w1, b1=grad_b1.unsqueeze(1), w2=grad_w2, b2=grad_b2.unsqueeze(1))
        
        # Build Hessian result
        hess_result = HessResult()
        hess_result.set(w1=hess_w1_full, b1=hess_b1_full, w2=hess_w2_full, b2=hess_b2_full)
        
        return forward_result, grad_result, hess_result


# ============================================================================
# 2. PyTorch Manual Implementation
# ============================================================================

class MLPClassificationTorch(SecondForwardBackwardBase):
    """
    Pure PyTorch manual implementation of MLP classification.
    
    This class implements forward and backward passes using manually
    derived formulas. It serves as a reference implementation for
    validation against CUDA kernels.
    
    Args:
        i: Input dimension
        m: Hidden layer dimension
        o: Output dimension (number of classes)
    """
    
    def __init__(self, i: int, m: int, o: int):
        self.i = i
        self.m = m
        self.o = o
        self.initialized = False
        
        # Cache intermediate variables
        self.x = None
        self.y = None
        self.w1 = None
        self.b1 = None
        self.w2 = None
        self.b2 = None
        self.a1 = None
        self.z1 = None
        self.a2 = None
        self.sigma = None
        self.loss = None
    
    def initialize(
        self,
        params: Params,
        config: Optional[Config] = None,
    ) -> None:
        """Initialize and validate parameters."""
        required_params = ['w1', 'b1', 'w2', 'b2']
        for name in required_params:
            if params.get(name) is None:
                raise ValueError(f"Missing required parameter: {name}")
        
        self.initialized = True
    
    def forward_only(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> ForwardResult:
        """Forward computation only.
        
        If targets are None in data, only computes predictions without loss.
        """
        # Extract parameters and data
        w1 = params.get('w1').data  # (m, i)
        b1 = params.get('b1').data  # (m, 1)
        w2 = params.get('w2').data  # (o, m)
        b2 = params.get('b2').data  # (o, 1)
        x = data.get('inputs')      # (batch_size, i)
        y = data.get_optional('targets')     # (batch_size, o) or None
        
        # Validate input dimensions
        assert x.dim() == 2, f"Input x must be 2D (batch_size, i), got shape {x.shape}"
        
        batch_size = x.shape[0]
        compute_loss = y is not None
        
        if compute_loss:
            assert y.dim() == 2, f"Target y must be 2D (batch_size, o), got shape {y.shape}"
        
        # Forward computation: transpose x to (i, batch_size) for matrix multiplication
        x_t = x.t()  # (i, batch_size)
        
        a1 = torch.mm(w1, x_t) + b1  # (m, batch_size)
        z1 = torch.tanh(a1)          # (m, batch_size)
        a2 = torch.mm(w2, z1) + b2   # (o, batch_size)
        
        # Softmax + cross-entropy (along class dimension)
        z_max = torch.max(a2, dim=0, keepdim=True)[0]  # (1, batch_size)
        exp_z = torch.exp(a2 - z_max)
        sigma = exp_z / torch.sum(exp_z, dim=0, keepdim=True)  # (o, batch_size)
        
        # Compute loss only if targets provided
        if compute_loss:
            y_t = y.t()  # (o, batch_size)
            loss = -torch.sum(y_t * torch.log(sigma + 1e-10)) / batch_size
        else:
            loss = None
        
        return ForwardResult(
            outputs={'loss': loss, 'predictions': sigma.t(), 'logits': a2.t()},
            intermediates={'a1': a1.t(), 'z1': z1.t(), 'a2': a2.t(), 'sigma': sigma.t()}
        )
    
    def forward_backward1(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> tuple[ForwardResult, GradResult]:
        """Forward + first-order backward computation."""
        # Extract parameters and data
        self.w1 = params.get('w1').data  # (m, i)
        self.b1 = params.get('b1').data  # (m, 1)
        self.w2 = params.get('w2').data  # (o, m)
        self.b2 = params.get('b2').data  # (o, 1)
        self.x = data.get('inputs')      # (batch_size, i)
        self.y = data.get('targets')     # (batch_size, o)
        
        # Validate input dimensions
        assert self.x.dim() == 2, f"Input x must be 2D (batch_size, i), got shape {self.x.shape}"
        assert self.y.dim() == 2, f"Target y must be 2D (batch_size, o), got shape {self.y.shape}"
        
        batch_size = self.x.shape[0]
        
        # Forward computation: transpose x to (i, batch_size) for matrix multiplication
        x_t = self.x.t()  # (i, batch_size)
        y_t = self.y.t()  # (o, batch_size)
        
        self.a1 = torch.mm(self.w1, x_t) + self.b1  # (m, batch_size)
        self.z1 = torch.tanh(self.a1)               # (m, batch_size)
        self.a2 = torch.mm(self.w2, self.z1) + self.b2  # (o, batch_size)
        
        # Softmax + cross-entropy (along class dimension)
        z_max = torch.max(self.a2, dim=0, keepdim=True)[0]  # (1, batch_size)
        exp_z = torch.exp(self.a2 - z_max)
        self.sigma = exp_z / torch.sum(exp_z, dim=0, keepdim=True)  # (o, batch_size)
        self.loss = -torch.sum(y_t * torch.log(self.sigma + 1e-10)) / batch_size
        
        # Forward result
        fwd_result = ForwardResult(
            outputs={'loss': self.loss, 'predictions': self.sigma.t(), 'logits': self.a2.t()},
            intermediates={'a1': self.a1.t(), 'z1': self.z1.t(), 'a2': self.a2.t(), 'sigma': self.sigma.t()}
        )
        
        # Backward: elegant form (sigma - y)
        error = self.sigma - y_t  # (o, batch_size)
        
        # Second layer gradients (average over batch)
        grad_w2 = torch.mm(error, self.z1.t()) / batch_size  # (o, m)
        grad_b2 = torch.sum(error, dim=1, keepdim=True) / batch_size  # (o, 1)
        
        # First layer gradients (average over batch)
        s = torch.mm(self.w2.t(), error)  # (m, batch_size)
        d = 1 - self.z1 ** 2              # (m, batch_size)
        grad_a1 = s * d                   # (m, batch_size)
        grad_w1 = torch.mm(grad_a1, x_t.t()) / batch_size  # (m, i)
        grad_b1 = torch.sum(grad_a1, dim=1, keepdim=True) / batch_size  # (m, 1)
        
        # Gradient result
        grad_result = GradResult()
        grad_result.set('w1', grad_w1)
        grad_result.set('b1', grad_b1)
        grad_result.set('w2', grad_w2)
        grad_result.set('b2', grad_b2)
        
        return fwd_result, grad_result
    
    def forward_backward2(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> tuple[ForwardResult, GradResult, HessResult]:
        """Forward + first-order + second-order backward computation.
        
        Supports batch processing: computes Hessian of average loss over batch.
        H(θ) = (1/B) Σ_{b=1}^B H_b(θ)
        """
        # First compute forward and first-order
        fwd_result, grad_result = self.forward_backward1(params, data, config)
        
        # Get batch size and dimensions
        batch_size = self.x.shape[0]
        m, i = self.w1.shape
        o = self.w2.shape[0]
        device = self.x.device
        
        # Initialize accumulated Hessians (zero tensors)
        hess_b2_sum = torch.zeros(o, o, device=device)
        hess_b1_sum = torch.zeros(m, m, device=device)
        hess_w2_sum = torch.zeros(o*m, o*m, device=device)
        hess_w1_sum = torch.zeros(m*i, m*i, device=device)
        
        # Loop over batch samples to compute and accumulate Hessians
        for b in range(batch_size):
            # Extract data for sample b
            # self.sigma: (o, batch_size), self.z1: (m, batch_size)
            # self.x: (batch_size, i), self.y: (batch_size, o)
            sigma_b = self.sigma[:, b]  # (o,)
            z1_b = self.z1[:, b]        # (m,)
            x_b = self.x[b, :]          # (i,)
            y_b = self.y[b, :]          # (o,)
            
            # Hessian of softmax + cross-entropy: H = diag(σ) - σσ^T
            hess_a2_b = torch.diag(sigma_b) - torch.outer(sigma_b, sigma_b)
            
            # Hessian for b2 (same as a2)
            hess_b2_sum += hess_a2_b
            
            # Hessian for w2 (Kronecker product)
            zzT_b = torch.outer(z1_b, z1_b)
            hess_w2_b = torch.kron(hess_a2_b, zzT_b)
            hess_w2_sum += hess_w2_b
            
            # Backpropagate to first layer
            # Chain rule: d²L/da1² = D * (W2^T * H_a2 * W2) * D + second_term
            d_b = 1 - z1_b ** 2
            D_b = torch.diag(d_b)
            
            # First term
            w2T_h_w2_b = torch.mm(torch.mm(self.w2.t(), hess_a2_b), self.w2)
            first_term_b = torch.mm(torch.mm(D_b, w2T_h_w2_b), D_b)
            
            # Second term (from tanh's second derivative)
            error_b = sigma_b - y_b
            s_b = torch.mv(self.w2.t(), error_b)
            second_term_b = -2 * torch.diag(s_b * z1_b * d_b)
            
            hess_a1_b = first_term_b + second_term_b
            
            # Hessian for b1
            hess_b1_sum += hess_a1_b
            
            # Hessian for w1 (Kronecker product)
            xxT_b = torch.outer(x_b, x_b)
            hess_w1_b = torch.kron(hess_a1_b, xxT_b)
            hess_w1_sum += hess_w1_b
        
        # Average Hessians over batch
        hess_b2_avg = hess_b2_sum / batch_size
        hess_b1_avg = hess_b1_sum / batch_size
        hess_w2_avg = hess_w2_sum / batch_size
        hess_w1_avg = hess_w1_sum / batch_size
        
        # Hessian result
        hess_result = HessResult()
        hess_result.set(w1=hess_w1_avg, b1=hess_b1_avg, w2=hess_w2_avg, b2=hess_b2_avg)
        
        return fwd_result, grad_result, hess_result


# ============================================================================
# 3. PyTorch Autograd Implementation
# ============================================================================

class SimpleNetwork(nn.Module):
    """Simple two-layer network for autograd computation."""
    
    def __init__(self, i: int, m: int, o: int):
        super(SimpleNetwork, self).__init__()
        self.linear1 = nn.Linear(i, m, bias=True)
        self.linear2 = nn.Linear(m, o, bias=True)
    
    def forward(self, x):
        x = torch.tanh(self.linear1(x))
        x = self.linear2(x)
        return x


class MLPClassificationAutograd(SecondForwardBackwardBase):
    """
    PyTorch autograd-based MLP classification implementation.
    
    This class uses PyTorch's automatic differentiation to compute
    gradients and Hessians. It serves as the ground truth for validation.
    
    Args:
        i: Input dimension
        m: Hidden layer dimension
        o: Output dimension (number of classes)
    """
    
    def __init__(self, i: int, m: int, o: int):
        self.i = i
        self.m = m
        self.o = o
        self.model = SimpleNetwork(i, m, o)
        self.initialized = False
        self.x = None
        self.y = None
    
    def initialize(
        self,
        params: Params,
        config: Optional[Config] = None,
    ) -> None:
        """Initialize and validate parameters."""
        required_params = ['w1', 'b1', 'w2', 'b2']
        for name in required_params:
            if params.get(name) is None:
                raise ValueError(f"Missing required parameter: {name}")
        
        self.initialized = True
    
    def forward_only(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> ForwardResult:
        """Forward computation only.
        
        If targets are None in data, only computes predictions without loss.
        """
        # Extract parameters and data
        w1 = params.get('w1').data  # (m, i)
        b1 = params.get('b1').data  # (m, 1)
        w2 = params.get('w2').data  # (o, m)
        b2 = params.get('b2').data  # (o, 1)
        x = data.get('inputs')      # (batch_size, i)
        y = data.get_optional('targets')     # (batch_size, o) or None
        
        # Validate input dimensions
        assert x.dim() == 2, f"Input x must be 2D (batch_size, i), got shape {x.shape}"
        
        batch_size = x.shape[0]
        compute_loss = y is not None
        
        if compute_loss:
            assert y.dim() == 2, f"Target y must be 2D (batch_size, o), got shape {y.shape}"
        
        # Set parameters to model
        with torch.no_grad():
            self.model.linear1.weight.copy_(w1)
            self.model.linear1.bias.copy_(b1.squeeze())
            self.model.linear2.weight.copy_(w2)
            self.model.linear2.bias.copy_(b2.squeeze())
        
        # Forward pass: nn.Linear expects (batch_size, features)
        logits = self.model(x)  # (batch_size, o)
        sigma = torch.softmax(logits, dim=1)  # (batch_size, o)
        
        # Compute loss only if targets provided
        if compute_loss:
            loss = -torch.sum(y * torch.log(sigma + 1e-10)) / batch_size
        else:
            loss = None
        
        return ForwardResult(
            outputs={'loss': loss, 'predictions': sigma, 'logits': logits},
            intermediates={}
        )
    
    def forward_backward1(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> tuple[ForwardResult, GradResult]:
        """Forward + first-order backward computation."""
        # Save input and target
        self.x = data.get('inputs')      # (batch_size, i)
        self.y = data.get('targets')     # (batch_size, o)
        
        # Validate input dimensions
        assert self.x.dim() == 2, f"Input x must be 2D (batch_size, i), got shape {self.x.shape}"
        assert self.y.dim() == 2, f"Target y must be 2D (batch_size, o), got shape {self.y.shape}"
        
        batch_size = self.x.shape[0]
        
        # Extract parameters
        w1 = params.get('w1').data
        b1 = params.get('b1').data
        w2 = params.get('w2').data
        b2 = params.get('b2').data
        
        # Set parameters to model
        with torch.no_grad():
            self.model.linear1.weight.copy_(w1)
            self.model.linear1.bias.copy_(b1.squeeze())
            self.model.linear2.weight.copy_(w2)
            self.model.linear2.bias.copy_(b2.squeeze())
        
        # Clear gradients
        self.model.zero_grad()
        
        # Forward pass: nn.Linear expects (batch_size, features)
        logits = self.model(self.x)  # (batch_size, o)
        sigma = torch.softmax(logits, dim=1)  # (batch_size, o)
        loss = -torch.sum(self.y * torch.log(sigma + 1e-10)) / batch_size
        
        # Forward result
        fwd_result = ForwardResult(
            outputs={'loss': loss, 'predictions': sigma, 'logits': logits},
            intermediates={}
        )
        
        # Backward pass
        loss.backward()
        
        # Extract gradients
        grad_result = GradResult()
        grad_result.set('w1', self.model.linear1.weight.grad.clone())
        grad_result.set('b1', self.model.linear1.bias.grad.unsqueeze(1).clone())
        grad_result.set('w2', self.model.linear2.weight.grad.clone())
        grad_result.set('b2', self.model.linear2.bias.grad.unsqueeze(1).clone())
        
        # Clear gradients
        self.model.zero_grad()
        
        return fwd_result, grad_result
    
    def forward_backward2(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> tuple[ForwardResult, GradResult, HessResult]:
        """Forward + first-order + second-order backward computation."""
        # First compute forward and first-order
        fwd_result, grad_result = self.forward_backward1(params, data, config)
        
        # Compute second-order Hessian using autograd
        hess_result = self._compute_hessians_autograd()
        
        return fwd_result, grad_result, hess_result
    
    def _compute_hessians_autograd(self):
        """Compute Hessians using autograd.functional.hessian.
        
        Supports batch processing: computes Hessian of average loss over batch.
        """
        # Get current parameters
        w1 = self.model.linear1.weight
        b1 = self.model.linear1.bias
        w2 = self.model.linear2.weight
        b2 = self.model.linear2.bias
        
        m, i = w1.shape
        o, _ = w2.shape
        batch_size = self.x.shape[0]
        
        # Flatten parameters
        flat_w1 = w1.reshape(-1)
        flat_b1 = b1.reshape(-1)
        flat_w2 = w2.reshape(-1)
        flat_b2 = b2.reshape(-1)
        
        flat_params = torch.cat([flat_w1, flat_b1, flat_w2, flat_b2])
        
        sizes = {
            'w1': flat_w1.numel(),
            'b1': flat_b1.numel(),
            'w2': flat_w2.numel(),
            'b2': flat_b2.numel(),
        }
        
        x = self.x  # (batch_size, i)
        y = self.y  # (batch_size, o)
        
        def loss_from_flat(flat):
            """Compute average loss over batch from flattened parameters."""
            offset = 0
            
            num_w1 = sizes['w1']
            num_b1 = sizes['b1']
            num_w2 = sizes['w2']
            num_b2 = sizes['b2']
            
            w1_flat = flat[offset: offset + num_w1]
            offset += num_w1
            b1_flat = flat[offset: offset + num_b1]
            offset += num_b1
            w2_flat = flat[offset: offset + num_w2]
            offset += num_w2
            b2_flat = flat[offset: offset + num_b2]
            
            W1 = w1_flat.view(m, i)
            b1_vec = b1_flat.view(m)
            W2 = w2_flat.view(o, m)
            b2_vec = b2_flat.view(o)
            
            # Compute loss for each sample in batch and average
            total_loss = 0.0
            for b in range(batch_size):
                x_sample = x[b]  # (i,)
                y_sample = y[b]  # (o,)
                
                a1 = torch.nn.functional.linear(x_sample, W1, b1_vec)  # (m,)
                z1 = torch.tanh(a1)
                a2 = torch.nn.functional.linear(z1, W2, b2_vec)  # (o,)
                
                # Softmax + cross-entropy
                sigma = torch.softmax(a2, dim=0)
                loss_b = -torch.sum(y_sample * torch.log(sigma + 1e-10))
                total_loss += loss_b
            
            # Return average loss
            avg_loss = total_loss / batch_size
            return avg_loss
        
        # Compute full Hessian
        H_full = torch.autograd.functional.hessian(loss_from_flat, flat_params)
        
        # Extract diagonal blocks
        hess_result = HessResult()
        offset = 0
        for name in ['w1', 'b1', 'w2', 'b2']:
            n = sizes[name]
            H_block = H_full[offset: offset + n, offset: offset + n]
            hess_result.set(name, H_block)
            offset += n
        
        return hess_result


# ============================================================================
# Factory Function
# ============================================================================

def create_mlp_classification(
    i: int,
    m: int,
    o: int,
    backend: str = 'auto'
) -> SecondForwardBackwardBase:
    """
    Factory function to create MLP classification operator.
    
    Args:
        i: Input dimension
        m: Hidden layer dimension
        o: Output dimension (number of classes)
        backend: Backend to use ('cuda', 'torch', 'autograd', or 'auto')
            - 'cuda': Use CUDA implementation (requires compilation)
            - 'torch': Use pure PyTorch manual implementation
            - 'autograd': Use PyTorch autograd implementation
            - 'auto': Automatically select CUDA if available, else Torch
    
    Returns:
        An instance of SecondForwardBackwardBase
    """
    if backend == 'auto':
        backend = 'cuda' if _is_cuda_available() else 'torch'
    
    if backend == 'cuda':
        return MLPClassificationCUDA(i, m, o)
    elif backend == 'torch':
        return MLPClassificationTorch(i, m, o)
    elif backend == 'autograd':
        return MLPClassificationAutograd(i, m, o)
    else:
        raise ValueError(f"Unknown backend: {backend}. Choose 'cuda', 'torch', 'autograd', or 'auto'")


__all__ = [
    'MLPClassificationCUDA',
    'MLPClassificationTorch',
    'MLPClassificationAutograd',
    'create_mlp_classification',
]
