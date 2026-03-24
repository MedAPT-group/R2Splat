"""
ConicsMu operators for CT reconstruction.

This module provides CUDA and PyTorch autograd implementations of the
ConicsMu computation: from projected 3x3 covariance matrices, compute
the inverse of the blurred upper-left 2x2 sub-block (conics) and the
compensation factor (mu).

Mathematical notation (with blur):
    tilde{Sigma} = [a, b, c; b, d, e; c, e, f]   (symmetric 3x3)
    hat{Sigma}   = [a, b; b, d]                    (upper-left 2x2)
    hat{Sigma}_b = hat{Sigma} + eps * I_2           (blurred 2x2)
    tilde{Sigma}_b = tilde{Sigma} + eps * I_3       (blurred 3x3)
    D2b = det(hat{Sigma}_b)
    D3b = det(tilde{Sigma}_b)
    conics = hat{Sigma}_b^{-1}  stored as (u11, u12, u22)
    mu = sqrt(2*pi * D3b / D2b)

Supports:
- ConicsMuCUDA: CUDA-accelerated implementation
- ConicsMuAutograd: Pure PyTorch reference using torch.autograd

Author: Dy164
Date: 2026-02-10
"""

import math
import torch
from typing import Optional, Tuple
from .base import SecondForwardBackwardBase
from .data import Params, Data, Config, ForwardResult, GradResult, HessResult


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
# CUDA Implementation
# ============================================================================

class ConicsMuCUDA(SecondForwardBackwardBase):
    """
    CUDA-accelerated ConicsMu implementation.
    
    Computes conics and mu from projected 3x3 covariance matrices.
    Uses blur (eps * I) for numerical stability.
    
    This class computes:
    - Forward: conics (C, N, 3) and mus (C, N)
    - Backward1: gradient w.r.t. cov2d_3x3 (C, N, 6)
    - Backward2: not implemented (raises NotImplementedError)
    
    Expected params:
        - cov2d_3x3: (C, N, 6) projected 3x3 covariance in upper-triangular
                      storage (a, b, c, d, e, f)
    
    Expected config:
        - eps: float, blur epsilon (default 0.3)
    """
    
    def __init__(self):
        if not _is_cuda_available():
            raise RuntimeError(
                "CUDA extension not available. "
                "Please compile the CUDA extension first with 'pip install -e .'"
            )
        self.initialized = False
    
    def initialize(
        self,
        params: Params,
        config: Optional[Config] = None,
    ) -> None:
        """Initialize and validate parameters."""
        if params.get('cov2d_3x3') is None:
            raise ValueError("Missing required parameter: cov2d_3x3")
        
        cov2d_3x3 = params.get('cov2d_3x3').data
        
        # Must be 3D: (C, N, 6)
        if cov2d_3x3.dim() != 3 or cov2d_3x3.size(-1) != 6:
            raise ValueError(
                f"cov2d_3x3 must be 3D with shape (C, N, 6), got {cov2d_3x3.shape}"
            )
        
        # Ensure CUDA tensors
        if not cov2d_3x3.is_cuda:
            raise ValueError("cov2d_3x3 must be a CUDA tensor")
        
        self.initialized = True
    
    def forward_only(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> ForwardResult:
        """Forward computation only (no gradients).
        
        Args:
            params: Must contain 'cov2d_3x3' (C, N, 6)
            data: Not used
            config: Must contain 'eps' (float, default 0.3)
        
        Returns:
            ForwardResult with:
                outputs: {'conics': (C, N, 3), 'mus': (C, N)}
        """
        from ..cuda._wrapper import conics_mu_fwd
        
        cov2d_3x3 = params.get('cov2d_3x3').data
        eps = config.get('eps', 0.3) if config is not None else 0.3
        
        conics, mus = conics_mu_fwd(cov2d_3x3, eps)
        
        return ForwardResult(
            outputs={'conics': conics, 'mus': mus},
            intermediates={}
        )
    
    def forward_backward1(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> Tuple[ForwardResult, GradResult]:
        """Forward + first-order backward using CUDA kernels.
        
        Args:
            params: Must contain 'cov2d_3x3' (C, N, 6)
            data: Must contain:
                - v_conics: (C, N, 3) upstream gradient dL/d(conics)
                - v_mus: (C, N) upstream gradient dL/d(mus)
                If not provided, uses ones as default.
            config: Must contain 'eps' (float, default 0.3)
        
        Returns:
            ForwardResult with conics and mus
            GradResult with gradient for cov2d_3x3
        """
        from ..cuda._wrapper import conics_mu_fwd, conics_mu_bwd
        
        cov2d_3x3 = params.get('cov2d_3x3').data
        eps = config.get('eps', 0.3) if config is not None else 0.3
        
        # Forward pass
        conics, mus = conics_mu_fwd(cov2d_3x3, eps)
        
        # Get upstream gradients (default to ones if not provided)
        v_conics = data.get('v_conics', None)
        if v_conics is None:
            v_conics = torch.ones_like(conics)
        
        v_mus = data.get('v_mus', None)
        if v_mus is None:
            v_mus = torch.ones_like(mus)
        
        # Backward pass
        v_cov2d_3x3 = conics_mu_bwd(
            cov2d_3x3, eps, conics, mus, v_conics, v_mus
        )
        
        forward_result = ForwardResult(
            outputs={'conics': conics, 'mus': mus},
            intermediates={}
        )
        
        grad_result = GradResult(
            param_grads={'cov2d_3x3': v_cov2d_3x3}
        )
        
        return forward_result, grad_result
    
    def forward_backward2(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> Tuple[ForwardResult, GradResult, HessResult]:
        """Forward + second-order backward using CUDA kernels.
        
        Computes forward outputs, first-order gradient dL/d(cov2d_3x3),
        and second-order Hessians H_{sigma,sigma}, H_{sigma,means2d},
        H_{sigma,opacity}.
        
        Args:
            params: Must contain 'cov2d_3x3' (C, N, 6)
            data: Must contain:
                - v_conics: (C, N, 3)
                - v_mus: (C, N)
                - h_conics: (C, N, 6)
                - h_conics_mu: (C, N, 3)
                - h_mu_mu: (C, N)
                - h_means2d_conics: (C, N, 6) H_{means2d,conics} [2x3 row-major]
                - h_conics_opacity: (C, N, 3) H_{conics,opacity}
                - h_mu_means2d: (C, N, 2) H_{mu,means2d}
                - h_mu_opacity: (C, N) H_{mu,opacity}
            config: Must contain 'eps' (float, default 0.3)
        
        Returns:
            ForwardResult with conics and mus
            GradResult with gradient for cov2d_3x3
            HessResult with:
                - 'cov_cov': H_{sigma,sigma} upper-tri (C, N, 21)
                - 'cov_means2d': H_{sigma,means2d} (C, N, 12)
                - 'cov_opacity': H_{sigma,opacity} (C, N, 6)
        """
        from ..cuda._wrapper import conics_mu_fwd, conics_mu_bwd2
        
        cov2d_3x3 = params.get('cov2d_3x3').data
        eps = config.get('eps', 0.3) if config is not None else 0.3
        C, N = cov2d_3x3.shape[:2]
        
        # Forward pass
        conics, mus = conics_mu_fwd(cov2d_3x3, eps)
        
        # Get downstream first-order gradients
        v_conics = data.get('v_conics')
        v_mus = data.get('v_mus')
        if v_conics is None:
            v_conics = torch.zeros_like(conics)
        if v_mus is None:
            v_mus = torch.zeros_like(mus)
        
        # Get downstream Hessian blocks (default to zeros)
        def _get_or_zeros(key, shape):
            val = data.get(key)
            if val is None:
                return torch.zeros(shape, dtype=cov2d_3x3.dtype,
                                   device=cov2d_3x3.device)
            return val
        
        h_conics = _get_or_zeros('h_conics', (C, N, 6))
        h_conics_mu = _get_or_zeros('h_conics_mu', (C, N, 3))
        h_mu_mu = _get_or_zeros('h_mu_mu', (C, N,))
        h_means2d_conics = _get_or_zeros('h_means2d_conics', (C, N, 6))
        h_conics_opacity = _get_or_zeros('h_conics_opacity', (C, N, 3))
        h_mu_means2d = _get_or_zeros('h_mu_means2d', (C, N, 2))
        h_mu_opacity = _get_or_zeros('h_mu_opacity', (C, N,))
        
        # Second-order backward pass
        v_cov2d_3x3, h_cov_cov, h_cov_means2d, h_cov_opacity = conics_mu_bwd2(
            cov2d_3x3, eps, conics, mus,
            v_conics, v_mus,
            h_conics, h_conics_mu, h_mu_mu,
            h_means2d_conics, h_conics_opacity, h_mu_means2d, h_mu_opacity,
        )
        
        forward_result = ForwardResult(
            outputs={'conics': conics, 'mus': mus},
            intermediates={}
        )
        
        grad_result = GradResult(
            param_grads={'cov2d_3x3': v_cov2d_3x3}
        )
        
        hess_result = HessResult(
            param_hess={
                'cov_cov': h_cov_cov,
                'cov_means2d': h_cov_means2d,
                'cov_opacity': h_cov_opacity,
            }
        )
        
        return forward_result, grad_result, hess_result


# ============================================================================
# PyTorch Autograd Implementation
# ============================================================================

def _compute_conics_mu(cov2d_3x3: torch.Tensor, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute conics and mus from cov2d_3x3 (pure PyTorch implementation).
    
    This function is differentiable via PyTorch's autograd.
    
    Args:
        cov2d_3x3: (C, N, 6) upper-triangular storage (a, b, c, d, e, f)
        eps: blur epsilon
    
    Returns:
        conics: (C, N, 3) as (u11, u12, u22)
        mus: (C, N)
    """
    # Unpack upper-triangular elements
    a = cov2d_3x3[..., 0]  # Sigma[0,0]
    b = cov2d_3x3[..., 1]  # Sigma[0,1] = Sigma[1,0]
    c = cov2d_3x3[..., 2]  # Sigma[0,2] = Sigma[2,0]
    d = cov2d_3x3[..., 3]  # Sigma[1,1]
    e = cov2d_3x3[..., 4]  # Sigma[1,2] = Sigma[2,1]
    f = cov2d_3x3[..., 5]  # Sigma[2,2]
    
    # Apply blur
    ab = a + eps
    db = d + eps
    fb = f + eps
    
    # D2b = det(hat{Sigma}_b) = (a+eps)(d+eps) - b^2
    D2b = ab * db - b * b
    
    # Conics = hat{Sigma}_b^{-1}
    inv_D2b = 1.0 / D2b
    u11 = db * inv_D2b
    u12 = -b * inv_D2b
    u22 = ab * inv_D2b
    
    conics = torch.stack([u11, u12, u22], dim=-1)
    
    # D3b = det(tilde{Sigma}_b)
    # Using cofactor expansion along first row:
    # D3b = ab * (db*fb - e^2) - b * (b*fb - e*c) + c * (b*e - db*c)
    D3b = (ab * (db * fb - e * e)
           - b * (b * fb - e * c)
           + c * (b * e - db * c))
    
    # mu = sqrt(2*pi * D3b / D2b)
    ratio = D3b / D2b
    # Clamp to avoid sqrt of negative (degenerate covariance)
    ratio_clamped = torch.clamp(ratio, min=0.0)
    mus = torch.sqrt(2.0 * math.pi * ratio_clamped)
    
    return conics, mus


class ConicsMuAutograd(SecondForwardBackwardBase):
    """
    PyTorch autograd-based ConicsMu implementation.
    
    This is a reference implementation for validating the CUDA backward pass.
    Uses PyTorch's automatic differentiation - gradients are computed
    automatically without manual implementation.
    
    Expected params:
        - cov2d_3x3: (C, N, 6) projected 3x3 covariance in upper-triangular
                      storage (a, b, c, d, e, f)
    
    Expected config:
        - eps: float, blur epsilon (default 0.3)
    """
    
    def __init__(self):
        self.initialized = False
    
    def initialize(
        self,
        params: Params,
        config: Optional[Config] = None,
    ) -> None:
        """Initialize and validate parameters."""
        if params.get('cov2d_3x3') is None:
            raise ValueError("Missing required parameter: cov2d_3x3")
        
        cov2d_3x3 = params.get('cov2d_3x3').data
        
        # Must be 3D: (C, N, 6)
        if cov2d_3x3.dim() != 3 or cov2d_3x3.size(-1) != 6:
            raise ValueError(
                f"cov2d_3x3 must be 3D with shape (C, N, 6), got {cov2d_3x3.shape}"
            )
        
        self.initialized = True
    
    def forward_only(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> ForwardResult:
        """Forward computation only (no gradients).
        
        Args:
            params: Must contain 'cov2d_3x3' (C, N, 6)
            data: Not used
            config: Must contain 'eps' (float, default 0.3)
        
        Returns:
            ForwardResult with:
                outputs: {'conics': (C, N, 3), 'mus': (C, N)}
        """
        cov2d_3x3 = params.get('cov2d_3x3').data
        eps = config.get('eps', 0.3) if config is not None else 0.3
        
        with torch.no_grad():
            conics, mus = _compute_conics_mu(cov2d_3x3, eps)
        
        return ForwardResult(
            outputs={'conics': conics, 'mus': mus},
            intermediates={}
        )
    
    def forward_backward1(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> Tuple[ForwardResult, GradResult]:
        """Forward + first-order backward using torch.autograd.
        
        Args:
            params: Must contain 'cov2d_3x3' (C, N, 6)
            data: Must contain:
                - v_conics: (C, N, 3) upstream gradient dL/d(conics)
                - v_mus: (C, N) upstream gradient dL/d(mus)
                If not provided, uses ones as default.
            config: Must contain 'eps' (float, default 0.3)
        
        Returns:
            ForwardResult with conics and mus
            GradResult with gradient for cov2d_3x3
        """
        cov2d_3x3 = params.get('cov2d_3x3').data.requires_grad_(True)
        eps = config.get('eps', 0.3) if config is not None else 0.3
        
        # Forward pass - PyTorch will track all operations for autograd
        conics, mus = _compute_conics_mu(cov2d_3x3, eps)
        
        # Get upstream gradients (default to ones if not provided)
        v_conics = data.get('v_conics', None)
        if v_conics is None:
            v_conics = torch.ones_like(conics)
        
        v_mus = data.get('v_mus', None)
        if v_mus is None:
            v_mus = torch.ones_like(mus)
        
        # Backward pass - PyTorch automatically computes gradients
        torch.autograd.backward(
            tensors=[conics, mus],
            grad_tensors=[v_conics, v_mus],
        )
        
        forward_result = ForwardResult(
            outputs={'conics': conics.detach(), 'mus': mus.detach()},
            intermediates={}
        )
        
        grad_result = GradResult(
            param_grads={'cov2d_3x3': cov2d_3x3.grad.detach()}
        )
        
        return forward_result, grad_result
    
    def forward_backward2(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> Tuple[ForwardResult, GradResult, HessResult]:
        """Forward + second-order backward using torch.autograd (double backward).
        
        Computes full Hessian blocks via create_graph=True double backward.
        Used as ground-truth reference for validating the CUDA bwd2 kernel.
        
        Args:
            params: Must contain 'cov2d_3x3' (C, N, 6)
            data: Must contain:
                - v_conics: (C, N, 3)
                - v_mus: (C, N)
                - h_conics: (C, N, 6)
                - h_conics_mu: (C, N, 3)
                - h_mu_mu: (C, N)
                - h_means2d_conics: (C, N, 6) H_{means2d,conics} [2x3 row-major]
                - h_conics_opacity: (C, N, 3)
                - h_mu_means2d: (C, N, 2)
                - h_mu_opacity: (C, N)
            config: Must contain 'eps' (float, default 0.3)
        
        Returns:
            ForwardResult, GradResult, HessResult
        """
        cov2d_3x3_orig = params.get('cov2d_3x3').data
        eps = config.get('eps', 0.3) if config is not None else 0.3
        C, N = cov2d_3x3_orig.shape[:2]
        
        # Create leaf with requires_grad for double backward
        cov2d_3x3 = cov2d_3x3_orig.detach().clone().requires_grad_(True)
        
        # Forward pass (tracked by autograd)
        conics, mus = _compute_conics_mu(cov2d_3x3, eps)
        
        # Get downstream first-order gradients
        v_conics = data.get('v_conics')
        v_mus = data.get('v_mus')
        if v_conics is None:
            v_conics = torch.zeros_like(conics)
        if v_mus is None:
            v_mus = torch.zeros_like(mus)
        
        # First backward with create_graph=True to allow second backward
        # v_cov = J^T v_phi, as a function of cov2d_3x3 (graph retained)
        v_cov_graph, = torch.autograd.grad(
            outputs=[conics, mus],
            inputs=[cov2d_3x3],
            grad_outputs=[v_conics, v_mus],
            create_graph=True,
            retain_graph=True,
        )
        # v_cov_graph shape: (C, N, 6), still in the computation graph
        
        # Get downstream Hessian blocks
        def _get_or_zeros(key, shape):
            val = data.get(key)
            if val is None:
                return torch.zeros(shape, dtype=cov2d_3x3.dtype,
                                   device=cov2d_3x3.device)
            return val
        
        h_conics = _get_or_zeros('h_conics', (C, N, 6))
        h_conics_mu = _get_or_zeros('h_conics_mu', (C, N, 3))
        h_mu_mu = _get_or_zeros('h_mu_mu', (C, N,))
        h_means2d_conics = _get_or_zeros('h_means2d_conics', (C, N, 6))
        h_conics_opacity = _get_or_zeros('h_conics_opacity', (C, N, 3))
        h_mu_means2d = _get_or_zeros('h_mu_means2d', (C, N, 2))
        h_mu_opacity = _get_or_zeros('h_mu_opacity', (C, N,))
        
        # ================================================================
        # H_{sigma,sigma} = J^T H_{phi,phi} J + sum_k v_k Hess_k
        #
        # Term 2 (sum_k v_k Hess_k): obtained by differentiating v_cov_graph
        #   w.r.t. cov2d_3x3. Since v_cov_graph = J^T v_phi and v_phi is
        #   constant (detached), d(v_cov)/d(sigma) = sum_k v_k Hess_k.
        #
        # Term 1 (J^T H J): reconstruct the full H_{phi,phi} as a 4x4
        #   symmetric matrix, then for each column j of J (i.e. each
        #   e_j direction), compute J^T (H_{phi,phi} (J e_j)) via
        #   autograd. This gives column j of J^T H J.
        # ================================================================
        h_cov_cov = torch.zeros(C, N, 21, dtype=cov2d_3x3.dtype,
                                device=cov2d_3x3.device)
        h_cov_means2d = torch.zeros(C, N, 12, dtype=cov2d_3x3.dtype,
                                    device=cov2d_3x3.device)
        h_cov_opacity = torch.zeros(C, N, 6, dtype=cov2d_3x3.dtype,
                                    device=cov2d_3x3.device)
        
        # --- Term 2: sum_k v_k Hess_k (via second backward of v_cov_graph) ---
        term2 = torch.zeros(C, N, 6, 6, dtype=cov2d_3x3.dtype,
                            device=cov2d_3x3.device)
        for i in range(6):
            selector = torch.zeros_like(v_cov_graph)
            selector[..., i] = 1.0
            
            hess_row_i, = torch.autograd.grad(
                outputs=v_cov_graph,
                inputs=cov2d_3x3,
                grad_outputs=selector,
                retain_graph=True,
            )
            term2[..., i, :] = hess_row_i
        
        # --- Term 1: J^T H_{phi,phi} J ---
        # Compute J (the 4x6 Jacobian d(phi)/d(sigma)) explicitly via
        # 4 backward calls (one per output component k), then do the
        # matrix multiplication J^T @ H_{phi,phi} @ J per element.
        #
        # J[k, :] = d(phi_k)/d(sigma) = grad(phi_k.sum(), sigma)
        J = torch.zeros(C, N, 4, 6, dtype=cov2d_3x3.dtype,
                        device=cov2d_3x3.device)
        for k in range(4):
            # Select component k of phi
            if k < 3:
                grad_c = torch.zeros_like(conics)
                grad_c[..., k] = 1.0
                grad_m = torch.zeros_like(mus)
            else:
                grad_c = torch.zeros_like(conics)
                grad_m = torch.ones_like(mus)
            
            jt_col_k, = torch.autograd.grad(
                outputs=[conics, mus],
                inputs=[cov2d_3x3],
                grad_outputs=[grad_c, grad_m],
                retain_graph=True,
            )
            # jt_col_k shape: (C, N, 6) = J^T[:, k] = J[k, :]
            J[..., k, :] = jt_col_k
        
        # Now compute term1 = J^T H J per element.
        # H_phiphi is broadcast (constant for all elements in this test,
        # but we handle the general case where h_conics etc. vary per element).
        # Build per-element H_phiphi: (C, N, 4, 4)
        H_pp = torch.zeros(C, N, 4, 4, dtype=cov2d_3x3.dtype,
                           device=cov2d_3x3.device)
        H_pp[..., 0, 0] = h_conics[..., 0]
        H_pp[..., 0, 1] = H_pp[..., 1, 0] = h_conics[..., 1]
        H_pp[..., 0, 2] = H_pp[..., 2, 0] = h_conics[..., 2]
        H_pp[..., 1, 1] = h_conics[..., 3]
        H_pp[..., 1, 2] = H_pp[..., 2, 1] = h_conics[..., 4]
        H_pp[..., 2, 2] = h_conics[..., 5]
        H_pp[..., 0, 3] = H_pp[..., 3, 0] = h_conics_mu[..., 0]
        H_pp[..., 1, 3] = H_pp[..., 3, 1] = h_conics_mu[..., 1]
        H_pp[..., 2, 3] = H_pp[..., 3, 2] = h_conics_mu[..., 2]
        H_pp[..., 3, 3] = h_mu_mu
        
        # J: (C,N,4,6), H_pp: (C,N,4,4)
        # term1 = J^T @ H_pp @ J = (C,N,6,4) @ (C,N,4,4) @ (C,N,4,6)
        #       = (C,N,6,6)
        Jt = J.transpose(-1, -2)  # (C,N,6,4)
        HJ = torch.einsum('...ij,...jk->...ik', H_pp, J)  # (C,N,4,6)
        term1 = torch.einsum('...ij,...jk->...ik', Jt, HJ)  # (C,N,6,6)
        
        # Combine: H_sigma_sigma = term1 + term2
        full_H = term1.detach() + term2.detach()
        
        # Pack upper-triangular 21 elements
        for i in range(6):
            for j in range(i, 6):
                flat_idx = i * (11 - i) // 2 + j
                h_cov_cov[..., flat_idx] = full_H[..., i, j]
        
        # --- Cross-Hessians: H_{sigma,means2d} = J^T H_{phi,means2d} ---
        # h_means2d_conics layout (2x3 row-major):
        #   [H_{mx,u11}, H_{mx,u12}, H_{mx,u22}, H_{my,u11}, H_{my,u12}, H_{my,u22}]
        #   col=0 (mx): indices 0,1,2 for conics 0,1,2
        #   col=1 (my): indices 3,4,5 for conics 0,1,2
        for col in range(2):
            v_c = torch.zeros(C, N, 3, dtype=cov2d_3x3.dtype,
                              device=cov2d_3x3.device)
            v_m = torch.zeros(C, N, dtype=cov2d_3x3.dtype,
                              device=cov2d_3x3.device)
            v_c[..., 0] = h_means2d_conics[..., col * 3 + 0]
            v_c[..., 1] = h_means2d_conics[..., col * 3 + 1]
            v_c[..., 2] = h_means2d_conics[..., col * 3 + 2]
            v_m[...] = h_mu_means2d[..., col]
            
            jt_h_col, = torch.autograd.grad(
                outputs=[conics, mus],
                inputs=[cov2d_3x3],
                grad_outputs=[v_c, v_m],
                retain_graph=True,
            )
            for r in range(6):
                h_cov_means2d[..., r * 2 + col] = jt_h_col[..., r]
        
        # --- Cross-Hessians: H_{sigma,opacity} = J^T H_{phi,opacity} ---
        v_c_op = h_conics_opacity
        v_m_op = h_mu_opacity
        
        jt_h_op, = torch.autograd.grad(
            outputs=[conics, mus],
            inputs=[cov2d_3x3],
            grad_outputs=[v_c_op, v_m_op],
            retain_graph=True,
        )
        h_cov_opacity[...] = jt_h_op.detach()
        
        forward_result = ForwardResult(
            outputs={'conics': conics.detach(), 'mus': mus.detach()},
            intermediates={}
        )
        
        grad_result = GradResult(
            param_grads={'cov2d_3x3': v_cov_graph.detach()}
        )
        
        hess_result = HessResult(
            param_hess={
                'cov_cov': h_cov_cov.detach(),
                'cov_means2d': h_cov_means2d.detach(),
                'cov_opacity': h_cov_opacity.detach(),
            }
        )
        
        return forward_result, grad_result, hess_result


# ============================================================================
# Factory Function
# ============================================================================

def create_conics_mu(backend: str = "cuda") -> SecondForwardBackwardBase:
    """Factory function to create ConicsMu operator.
    
    Args:
        backend: "cuda" for CUDA implementation, "autograd" for PyTorch reference
    
    Returns:
        ConicsMu instance
    """
    if backend == "cuda":
        return ConicsMuCUDA()
    elif backend == "autograd":
        return ConicsMuAutograd()
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'cuda' or 'autograd'.")


__all__ = [
    "ConicsMuCUDA",
    "ConicsMuAutograd",
    "create_conics_mu",
]
