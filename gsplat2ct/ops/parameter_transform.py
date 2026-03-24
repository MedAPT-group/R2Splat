"""
ParameterTransform operators for CT reconstruction.

This module provides CUDA and PyTorch autograd implementations of the
parameter transform: from unit quaternions and raw scale parameters,
compute the 3D world-space covariance matrix (upper-triangular storage).

Mathematical pipeline:
    sig_i = sigmoid(so_i)
    s_i = sig_i * (scale_max - scale_min) + scale_min
    R = quat_to_rotation_matrix(q)       (unit quaternion, ||q||=1)
    M = R * diag(s)
    C = M * M^T = R * diag(s^2) * R^T
    covars = (C00, C01, C02, C11, C12, C22)  upper-triangular storage

Input parameters (theta):
    quats [N, 4]     -- unit quaternion (w, x, y, z)
    scale_ori [N, 3] -- raw scale (before sigmoid activation)

Output parameters (phi):
    covars [N, 6]    -- upper-triangular covariance

External parameters (psi) for cross-Hessians:
    means [N, 3]     -- Gaussian center positions
    opacity [N]      -- Gaussian opacity

Supports:
- ParameterTransformCUDA: CUDA-accelerated implementation (fwd/bwd/bwd2)
- ParameterTransformAutograd: Pure PyTorch reference using torch.autograd

Author: Dy164
Date: 2026-02-20
"""

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
# Pure PyTorch Forward Implementation (shared by Autograd class)
# ============================================================================

def _quat_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternion to rotation matrix (differentiable).

    Args:
        q: [..., 4] quaternion (w, x, y, z), ||q||=1

    Returns:
        R: [..., 3, 3] rotation matrix
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = torch.stack([
        1 - 2 * (y2 + z2), 2 * (xy - wz),     2 * (xz + wy),
        2 * (xy + wz),     1 - 2 * (x2 + z2), 2 * (yz - wx),
        2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (x2 + y2),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)

    return R


def _compute_covars(
    quats: torch.Tensor,
    scale_ori: torch.Tensor,
    scale_min: float,
    scale_max: float,
) -> torch.Tensor:
    """Compute covars from (quats, scale_ori) — pure PyTorch, differentiable.

    Args:
        quats: Unit quaternions (N, 4) as (w, x, y, z)
        scale_ori: Raw scale parameters (N, 3)
        scale_min: Minimum scale after activation
        scale_max: Maximum scale after activation

    Returns:
        covars: (N, 6) upper-triangular (C00,C01,C02,C11,C12,C22)
    """
    # Sigmoid activation
    sig = torch.sigmoid(scale_ori)
    s = sig * (scale_max - scale_min) + scale_min  # [N, 3]

    # Rotation matrix
    R = _quat_to_rotation_matrix(quats)  # [N, 3, 3]

    # M = R * diag(s)
    S = torch.diag_embed(s)  # [N, 3, 3]
    M = R @ S  # [N, 3, 3]

    # C = M * M^T
    C = M @ M.transpose(-1, -2)  # [N, 3, 3]

    # Upper-triangular storage
    covars = torch.stack([
        C[..., 0, 0], C[..., 0, 1], C[..., 0, 2],
        C[..., 1, 1], C[..., 1, 2], C[..., 2, 2],
    ], dim=-1)  # [N, 6]

    return covars


# ============================================================================
# CUDA Implementation
# ============================================================================

class ParameterTransformCUDA(SecondForwardBackwardBase):
    """
    CUDA-accelerated ParameterTransform implementation.

    Computes covars from (quats, scale_ori) using CUDA kernels for
    forward, first-order backward, and second-order backward passes.

    Expected params:
        - quats: (N, 4) unit quaternions (w, x, y, z), ||q||=1
        - scale_ori: (N, 3) raw scale parameters (before sigmoid)

    Expected config:
        - scale_min: float, minimum scale after activation
        - scale_max: float, maximum scale after activation
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
        if params.get('quats') is None:
            raise ValueError("Missing required parameter: quats")
        if params.get('scale_ori') is None:
            raise ValueError("Missing required parameter: scale_ori")

        quats = params.get('quats').data
        scale_ori = params.get('scale_ori').data

        # Must be 2D: (N, 4) and (N, 3)
        if quats.dim() != 2 or quats.size(-1) != 4:
            raise ValueError(
                f"quats must be 2D with shape (N, 4), got {quats.shape}"
            )
        if scale_ori.dim() != 2 or scale_ori.size(-1) != 3:
            raise ValueError(
                f"scale_ori must be 2D with shape (N, 3), got {scale_ori.shape}"
            )
        if quats.size(0) != scale_ori.size(0):
            raise ValueError(
                f"quats N={quats.size(0)} != scale_ori N={scale_ori.size(0)}"
            )

        # Ensure CUDA tensors
        if not quats.is_cuda:
            raise ValueError("quats must be a CUDA tensor")
        if not scale_ori.is_cuda:
            raise ValueError("scale_ori must be a CUDA tensor")

        self.initialized = True

    def forward_only(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> ForwardResult:
        """Forward computation only (no gradients).

        Args:
            params: Must contain 'quats' (N, 4) and 'scale_ori' (N, 3)
            data: Not used
            config: Must contain 'scale_min' and 'scale_max'

        Returns:
            ForwardResult with:
                outputs: {'covars': (N, 6)}
        """
        from ..cuda._wrapper import parameter_transform_fwd

        quats = params.get('quats').data
        scale_ori = params.get('scale_ori').data
        scale_min = config.get('scale_min', 0.01) if config is not None else 0.01
        scale_max = config.get('scale_max', 2.0) if config is not None else 2.0

        covars, max_scales = parameter_transform_fwd(quats, scale_ori, scale_min, scale_max)

        return ForwardResult(
            outputs={'covars': covars, 'max_scales': max_scales},
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
            params: Must contain 'quats' (N, 4) and 'scale_ori' (N, 3)
            data: Must contain:
                - v_covars: (N, 6) upstream gradient dL/d(covars)
                If not provided, uses ones as default.
            config: Must contain 'scale_min' and 'scale_max'

        Returns:
            ForwardResult with covars
            GradResult with gradients for quats and scale_ori
        """
        from ..cuda._wrapper import parameter_transform_fwd, parameter_transform_bwd

        quats = params.get('quats').data
        scale_ori = params.get('scale_ori').data
        scale_min = config.get('scale_min', 0.01) if config is not None else 0.01
        scale_max = config.get('scale_max', 2.0) if config is not None else 2.0

        # Forward pass
        covars, max_scales = parameter_transform_fwd(quats, scale_ori, scale_min, scale_max)

        # Get upstream gradients (default to ones if not provided)
        v_covars = data.get('v_covars', None)
        if v_covars is None:
            v_covars = torch.ones_like(covars)

        # Backward pass
        v_quats, v_scale_ori = parameter_transform_bwd(
            quats, scale_ori, scale_min, scale_max, v_covars
        )

        forward_result = ForwardResult(
            outputs={'covars': covars, 'max_scales': max_scales},
            intermediates={}
        )

        grad_result = GradResult(
            param_grads={
                'quats': v_quats,
                'scale_ori': v_scale_ori,
            }
        )

        return forward_result, grad_result

    def forward_backward2(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> Tuple[ForwardResult, GradResult, HessResult]:
        """Forward + second-order backward using CUDA kernels.

        Computes forward outputs, first-order gradients, and second-order
        Hessians w.r.t. (quats, scale_ori), plus cross-Hessians with
        means and opacity.

        Args:
            params: Must contain 'quats' (N, 4) and 'scale_ori' (N, 3)
            data: Must contain:
                - v_covars: (N, 6) first-order gradient dL/d(covars)
                - h_covars: (N, 21) H_{c,c} upper-tri of 6x6
                - h_covars_means: (N, 18) H_{c,mean} 6x3 row-major
                - h_covars_opacity: (N, 6) H_{c,opacity}
            config: Must contain 'scale_min' and 'scale_max'

        Returns:
            ForwardResult with covars
            GradResult with gradients for quats and scale_ori
            HessResult with:
                param_hess:
                    - 'quats': H_{q,q} upper-tri (N, 10)
                    - 'scale_ori': H_{s,s} diagonal (N, 3)
                cross_hess:
                    - 'quats_scale_ori': H_{q,s} (N, 12)
                    - 'quats_means': H_{q,mean} (N, 12)
                    - 'quats_opacity': H_{q,opacity} (N, 4)
                    - 'scale_ori_means': H_{s,mean} (N, 9)
                    - 'scale_ori_opacity': H_{s,opacity} (N, 3)
        """
        from ..cuda._wrapper import parameter_transform_fwd, parameter_transform_bwd2

        quats = params.get('quats').data
        scale_ori = params.get('scale_ori').data
        scale_min = config.get('scale_min', 0.01) if config is not None else 0.01
        scale_max = config.get('scale_max', 2.0) if config is not None else 2.0
        N = quats.size(0)

        # Forward pass
        covars, max_scales = parameter_transform_fwd(quats, scale_ori, scale_min, scale_max)

        # Get downstream first-order gradients
        v_covars = data.get('v_covars')
        if v_covars is None:
            v_covars = torch.zeros_like(covars)

        # Get downstream Hessian blocks (default to zeros)
        def _get_or_zeros(key, shape):
            val = data.get(key)
            if val is None:
                return torch.zeros(shape, dtype=quats.dtype, device=quats.device)
            return val

        h_covars = _get_or_zeros('h_covars', (N, 21))
        h_covars_means = _get_or_zeros('h_covars_means', (N, 18))
        h_covars_opacity = _get_or_zeros('h_covars_opacity', (N, 6))

        # Second-order backward pass
        (v_quats, v_scale_ori,
         h_quats, h_scale_ori, h_quats_scale_ori,
         h_quats_means, h_quats_opacity,
         h_scale_ori_means, h_scale_ori_opacity
        ) = parameter_transform_bwd2(
            quats, scale_ori, scale_min, scale_max,
            v_covars, h_covars, h_covars_means, h_covars_opacity,
        )

        forward_result = ForwardResult(
            outputs={'covars': covars, 'max_scales': max_scales},
            intermediates={}
        )

        grad_result = GradResult(
            param_grads={
                'quats': v_quats,
                'scale_ori': v_scale_ori,
            }
        )

        hess_result = HessResult(
            param_hess={
                'quats': h_quats,
                'scale_ori': h_scale_ori,
            },
            cross_hess={
                'quats_scale_ori': h_quats_scale_ori,
                'quats_means': h_quats_means,
                'quats_opacity': h_quats_opacity,
                'scale_ori_means': h_scale_ori_means,
                'scale_ori_opacity': h_scale_ori_opacity,
            }
        )

        return forward_result, grad_result, hess_result


# ============================================================================
# PyTorch Autograd Implementation
# ============================================================================

class ParameterTransformAutograd(SecondForwardBackwardBase):
    """
    PyTorch autograd-based ParameterTransform implementation.

    This is a reference implementation for validating the CUDA backward passes.
    Uses PyTorch's automatic differentiation — gradients are computed
    automatically without manual implementation.

    Expected params:
        - quats: (N, 4) unit quaternions (w, x, y, z)
        - scale_ori: (N, 3) raw scale parameters

    Expected config:
        - scale_min: float
        - scale_max: float
    """

    def __init__(self):
        self.initialized = False

    def initialize(
        self,
        params: Params,
        config: Optional[Config] = None,
    ) -> None:
        """Initialize and validate parameters."""
        if params.get('quats') is None:
            raise ValueError("Missing required parameter: quats")
        if params.get('scale_ori') is None:
            raise ValueError("Missing required parameter: scale_ori")

        quats = params.get('quats').data
        scale_ori = params.get('scale_ori').data

        if quats.dim() != 2 or quats.size(-1) != 4:
            raise ValueError(
                f"quats must be 2D with shape (N, 4), got {quats.shape}"
            )
        if scale_ori.dim() != 2 or scale_ori.size(-1) != 3:
            raise ValueError(
                f"scale_ori must be 2D with shape (N, 3), got {scale_ori.shape}"
            )

        self.initialized = True

    def forward_only(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> ForwardResult:
        """Forward computation only (no gradients).

        Returns:
            ForwardResult with:
                outputs: {'covars': (N, 6)}
        """
        quats = params.get('quats').data
        scale_ori = params.get('scale_ori').data
        scale_min = config.get('scale_min', 0.01) if config is not None else 0.01
        scale_max = config.get('scale_max', 2.0) if config is not None else 2.0

        with torch.no_grad():
            covars = _compute_covars(quats, scale_ori, scale_min, scale_max)

        return ForwardResult(
            outputs={'covars': covars},
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
            params: Must contain 'quats' (N, 4) and 'scale_ori' (N, 3)
            data: Must contain:
                - v_covars: (N, 6) upstream gradient dL/d(covars)
                If not provided, uses ones as default.
            config: Must contain 'scale_min' and 'scale_max'

        Returns:
            ForwardResult with covars
            GradResult with gradients for quats and scale_ori
        """
        quats = params.get('quats').data.detach().clone().requires_grad_(True)
        scale_ori = params.get('scale_ori').data.detach().clone().requires_grad_(True)
        scale_min = config.get('scale_min', 0.01) if config is not None else 0.01
        scale_max = config.get('scale_max', 2.0) if config is not None else 2.0

        # Forward pass — PyTorch tracks all operations for autograd
        covars = _compute_covars(quats, scale_ori, scale_min, scale_max)

        # Get upstream gradients (default to ones if not provided)
        v_covars = data.get('v_covars', None)
        if v_covars is None:
            v_covars = torch.ones_like(covars)

        # Backward pass — PyTorch automatically computes gradients
        torch.autograd.backward(
            tensors=[covars],
            grad_tensors=[v_covars],
        )

        forward_result = ForwardResult(
            outputs={'covars': covars.detach()},
            intermediates={}
        )

        grad_result = GradResult(
            param_grads={
                'quats': quats.grad.detach(),
                'scale_ori': scale_ori.grad.detach(),
            }
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
            params: Must contain 'quats' (N, 4) and 'scale_ori' (N, 3)
            data: Must contain:
                - v_covars: (N, 6) first-order gradient dL/d(covars)
                - h_covars: (N, 21) H_{c,c} upper-tri of 6x6
                - h_covars_means: (N, 18) H_{c,mean} 6x3 row-major
                - h_covars_opacity: (N, 6) H_{c,opacity}
            config: Must contain 'scale_min' and 'scale_max'

        Returns:
            ForwardResult, GradResult, HessResult
        """
        quats_orig = params.get('quats').data
        scale_ori_orig = params.get('scale_ori').data
        scale_min = config.get('scale_min', 0.01) if config is not None else 0.01
        scale_max = config.get('scale_max', 2.0) if config is not None else 2.0
        N = quats_orig.size(0)
        device = quats_orig.device
        dtype = quats_orig.dtype

        # Create leaf tensors with requires_grad for double backward
        quats = quats_orig.detach().clone().requires_grad_(True)
        scale_ori = scale_ori_orig.detach().clone().requires_grad_(True)

        # Forward pass (tracked by autograd)
        covars = _compute_covars(quats, scale_ori, scale_min, scale_max)

        # Get downstream first-order gradients
        v_covars = data.get('v_covars')
        if v_covars is None:
            v_covars = torch.zeros_like(covars)

        # First backward with create_graph=True to allow second backward
        # v_theta = J^T v_covars, as a function of (quats, scale_ori)
        v_q_graph, v_so_graph = torch.autograd.grad(
            outputs=[covars],
            inputs=[quats, scale_ori],
            grad_outputs=[v_covars],
            create_graph=True,
            retain_graph=True,
        )
        # v_q_graph: (N, 4), v_so_graph: (N, 3) — still in the computation graph

        # Get downstream Hessian blocks (default to zeros)
        def _get_or_zeros(key, shape):
            val = data.get(key)
            if val is None:
                return torch.zeros(shape, dtype=dtype, device=device)
            return val

        h_covars = _get_or_zeros('h_covars', (N, 21))
        h_covars_means = _get_or_zeros('h_covars_means', (N, 18))
        h_covars_opacity = _get_or_zeros('h_covars_opacity', (N, 6))

        # ================================================================
        # H_{theta,theta} = J^T H_{cc} J + sum_k v_k Hess_k
        #
        # Term 2 (curvature): obtained by differentiating v_theta_graph
        #   w.r.t. theta. Since v_theta = J^T v and v is constant,
        #   d(v_theta)/d(theta) = sum_k v_k Hess_k
        #
        # Term 1 (Gauss-Newton): J^T H_{cc} J, computed via explicit
        #   Jacobian J [6 x 7] and matrix multiplication.
        # ================================================================

        # --- Term 2: curvature via second backward ---
        # H_{q,q}: d(v_q)/d(q) [N, 4, 4]
        term2_qq = torch.zeros(N, 4, 4, dtype=dtype, device=device)
        for a in range(4):
            selector = torch.zeros_like(v_q_graph)
            selector[:, a] = 1.0
            grads = torch.autograd.grad(
                v_q_graph, quats, grad_outputs=selector,
                retain_graph=True,
            )[0]  # (N, 4)
            term2_qq[:, a, :] = grads

        # H_{so,so}: d(v_so)/d(so) [N, 3, 3]
        term2_ss = torch.zeros(N, 3, 3, dtype=dtype, device=device)
        for a in range(3):
            selector = torch.zeros_like(v_so_graph)
            selector[:, a] = 1.0
            grads = torch.autograd.grad(
                v_so_graph, scale_ori, grad_outputs=selector,
                retain_graph=True,
            )[0]  # (N, 3)
            term2_ss[:, a, :] = grads

        # H_{q,so}: d(v_q)/d(so) [N, 4, 3]
        term2_qs = torch.zeros(N, 4, 3, dtype=dtype, device=device)
        for a in range(4):
            selector = torch.zeros_like(v_q_graph)
            selector[:, a] = 1.0
            grads = torch.autograd.grad(
                v_q_graph, scale_ori, grad_outputs=selector,
                retain_graph=True,
            )[0]  # (N, 3)
            term2_qs[:, a, :] = grads

        # --- Term 1: Gauss-Newton J^T H_{cc} J ---
        # Compute Jacobian J [N, 6, 7]: d(covars)/d(theta)
        # theta = (q_0..q_3, so_0..so_2) concatenated as 7-vector
        J = torch.zeros(N, 6, 7, dtype=dtype, device=device)
        for k in range(6):
            grad_c = torch.zeros_like(covars)
            grad_c[:, k] = 1.0
            j_q, j_so = torch.autograd.grad(
                outputs=[covars],
                inputs=[quats, scale_ori],
                grad_outputs=[grad_c],
                retain_graph=True,
            )
            # j_q: (N, 4), j_so: (N, 3)
            J[:, k, :4] = j_q
            J[:, k, 4:] = j_so

        # Reconstruct full H_{cc} 6x6 symmetric from upper-tri 21
        H_cc = torch.zeros(N, 6, 6, dtype=dtype, device=device)
        idx = 0
        for i in range(6):
            for j in range(i, 6):
                H_cc[:, i, j] = h_covars[:, idx]
                H_cc[:, j, i] = h_covars[:, idx]
                idx += 1

        # term1 = J^T @ H_cc @ J  [N, 7, 7]
        Jt = J.transpose(-1, -2)  # [N, 7, 6]
        HJ = torch.einsum('...ij,...jk->...ik', H_cc, J)  # [N, 6, 7]
        term1 = torch.einsum('...ij,...jk->...ik', Jt, HJ)  # [N, 7, 7]

        # Combine term1 + term2 for each block
        # term1 is [N, 7, 7], blocks: q-q [0:4,0:4], q-so [0:4,4:7], so-so [4:7,4:7]
        term1_det = term1.detach()

        full_qq = term1_det[:, :4, :4] + term2_qq.detach()
        full_ss = term1_det[:, 4:, 4:] + term2_ss.detach()
        full_qs = term1_det[:, :4, 4:] + term2_qs.detach()

        # Pack h_quats: upper-tri of 4x4 -> 10 elements
        h_quats = torch.zeros(N, 10, dtype=dtype, device=device)
        idx = 0
        for i in range(4):
            for j in range(i, 4):
                h_quats[:, idx] = full_qq[:, i, j]
                idx += 1

        # h_scale_ori: diagonal of 3x3
        h_scale_ori = torch.stack([full_ss[:, i, i] for i in range(3)], dim=-1)

        # h_quats_scale_ori: 4x3 row-major -> 12 elements
        h_quats_scale_ori = full_qs.reshape(N, 12)

        # ================================================================
        # Cross-Hessians: H_{theta,psi} = J^T @ H_{covar,psi}
        # ================================================================

        # --- theta-mean cross: J^T @ h_covars_means [N, 7, 3] ---
        # h_covars_means is [N, 18] = 6x3 row-major
        H_cm = h_covars_means.reshape(N, 6, 3)
        JtHcm = torch.einsum('...ij,...jk->...ik', Jt, H_cm)  # [N, 7, 3]

        h_quats_means = JtHcm[:, :4, :].reshape(N, 12)     # 4x3 -> 12
        h_scale_ori_means = JtHcm[:, 4:, :].reshape(N, 9)  # 3x3 -> 9

        # --- theta-opacity cross: J^T @ h_covars_opacity [N, 7, 1] ---
        # h_covars_opacity is [N, 6]
        H_co = h_covars_opacity.unsqueeze(-1)  # [N, 6, 1]
        JtHco = torch.einsum('...ij,...jk->...ik', Jt, H_co).squeeze(-1)  # [N, 7]

        h_quats_opacity = JtHco[:, :4]      # [N, 4]
        h_scale_ori_opacity = JtHco[:, 4:]  # [N, 3]

        # ================================================================
        # Assemble results
        # ================================================================
        forward_result = ForwardResult(
            outputs={'covars': covars.detach()},
            intermediates={}
        )

        grad_result = GradResult(
            param_grads={
                'quats': v_q_graph.detach(),
                'scale_ori': v_so_graph.detach(),
            }
        )

        hess_result = HessResult(
            param_hess={
                'quats': h_quats,
                'scale_ori': h_scale_ori,
            },
            cross_hess={
                'quats_scale_ori': h_quats_scale_ori,
                'quats_means': h_quats_means,
                'quats_opacity': h_quats_opacity,
                'scale_ori_means': h_scale_ori_means,
                'scale_ori_opacity': h_scale_ori_opacity,
            }
        )

        return forward_result, grad_result, hess_result


# ============================================================================
# Factory Function
# ============================================================================

def create_parameter_transform(backend: str = "cuda") -> SecondForwardBackwardBase:
    """Factory function to create ParameterTransform operator.

    Args:
        backend: "cuda" for CUDA implementation, "autograd" for PyTorch reference

    Returns:
        ParameterTransform instance
    """
    if backend == "cuda":
        return ParameterTransformCUDA()
    elif backend == "autograd":
        return ParameterTransformAutograd()
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'cuda' or 'autograd'.")


__all__ = [
    "ParameterTransformCUDA",
    "ParameterTransformAutograd",
    "create_parameter_transform",
    "_compute_covars",
]
