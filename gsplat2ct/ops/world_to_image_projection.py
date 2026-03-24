"""
WorldToImageProjection operators for CT reconstruction.

This module provides CUDA and PyTorch autograd implementations of the
fused world-to-image projection: from world-space Gaussian means and
covariances, compute projected 2D means and 3x3 projected covariances.

Pipeline:
    F1. Camera transform:    p_c = R * p + t
    F2. Perspective project:  means2d = (fx * x / z + cx, fy * y / z + cy)
    F3. FOV hard clamp:       tx = z * clamp(x/z, lim), ty = z * clamp(y/z, lim)
    F4. Build J (3x3):        J = [[fx/z, 0, -fx*tx/z^2],
                                    [0, fy/z, -fy*ty/z^2],
                                    [tx/l, ty/l, z/l]]
                               where l = sqrt(tx^2 + ty^2 + z^2)
    F5. Camera covariance:    Sigma_c = R * Sigma * R^T
    F6. Projected covariance: cov2d_3x3 = J * Sigma_c * J^T

Upper-triangular storage for 3x3 symmetric matrix:
    (a, b, c, d, e, f) = (S00, S01, S02, S11, S12, S22)

Supports:
- WorldToImageProjectionCUDA: CUDA-accelerated implementation
- WorldToImageProjectionAutograd: Pure PyTorch reference using torch.autograd

Author: Dy164
Date: 2026-02-19
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
# Pure PyTorch Forward Computation (shared by CUDA and Autograd)
# ============================================================================

def _compute_world_to_image_projection(
    means: torch.Tensor,
    covars: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    dim1: int,
    dim0: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute world-to-image projection (pure PyTorch implementation).

    This function is differentiable via PyTorch's autograd. It replicates
    the CUDA forward kernel exactly.

    Args:
        means: World-space Gaussian centers (N, 3), shared across cameras
        covars: World-space covariance upper-tri (N, 6), shared across cameras
        viewmats: Camera extrinsic matrices (C, 4, 4) row-major
        Ks: Camera intrinsic matrices (C, 3, 3) row-major
        dim1: Image width in pixels
        dim0: Image height in pixels

    Returns:
        means2d: Projected 2D centers (C, N, 2)
        cov2d_3x3: Projected 3x3 covariance upper-tri (C, N, 6)
    """
    N = means.shape[0]
    C = viewmats.shape[0]
    device = means.device
    dtype = means.dtype

    # Extract rotation R (C, 3, 3) and translation t (C, 3)
    R = viewmats[:, :3, :3]  # (C, 3, 3) row-major
    t = viewmats[:, :3, 3]   # (C, 3)

    # Extract intrinsics
    fx = Ks[:, 0, 0]  # (C,)
    fy = Ks[:, 1, 1]  # (C,)
    cx = Ks[:, 0, 2]  # (C,)
    cy = Ks[:, 1, 2]  # (C,)

    # Expand for broadcasting: (C, 1) for element-wise ops with (C, N)
    fx = fx.unsqueeze(1)
    fy = fy.unsqueeze(1)
    cx = cx.unsqueeze(1)
    cy = cy.unsqueeze(1)

    # ================================================================
    # Step 1: Camera transform  p_c = R * p + t
    # ================================================================
    # means: (N, 3), R: (C, 3, 3), t: (C, 3)
    # p_c = R @ p^T + t  =>  p_c = einsum('cij,nj->cni', R, means) + t[:,None,:]
    p_c = torch.einsum('cij,nj->cni', R, means) + t.unsqueeze(1)  # (C, N, 3)

    x = p_c[..., 0]  # (C, N)
    y = p_c[..., 1]
    z = p_c[..., 2]

    # ================================================================
    # Step 2: Perspective projection for means2d
    # ================================================================
    # Clamp z to avoid division by zero
    z_safe = z.clamp(min=0.01)
    rz = 1.0 / z_safe
    rz2 = rz * rz

    mx = fx * x * rz + cx  # (C, N)
    my = fy * y * rz + cy

    means2d = torch.stack([mx, my], dim=-1)  # (C, N, 2)

    # ================================================================
    # Step 3: FOV clipping (hard clamp)
    # ================================================================
    W = torch.tensor(dim1, dtype=dtype, device=device)
    H = torch.tensor(dim0, dtype=dtype, device=device)
    tan_fovx = 0.5 * W / fx  # (C, 1)
    tan_fovy = 0.5 * H / fy
    lim_x_pos = (W - cx) / fx + 0.3 * tan_fovx  # (C, 1)
    lim_x_neg = cx / fx + 0.3 * tan_fovx
    lim_y_pos = (H - cy) / fy + 0.3 * tan_fovy
    lim_y_neg = cy / fy + 0.3 * tan_fovy

    xrz = x * rz  # (C, N)
    yrz = y * rz
    tx = z_safe * torch.clamp(xrz, min=-lim_x_neg, max=lim_x_pos)
    ty = z_safe * torch.clamp(yrz, min=-lim_y_neg, max=lim_y_pos)

    # ================================================================
    # Step 4: Build Jacobian J (3x3) per Gaussian
    # ================================================================
    l = torch.sqrt(tx * tx + ty * ty + z_safe * z_safe)  # (C, N)
    rl = 1.0 / l

    # J is (C, N, 3, 3)
    zeros = torch.zeros_like(x)

    J00 = fx * rz           # (C, N)
    J02 = -fx * tx * rz2
    J11 = fy * rz
    J12 = -fy * ty * rz2
    J20 = tx * rl
    J21 = ty * rl
    J22 = z_safe * rl

    # Build J as (C, N, 3, 3)
    J = torch.stack([
        torch.stack([J00,   zeros, J02], dim=-1),
        torch.stack([zeros, J11,   J12], dim=-1),
        torch.stack([J20,   J21,   J22], dim=-1),
    ], dim=-2)  # (C, N, 3, 3)

    # ================================================================
    # Step 5: Camera covariance  Sigma_c = R * Sigma * R^T
    # ================================================================
    # Unpack covars upper-tri to full symmetric (N, 3, 3)
    s00 = covars[..., 0]
    s01 = covars[..., 1]
    s02 = covars[..., 2]
    s11 = covars[..., 3]
    s12 = covars[..., 4]
    s22 = covars[..., 5]

    Sigma = torch.stack([
        torch.stack([s00, s01, s02], dim=-1),
        torch.stack([s01, s11, s12], dim=-1),
        torch.stack([s02, s12, s22], dim=-1),
    ], dim=-2)  # (N, 3, 3)

    # R: (C, 3, 3) -> (C, 1, 3, 3) for broadcasting with Sigma (1, N, 3, 3)
    R_exp = R.unsqueeze(1)  # (C, 1, 3, 3)
    Rt_exp = R_exp.transpose(-1, -2)  # (C, 1, 3, 3)
    Sigma_exp = Sigma.unsqueeze(0)  # (1, N, 3, 3)

    # Sigma_c = R @ Sigma @ R^T, broadcasts to (C, N, 3, 3)
    Sigma_c = R_exp @ Sigma_exp @ Rt_exp  # (C, N, 3, 3)

    # ================================================================
    # Step 6: Projected covariance  cov2d_3x3 = J * Sigma_c * J^T
    # ================================================================
    Jt = J.transpose(-1, -2)  # (C, N, 3, 3)
    C2 = J @ Sigma_c @ Jt    # (C, N, 3, 3)

    # Pack upper-triangular: (a, b, c, d, e, f)
    cov2d_3x3 = torch.stack([
        C2[..., 0, 0],
        C2[..., 0, 1],
        C2[..., 0, 2],
        C2[..., 1, 1],
        C2[..., 1, 2],
        C2[..., 2, 2],
    ], dim=-1)  # (C, N, 6)

    # Handle behind-camera: set outputs to zero where z <= 0.01
    behind_mask = (z <= 0.01).unsqueeze(-1)  # (C, N, 1)
    means2d = means2d.masked_fill(behind_mask, 0.0)
    cov2d_3x3 = cov2d_3x3.masked_fill(behind_mask, 0.0)

    return means2d, cov2d_3x3


# ============================================================================
# CUDA Implementation
# ============================================================================

class WorldToImageProjectionCUDA(SecondForwardBackwardBase):
    """
    CUDA-accelerated WorldToImageProjection implementation.

    Computes means2d and cov2d_3x3 from world-space Gaussian parameters.

    Expected params:
        - means: (N, 3) world-space Gaussian centers (shared across cameras)
        - covars: (N, 6) world-space covariance upper-tri (shared across cameras)

    Expected config:
        - dim1: int
        - dim0: int

    Expected data (for backward):
        - viewmats: (C, 4, 4) camera extrinsic matrices
        - Ks: (C, 3, 3) camera intrinsic matrices
        - v_means2d: (C, N, 2) upstream gradient
        - v_cov2d_3x3: (C, N, 6) upstream gradient
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
        if params.get('means') is None:
            raise ValueError("Missing required parameter: means")
        if params.get('covars') is None:
            raise ValueError("Missing required parameter: covars")

        means = params.get('means').data
        covars = params.get('covars').data

        if means.dim() != 2 or means.size(-1) != 3:
            raise ValueError(
                f"means must be 2D with shape (N, 3), got {means.shape}"
            )
        if covars.dim() != 2 or covars.size(-1) != 6:
            raise ValueError(
                f"covars must be 2D with shape (N, 6), got {covars.shape}"
            )
        if not means.is_cuda:
            raise ValueError("means must be a CUDA tensor")

        self.initialized = True

    def forward_only(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> ForwardResult:
        """Forward computation only.

        Args:
            params: Must contain 'means' (N, 3) and 'covars' (N, 6)
            data: Must contain 'viewmats' (C, 4, 4) and 'Ks' (C, 3, 3)
            config: Must contain 'dim1' and 'dim0'; optional 'eps_blur'

        Returns:
            ForwardResult with outputs: {'means2d': (C,N,2), 'cov2d_3x3': (C,N,6), 'radii': (C,N)}
        """
        from ..cuda._wrapper import world_to_image_projection_fwd

        means = params.get('means').data.contiguous()
        covars = params.get('covars').data.contiguous()
        viewmats = data.get('viewmats').contiguous()
        Ks = data.get('Ks').contiguous()
        dim1 = config.get('dim1')
        dim0 = config.get('dim0')
        eps_blur = config.get('eps_blur', 0.3)

        means2d, cov2d_3x3, radii = world_to_image_projection_fwd(
            means, covars, viewmats, Ks, dim1, dim0, eps_blur
        )

        return ForwardResult(
            outputs={'means2d': means2d, 'cov2d_3x3': cov2d_3x3, 'radii': radii},
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
            params: Must contain 'means' and 'covars'
            data: Must contain 'viewmats', 'Ks', 'v_means2d', 'v_cov2d_3x3'
            config: Must contain 'dim1', 'dim0'

        Returns:
            ForwardResult, GradResult
        """
        from ..cuda._wrapper import (
            world_to_image_projection_fwd,
            world_to_image_projection_bwd,
        )

        means = params.get('means').data.contiguous()
        covars = params.get('covars').data.contiguous()
        viewmats = data.get('viewmats').contiguous()
        Ks = data.get('Ks').contiguous()
        dim1 = config.get('dim1')
        dim0 = config.get('dim0')
        eps_blur = config.get('eps_blur', 0.3)

        # Forward
        means2d, cov2d_3x3, radii = world_to_image_projection_fwd(
            means, covars, viewmats, Ks, dim1, dim0, eps_blur
        )

        # Get upstream gradients
        v_means2d = data.get('v_means2d')
        v_cov2d_3x3 = data.get('v_cov2d_3x3')
        if v_means2d is None:
            v_means2d = torch.ones_like(means2d)
        if v_cov2d_3x3 is None:
            v_cov2d_3x3 = torch.ones_like(cov2d_3x3)
        v_means2d = v_means2d.contiguous()
        v_cov2d_3x3 = v_cov2d_3x3.contiguous()

        # Backward
        v_means, v_covars = world_to_image_projection_bwd(
            means, covars, viewmats, Ks, dim1, dim0,
            v_means2d, v_cov2d_3x3,
        )

        forward_result = ForwardResult(
            outputs={'means2d': means2d, 'cov2d_3x3': cov2d_3x3},
            intermediates={}
        )
        grad_result = GradResult(
            param_grads={'means': v_means, 'covars': v_covars}
        )

        return forward_result, grad_result

    def forward_backward2(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> Tuple[ForwardResult, GradResult, HessResult]:
        """Forward + second-order backward using CUDA kernels.

        Args:
            params: Must contain 'means' and 'covars'
            data: Must contain 'viewmats', 'Ks', downstream grads and Hessians
            config: Must contain 'dim1', 'dim0'

        Returns:
            ForwardResult, GradResult, HessResult
        """
        from ..cuda._wrapper import (
            world_to_image_projection_fwd,
            world_to_image_projection_bwd2,
        )

        means = params.get('means').data.contiguous()
        covars = params.get('covars').data.contiguous()
        viewmats = data.get('viewmats').contiguous()
        Ks = data.get('Ks').contiguous()
        dim1 = config.get('dim1')
        dim0 = config.get('dim0')
        eps_blur = config.get('eps_blur', 0.3)
        N = means.shape[0]
        C = viewmats.shape[0]

        # Forward
        means2d, cov2d_3x3, radii = world_to_image_projection_fwd(
            means, covars, viewmats, Ks, dim1, dim0, eps_blur
        )

        # Get downstream first-order gradients
        v_means2d = data.get('v_means2d')
        v_cov2d_3x3 = data.get('v_cov2d_3x3')
        if v_means2d is None:
            v_means2d = torch.zeros_like(means2d)
        if v_cov2d_3x3 is None:
            v_cov2d_3x3 = torch.zeros_like(cov2d_3x3)
        v_means2d = v_means2d.contiguous()
        v_cov2d_3x3 = v_cov2d_3x3.contiguous()

        # Get downstream Hessians
        def _get_or_zeros(key, shape):
            val = data.get(key)
            if val is None:
                return torch.zeros(shape, dtype=means.dtype,
                                   device=means.device)
            return val.contiguous()

        h_means2d = _get_or_zeros('h_means2d', (C, N, 3))
        h_cov2d_3x3 = _get_or_zeros('h_cov2d_3x3', (C, N, 21))
        h_means2d_cov2d_3x3 = _get_or_zeros('h_means2d_cov2d_3x3', (C, N, 12))
        h_means2d_opacity = _get_or_zeros('h_means2d_opacity', (C, N, 2))
        h_cov2d_3x3_opacity = _get_or_zeros('h_cov2d_3x3_opacity', (C, N, 6))

        # Second-order backward
        (v_means, v_covars,
         h_means, h_covars, h_means_covars,
         h_means_opacity, h_covars_opacity) = world_to_image_projection_bwd2(
            means, covars, viewmats, Ks, dim1, dim0,
            v_means2d, v_cov2d_3x3,
            h_means2d, h_cov2d_3x3, h_means2d_cov2d_3x3,
            h_means2d_opacity, h_cov2d_3x3_opacity,
        )

        forward_result = ForwardResult(
            outputs={'means2d': means2d, 'cov2d_3x3': cov2d_3x3},
            intermediates={}
        )
        grad_result = GradResult(
            param_grads={'means': v_means, 'covars': v_covars}
        )
        hess_result = HessResult(
            param_hess={
                'means': h_means,
                'covars': h_covars,
                'means_covars': h_means_covars,
                'means_opacity': h_means_opacity,
                'covars_opacity': h_covars_opacity,
            }
        )

        return forward_result, grad_result, hess_result


# ============================================================================
# PyTorch Autograd Implementation
# ============================================================================

class WorldToImageProjectionAutograd(SecondForwardBackwardBase):
    """
    PyTorch autograd-based WorldToImageProjection implementation.

    Reference implementation for validating the CUDA forward/backward passes.
    Uses PyTorch's automatic differentiation for gradient and Hessian computation.

    Expected params:
        - means: (N, 3) world-space Gaussian centers (shared across cameras)
        - covars: (N, 6) world-space covariance upper-tri (shared across cameras)

    Expected config:
        - dim1: int
        - dim0: int
    """

    def __init__(self):
        self.initialized = False

    def initialize(
        self,
        params: Params,
        config: Optional[Config] = None,
    ) -> None:
        """Initialize and validate parameters."""
        if params.get('means') is None:
            raise ValueError("Missing required parameter: means")
        if params.get('covars') is None:
            raise ValueError("Missing required parameter: covars")

        means = params.get('means').data
        covars = params.get('covars').data

        if means.dim() != 2 or means.size(-1) != 3:
            raise ValueError(
                f"means must be 2D with shape (N, 3), got {means.shape}"
            )
        if covars.dim() != 2 or covars.size(-1) != 6:
            raise ValueError(
                f"covars must be 2D with shape (N, 6), got {covars.shape}"
            )

        self.initialized = True

    def forward_only(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> ForwardResult:
        """Forward computation only (no gradients)."""
        means = params.get('means').data
        covars = params.get('covars').data
        viewmats = data.get('viewmats')
        Ks = data.get('Ks')
        dim1 = config.get('dim1')
        dim0 = config.get('dim0')

        with torch.no_grad():
            means2d, cov2d_3x3 = _compute_world_to_image_projection(
                means, covars, viewmats, Ks, dim1, dim0
            )

        return ForwardResult(
            outputs={'means2d': means2d, 'cov2d_3x3': cov2d_3x3},
            intermediates={}
        )

    def forward_backward1(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> Tuple[ForwardResult, GradResult]:
        """Forward + first-order backward using torch.autograd."""
        means_orig = params.get('means').data
        covars_orig = params.get('covars').data
        viewmats = data.get('viewmats')
        Ks = data.get('Ks')
        dim1 = config.get('dim1')
        dim0 = config.get('dim0')

        # Create leaves for autograd
        means = means_orig.detach().clone().requires_grad_(True)
        covars = covars_orig.detach().clone().requires_grad_(True)

        # Forward pass (tracked by autograd)
        means2d, cov2d_3x3 = _compute_world_to_image_projection(
            means, covars, viewmats, Ks, dim1, dim0
        )

        # Get upstream gradients
        v_means2d = data.get('v_means2d')
        v_cov2d_3x3 = data.get('v_cov2d_3x3')
        if v_means2d is None:
            v_means2d = torch.ones_like(means2d)
        if v_cov2d_3x3 is None:
            v_cov2d_3x3 = torch.ones_like(cov2d_3x3)

        # Backward
        torch.autograd.backward(
            tensors=[means2d, cov2d_3x3],
            grad_tensors=[v_means2d, v_cov2d_3x3],
        )

        forward_result = ForwardResult(
            outputs={'means2d': means2d.detach(), 'cov2d_3x3': cov2d_3x3.detach()},
            intermediates={}
        )
        grad_result = GradResult(
            param_grads={
                'means': means.grad.detach(),
                'covars': covars.grad.detach(),
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

        Note: Since means and covars are shared across cameras, the Hessians
        are accumulated (summed) over C cameras, giving shapes:
            means: (N, 6), covars: (N, 21), means_covars: (N, 18)
        The opacity cross-Hessians remain per-camera:
            means_opacity: (C, N, 3), covars_opacity: (C, N, 6)

        Returns:
            ForwardResult, GradResult, HessResult with:
                param_hess keys:
                    'means': H_{p,p} upper-tri 3x3 (N, 6)
                    'covars': H_{s,s} upper-tri 6x6 (N, 21)
                    'means_covars': H_{p,s} 3x6 row-major (N, 18)
                    'means_opacity': H_{p,opacity} (C, N, 3)
                    'covars_opacity': H_{s,opacity} (C, N, 6)
        """
        means_orig = params.get('means').data
        covars_orig = params.get('covars').data
        viewmats = data.get('viewmats')
        Ks = data.get('Ks')
        dim1 = config.get('dim1')
        dim0 = config.get('dim0')
        N = means_orig.shape[0]
        C = viewmats.shape[0]

        # Create leaves for autograd
        means = means_orig.detach().clone().requires_grad_(True)
        covars = covars_orig.detach().clone().requires_grad_(True)

        # Forward pass (tracked by autograd)
        means2d, cov2d_3x3 = _compute_world_to_image_projection(
            means, covars, viewmats, Ks, dim1, dim0
        )

        # Get downstream first-order gradients
        v_means2d = data.get('v_means2d')
        v_cov2d_3x3 = data.get('v_cov2d_3x3')
        if v_means2d is None:
            v_means2d = torch.zeros_like(means2d)
        if v_cov2d_3x3 is None:
            v_cov2d_3x3 = torch.zeros_like(cov2d_3x3)

        # First backward with create_graph=True
        v_means_graph, v_covars_graph = torch.autograd.grad(
            outputs=[means2d, cov2d_3x3],
            inputs=[means, covars],
            grad_outputs=[v_means2d, v_cov2d_3x3],
            create_graph=True,
            retain_graph=True,
        )

        # Get downstream Hessians
        def _get_or_zeros(key, shape):
            val = data.get(key)
            if val is None:
                return torch.zeros(shape, dtype=means.dtype,
                                   device=means.device)
            return val

        h_means2d = _get_or_zeros('h_means2d', (C, N, 3))
        h_cov2d_3x3 = _get_or_zeros('h_cov2d_3x3', (C, N, 21))
        h_means2d_cov2d_3x3 = _get_or_zeros('h_means2d_cov2d_3x3', (C, N, 12))
        h_means2d_opacity = _get_or_zeros('h_means2d_opacity', (C, N, 2))
        h_cov2d_3x3_opacity = _get_or_zeros('h_cov2d_3x3_opacity', (C, N, 6))

        # Total number of upstream params: 9 (3 for means + 6 for covars)
        # We compute H_{theta,theta} column by column via second backward.
        # Since means (N,3) and covars (N,6) are shared across cameras,
        # the autograd Hessians are naturally (N,...) - summed over C.

        # Build per-element Hessians (N, ...) - autograd sums over C automatically
        h_pp = torch.zeros(N, 3, 3, dtype=means.dtype, device=means.device)
        h_ss = torch.zeros(N, 6, 6, dtype=means.dtype, device=means.device)
        h_ps = torch.zeros(N, 3, 6, dtype=means.dtype, device=means.device)

        # ================================================================
        # Term 2: sum_k v_k * Hess_k via double backward
        # ================================================================
        # Differentiate v_theta = (v_means_graph, v_covars_graph) w.r.t.
        # (means, covars) to get the second-order terms.

        # For means Hessian: differentiate v_means w.r.t. means & covars
        for j in range(3):
            selector_p = torch.zeros_like(v_means_graph)
            selector_p[..., j] = 1.0

            grads = torch.autograd.grad(
                outputs=v_means_graph,
                inputs=[means, covars],
                grad_outputs=selector_p,
                retain_graph=True,
                allow_unused=True,
            )
            g0 = grads[0] if grads[0] is not None else torch.zeros_like(means)
            g1 = grads[1] if grads[1] is not None else torch.zeros_like(covars)
            h_pp[..., :, j] += g0  # d(v_p)/d(p_j)
            h_ps[..., j, :] += g1  # d(v_p_j)/d(s) = H_{p_j, s}...wait
            # Actually grads[1] = d(v_means3d)/d(covars) when selecting j-th component
            # This is (C, N, 6), which gives H_{pp_j, s} = row j of the p-s cross hess
            # But we also need d(v_covars)/d(means3d) for the full cross term.

        # For covars part of the cross Hessian: differentiate v_covars w.r.t. means
        for j in range(6):
            selector_s = torch.zeros_like(v_covars_graph)
            selector_s[..., j] = 1.0

            grads = torch.autograd.grad(
                outputs=v_covars_graph,
                inputs=[means, covars],
                grad_outputs=selector_s,
                retain_graph=True,
                allow_unused=True,
            )
            g1 = grads[1] if grads[1] is not None else torch.zeros_like(covars)
            # grads[0] = d(v_s_j)/d(p), shape (C, N, 3) => column j of H_{p,s}
            # But H_{p,s} should be symmetric with what we got above...
            # Actually for the cross Hessian: H_{p_i, s_j} = d^2L / d(p_i)d(s_j)
            # From v_means3d differentiation: d(v_p_i)/d(s_j)  (row i fixed)
            # From v_covars differentiation: d(v_s_j)/d(p_i)   (col j fixed)
            # These should be the same (Schwarz's theorem). Let's use the second.
            # Actually, they compute different things because v_p and v_s depend
            # on v_phi which is constant. Let's just use both and average.

            h_ss[..., :, j] += g1  # d(v_s)/d(s_j)

        # The h_ps from the means3d loop gives d(v_p_j)/d(s) but that's actually
        # transposed. Let me re-derive:
        # v_p = J_p^T @ v_phi  (J_p is d(phi)/d(p), 8x3)
        # d(v_p_i)/d(s_j) involves d(J_p)/d(s_j) which can be nonzero.
        # v_s = J_s^T @ v_phi  (J_s is d(phi)/d(s), 8x6)
        # d(v_s_j)/d(p_i) involves d(J_s)/d(p_i) which can also be nonzero.
        # These are both d^2(phi)/d(p_i)d(s_j) dotted with v_phi + cross terms.
        # By symmetry of mixed partials, d(v_p_i)/d(s_j) should equal d(v_s_j)/d(p_i).
        # Let's use the version from the means3d loop (grads[1] when selecting p component j).

        # Reset h_ps and recompute from the means loop data
        h_ps_t2 = torch.zeros(N, 3, 6, dtype=means.dtype, device=means.device)
        for j in range(3):
            selector_p = torch.zeros_like(v_means_graph)
            selector_p[..., j] = 1.0
            grads = torch.autograd.grad(
                outputs=v_means_graph,
                inputs=[means, covars],
                grad_outputs=selector_p,
                retain_graph=True,
                allow_unused=True,
            )
            g1 = grads[1] if grads[1] is not None else torch.zeros_like(covars)
            # grads[1] = d(v_p_j)/d(s), shape (N, 6) => row j of H_{p,s} term2
            h_ps_t2[..., j, :] = g1

        # ================================================================
        # Term 1: sum_c J_c^T @ H_c @ J_c (Gauss-Newton term)
        # ================================================================
        # For each camera c, the Jacobian J_c maps (means[N,3], covars[N,6]) -> phi_c[N,8].
        # Since means/covars are shared, we need per-camera Jacobians.
        # We compute them by masking out other cameras when computing gradients.
        # phi = (means2d[0], means2d[1], cov2d_3x3[0..5]) -> 8 outputs per camera
        # theta = (means[0..2], covars[0..5]) -> 9 inputs

        # Build H_{phi,phi} (8 x 8) per (C, N) element
        H_pp_full = torch.zeros(C, N, 8, 8, dtype=means.dtype, device=means.device)

        # means2d block (2x2): h_means2d has 3 upper-tri elements
        H_pp_full[..., 0, 0] = h_means2d[..., 0]
        H_pp_full[..., 0, 1] = H_pp_full[..., 1, 0] = h_means2d[..., 1]
        H_pp_full[..., 1, 1] = h_means2d[..., 2]

        # cov2d block (6x6): h_cov2d_3x3 has 21 upper-tri elements
        for i in range(6):
            for j in range(i, 6):
                flat_idx = i * (11 - i) // 2 + j
                H_pp_full[..., 2 + i, 2 + j] = h_cov2d_3x3[..., flat_idx]
                if i != j:
                    H_pp_full[..., 2 + j, 2 + i] = h_cov2d_3x3[..., flat_idx]

        # means2d-cov2d cross block (2x6): h_means2d_cov2d_3x3, row-major
        for i in range(2):
            for j in range(6):
                val = h_means2d_cov2d_3x3[..., i * 6 + j]
                H_pp_full[..., i, 2 + j] = val
                H_pp_full[..., 2 + j, i] = val

        # Compute per-camera Jacobians J_c (C, N, 8, 9) by selecting each camera
        J_full = torch.zeros(C, N, 8, 9, dtype=means.dtype, device=means.device)

        for c in range(C):
            for k in range(8):
                # Select only camera c's output
                if k < 2:
                    grad_m = torch.zeros_like(means2d)
                    grad_m[c, :, k] = 1.0
                    grad_c_tensor = torch.zeros_like(cov2d_3x3)
                else:
                    grad_m = torch.zeros_like(means2d)
                    grad_c_tensor = torch.zeros_like(cov2d_3x3)
                    grad_c_tensor[c, :, k - 2] = 1.0

                jt_col_k = torch.autograd.grad(
                    outputs=[means2d, cov2d_3x3],
                    inputs=[means, covars],
                    grad_outputs=[grad_m, grad_c_tensor],
                    retain_graph=True,
                    allow_unused=True,
                )
                jt_col_k = [
                    g if g is not None else torch.zeros_like(means)
                    for g in jt_col_k[:1]
                ] + [
                    g if g is not None else torch.zeros_like(covars)
                    for g in jt_col_k[1:]
                ]
                # jt_col_k[0] shape (N, 3) - d(phi_c_k)/d(means), only from camera c
                # jt_col_k[1] shape (N, 6) - d(phi_c_k)/d(covars), only from camera c
                J_full[c, :, k, :3] = jt_col_k[0]
                J_full[c, :, k, 3:] = jt_col_k[1]

        # term1 = sum_c J_c^T @ H_c @ J_c, computed per-camera then summed
        Jt_full = J_full.transpose(-1, -2)  # (C, N, 9, 8)
        HJ = torch.einsum('...ij,...jk->...ik', H_pp_full, J_full)  # (C, N, 8, 9)
        term1_per_cam = torch.einsum('...ij,...jk->...ik', Jt_full, HJ)  # (C, N, 9, 9)
        term1 = term1_per_cam.sum(dim=0)  # (N, 9, 9) - sum over cameras

        # ================================================================
        # Combine: H_theta = term1 + term2
        # ================================================================
        # Extract blocks
        full_H = term1.detach()

        # Add term2 to the appropriate blocks
        full_H[..., :3, :3] += h_pp
        full_H[..., 3:, 3:] += h_ss
        full_H[..., :3, 3:] += h_ps_t2

        # Symmetrize the cross block
        full_H[..., 3:, :3] = full_H[..., :3, 3:].transpose(-1, -2)

        # Pack outputs - all (N, ...) since accumulated over cameras
        h_means_out = torch.zeros(N, 6, dtype=means.dtype, device=means.device)
        for i in range(3):
            for j in range(i, 3):
                flat_idx = i * (5 - i) // 2 + j
                h_means_out[..., flat_idx] = full_H[..., i, j]

        h_covars_out = torch.zeros(N, 21, dtype=means.dtype, device=means.device)
        for i in range(6):
            for j in range(i, 6):
                flat_idx = i * (11 - i) // 2 + j
                h_covars_out[..., flat_idx] = full_H[..., 3 + i, 3 + j]

        h_means_covars_out = torch.zeros(N, 18, dtype=means.dtype,
                                         device=means.device)
        for i in range(3):
            for j in range(6):
                h_means_covars_out[..., i * 6 + j] = full_H[..., i, 3 + j]

        # ================================================================
        # Cross-Hessians with opacity: H_{theta, psi} = J_full^T @ H_{phi, psi}
        # ================================================================
        # These remain per-camera since opacity is per-camera.
        # H_{phi, opacity} is 8x1: (h_means2d_opacity[0..1], h_cov2d_3x3_opacity[0..5])
        h_phi_opacity = torch.cat([
            h_means2d_opacity,      # (C, N, 2)
            h_cov2d_3x3_opacity,    # (C, N, 6)
        ], dim=-1)  # (C, N, 8)

        # H_{theta, opacity} = J_full^T @ h_phi_opacity  per-camera (C, N, 9)
        h_theta_opacity = torch.einsum('...ji,...j->...i', J_full, h_phi_opacity)
        # (C, N, 9)

        h_means_opacity_out = h_theta_opacity[..., :3]   # (C, N, 3)
        h_covars_opacity_out = h_theta_opacity[..., 3:]   # (C, N, 6)

        forward_result = ForwardResult(
            outputs={'means2d': means2d.detach(), 'cov2d_3x3': cov2d_3x3.detach()},
            intermediates={}
        )
        grad_result = GradResult(
            param_grads={
                'means': v_means_graph.detach(),
                'covars': v_covars_graph.detach(),
            }
        )
        hess_result = HessResult(
            param_hess={
                'means': h_means_out.detach(),
                'covars': h_covars_out.detach(),
                'means_covars': h_means_covars_out.detach(),
                'means_opacity': h_means_opacity_out.detach(),
                'covars_opacity': h_covars_opacity_out.detach(),
            }
        )

        return forward_result, grad_result, hess_result


# ============================================================================
# Factory Function
# ============================================================================

def create_world_to_image_projection(
    backend: str = "cuda",
) -> SecondForwardBackwardBase:
    """Factory function to create WorldToImageProjection operator.

    Args:
        backend: "cuda" for CUDA implementation, "autograd" for PyTorch reference

    Returns:
        WorldToImageProjection instance
    """
    if backend == "cuda":
        return WorldToImageProjectionCUDA()
    elif backend == "autograd":
        return WorldToImageProjectionAutograd()
    else:
        raise ValueError(
            f"Unknown backend: {backend}. Use 'cuda' or 'autograd'."
        )


__all__ = [
    "WorldToImageProjectionCUDA",
    "WorldToImageProjectionAutograd",
    "create_world_to_image_projection",
]
