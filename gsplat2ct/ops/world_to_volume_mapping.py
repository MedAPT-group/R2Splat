"""
WorldToVolumeMapping operators for CT reconstruction.

This module provides CUDA and PyTorch autograd implementations of the
world-to-volume mapping: from world-space Gaussian means and covariances,
compute voxel-space means, conics (inverse covariance), and radii.

Pipeline:
    F1. Compute scaling factors: r[k] = dim[k] / range[k]
    F2. Affine transform means: means3d_v[k] = (means[k] - center[k]) * r[k] + dim[k]/2
    F3. Scale covariance: Sigma_v[i][j] = r[i]*r[j] * covars[i][j]
    F4. Add blur: Sigma_v[i][i] += eps3d
    F5. Matrix inverse: conics = Sigma_v^{-1}
    F6. Compute radii: radii[k] = ceil(radius_clip * max_scales * r[k])

Key differences from WorldToImageProjection:
  - NO batch dimension C (single volume)
  - R is diagonal (not general rotation) → massive simplification
  - NO compensation factor
  - conics stored as 3x3 upper-tri [N, 6] (not 2x2)
  - radii are 3D [N, 3] (not 2D)
  - max_scales is non-differentiable → no gradient path

Upper-triangular storage for 3x3 symmetric matrix:
    (a, b, c, d, e, f) = (S00, S01, S02, S11, S12, S22)

Supports:
- WorldToVolumeMappingCUDA: CUDA-accelerated implementation
- WorldToVolumeMappingAutograd: Pure PyTorch reference using torch.autograd

Author: Dy164
Date: 2026-02-21
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

def _compute_world_to_volume_mapping(
    means: torch.Tensor,
    covars: torch.Tensor,
    max_scales: torch.Tensor,
    ranges: torch.Tensor,
    centers: torch.Tensor,
    dim2: int,
    dim1: int,
    dim0: int,
    eps3d: float = 1e-4,
    radius_clip: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute world-to-volume mapping (pure PyTorch implementation).

    This function is differentiable via PyTorch's autograd (for means3d_v and conics).
    Radii are computed with integer ops and are non-differentiable.

    Args:
        means: World-space Gaussian centers (N, 3)
        covars: World-space covariance upper-tri (N, 6)
        max_scales: Max activated scale per Gaussian (N)
        ranges: Physical extent per axis (3,)
        centers: Physical center per axis (3,)
        dim2: Voxel grid x dimension (fastest)
        dim1: Voxel grid y dimension (middle)
        dim0: Voxel grid z dimension (slowest)
        eps3d: Blur epsilon
        radius_clip: Radius multiplier

    Returns:
        means3d_v: Voxel-space centers (N, 3)
        conics: Inverse covariance upper-tri (N, 6)
        radii: Integer radii per axis (N, 3) as int32
    """
    N = means.shape[0]
    device = means.device
    dtype = means.dtype

    # Scaling factors
    dims = torch.tensor([dim2, dim1, dim0], dtype=dtype, device=device)
    r = dims / ranges  # (3,)

    # F2: Affine transform for means
    means3d_v = (means - centers.unsqueeze(0)) * r.unsqueeze(0) + (dims / 2.0).unsqueeze(0)

    # F3: Covariance transform (diagonal R)
    # Unpack upper-tri
    s00, s01, s02, s11, s12, s22 = (
        covars[:, 0], covars[:, 1], covars[:, 2],
        covars[:, 3], covars[:, 4], covars[:, 5],
    )

    # Sigma_v[i][j] = r[i]*r[j] * covars[i][j]
    Sv00 = r[0] * r[0] * s00
    Sv01 = r[0] * r[1] * s01
    Sv02 = r[0] * r[2] * s02
    Sv11 = r[1] * r[1] * s11
    Sv12 = r[1] * r[2] * s12
    Sv22 = r[2] * r[2] * s22

    # F4: Add blur
    Sv00 = Sv00 + eps3d
    Sv11 = Sv11 + eps3d
    Sv22 = Sv22 + eps3d

    # F5: Matrix inverse via cofactors
    c00 = Sv11 * Sv22 - Sv12 * Sv12
    c01 = Sv02 * Sv12 - Sv01 * Sv22
    c02 = Sv01 * Sv12 - Sv02 * Sv11
    c11 = Sv00 * Sv22 - Sv02 * Sv02
    c12 = Sv01 * Sv02 - Sv00 * Sv12
    c22 = Sv00 * Sv11 - Sv01 * Sv01

    det = Sv00 * c00 + Sv01 * c01 + Sv02 * c02
    inv_det = 1.0 / det.clamp(min=1e-12)

    conics = torch.stack([
        c00 * inv_det, c01 * inv_det, c02 * inv_det,
        c11 * inv_det, c12 * inv_det, c22 * inv_det,
    ], dim=-1)  # (N, 6)

    # F6: Compute radii (non-differentiable)
    # radii[k] = ceil(radius_clip * max_scales * r[k])
    radii = torch.ceil(
        radius_clip * max_scales.unsqueeze(-1) * r.unsqueeze(0)
    ).to(torch.int32)  # (N, 3)

    return means3d_v, conics, radii


# ============================================================================
# CUDA Implementation
# ============================================================================

class WorldToVolumeMappingCUDA(SecondForwardBackwardBase):
    """
    CUDA-accelerated WorldToVolumeMapping implementation.

    Computes means3d_v, conics, and radii from world-space Gaussian parameters.

    Expected params:
        - means: (N, 3) world-space Gaussian centers
        - covars: (N, 6) world-space covariance upper-tri
        - max_scales: (N,) max activated scale per Gaussian

    Expected config:
        - ranges: (3,) physical extent per axis
        - centers: (3,) physical center per axis
        - dim2: int  (voxel grid x)
        - dim1: int  (voxel grid y)
        - dim0: int  (voxel grid z)
        - eps3d: float (default 1e-4)
        - radius_clip: float (default 3.0)

    Expected data (for backward):
        - v_means3d_v: (N, 3) upstream gradient
        - v_conics: (N, 6) upstream gradient
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
            params: Must contain 'means' (N, 3), 'covars' (N, 6), 'max_scales' (N,)
            data: Must contain 'ranges' (3,) and 'centers' (3,)
            config: Must contain 'dim2', 'dim1', 'dim0', optionally 'eps3d', 'radius_clip'

        Returns:
            ForwardResult with outputs: {
                'means3d_v': (N, 3),
                'conics': (N, 6),
                'radii': (N, 3) int32
            }
        """
        from ..cuda._wrapper import world_to_volume_mapping_fwd

        means = params.get('means').data.contiguous()
        covars = params.get('covars').data.contiguous()
        max_scales = params.get('max_scales').data.contiguous()
        ranges = data.get('ranges').contiguous()
        centers = data.get('centers').contiguous()
        dim2 = config.get('dim2')
        dim1 = config.get('dim1')
        dim0 = config.get('dim0')
        eps3d = config.get('eps3d', 1e-4)
        radius_clip = config.get('radius_clip', 3.0)

        means3d_v, conics, radii = world_to_volume_mapping_fwd(
            means, covars, max_scales, ranges, centers,
            dim2, dim1, dim0, eps3d, radius_clip
        )

        return ForwardResult(
            outputs={
                'means3d_v': means3d_v,
                'conics': conics,
                'radii': radii,
            },
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
            params: Must contain 'means', 'covars', 'max_scales'
            data: Must contain 'ranges', 'centers', 'v_means3d_v', 'v_conics'
            config: Must contain 'dim2', 'dim1', 'dim0', optionally 'eps3d', 'radius_clip'

        Returns:
            ForwardResult, GradResult
        """
        from ..cuda._wrapper import (
            world_to_volume_mapping_fwd,
            world_to_volume_mapping_bwd,
        )

        means = params.get('means').data.contiguous()
        covars = params.get('covars').data.contiguous()
        max_scales = params.get('max_scales').data.contiguous()
        ranges = data.get('ranges').contiguous()
        centers = data.get('centers').contiguous()
        dim2 = config.get('dim2')
        dim1 = config.get('dim1')
        dim0 = config.get('dim0')
        eps3d = config.get('eps3d', 1e-4)
        radius_clip = config.get('radius_clip', 3.0)

        # Forward
        means3d_v, conics, radii = world_to_volume_mapping_fwd(
            means, covars, max_scales, ranges, centers,
            dim2, dim1, dim0, eps3d, radius_clip
        )

        # Get upstream gradients
        v_means3d_v = data.get('v_means3d_v')
        v_conics = data.get('v_conics')
        if v_means3d_v is None:
            v_means3d_v = torch.ones_like(means3d_v)
        if v_conics is None:
            v_conics = torch.ones_like(conics)
        v_means3d_v = v_means3d_v.contiguous()
        v_conics = v_conics.contiguous()

        # Backward
        v_means, v_covars = world_to_volume_mapping_bwd(
            means, covars, ranges, centers,
            dim2, dim1, dim0, eps3d,
            v_means3d_v, v_conics,
        )

        forward_result = ForwardResult(
            outputs={
                'means3d_v': means3d_v,
                'conics': conics,
                'radii': radii,
            },
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
            params: Must contain 'means', 'covars', 'max_scales'
            data: Must contain 'ranges', 'centers', downstream grads and Hessians
            config: Must contain 'dim2', 'dim1', 'dim0', optionally 'eps3d'

        Returns:
            ForwardResult, GradResult, HessResult
        """
        from ..cuda._wrapper import (
            world_to_volume_mapping_fwd,
            world_to_volume_mapping_bwd2,
        )

        means = params.get('means').data.contiguous()
        covars = params.get('covars').data.contiguous()
        max_scales = params.get('max_scales').data.contiguous()
        ranges = data.get('ranges').contiguous()
        centers = data.get('centers').contiguous()
        dim2 = config.get('dim2')
        dim1 = config.get('dim1')
        dim0 = config.get('dim0')
        eps3d = config.get('eps3d', 1e-4)
        radius_clip = config.get('radius_clip', 3.0)
        N = means.shape[0]

        # Forward
        means3d_v, conics, radii = world_to_volume_mapping_fwd(
            means, covars, max_scales, ranges, centers,
            dim2, dim1, dim0, eps3d, radius_clip
        )

        # Get downstream first-order gradients
        v_means3d_v = data.get('v_means3d_v')
        v_conics = data.get('v_conics')
        if v_means3d_v is None:
            v_means3d_v = torch.zeros_like(means3d_v)
        if v_conics is None:
            v_conics = torch.zeros_like(conics)
        v_means3d_v = v_means3d_v.contiguous()
        v_conics = v_conics.contiguous()

        # Get downstream Hessians
        def _get_or_zeros(key, shape):
            val = data.get(key)
            if val is None:
                return torch.zeros(shape, dtype=means.dtype, device=means.device)
            return val.contiguous()

        h_means3d_v = _get_or_zeros('h_means3d_v', (N, 6))
        h_conics = _get_or_zeros('h_conics', (N, 21))
        h_means3d_v_conics = _get_or_zeros('h_means3d_v_conics', (N, 18))
        h_means3d_v_opacity = _get_or_zeros('h_means3d_v_opacity', (N, 3))
        h_conics_opacity = _get_or_zeros('h_conics_opacity', (N, 6))

        # Second-order backward
        (v_means, v_covars,
         h_means, h_covars, h_means_covars,
         h_means_opacity, h_covars_opacity) = world_to_volume_mapping_bwd2(
            means, covars, ranges, centers,
            dim2, dim1, dim0, eps3d,
            v_means3d_v, v_conics,
            h_means3d_v, h_conics, h_means3d_v_conics,
            h_means3d_v_opacity, h_conics_opacity,
        )

        forward_result = ForwardResult(
            outputs={
                'means3d_v': means3d_v,
                'conics': conics,
                'radii': radii,
            },
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

class WorldToVolumeMappingAutograd(SecondForwardBackwardBase):
    """
    PyTorch autograd-based WorldToVolumeMapping implementation.

    Reference implementation for validating the CUDA forward/backward passes.
    Uses PyTorch's automatic differentiation for gradient and Hessian computation.

    Expected params:
        - means: (N, 3) world-space Gaussian centers
        - covars: (N, 6) world-space covariance upper-tri
        - max_scales: (N,) max activated scale per Gaussian

    Expected config:
        - ranges, centers, dim2, dim1, dim0, eps3d, radius_clip
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
        max_scales = params.get('max_scales').data
        ranges = data.get('ranges')
        centers = data.get('centers')
        dim2 = config.get('dim2')
        dim1 = config.get('dim1')
        dim0 = config.get('dim0')
        eps3d = config.get('eps3d', 1e-4)
        radius_clip = config.get('radius_clip', 3.0)

        with torch.no_grad():
            means3d_v, conics, radii = _compute_world_to_volume_mapping(
                means, covars, max_scales, ranges, centers,
                dim2, dim1, dim0, eps3d, radius_clip
            )

        return ForwardResult(
            outputs={
                'means3d_v': means3d_v,
                'conics': conics,
                'radii': radii,
            },
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
        max_scales = params.get('max_scales').data
        ranges = data.get('ranges')
        centers = data.get('centers')
        dim2 = config.get('dim2')
        dim1 = config.get('dim1')
        dim0 = config.get('dim0')
        eps3d = config.get('eps3d', 1e-4)
        radius_clip = config.get('radius_clip', 3.0)

        # Create leaves for autograd
        means = means_orig.detach().clone().requires_grad_(True)
        covars = covars_orig.detach().clone().requires_grad_(True)

        # Forward pass (tracked by autograd)
        means3d_v, conics, radii = _compute_world_to_volume_mapping(
            means, covars, max_scales, ranges, centers,
            dim2, dim1, dim0, eps3d, radius_clip
        )

        # Get upstream gradients
        v_means3d_v = data.get('v_means3d_v')
        v_conics = data.get('v_conics')
        if v_means3d_v is None:
            v_means3d_v = torch.ones_like(means3d_v)
        if v_conics is None:
            v_conics = torch.ones_like(conics)

        # Backward
        torch.autograd.backward(
            tensors=[means3d_v, conics],
            grad_tensors=[v_means3d_v, v_conics],
        )

        forward_result = ForwardResult(
            outputs={
                'means3d_v': means3d_v.detach(),
                'conics': conics.detach(),
                'radii': radii,
            },
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

        Computes Hessian blocks using the chain rule:
            H_theta = J^T @ H_phi @ J + sum_k v_k * Hess_k(theta)

        The first term (GN term) is computed analytically from downstream
        Hessians, and the second term (curvature) is computed via
        torch.autograd double backward.

        Expected downstream Hessians in data:
            h_means3d_v:          (N, 6)  upper-tri 3x3
            h_conics:             (N, 21) upper-tri 6x6
            h_means3d_v_conics:   (N, 18) 3x6 row-major
            h_means3d_v_opacity:  (N, 3)
            h_conics_opacity:     (N, 6)

        Returns:
            ForwardResult, GradResult, HessResult with:
                param_hess keys:
                    'means': H_{p,p} upper-tri 3x3 (N, 6)
                    'covars': H_{s,s} upper-tri 6x6 (N, 21)
                    'means_covars': H_{p,s} 3x6 row-major (N, 18)
                    'means_opacity': H_{p,opacity} (N, 3)
                    'covars_opacity': H_{s,opacity} (N, 6)
        """
        means_orig = params.get('means').data
        covars_orig = params.get('covars').data
        max_scales = params.get('max_scales').data
        ranges = data.get('ranges')
        centers = data.get('centers')
        dim2 = config.get('dim2')
        dim1 = config.get('dim1')
        dim0 = config.get('dim0')
        eps3d = config.get('eps3d', 1e-4)
        radius_clip = config.get('radius_clip', 3.0)
        N = means_orig.shape[0]
        dtype = means_orig.dtype
        device = means_orig.device

        # Create leaves for autograd
        means = means_orig.detach().clone().requires_grad_(True)
        covars = covars_orig.detach().clone().requires_grad_(True)

        # Forward pass
        means3d_v, conics, radii = _compute_world_to_volume_mapping(
            means, covars, max_scales, ranges, centers,
            dim2, dim1, dim0, eps3d, radius_clip
        )

        # Get upstream gradients
        v_means3d_v = data.get('v_means3d_v')
        v_conics = data.get('v_conics')
        if v_means3d_v is None:
            v_means3d_v = torch.zeros(N, 3, dtype=dtype, device=device)
        if v_conics is None:
            v_conics = torch.zeros(N, 6, dtype=dtype, device=device)

        # Get downstream Hessians
        def _get_or_zeros(key, shape):
            val = data.get(key)
            if val is None:
                return torch.zeros(shape, dtype=dtype, device=device)
            return val

        h_means3d_v_in = _get_or_zeros('h_means3d_v', (N, 6))
        h_conics_in = _get_or_zeros('h_conics', (N, 21))
        h_means3d_v_conics_in = _get_or_zeros('h_means3d_v_conics', (N, 18))
        h_means3d_v_opacity_in = _get_or_zeros('h_means3d_v_opacity', (N, 3))
        h_conics_opacity_in = _get_or_zeros('h_conics_opacity', (N, 6))

        # First backward with create_graph=True
        grads = torch.autograd.grad(
            outputs=[means3d_v, conics],
            inputs=[means, covars],
            grad_outputs=[v_means3d_v, v_conics],
            create_graph=True,
            allow_unused=True,
        )
        v_means = grads[0] if grads[0] is not None else torch.zeros_like(means)
        v_covars = grads[1] if grads[1] is not None else torch.zeros_like(covars)

        # ==================================================================
        # Compute curvature terms via double backward
        # ==================================================================
        # Upper-tri index helpers
        def utri3_idx(i, j):
            if i > j:
                i, j = j, i
            return i * (5 - i) // 2 + j

        def utri6_idx(i, j):
            if i > j:
                i, j = j, i
            return i * (11 - i) // 2 + j

        # Curvature H_{covars, covars}: d(v_covars)/d(covars)
        curv_covars = torch.zeros(N, 21, dtype=dtype, device=device)
        if v_covars.requires_grad or v_covars.grad_fn is not None:
            for k in range(6):
                e_k = torch.zeros(N, 6, dtype=dtype, device=device)
                e_k[:, k] = 1.0
                g2 = torch.autograd.grad(
                    v_covars, covars, grad_outputs=e_k,
                    retain_graph=True, allow_unused=True,
                )[0]
                if g2 is not None:
                    for l in range(k, 6):
                        curv_covars[:, utri6_idx(k, l)] = g2[:, l]

        # Curvature H_{means, means} and H_{means, covars} are zero
        # because means → means3d_v is affine (linear Jacobian, no curvature).
        # means does not appear in the conics path at all.

        # ==================================================================
        # Compute GN terms analytically
        # ==================================================================
        # The Jacobians for this module:
        #   J_A: means → means3d_v:  J_A[k][l] = r[k] * delta_{kl}  (diagonal)
        #   J_B: covars → conics:    handled via J_inv (6x6 matrix)
        #
        # For J_A (diagonal with entries r[k]):
        #   H_{means,means} = J_A^T @ h_means3d_v @ J_A
        #                   = r[k]*r[l] * h_means3d_v[utri(k,l)]
        #
        # For J_B (covars → Sv → conics), we need the full 6x6 Jacobian.
        # Rather than rebuilding J_inv here, we use a simpler approach:
        # Compute J_B numerically via autograd Jacobian.

        dims_t = torch.tensor([dim2, dim1, dim0], dtype=dtype, device=device)
        r = dims_t / ranges  # (3,)

        # --- GN: H_{means, means} = r[i]*r[j] * h_means3d_v_in[utri(i,j)] ---
        h_means_gn = torch.zeros(N, 6, dtype=dtype, device=device)
        idx = 0
        for i in range(3):
            for j in range(i, 3):
                h_means_gn[:, idx] = r[i] * r[j] * h_means3d_v_in[:, idx]
                idx += 1

        # --- GN: H_{covars, covars} = J_B^T @ h_conics_in @ J_B ---
        # We compute J_B per Gaussian using autograd.
        # J_B[out_flat, in_flat] = d(conics[n, out_flat]) / d(covars[n, in_flat])
        # Since we already have conics as a function of covars with create_graph,
        # we can extract J_B via autograd.

        # Build J_B: (N, 6, 6) where J_B[n, a, b] = d(conics[n,a])/d(covars[n,b])
        J_B = torch.zeros(N, 6, 6, dtype=dtype, device=device)
        for a in range(6):
            e_a = torch.zeros(N, 6, dtype=dtype, device=device)
            e_a[:, a] = 1.0
            g = torch.autograd.grad(
                conics, covars, grad_outputs=e_a,
                retain_graph=True, allow_unused=True,
            )[0]
            if g is not None:
                J_B[:, a, :] = g

        # Expand h_conics_in from upper-tri (N, 21) to full (N, 6, 6) symmetric
        h_conics_full = torch.zeros(N, 6, 6, dtype=dtype, device=device)
        idx = 0
        for i in range(6):
            for j in range(i, 6):
                h_conics_full[:, i, j] = h_conics_in[:, idx]
                h_conics_full[:, j, i] = h_conics_in[:, idx]
                idx += 1

        # GN term: J_B^T @ h_conics_full @ J_B  → (N, 6, 6)
        h_covars_gn_full = torch.bmm(J_B.transpose(1, 2), torch.bmm(h_conics_full, J_B))

        # Extract upper-tri (N, 21)
        h_covars_gn = torch.zeros(N, 21, dtype=dtype, device=device)
        idx = 0
        for i in range(6):
            for j in range(i, 6):
                h_covars_gn[:, idx] = h_covars_gn_full[:, i, j]
                idx += 1

        # --- GN: H_{means, covars} = J_A^T @ h_means3d_v_conics_in @ J_B ---
        # h_means3d_v_conics_in is (N, 18) in 3x6 row-major
        # J_A is diag(r[0], r[1], r[2])
        # Result is (N, 3, 6) → flattened to (N, 18)
        h_mc_in = h_means3d_v_conics_in.reshape(N, 3, 6)  # (N, 3, 6)
        # J_A^T @ h_mc_in = diag(r) @ h_mc_in → scale rows by r
        JA_h_mc = h_mc_in * r.unsqueeze(0).unsqueeze(-1)  # (N, 3, 6)
        # @ J_B: (N, 3, 6) @ (N, 6, 6) → (N, 3, 6)
        h_means_covars_gn = torch.bmm(JA_h_mc, J_B)  # (N, 3, 6)

        # --- Cross: H_{means, opacity} = J_A^T @ h_means3d_v_opacity_in ---
        # h_means3d_v_opacity_in is (N, 3)
        # J_A^T = diag(r)
        h_means_opacity = h_means3d_v_opacity_in * r.unsqueeze(0)  # (N, 3)

        # --- Cross: H_{covars, opacity} = J_B^T @ h_conics_opacity_in ---
        # h_conics_opacity_in is (N, 6)
        # J_B^T @ h_conics_opacity: (N, 6, 6) @ (N, 6, 1) → (N, 6)
        h_covars_opacity = torch.bmm(
            J_B.transpose(1, 2),
            h_conics_opacity_in.unsqueeze(-1)
        ).squeeze(-1)  # (N, 6)

        # ==================================================================
        # Combine: total Hessian = GN term + curvature term
        # ==================================================================
        h_means = h_means_gn  # curvature for means is zero (affine)
        h_covars_out = h_covars_gn + curv_covars
        h_means_covars = h_means_covars_gn.reshape(N, 18)
        # h_means_opacity and h_covars_opacity have no curvature contribution
        # (opacity doesn't appear in this module's forward)

        forward_result = ForwardResult(
            outputs={
                'means3d_v': means3d_v.detach(),
                'conics': conics.detach(),
                'radii': radii,
            },
            intermediates={}
        )
        grad_result = GradResult(
            param_grads={
                'means': v_means.detach(),
                'covars': v_covars.detach(),
            }
        )
        hess_result = HessResult(
            param_hess={
                'means': h_means,
                'covars': h_covars_out,
                'means_covars': h_means_covars,
                'means_opacity': h_means_opacity,
                'covars_opacity': h_covars_opacity,
            }
        )

        return forward_result, grad_result, hess_result


# ============================================================================
# Factory Functions
# ============================================================================

def create_world_to_volume_mapping(
    backend: str = "cuda",
) -> SecondForwardBackwardBase:
    """Create a WorldToVolumeMapping operator.

    Args:
        backend: "cuda" for CUDA implementation, "autograd" for PyTorch reference

    Returns:
        WorldToVolumeMapping operator instance
    """
    if backend == "cuda":
        return WorldToVolumeMappingCUDA()
    elif backend == "autograd":
        return WorldToVolumeMappingAutograd()
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'cuda' or 'autograd'.")
