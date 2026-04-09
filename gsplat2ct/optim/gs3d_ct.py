"""
GS3DCT-specific optimizers for 3DGS-based CT reconstruction.

Provides concrete Adam and Newton optimizers that understand the Hessian
formats produced by ``GS3DCTLossCUDA.forward_backward2``.

Parameter groups and their Hessian formats:
    means:     grad (N,3),  self-hess (N,6)  upper-tri 3×3
    quats:     grad (N,4),  self-hess (N,10) upper-tri 4×4
    scale_ori: grad (N,3),  self-hess (N,3)  diagonal 3×3
    opacity:   grad (N,),   self-hess (N,)   scalar

Cross-Hessian map (name -> shape -> semantic):
    quats_means:        (N,12)  4×3   ∂²L/∂quats∂means
    quats_scale_ori:    (N,12)  4×3   ∂²L/∂quats∂scale_ori
    quats_opacity:      (N,4)   4×1   ∂²L/∂quats∂opacity
    scale_ori_means:    (N,9)   3×3   ∂²L/∂scale_ori∂means
    scale_ori_opacity:  (N,3)   3×1   ∂²L/∂scale_ori∂opacity
    means_opacity:      (N,3)   3×1   ∂²L/∂means∂opacity

Full Hessian layout (D=11 per Gaussian):
    [means(3) | quats(4) | scale_ori(3) | opacity(1)]

Author: Dy164
Date: 2026-03-25
"""

from typing import Dict, Optional, Tuple

import torch

from gsplat2ct.ops.data import Params, GradResult
from .base import (
    FirstOrderOptimizerBase,
    FirstOrderConfig,
    NewtonOptimizerBase,
    SecondOrderConfig,
    HessianMode,
    SignalStrategy,
)


# ============================================================================
# Constants
# ============================================================================

PARAM_NAMES = ["means", "quats", "scale_ori", "opacity"]
PARAM_DIMS = [3, 4, 3, 1]

CROSS_HESS_NAMES = [
    "quats_means",
    "quats_scale_ori",
    "quats_opacity",
    "scale_ori_means",
    "scale_ori_opacity",
    "means_opacity",
]

# Upper-tri indices that correspond to the diagonal of the full matrix.
# For d=3: positions of H00, H11, H22 in the 6-element upper-tri vector.
# For d=4: positions of H00, H11, H22, H33 in the 10-element vector.
_DIAG_INDICES = {
    "means": [0, 3, 5],
    "quats": [0, 4, 7, 9],
}

# Cross-Hessian specification: (row_param, col_param, n_rows, n_cols)
_CROSS_MAP = {
    "quats_means":       ("quats",     "means",     4, 3),
    "quats_scale_ori":   ("quats",     "scale_ori", 4, 3),
    "quats_opacity":     ("quats",     "opacity",   4, 1),
    "scale_ori_means":   ("scale_ori", "means",     3, 3),
    "scale_ori_opacity": ("scale_ori", "opacity",   3, 1),
    "means_opacity":     ("means",     "opacity",   3, 1),
}


# ============================================================================
# Utility
# ============================================================================

def upper_tri_to_symmetric(h: torch.Tensor, d: int) -> torch.Tensor:
    """Convert (N, d*(d+1)/2) upper-tri to (N, d, d) symmetric matrix."""
    N = h.shape[0]
    mat = torch.zeros(N, d, d, device=h.device, dtype=h.dtype)
    idx = 0
    for i in range(d):
        for j in range(i, d):
            mat[:, i, j] = h[:, idx]
            if i != j:
                mat[:, j, i] = h[:, idx]
            idx += 1
    return mat


def _renormalize_quats(params: Params, eps: float = 1e-12) -> None:
    """Project quaternions back to unit sphere in-place."""
    p = params.get("quats")
    if p is None:
        return
    q = p.data
    if q.dim() != 2 or q.shape[-1] != 4:
        return
    q_norm = q.norm(dim=-1, keepdim=True).clamp_min(eps)
    q.div_(q_norm)


# ============================================================================
# First-Order: Adam (Scheme A)
# ============================================================================

class GS3DCTAdamOptimizer(FirstOrderOptimizerBase):
    """Adam optimizer for GS3DCT parameters (Scheme A baseline).

    Example::

        config = FirstOrderConfig(
            lr_map={'means': 1e-3, 'quats': 1e-3,
                    'scale_ori': 1e-3, 'opacity': 5e-3},
        )
        opt = GS3DCTAdamOptimizer(params, config)
        # training loop:
        fwd, grad = operator.forward_backward1(params, data, cfg)
        opt.add_grad(grad)
        opt.step()
        opt.zero_grad()
    """

    def __init__(self, params: Params, config: FirstOrderConfig):
        super().__init__(params, config, PARAM_NAMES)

    def _apply_constraints(self) -> None:
        _renormalize_quats(self.params)
        p = self.params.get("opacity")
        if p is not None:
            p.data.clamp_(min=0.0)


# ============================================================================
# Second-Order: Newton (Schemes B / C / D / E × Modes 1 / 2 / 3)
# ============================================================================

class GS3DCTNewtonOptimizer(NewtonOptimizerBase):
    """Newton-family optimizer for GS3DCT parameters.

    A single class that covers **all** combinations of signal-processing
    strategy (B/C/D/E) and Hessian-utilisation mode (DIAGONAL/BLOCK/FULL),
    selected entirely through ``SecondOrderConfig``.

    Example — Scheme D (Adam-norm Newton) with BLOCK mode::

        config = SecondOrderConfig(
            lr_map={'means': 1.0, 'quats': 1.0,
                    'scale_ori': 1.0, 'opacity': 1.0},
            damping_map={'means': 1e-3, 'quats': 1e-3,
                         'scale_ori': 1e-3, 'opacity': 1e-3},
            hess_mode=HessianMode.BLOCK,
            grad_strategy=SignalStrategy.ADAM_NORM,
            hess_strategy=SignalStrategy.ADAM_NORM,
        )
        opt = GS3DCTNewtonOptimizer(params, config)
        # training loop:
        fwd, grad, hess = operator.forward_backward2(params, data, cfg)
        opt.add_grad_hess(grad, hess)
        opt.step()
        opt.zero_grad()
    """

    def __init__(self, params: Params, config: SecondOrderConfig):
        super().__init__(
            params, config, PARAM_NAMES, PARAM_DIMS, CROSS_HESS_NAMES,
        )
        # Pre-compute offsets for full-block assembly
        self._offsets: Dict[str, Tuple[int, int]] = {}
        off = 0
        for name, d in zip(PARAM_NAMES, PARAM_DIMS):
            self._offsets[name] = (off, off + d)
            off += d

    # ---- Hessian format helpers (DIAGONAL) ---------------------------------

    def _extract_hess_diagonal(
        self, name: str, h: torch.Tensor,
    ) -> torch.Tensor:
        if name in _DIAG_INDICES:
            return h[:, _DIAG_INDICES[name]]
        # scale_ori (N,3) and opacity (N,) are already element-wise
        return h

    # ---- Hessian format helpers (BLOCK) ------------------------------------

    def _reconstruct_block(
        self, name: str, h: torch.Tensor,
    ) -> torch.Tensor:
        if name == "means":
            return upper_tri_to_symmetric(h, 3)
        elif name == "quats":
            return upper_tri_to_symmetric(h, 4)
        elif name == "scale_ori":
            return torch.diag_embed(h)           # (N,3) → (N,3,3)
        elif name == "opacity":
            return h.unsqueeze(-1).unsqueeze(-1)  # (N,) → (N,1,1)
        raise ValueError(f"Unknown parameter: {name}")

    # ---- Hessian format helpers (FULL) -------------------------------------

    def _assemble_full_hessian(
        self,
        h: Dict[str, torch.Tensor],
        ch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Build the full (N, 11, 11) Hessian per Gaussian.

        Parameter order: means(3), quats(4), scale_ori(3), opacity(1).
        """
        N = next(iter(h.values())).shape[0]
        D = self.total_dim  # 11
        device = next(iter(h.values())).device
        dtype = next(iter(h.values())).dtype

        H = torch.zeros(N, D, D, device=device, dtype=dtype)

        # Diagonal (self) blocks
        for name in PARAM_NAMES:
            if name not in h:
                continue
            i0, i1 = self._offsets[name]
            H[:, i0:i1, i0:i1] = self._reconstruct_block(name, h[name])

        # Off-diagonal (cross) blocks — symmetric
        for cross_name, (name_a, name_b, da, db) in _CROSS_MAP.items():
            if cross_name not in ch:
                continue
            a0, a1 = self._offsets[name_a]
            b0, b1 = self._offsets[name_b]
            block = ch[cross_name].reshape(N, da, db)
            H[:, a0:a1, b0:b1] = block
            H[:, b0:b1, a0:a1] = block.transpose(-1, -2)

        return H

    # ---- constraints -------------------------------------------------------

    def _apply_constraints(self) -> None:
        _renormalize_quats(self.params)
        p = self.params.get("opacity")
        if p is not None:
            p.data.clamp_(min=0.0)


# ============================================================================
# Factory helpers
# ============================================================================

def create_gs3d_ct_adam(
    params: Params,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    **per_param_lr,
) -> GS3DCTAdamOptimizer:
    """Create an Adam optimizer with uniform or per-parameter learning rates.

    Args:
        params: parameter container
        lr: default learning rate for all parameters
        per_param_lr: overrides, e.g. ``opacity=5e-3``
    """
    lr_map = {name: lr for name in PARAM_NAMES}
    lr_map.update(per_param_lr)
    config = FirstOrderConfig(lr_map=lr_map, beta1=beta1, beta2=beta2, eps=eps)
    return GS3DCTAdamOptimizer(params, config)


def create_gs3d_ct_newton(
    params: Params,
    lr: float = 1.0,
    damping: float = 1e-3,
    hess_mode: HessianMode = HessianMode.DIAGONAL,
    grad_strategy: SignalStrategy = SignalStrategy.RAW,
    hess_strategy: SignalStrategy = SignalStrategy.RAW,
    **kwargs,
) -> GS3DCTNewtonOptimizer:
    """Create a Newton optimizer with sensible defaults.

    Args:
        params: parameter container
        lr: default learning rate
        damping: default damping coefficient
        hess_mode: DIAGONAL / BLOCK / FULL
        grad_strategy: RAW / EMA / ADAM_NORM
        hess_strategy: RAW / EMA / ADAM_NORM
        **kwargs: forwarded to SecondOrderConfig (e.g. beta1_g, eps_h, ...)
    """
    lr_map = {name: lr for name in PARAM_NAMES}
    damping_map = {name: damping for name in PARAM_NAMES}
    config = SecondOrderConfig(
        lr_map=lr_map,
        damping_map=damping_map,
        hess_mode=hess_mode,
        grad_strategy=grad_strategy,
        hess_strategy=hess_strategy,
        **kwargs,
    )
    return GS3DCTNewtonOptimizer(params, config)


__all__ = [
    "PARAM_NAMES",
    "PARAM_DIMS",
    "CROSS_HESS_NAMES",
    "upper_tri_to_symmetric",
    "GS3DCTAdamOptimizer",
    "GS3DCTNewtonOptimizer",
    "create_gs3d_ct_adam",
    "create_gs3d_ct_newton",
]
