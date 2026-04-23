"""
GS3DCT diagonal-clamped Newton optimizer.

This optimizer is implemented directly on top of ``NewtonOptimizerBase`` and
uses per-parameter diagonal Hessian updates only:

    delta_i = -g_i / (damping + clamp(h_ii, min=clamp_min))

Compared with the generic Newton optimizer, this variant explicitly enforces a
DIAGONAL update path and supports per-parameter-group Hessian diagonal clamp.
"""

import math
from dataclasses import dataclass, field
from typing import Dict

import torch

from gsplat2ct.ops.data import Params
from .base import HessianMode, NewtonOptimizerBase, SecondOrderConfig, SignalStrategy


PARAM_NAMES = ["means", "quats", "scale_ori", "opacity"]
PARAM_DIMS = [3, 4, 3, 1]

# Upper-tri diagonal indices for compact Hessians.
_DIAG_INDICES = {
    "means": [0, 3, 5],
    "quats": [0, 4, 7, 9],
    "scale_ori": [0, 3, 5],
}


def _upper_tri_to_symmetric(h: torch.Tensor, d: int) -> torch.Tensor:
    """Convert (N, d*(d+1)/2) upper-tri tensor to (N, d, d) symmetric matrix."""
    n_rows = h.shape[0]
    mat = torch.zeros(n_rows, d, d, device=h.device, dtype=h.dtype)
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


@dataclass
class DiagonalClampSecondOrderConfig(SecondOrderConfig):
    """Second-order config with per-parameter Hessian diagonal clamp."""

    clamp_map: Dict[str, float] = field(default_factory=dict)
    max_update_map: Dict[str, float] = field(default_factory=dict)

    def get_clamp(self, name: str, default: float = 1e-5) -> float:
        return float(self.clamp_map.get(name, default))

    def set_clamp(self, name: str, clamp_value: float) -> None:
        self.clamp_map[name] = float(clamp_value)

    def get_max_update(self, name: str, default: float = float("inf")) -> float:
        return float(self.max_update_map.get(name, default))

    def set_max_update(self, name: str, max_update: float) -> None:
        self.max_update_map[name] = float(max_update)


class GS3DCTNewtonDiagClampOptimizer(NewtonOptimizerBase):
    """Diagonal-clamped damped Newton optimizer for GS3DCT.

    Notes:
    - This optimizer only supports DIAGONAL Hessian mode.
    - Hessian compact formats follow GS3DCT conventions:
      means (N,6), quats (N,10), scale_ori (N,6), opacity (N,).
    """

    def __init__(self, params: Params, config: DiagonalClampSecondOrderConfig):
        if config.hess_mode != HessianMode.DIAGONAL:
            raise ValueError(
                "GS3DCTNewtonDiagClampOptimizer only supports hess_mode=DIAGONAL"
            )
        super().__init__(params, config, PARAM_NAMES, PARAM_DIMS, cross_hess_names=[])
        self.config: DiagonalClampSecondOrderConfig = config

    def _verbose_dump_step(
        self,
        *,
        name: str,
        grad_raw: torch.Tensor,
        grad_proc: torch.Tensor,
        hess_raw_diag: torch.Tensor,
        hess_proc_diag: torch.Tensor,
        hess_clamped: torch.Tensor,
        delta: torch.Tensor,
        update_raw: torch.Tensor,
        update: torch.Tensor,
        hess_clamp_hits: int,
        hess_total: int,
        update_clip_hits: int,
        update_total: int,
        clamp_min: float,
        max_update: float,
        update_clip_enabled: bool,
        param_before: torch.Tensor,
        param_after: torch.Tensor,
    ) -> None:
        if not bool(getattr(self.config, "verbose", False)):
            return

        print(f"[NewtonDiagClamp][step={self._step_count}][{name}]")
        print(f"  param_orig= {self._tensor_quantile_str(param_before)}")
        print(f"  grad_raw  = {self._tensor_quantile_str(grad_raw)}")
        print(f"  grad_proc = {self._tensor_quantile_str(grad_proc)}")
        print(f"  hess_raw  = {self._tensor_quantile_str(hess_raw_diag)}")
        print(f"  hess_proc = {self._tensor_quantile_str(hess_proc_diag)}")
        print(f"  h_clamped = {self._tensor_quantile_str(hess_clamped)}")
        print(f"  delta     = {self._tensor_quantile_str(delta)}")
        print(f"  update_raw= {self._tensor_quantile_str(update_raw)}")
        print(f"  update    = {self._tensor_quantile_str(update)}")
        print(
            "  h_clamp_hits"
            f"= {hess_clamp_hits}/{hess_total}"
            f" (clamp_min={clamp_min:.4e})"
        )
        if update_clip_enabled:
            print(
                "  update_clip_hits"
                f"= {update_clip_hits}/{update_total}"
                f" (max_update={max_update:.4e})"
            )
        else:
            print(f"  update_clip_hits= {update_clip_hits}/{update_total} (disabled)")
        print(f"  param_post= {self._tensor_quantile_str(param_after)}")

    def step(self) -> None:
        """Process signals and apply per-element diagonal-clamped Newton step."""
        self._step_count += 1
        if bool(getattr(self.config, "warmup_beta_verbose", False)):
            beta_g, local_g, steps_g = self._resolve_beta1("grad", return_meta=True)
            beta_h, local_h, steps_h = self._resolve_beta1("hess", return_meta=True)
            print(
                f"[WarmupBeta][step={self._step_count}] "
                f"beta_g={beta_g:.6f} (local={local_g}/{max(steps_g, 1)}), "
                f"beta_h={beta_h:.6f} (local={local_h}/{max(steps_h, 1)})"
            )

        if (
            bool(getattr(self.config, "topology_damping_reanneal_verbose", False))
            and self._damping_reanneal_active
        ):
            mult = self._resolve_damping_multiplier()
            local = int(self._step_count) - int(self._damping_reanneal_anchor_step)
            print(
                f"[DampingReanneal][step={self._step_count}] "
                f"multiplier={mult:.4g} (local={local}/"
                f"{max(self._damping_reanneal_steps, 1)})"
            )

        g_proc = self._process_signal(self._grad, "grad")
        h_proc = self._process_signal(self._hess, "hess")

        for name in self.param_names:
            if name not in g_proc or name not in h_proc:
                continue

            grad_raw = self._grad[name]
            grad = g_proc[name]

            hess_raw_diag = self._extract_hess_diagonal(name, self._hess[name])
            hess_diag = self._extract_hess_diagonal(name, h_proc[name])

            lr = self.config.get_lr(name)
            damping = self._effective_damping(name)
            clamp_min = self.config.get_clamp(name)
            max_update = self.config.get_max_update(name)
            verbose = bool(getattr(self.config, "verbose", False))

            hess_clamped = torch.clamp(hess_diag, min=clamp_min)
            hess_clamp_hits = int((hess_diag < clamp_min).sum().item())
            hess_total = int(hess_diag.numel())
            delta = -grad / (damping + hess_clamped)
            update_raw = lr * delta
            update = update_raw
            update_total = int(update_raw.numel())
            update_clip_hits = 0
            update_clip_enabled = bool(math.isfinite(max_update) and max_update > 0.0)

            if update_clip_enabled:
                update_clip_hits = int((update_raw.abs() > max_update).sum().item())
                update = torch.clamp(update_raw, min=-max_update, max=max_update)

            p = self.params.get(name).data
            param_before = p.clone() if verbose else p
            p += update

            if verbose:
                self._verbose_dump_step(
                    name=name,
                    grad_raw=grad_raw,
                    grad_proc=grad,
                    hess_raw_diag=hess_raw_diag,
                    hess_proc_diag=hess_diag,
                    hess_clamped=hess_clamped,
                    delta=delta,
                    update_raw=update_raw,
                    update=update,
                    hess_clamp_hits=hess_clamp_hits,
                    hess_total=hess_total,
                    update_clip_hits=update_clip_hits,
                    update_total=update_total,
                    clamp_min=clamp_min,
                    max_update=max_update,
                    update_clip_enabled=update_clip_enabled,
                    param_before=param_before,
                    param_after=p,
                )

        self._apply_constraints()

    def _extract_hess_diagonal(self, name: str, h: torch.Tensor) -> torch.Tensor:
        if name in _DIAG_INDICES:
            return h[:, _DIAG_INDICES[name]]
        # opacity is already scalar (N,)
        return h

    def _reconstruct_block(self, name: str, h: torch.Tensor) -> torch.Tensor:
        if name == "means":
            return _upper_tri_to_symmetric(h, 3)
        if name == "quats":
            return _upper_tri_to_symmetric(h, 4)
        if name == "scale_ori":
            return _upper_tri_to_symmetric(h, 3)
        if name == "opacity":
            return h.unsqueeze(-1).unsqueeze(-1)
        raise ValueError(f"Unknown parameter: {name}")

    def _assemble_full_hessian(
        self,
        h: Dict[str, torch.Tensor],
        ch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError(
            "GS3DCTNewtonDiagClampOptimizer is diagonal-only and does not assemble full Hessians"
        )

    def _apply_constraints(self) -> None:
        _renormalize_quats(self.params)
        p = self.params.get("opacity")
        allow_negative_opacity = bool(getattr(self.config, "allow_negative_opacity", False))
        if p is not None and not allow_negative_opacity:
            p.data.clamp_(min=0.0)

    def state_dict(self) -> Dict:
        state = super().state_dict()
        state["clamp_map"] = self.config.clamp_map.copy()
        state["max_update_map"] = self.config.max_update_map.copy()
        return state

    def load_state_dict(self, state: Dict) -> None:
        super().load_state_dict(state)
        if "clamp_map" in state:
            self.config.clamp_map = state["clamp_map"].copy()
        if "max_update_map" in state:
            self.config.max_update_map = state["max_update_map"].copy()


def create_gs3d_ct_newton_diag_clamp(
    params: Params,
    lr: float = 1.0,
    damping: float = 1e-3,
    clamp_min: float = 1e-5,
    max_update: float = float("inf"),
    grad_strategy: SignalStrategy = SignalStrategy.RAW,
    hess_strategy: SignalStrategy = SignalStrategy.RAW,
    **kwargs,
) -> GS3DCTNewtonDiagClampOptimizer:
    """Create diagonal-clamped GS3DCT Newton optimizer with uniform defaults."""
    lr_map = {name: lr for name in PARAM_NAMES}
    damping_map = {name: damping for name in PARAM_NAMES}
    clamp_map = {name: clamp_min for name in PARAM_NAMES}
    max_update_map = {name: max_update for name in PARAM_NAMES}

    config = DiagonalClampSecondOrderConfig(
        lr_map=lr_map,
        damping_map=damping_map,
        clamp_map=clamp_map,
        max_update_map=max_update_map,
        hess_mode=HessianMode.DIAGONAL,
        grad_strategy=grad_strategy,
        hess_strategy=hess_strategy,
        **kwargs,
    )
    return GS3DCTNewtonDiagClampOptimizer(params, config)


__all__ = [
    "PARAM_NAMES",
    "PARAM_DIMS",
    "DiagonalClampSecondOrderConfig",
    "GS3DCTNewtonDiagClampOptimizer",
    "create_gs3d_ct_newton_diag_clamp",
]
