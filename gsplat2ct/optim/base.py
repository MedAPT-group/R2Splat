"""
Optimizer base classes for 3DGS-based CT reconstruction.

Architecture:

    OptimizerConfig ─── FirstOrderConfig
                    └── SecondOrderConfig

    OptimizerBase ─── FirstOrderOptimizerBase  (Adam family, Scheme A)
                  └── NewtonOptimizerBase       (Newton family, Schemes B/C/D/E)

HessianMode controls how second-order information structures the Newton step:
    DIAGONAL: per-element  δ_i = -g_i / (λ + |h_ii|)
    BLOCK:    per-group    δ = (H_block + λI)^{-1} (-g)
    FULL:     per-Gaussian δ = (H_full + Λ)^{-1} (-g)

SignalStrategy controls how raw gradient/Hessian signals are pre-processed:
    RAW:       use directly (Scheme B — Naive Newton)
    EMA:       exponential moving average (bias correction configurable) (Scheme C)
    ADAM_NORM: Adam-style m1_hat / (√m2_hat + ε) normalization (Scheme D)

Predefined optimizer schemes (configured via SecondOrderConfig):
    B (Naive):     grad_strategy=RAW,       hess_strategy=RAW
    C (EMA):       grad_strategy=EMA,       hess_strategy=EMA
    D (Adam-Norm): grad_strategy=ADAM_NORM,  hess_strategy=ADAM_NORM
    E (Hybrid):    grad_strategy=ADAM_NORM,  hess_strategy=EMA

Legacy classes (OptimizerConfig, SecondOrderOptimizerBase) are preserved at the
bottom of this file for backward compatibility with gs2d, gs3d_smoothing, etc.

Author: Dy164
Date: 2026-03-25
"""

from abc import ABC, abstractmethod
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import torch

from gsplat2ct.ops.data import Params, GradResult, HessResult


# ============================================================================
# Enumerations
# ============================================================================

class HessianMode(Enum):
    """How second-order information is utilized in the Newton step."""
    DIAGONAL = 1
    BLOCK = 2
    FULL = 3


class SignalStrategy(Enum):
    """How gradient / Hessian signals are pre-processed before the step."""
    RAW = 0
    EMA = 1
    ADAM_NORM = 2


# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class OptimizerConfig:
    """Base configuration with per-parameter learning rates."""
    lr_map: Dict[str, float] = field(default_factory=dict)
    damping_map: Dict[str, float] = field(default_factory=dict)

    def get_lr(self, name: str, default: float = 1e-3) -> float:
        return self.lr_map.get(name, default)

    def set_lr(self, name: str, lr: float) -> None:
        self.lr_map[name] = lr

    def get_damping(self, name: str, default: float = 1e-3) -> float:
        return self.damping_map.get(name, default)

    def set_damping(self, name: str, damping: float) -> None:
        self.damping_map[name] = damping


@dataclass
class FirstOrderConfig(OptimizerConfig):
    """Configuration for Adam optimizer."""
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    verbose: bool = False


@dataclass
class SecondOrderConfig(OptimizerConfig):
    """Configuration for Newton-family optimizers.

    The combination of grad_strategy and hess_strategy selects the scheme:
        B: RAW       + RAW        (no moments)
        C: EMA       + EMA        (1 moment each)
        D: ADAM_NORM + ADAM_NORM  (2 moments each)
        E: ADAM_NORM + EMA        (2 grad moments + 1 hess moment)
    """
    hess_mode: HessianMode = HessianMode.DIAGONAL
    grad_strategy: SignalStrategy = SignalStrategy.RAW
    hess_strategy: SignalStrategy = SignalStrategy.RAW

    beta1_g: float = 0.9
    beta2_g: float = 0.999
    eps_g: float = 1e-8

    beta1_h: float = 0.9
    beta2_h: float = 0.999
    eps_h: float = 1e-8

    # Optional warmup beta schedule for grad/hess EMA-like first moments.
    # When enabled, beta1 is scheduled from *_start to *_end during the first
    # warmup_beta_steps optimization steps.
    warmup_beta_enabled: bool = False
    warmup_beta_schedule: str = "COSINE"  # options: COSINE | LINEAR
    warmup_beta_steps: int = 0
    warmup_beta1_g_start: float = 0.1
    warmup_beta1_g_end: float = 0.9
    warmup_beta1_h_start: float = 0.1
    warmup_beta1_h_end: float = 0.9
    # Recommended default for warmup-beta mode: disable bias correction.
    warmup_beta_bias_correction: bool = False
    # Verbose logs for beta warmup/reanneal schedule only.
    warmup_beta_verbose: bool = False

    # Topology-triggered damping reanneal.
    # When enabled, after a split/duplicate/prune event damping is multiplied
    # by ``topology_damping_reanneal_factor`` (e.g. 100x) and decays back to
    # the base value over ``topology_damping_reanneal_steps`` optimizer steps.
    # This stabilizes Newton updates right after the geometry changes when
    # the diagonal Hessian is noisiest.
    topology_damping_reanneal_enabled: bool = False
    topology_damping_reanneal_factor: float = 100.0
    topology_damping_reanneal_steps: int = 50
    topology_damping_reanneal_schedule: str = "COSINE"  # COSINE | LINEAR | EXP
    topology_damping_reanneal_verbose: bool = False

    # Verbose per-parameter tensor dumps (optimizer-specific formatting).
    verbose: bool = False


# ============================================================================
# Abstract Base
# ============================================================================

class OptimizerBase(ABC):
    """Root abstract base class for all new optimizers."""

    def __init__(
        self,
        params: Params,
        config: OptimizerConfig,
        param_names: List[str],
    ):
        self.params = params
        self.config = config
        self.param_names = param_names
        self._step_count = 0

    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def zero_grad(self) -> None:
        raise NotImplementedError

    def _apply_constraints(self) -> None:
        """Override in subclasses to enforce parameter constraints."""
        pass

    def state_dict(self) -> Dict:
        return {"step_count": self._step_count}

    def load_state_dict(self, state: Dict) -> None:
        self._step_count = state["step_count"]

    @staticmethod
    def _tensor_quantile_str(x: torch.Tensor) -> str:
        """Return a compact fixed-quantile summary for verbose logging."""
        flat = x.detach().reshape(-1).to(torch.float64).cpu().numpy()
        total = int(flat.size)
        if total == 0:
            return "empty"

        finite_mask = np.isfinite(flat)
        finite = flat[finite_mask]
        n_finite = int(finite.size)
        n_nan = int(np.isnan(flat).sum())
        n_posinf = int(np.isposinf(flat).sum())
        n_neginf = int(np.isneginf(flat).sum())

        if n_finite == 0:
            return (
                "all_non_finite"
                f" (numel={total}, nan={n_nan}, +inf={n_posinf}, -inf={n_neginf})"
            )

        q_values = np.percentile(
            finite,
            [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100],
        )
        q_labels = [
            "min", "p05", "p10", "p20", "p30", "p40", "p50",
            "p60", "p70", "p80", "p90", "p95", "max",
        ]
        stats = ", ".join(
            f"{label}={float(value):.4e}"
            for label, value in zip(q_labels, q_values)
        )

        if n_finite != total:
            return (
                f"{stats}"
                f" (numel={total}, finite={n_finite}, nan={n_nan}, +inf={n_posinf}, -inf={n_neginf})"
            )
        return f"{stats} (numel={total})"


# ============================================================================
# First-Order Optimizer Base (Adam)
# ============================================================================

class FirstOrderOptimizerBase(OptimizerBase):
    """Adam optimizer base.

    Manages per-parameter first-moment (m1) and second-moment (m2) buffers.
    Subclasses only need to override ``_apply_constraints`` for task-specific
    parameter clamping.
    """

    def __init__(
        self,
        params: Params,
        config: FirstOrderConfig,
        param_names: List[str],
    ):
        super().__init__(params, config, param_names)
        self.config: FirstOrderConfig = config
        self._grad: Dict[str, torch.Tensor] = {}
        self._m1: Dict[str, torch.Tensor] = {}
        self._m2: Dict[str, torch.Tensor] = {}

    # ---- gradient interface ------------------------------------------------

    def add_grad(self, grad_result: GradResult) -> None:
        for name in self.param_names:
            g = grad_result.get(name)
            if g is not None:
                self._grad[name] = g

    def zero_grad(self) -> None:
        self._grad.clear()

    def reset_moments(self) -> None:
        self._m1.clear()
        self._m2.clear()
        self._step_count = 0

    @staticmethod
    def _append_zero_rows(state: torch.Tensor, n_new: int) -> torch.Tensor:
        if n_new <= 0:
            return state
        pad_shape = (int(n_new),) + tuple(state.shape[1:])
        pad = torch.zeros(pad_shape, device=state.device, dtype=state.dtype)
        return torch.cat([state, pad], dim=0).contiguous()

    @torch.no_grad()
    def apply_topology_transition(
        self,
        *,
        duplicate_sel: Optional[torch.Tensor] = None,
        split_keep_parent: Optional[torch.Tensor] = None,
        split_children: int = 0,
        prune_keep: Optional[torch.Tensor] = None,
        append_new: int = 0,
    ) -> None:
        """Migrate Adam moments across topology edits.

        This mirrors the 3DGS/r2-style behavior:
        - existing points keep their moments,
        - newly created points get zero-initialized moments,
        - removed points drop their moments.
        """
        if not self._m1 or not self._m2:
            return

        for name in self.param_names:
            m1 = self._m1.get(name)
            m2 = self._m2.get(name)
            if m1 is None or m2 is None:
                continue

            if duplicate_sel is not None and duplicate_sel.numel() > 0:
                m1 = self._append_zero_rows(m1, int(duplicate_sel.numel()))
                m2 = self._append_zero_rows(m2, int(duplicate_sel.numel()))

            if split_keep_parent is not None:
                keep_parent = split_keep_parent.to(device=m1.device, dtype=torch.bool)
                if keep_parent.numel() != m1.shape[0]:
                    self._m1.pop(name, None)
                    self._m2.pop(name, None)
                    continue
                m1 = m1[keep_parent].contiguous()
                m2 = m2[keep_parent].contiguous()
                if split_children > 0:
                    m1 = self._append_zero_rows(m1, int(split_children))
                    m2 = self._append_zero_rows(m2, int(split_children))

            if prune_keep is not None:
                keep = prune_keep.to(device=m1.device, dtype=torch.bool)
                if keep.numel() != m1.shape[0]:
                    self._m1.pop(name, None)
                    self._m2.pop(name, None)
                    continue
                m1 = m1[keep].contiguous()
                m2 = m2[keep].contiguous()

            if append_new > 0:
                m1 = self._append_zero_rows(m1, int(append_new))
                m2 = self._append_zero_rows(m2, int(append_new))

            self._m1[name] = m1
            self._m2[name] = m2

    # ---- optimisation step -------------------------------------------------

    def _verbose_dump_step(
        self,
        *,
        name: str,
        grad: torch.Tensor,
        update: torch.Tensor,
        param_before: torch.Tensor,
        param_after: torch.Tensor,
    ) -> None:
        if not bool(getattr(self.config, "verbose", False)):
            return
        print(f"[Adam][step={self._step_count}][{name}]")
        print(f"  param_orig= {self._tensor_quantile_str(param_before)}")
        print(f"  grad      = {self._tensor_quantile_str(grad)}")
        print(f"  update    = {self._tensor_quantile_str(update)}")
        print(f"  param_post= {self._tensor_quantile_str(param_after)}")

    def step(self) -> None:
        self._step_count += 1
        b1, b2, eps = self.config.beta1, self.config.beta2, self.config.eps
        verbose = bool(getattr(self.config, "verbose", False))

        for name in self.param_names:
            if name not in self._grad:
                continue
            g = self._grad[name]
            lr = self.config.get_lr(name)

            if name not in self._m1:
                self._m1[name] = torch.zeros_like(g)
                self._m2[name] = torch.zeros_like(g)
            elif self._m1[name].shape != g.shape or self._m2[name].shape != g.shape:
                self._m1[name] = torch.zeros_like(g)
                self._m2[name] = torch.zeros_like(g)

            self._m1[name].mul_(b1).add_(g, alpha=1 - b1)
            self._m2[name].mul_(b2).addcmul_(g, g, value=1 - b2)

            m1_hat = self._m1[name] / (1 - b1 ** self._step_count)
            m2_hat = self._m2[name] / (1 - b2 ** self._step_count)
            update = -lr * m1_hat / (m2_hat.sqrt() + eps)

            p = self.params.get(name).data
            param_before = p.clone() if verbose else p
            p += update

            if verbose:
                self._verbose_dump_step(
                    name=name,
                    grad=g,
                    update=update,
                    param_before=param_before,
                    param_after=p,
                )

        self._apply_constraints()

    # ---- serialisation -----------------------------------------------------

    def state_dict(self) -> Dict:
        return {
            "step_count": self._step_count,
            "m1": {k: v.clone() for k, v in self._m1.items()},
            "m2": {k: v.clone() for k, v in self._m2.items()},
        }

    def load_state_dict(self, state: Dict) -> None:
        self._step_count = state["step_count"]
        self._m1 = {k: v.clone() for k, v in state["m1"].items()}
        self._m2 = {k: v.clone() for k, v in state["m2"].items()}


# ============================================================================
# Second-Order (Newton) Optimizer Base
# ============================================================================

class NewtonOptimizerBase(OptimizerBase):
    """Newton-family optimizer base with configurable signal processing.

    The optimisation step is decomposed into two orthogonal axes:

    1. **Signal processing** (``_process_signal``):
       Transforms raw gradient / Hessian tensors before they enter the
       Newton step.  Controlled by ``config.grad_strategy`` and
       ``config.hess_strategy`` (RAW / EMA / ADAM_NORM).

    2. **Newton step mode** (``config.hess_mode``):
       Determines how the (possibly processed) Hessian is used:
       DIAGONAL — per-element division;
       BLOCK — per-parameter-group matrix solve;
       FULL — per-Gaussian full-block matrix solve.

    Subclasses must implement three format-specific helpers:
        ``_extract_hess_diagonal``  — for DIAGONAL mode
        ``_reconstruct_block``      — for BLOCK mode
        ``_assemble_full_hessian``  — for FULL mode
    """

    def __init__(
        self,
        params: Params,
        config: SecondOrderConfig,
        param_names: List[str],
        param_dims: List[int],
        cross_hess_names: Optional[List[str]] = None,
    ):
        super().__init__(params, config, param_names)
        self.config: SecondOrderConfig = config
        self.param_dims = param_dims
        self.cross_hess_names = cross_hess_names or []
        self.total_dim = sum(param_dims)

        self._grad: Dict[str, torch.Tensor] = {}
        self._hess: Dict[str, torch.Tensor] = {}
        self._cross_hess: Dict[str, torch.Tensor] = {}
        self._moments: Dict[str, Dict[str, torch.Tensor]] = {}
        # Anchor used to support repeated beta warmup restarts after topology changes.
        self._beta_warmup_anchor_step: int = 0
        self._beta_warmup_steps_override: Optional[int] = None
        # Topology-triggered damping reanneal state.
        # When active, effective damping = base_damping * multiplier(local_step),
        # with multiplier decaying from ``_damping_reanneal_factor`` down to 1.0
        # over ``_damping_reanneal_steps`` optimizer steps.
        self._damping_reanneal_anchor_step: int = 0
        self._damping_reanneal_factor: float = 1.0
        self._damping_reanneal_steps: int = 0
        self._damping_reanneal_schedule: str = "COSINE"
        self._damping_reanneal_active: bool = False

    # ---- gradient / Hessian interface --------------------------------------

    def add_grad(self, grad_result: GradResult) -> None:
        """Store first-order gradients (for first-order-only Newton, Scheme B)."""
        for name in self.param_names:
            g = grad_result.get(name)
            if g is not None:
                self._grad[name] = g

    def add_grad_hess(
        self,
        grad_result: GradResult,
        hess_result: HessResult,
    ) -> None:
        """Store gradients, self-Hessians, and cross-Hessians."""
        for name in self.param_names:
            g = grad_result.get(name)
            if g is not None:
                self._grad[name] = g
            h = hess_result.get(name)
            if h is not None:
                self._hess[name] = h

        for name in self.cross_hess_names:
            ch = hess_result.get_cross(name)
            if ch is not None:
                self._cross_hess[name] = ch

    def zero_grad(self) -> None:
        self._grad.clear()
        self._hess.clear()
        self._cross_hess.clear()

    def reset_moments(self) -> None:
        self._moments.clear()
        self._step_count = 0
        self._beta_warmup_anchor_step = 0
        self._beta_warmup_steps_override = None
        self._damping_reanneal_anchor_step = 0
        self._damping_reanneal_factor = 1.0
        self._damping_reanneal_steps = 0
        self._damping_reanneal_active = False

    def restart_damping_reanneal(
        self,
        *,
        factor: Optional[float] = None,
        steps: Optional[int] = None,
        schedule: Optional[str] = None,
    ) -> None:
        """Start a damping reanneal cycle anchored at the current step.

        Effective damping becomes ``base_damping * multiplier(local_step)`` where
        ``multiplier`` decays from ``factor`` down to 1.0 over ``steps`` optimizer
        steps according to ``schedule``. The base damping (as modified by any
        ``OptimizerScheduler``) is preserved; the reanneal is a multiplicative
        transient on top of it.

        Args:
            factor: Peak damping multiplier right after restart (default
                ``self.config.topology_damping_reanneal_factor``).
            steps: Number of optimizer steps for the reanneal to decay back to
                1.0 (default ``self.config.topology_damping_reanneal_steps``).
            schedule: Decay shape: ``"COSINE"``, ``"LINEAR"``, or ``"EXP"``
                (default ``self.config.topology_damping_reanneal_schedule``).
        """
        cfg_factor = float(
            getattr(self.config, "topology_damping_reanneal_factor", 100.0)
        )
        cfg_steps = int(
            getattr(self.config, "topology_damping_reanneal_steps", 50)
        )
        cfg_schedule = str(
            getattr(self.config, "topology_damping_reanneal_schedule", "COSINE")
        )
        self._damping_reanneal_factor = (
            float(factor) if factor is not None else cfg_factor
        )
        self._damping_reanneal_steps = max(
            int(steps) if steps is not None else cfg_steps, 0
        )
        self._damping_reanneal_schedule = (
            str(schedule).upper() if schedule is not None else str(cfg_schedule).upper()
        )
        self._damping_reanneal_anchor_step = int(self._step_count)
        self._damping_reanneal_active = bool(
            self._damping_reanneal_steps > 0 and self._damping_reanneal_factor > 1.0
        )

        if bool(getattr(self.config, "topology_damping_reanneal_verbose", False)):
            mult = self._resolve_damping_multiplier()
            print(
                "[DampingReanneal] restart "
                f"anchor_step={self._damping_reanneal_anchor_step}, "
                f"factor={self._damping_reanneal_factor:.4g}, "
                f"steps={self._damping_reanneal_steps}, "
                f"schedule={self._damping_reanneal_schedule}, "
                f"multiplier_now={mult:.4g}"
            )

    def _resolve_damping_multiplier(self) -> float:
        """Return the current damping reanneal multiplier (>= 1.0).

        Returns 1.0 when no reanneal is active or when the reanneal has
        fully decayed.
        """
        if not self._damping_reanneal_active:
            return 1.0
        steps = int(self._damping_reanneal_steps)
        if steps <= 0:
            return 1.0
        factor = float(self._damping_reanneal_factor)
        if factor <= 1.0:
            return 1.0

        local_step = int(self._step_count) - int(self._damping_reanneal_anchor_step)
        # local_step == 0 at the instant of restart, before any optimizer.step();
        # local_step == 1 at the first step after restart (peak multiplier);
        # local_step == steps at the end of the reanneal window (multiplier = 1).
        if local_step <= 0:
            return factor
        if local_step >= steps:
            self._damping_reanneal_active = False
            return 1.0

        denom = float(max(steps - 1, 1))
        progress = float(local_step - 1) / denom if steps > 1 else 1.0
        progress = max(0.0, min(progress, 1.0))

        schedule = self._damping_reanneal_schedule.upper()
        if schedule == "COSINE":
            alpha = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
            mult = 1.0 + (factor - 1.0) * alpha
        elif schedule == "LINEAR":
            alpha = 1.0 - progress  # 1 -> 0
            mult = 1.0 + (factor - 1.0) * alpha
        elif schedule in ("EXP", "EXPONENTIAL", "GEOMETRIC"):
            # Geometric (log-linear) decay: factor -> 1 in log space.
            mult = factor ** (1.0 - progress)
        else:
            raise ValueError(
                f"Unsupported damping reanneal schedule: {schedule}"
            )
        return float(mult)

    def _effective_damping(self, name: str) -> float:
        """Return base damping scaled by the current reanneal multiplier."""
        base = float(self.config.get_damping(name))
        return base * self._resolve_damping_multiplier()

    def restart_beta_warmup(
        self,
        *,
        reset_moments: bool = False,
        warmup_steps: Optional[int] = None,
    ) -> None:
        """Restart beta warmup from current optimizer step.

        Args:
            reset_moments: Whether to clear grad/hess moments before reannealing.
            warmup_steps: Optional per-restart warmup length override.
        """
        if reset_moments:
            self._moments.clear()
        self._beta_warmup_anchor_step = int(self._step_count)
        self._beta_warmup_steps_override = (
            None if warmup_steps is None else max(int(warmup_steps), 0)
        )
        if bool(getattr(self.config, "warmup_beta_verbose", False)):
            beta_g, local_g, steps_g = self._resolve_beta1("grad", return_meta=True)
            beta_h, local_h, steps_h = self._resolve_beta1("hess", return_meta=True)
            print(
                "[WarmupBeta] restart "
                f"reset_moments={bool(reset_moments)}, "
                f"anchor_step={self._beta_warmup_anchor_step}, "
                f"local_step_g={local_g}/{max(steps_g, 1)}, "
                f"local_step_h={local_h}/{max(steps_h, 1)}, "
                f"beta_g={beta_g:.6f}, beta_h={beta_h:.6f}"
            )

    @staticmethod
    def _append_zero_rows(state: torch.Tensor, n_new: int) -> torch.Tensor:
        if n_new <= 0:
            return state
        pad_shape = (int(n_new),) + tuple(state.shape[1:])
        pad = torch.zeros(pad_shape, device=state.device, dtype=state.dtype)
        return torch.cat([state, pad], dim=0).contiguous()

    @torch.no_grad()
    def _migrate_moment_buckets(
        self,
        buckets: Dict[str, Dict[str, torch.Tensor]],
        *,
        duplicate_sel: Optional[torch.Tensor] = None,
        split_keep_parent: Optional[torch.Tensor] = None,
        split_children: int = 0,
        prune_keep: Optional[torch.Tensor] = None,
        append_new: int = 0,
    ) -> None:
        if not buckets:
            return

        for bucket_name in list(buckets.keys()):
            slot = buckets.get(bucket_name)
            if slot is None:
                continue

            drop_bucket = False
            migrated_slot: Dict[str, torch.Tensor] = {}

            for state_name, state_tensor in slot.items():
                # Scalar tracking values (e.g. beta products) are global states
                # and do not need per-Gaussian topology remapping.
                if state_tensor.dim() == 0:
                    migrated_slot[state_name] = state_tensor
                    continue

                state = state_tensor

                if duplicate_sel is not None and duplicate_sel.numel() > 0:
                    state = self._append_zero_rows(state, int(duplicate_sel.numel()))

                if split_keep_parent is not None:
                    keep_parent = split_keep_parent.to(device=state.device, dtype=torch.bool)
                    if keep_parent.numel() != state.shape[0]:
                        drop_bucket = True
                        break
                    state = state[keep_parent].contiguous()
                    if split_children > 0:
                        state = self._append_zero_rows(state, int(split_children))

                if prune_keep is not None:
                    keep = prune_keep.to(device=state.device, dtype=torch.bool)
                    if keep.numel() != state.shape[0]:
                        drop_bucket = True
                        break
                    state = state[keep].contiguous()

                if append_new > 0:
                    state = self._append_zero_rows(state, int(append_new))

                migrated_slot[state_name] = state

            if drop_bucket:
                buckets.pop(bucket_name, None)
            else:
                buckets[bucket_name] = migrated_slot

    @torch.no_grad()
    def apply_topology_transition(
        self,
        *,
        duplicate_sel: Optional[torch.Tensor] = None,
        split_keep_parent: Optional[torch.Tensor] = None,
        split_children: int = 0,
        prune_keep: Optional[torch.Tensor] = None,
        append_new: int = 0,
    ) -> None:
        """Migrate second-order moment buffers across topology edits.

        Existing Gaussians keep their moment history. Newly added Gaussians
        receive zero-initialized moments.
        """
        self._migrate_moment_buckets(
            self._moments,
            duplicate_sel=duplicate_sel,
            split_keep_parent=split_keep_parent,
            split_children=split_children,
            prune_keep=prune_keep,
            append_new=append_new,
        )

    # ---- signal processing -------------------------------------------------

    def _process_signal(
        self,
        buf: Dict[str, torch.Tensor],
        signal_type: str,
    ) -> Dict[str, torch.Tensor]:
        """Apply EMA or Adam-norm to a dict of tensors.

        Args:
            buf: raw signal tensors (gradient or Hessian buffers)
            signal_type: ``'grad'`` or ``'hess'`` — selects hyperparameters

        Returns:
            Processed signal dict (same keys, transformed values).
            For RAW strategy the original dict is returned as-is.
        """
        strategy = (
            self.config.grad_strategy
            if signal_type == "grad"
            else self.config.hess_strategy
        )

        if strategy == SignalStrategy.RAW:
            return buf

        b1 = self._resolve_beta1(signal_type)
        b2 = self.config.beta2_g if signal_type == "grad" else self.config.beta2_h
        eps = self.config.eps_g if signal_type == "grad" else self.config.eps_h
        apply_bias_correction = self._should_apply_moment_bias_correction()

        result = {}
        for name, val in buf.items():
            key = f"{signal_type}_{name}"
            if key not in self._moments:
                self._moments[key] = {}
            m = self._moments[key]

            if "m1" not in m or m["m1"].shape != val.shape:
                m["m1"] = torch.zeros_like(val)
                m.pop("m2", None)
                m["beta1_prod"] = torch.ones((), device=val.device, dtype=val.dtype)
                m["beta2_prod"] = torch.ones((), device=val.device, dtype=val.dtype)

            if (
                "beta1_prod" not in m
                or m["beta1_prod"].device != val.device
                or m["beta1_prod"].dtype != val.dtype
            ):
                m["beta1_prod"] = torch.ones((), device=val.device, dtype=val.dtype)
            m["m1"].mul_(b1).add_(val, alpha=1 - b1)
            m["beta1_prod"] = m["beta1_prod"] * b1

            if apply_bias_correction:
                denom1 = (1.0 - m["beta1_prod"]).clamp_min(1e-12)
                m1_used = m["m1"] / denom1
            else:
                m1_used = m["m1"]

            if strategy == SignalStrategy.EMA:
                result[name] = m1_used

            elif strategy == SignalStrategy.ADAM_NORM:
                if "m2" not in m or m["m2"].shape != val.shape:
                    m["m2"] = torch.zeros_like(val)
                    m["beta2_prod"] = torch.ones((), device=val.device, dtype=val.dtype)
                if (
                    "beta2_prod" not in m
                    or m["beta2_prod"].device != val.device
                    or m["beta2_prod"].dtype != val.dtype
                ):
                    m["beta2_prod"] = torch.ones((), device=val.device, dtype=val.dtype)
                m["m2"].mul_(b2).addcmul_(val, val, value=1 - b2)
                m["beta2_prod"] = m["beta2_prod"] * b2

                if apply_bias_correction:
                    denom2 = (1.0 - m["beta2_prod"]).clamp_min(1e-12)
                    m2_used = m["m2"] / denom2
                else:
                    m2_used = m["m2"]
                result[name] = m1_used / (m2_used.sqrt() + eps)

        return result

    def _resolve_beta1(
        self, signal_type: str, return_meta: bool = False
    ) -> float | tuple[float, int, int]:
        """Resolve beta1, optionally using warmup_beta schedule."""
        default_beta = self.config.beta1_g if signal_type == "grad" else self.config.beta1_h
        if not bool(getattr(self.config, "warmup_beta_enabled", False)):
            beta = float(default_beta)
            if return_meta:
                return beta, 0, 0
            return beta

        cfg_steps = max(int(getattr(self.config, "warmup_beta_steps", 0)), 0)
        steps = (
            cfg_steps
            if self._beta_warmup_steps_override is None
            else int(self._beta_warmup_steps_override)
        )
        if steps <= 0:
            beta = float(default_beta)
            if return_meta:
                return beta, 0, 0
            return beta

        if signal_type == "grad":
            beta_start = float(getattr(self.config, "warmup_beta1_g_start", default_beta))
            beta_end = float(getattr(self.config, "warmup_beta1_g_end", default_beta))
        else:
            beta_start = float(getattr(self.config, "warmup_beta1_h_start", default_beta))
            beta_end = float(getattr(self.config, "warmup_beta1_h_end", default_beta))

        local_step = max(int(self._step_count) - int(self._beta_warmup_anchor_step), 0)
        if steps == 1:
            progress = 1.0
        else:
            progress = min(max((local_step - 1) / float(steps - 1), 0.0), 1.0)

        schedule = str(getattr(self.config, "warmup_beta_schedule", "COSINE")).upper()
        if schedule == "COSINE":
            alpha = 0.5 * (1.0 - math.cos(math.pi * progress))
        elif schedule == "LINEAR":
            alpha = progress
        else:
            raise ValueError(f"Unsupported warmup_beta_schedule: {schedule}")

        beta = beta_start + (beta_end - beta_start) * alpha
        beta = float(min(max(beta, 1e-6), 1.0 - 1e-6))
        if return_meta:
            return beta, int(local_step), int(steps)
        return beta

    def _should_apply_moment_bias_correction(self) -> bool:
        """Bias-correction policy for grad/hess moments."""
        if bool(getattr(self.config, "warmup_beta_enabled", False)):
            return bool(getattr(self.config, "warmup_beta_bias_correction", False))
        return True

    # ---- optimisation step -------------------------------------------------

    def step(self) -> None:
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

        mode = self.config.hess_mode
        if mode == HessianMode.DIAGONAL:
            self._step_diagonal(g_proc, h_proc)
        elif mode == HessianMode.BLOCK:
            self._step_block(g_proc, h_proc)
        elif mode == HessianMode.FULL:
            ch_proc = self._process_signal(self._cross_hess, "hess")
            self._step_full(g_proc, h_proc, ch_proc)

        self._apply_constraints()

    def _step_diagonal(self, g: Dict, h: Dict) -> None:
        """Per-element damped Newton: δ_i = -g_i / (λ + |h_ii|)."""
        for name in self.param_names:
            if name not in g or name not in h:
                continue
            grad = g[name]
            hess_diag = self._extract_hess_diagonal(name, h[name])
            lr = self.config.get_lr(name)
            damping = self._effective_damping(name)

            delta = -grad / (damping + hess_diag.abs())
            self.params.get(name).data += lr * delta

    def _step_block(self, g: Dict, h: Dict) -> None:
        """Per-parameter-group block Newton: (H_block + λI)δ = -g."""
        for name, d in zip(self.param_names, self.param_dims):
            if name not in g or name not in h:
                continue
            grad = g[name]
            lr = self.config.get_lr(name)
            damping = self._effective_damping(name)

            if d == 1:
                hess_val = self._extract_hess_diagonal(name, h[name])
                delta = -grad / (damping + hess_val.abs())
            else:
                H_block = self._reconstruct_block(name, h[name])  # (N, d, d)
                N = grad.shape[0]
                eye = torch.eye(d, device=grad.device, dtype=grad.dtype)
                H_damped = H_block + damping * eye.unsqueeze(0)
                # Solve (H + λI) δ = -g  for each of the N Gaussians
                delta = torch.linalg.solve(H_damped, -grad.unsqueeze(-1)).squeeze(-1)

            self.params.get(name).data += lr * delta

    def _step_full(self, g: Dict, h: Dict, ch: Dict) -> None:
        """Full per-Gaussian block Newton: (H_full + Λ)δ = -g."""
        N = next(iter(g.values())).shape[0]
        device = next(iter(g.values())).device
        dtype = next(iter(g.values())).dtype
        D = self.total_dim

        # Concatenate gradients: (N, D)
        g_cat = torch.cat(
            [g[name].reshape(N, -1) for name in self.param_names], dim=-1
        )

        # Assemble full Hessian: (N, D, D)
        H_full = self._assemble_full_hessian(h, ch)

        # Per-parameter-group damping diagonal: (D,)
        damping_diag = torch.cat([
            torch.full((d,), self._effective_damping(name),
                       device=device, dtype=dtype)
            for name, d in zip(self.param_names, self.param_dims)
        ])
        H_damped = H_full + torch.diag(damping_diag).unsqueeze(0)

        # Solve (H + Λ) δ = -g
        delta_cat = torch.linalg.solve(
            H_damped, -g_cat.unsqueeze(-1)
        ).squeeze(-1)  # (N, D)

        # Split back and apply per-group learning rates
        offset = 0
        for name, d in zip(self.param_names, self.param_dims):
            lr = self.config.get_lr(name)
            delta = delta_cat[:, offset:offset + d]
            if d == 1:
                delta = delta.squeeze(-1)
            self.params.get(name).data += lr * delta
            offset += d

    # ---- abstract Hessian-format helpers (task-specific) -------------------

    @abstractmethod
    def _extract_hess_diagonal(
        self, name: str, h: torch.Tensor
    ) -> torch.Tensor:
        """Return diagonal elements from a self-Hessian tensor.

        The returned tensor must have the same shape as the parameter.
        """
        raise NotImplementedError

    @abstractmethod
    def _reconstruct_block(
        self, name: str, h: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct (N, d, d) symmetric matrix from compact Hessian."""
        raise NotImplementedError

    @abstractmethod
    def _assemble_full_hessian(
        self,
        h: Dict[str, torch.Tensor],
        ch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Assemble full (N, D, D) Hessian from self-blocks and cross-blocks."""
        raise NotImplementedError

    # ---- serialisation -----------------------------------------------------

    def state_dict(self) -> Dict:
        m_state = {}
        for key, m in self._moments.items():
            m_state[key] = {k: v.clone() for k, v in m.items()}
        return {
            "step_count": self._step_count,
            "beta_warmup_anchor_step": int(self._beta_warmup_anchor_step),
            "beta_warmup_steps_override": self._beta_warmup_steps_override,
            "damping_reanneal_anchor_step": int(self._damping_reanneal_anchor_step),
            "damping_reanneal_factor": float(self._damping_reanneal_factor),
            "damping_reanneal_steps": int(self._damping_reanneal_steps),
            "damping_reanneal_schedule": str(self._damping_reanneal_schedule),
            "damping_reanneal_active": bool(self._damping_reanneal_active),
            "moments": m_state,
            "lr_map": self.config.lr_map.copy(),
            "damping_map": self.config.damping_map.copy(),
        }

    def load_state_dict(self, state: Dict) -> None:
        self._step_count = state["step_count"]
        self._beta_warmup_anchor_step = int(state.get("beta_warmup_anchor_step", 0))
        self._beta_warmup_steps_override = state.get("beta_warmup_steps_override", None)
        self._damping_reanneal_anchor_step = int(
            state.get("damping_reanneal_anchor_step", 0)
        )
        self._damping_reanneal_factor = float(
            state.get("damping_reanneal_factor", 1.0)
        )
        self._damping_reanneal_steps = int(
            state.get("damping_reanneal_steps", 0)
        )
        self._damping_reanneal_schedule = str(
            state.get("damping_reanneal_schedule", "COSINE")
        )
        self._damping_reanneal_active = bool(
            state.get("damping_reanneal_active", False)
        )
        self._moments = {}
        for key, m in state["moments"].items():
            self._moments[key] = {k: v.clone() for k, v in m.items()}
        if "lr_map" in state:
            self.config.lr_map = state["lr_map"].copy()
        if "damping_map" in state:
            self.config.damping_map = state["damping_map"].copy()


# ============================================================================
# Legacy Classes (backward compatibility for gs2d, gs3d_smoothing, mlp, loss)
# ============================================================================

class _LegacySecondOrderOptimizerBase(ABC):
    """Legacy second-order optimizer base (kept for gs2d / gs3d_smoothing).

    New code should use ``NewtonOptimizerBase`` instead.
    """

    def __init__(self, params: Params, config: OptimizerConfig):
        self.params = params
        self.config = config
        self._step_count = 0

    def add_grad_hess(
        self, grad_result: GradResult, hess_result: HessResult,
    ) -> None:
        for name in grad_result.param_grads.keys():
            self.params.add_grad(name, grad_result.get(name))
            diag_name = f"{name}_diag"
            if hess_result.get(diag_name) is not None:
                self.params.add_hess(name, hess_result.get(diag_name))
            else:
                self.params.add_hess(name, hess_result.get(name))

    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError

    def zero_grad(self) -> None:
        self.params.clear_grads_hess()
        self._step_count += 1

    def state_dict(self) -> Dict:
        return {
            "step_count": self._step_count,
            "lr_map": self.config.lr_map.copy(),
            "damping_map": self.config.damping_map.copy(),
        }

    def load_state_dict(self, state: Dict) -> None:
        self._step_count = state["step_count"]
        self.config.lr_map = state["lr_map"].copy()
        self.config.damping_map = state["damping_map"].copy()


# Alias so that ``from .base import SecondOrderOptimizerBase`` still works
SecondOrderOptimizerBase = _LegacySecondOrderOptimizerBase


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Enums
    "HessianMode",
    "SignalStrategy",
    # Configs
    "OptimizerConfig",
    "FirstOrderConfig",
    "SecondOrderConfig",
    # New base classes
    "OptimizerBase",
    "FirstOrderOptimizerBase",
    "NewtonOptimizerBase",
    # Legacy
    "SecondOrderOptimizerBase",
]
