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
    EMA:       exponential moving average with bias correction (Scheme C)
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
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

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

    # ---- optimisation step -------------------------------------------------

    def step(self) -> None:
        self._step_count += 1
        b1, b2, eps = self.config.beta1, self.config.beta2, self.config.eps

        for name in self.param_names:
            if name not in self._grad:
                continue
            g = self._grad[name]
            lr = self.config.get_lr(name)

            if name not in self._m1:
                self._m1[name] = torch.zeros_like(g)
                self._m2[name] = torch.zeros_like(g)

            self._m1[name].mul_(b1).add_(g, alpha=1 - b1)
            self._m2[name].mul_(b2).addcmul_(g, g, value=1 - b2)

            m1_hat = self._m1[name] / (1 - b1 ** self._step_count)
            m2_hat = self._m2[name] / (1 - b2 ** self._step_count)

            self.params.get(name).data -= lr * m1_hat / (m2_hat.sqrt() + eps)

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

        b1 = self.config.beta1_g if signal_type == "grad" else self.config.beta1_h
        b2 = self.config.beta2_g if signal_type == "grad" else self.config.beta2_h
        eps = self.config.eps_g if signal_type == "grad" else self.config.eps_h
        t = self._step_count

        result = {}
        for name, val in buf.items():
            key = f"{signal_type}_{name}"
            if key not in self._moments:
                self._moments[key] = {}
            m = self._moments[key]

            if "m1" not in m:
                m["m1"] = torch.zeros_like(val)
            m["m1"].mul_(b1).add_(val, alpha=1 - b1)

            if strategy == SignalStrategy.EMA:
                result[name] = m["m1"] / (1 - b1 ** t)

            elif strategy == SignalStrategy.ADAM_NORM:
                if "m2" not in m:
                    m["m2"] = torch.zeros_like(val)
                m["m2"].mul_(b2).addcmul_(val, val, value=1 - b2)

                m1_hat = m["m1"] / (1 - b1 ** t)
                m2_hat = m["m2"] / (1 - b2 ** t)
                result[name] = m1_hat / (m2_hat.sqrt() + eps)

        return result

    # ---- optimisation step -------------------------------------------------

    def step(self) -> None:
        self._step_count += 1

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
            damping = self.config.get_damping(name)

            delta = -grad / (damping + hess_diag.abs())
            self.params.get(name).data += lr * delta

    def _step_block(self, g: Dict, h: Dict) -> None:
        """Per-parameter-group block Newton: (H_block + λI)δ = -g."""
        for name, d in zip(self.param_names, self.param_dims):
            if name not in g or name not in h:
                continue
            grad = g[name]
            lr = self.config.get_lr(name)
            damping = self.config.get_damping(name)

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
            torch.full((d,), self.config.get_damping(name),
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
            "moments": m_state,
            "lr_map": self.config.lr_map.copy(),
            "damping_map": self.config.damping_map.copy(),
        }

    def load_state_dict(self, state: Dict) -> None:
        self._step_count = state["step_count"]
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
