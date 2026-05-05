"""Optimizer base classes for first-order CT reconstruction training."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from gsplat2ct.ops.data import GradResult, Params


@dataclass
class OptimizerConfig:
    """Base optimizer configuration with per-parameter learning rates."""

    lr_map: Dict[str, float] = field(default_factory=dict)

    def get_lr(self, name: str, default: float = 1e-3) -> float:
        return float(self.lr_map.get(name, default))

    def set_lr(self, name: str, lr: float) -> None:
        self.lr_map[name] = float(lr)


@dataclass
class FirstOrderConfig(OptimizerConfig):
    """Configuration for Adam."""

    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8


class OptimizerBase(ABC):
    """Root abstract base class for optimizers."""

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

    def state_dict(self) -> Dict:
        return {"step_count": self._step_count}

    def load_state_dict(self, state: Dict) -> None:
        self._step_count = int(state["step_count"])


class FirstOrderOptimizerBase(OptimizerBase):
    """Adam optimizer base with topology-aware moment migration."""

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

    def add_grad(self, grad_result: GradResult) -> None:
        for name in self.param_names:
            grad = grad_result.get(name)
            if grad is not None:
                self._grad[name] = grad

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
        """Migrate Adam moments across duplicate/split/prune edits."""
        if not self._m1 or not self._m2:
            return

        for name in self.param_names:
            m1 = self._m1.get(name)
            m2 = self._m2.get(name)
            if m1 is None or m2 is None:
                continue

            if duplicate_sel is not None and duplicate_sel.numel() > 0:
                n_new = int(duplicate_sel.numel())
                m1 = self._append_zero_rows(m1, n_new)
                m2 = self._append_zero_rows(m2, n_new)

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

    def step(self) -> None:
        self._step_count += 1
        beta1, beta2, eps = self.config.beta1, self.config.beta2, self.config.eps

        for name in self.param_names:
            if name not in self._grad:
                continue

            grad = self._grad[name]
            lr = self.config.get_lr(name)

            if name not in self._m1 or self._m1[name].shape != grad.shape:
                self._m1[name] = torch.zeros_like(grad)
            if name not in self._m2 or self._m2[name].shape != grad.shape:
                self._m2[name] = torch.zeros_like(grad)

            self._m1[name].mul_(beta1).add_(grad, alpha=1.0 - beta1)
            self._m2[name].mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            m1_hat = self._m1[name] / (1.0 - beta1 ** self._step_count)
            m2_hat = self._m2[name] / (1.0 - beta2 ** self._step_count)
            update = -lr * m1_hat / (m2_hat.sqrt() + eps)
            self.params.get(name).data.add_(update)

        self._apply_constraints()

    def state_dict(self) -> Dict:
        return {
            "step_count": self._step_count,
            "m1": {k: v.clone() for k, v in self._m1.items()},
            "m2": {k: v.clone() for k, v in self._m2.items()},
        }

    def load_state_dict(self, state: Dict) -> None:
        self._step_count = int(state["step_count"])
        self._m1 = {k: v.clone() for k, v in state["m1"].items()}
        self._m2 = {k: v.clone() for k, v in state["m2"].items()}


__all__ = [
    "OptimizerConfig",
    "FirstOrderConfig",
    "OptimizerBase",
    "FirstOrderOptimizerBase",
]
