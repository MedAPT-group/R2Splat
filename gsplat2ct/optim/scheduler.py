from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import math

from .base import OptimizerBase


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _factor(
    *,
    schedule_type: str,
    step: int,
    max_steps: int,
    warmup_steps: int,
    min_factor: float,
) -> float:
    if max_steps <= 0:
        return 1.0
    warmup_steps = max(int(warmup_steps), 0)
    min_factor = float(min_factor)

    if warmup_steps > 0 and step < warmup_steps:
        return _clamp((step + 1) / warmup_steps, 0.0, 1.0)

    # Progress in [0, 1]
    denom = max(max_steps - warmup_steps, 1)
    t = _clamp((step - warmup_steps) / denom, 0.0, 1.0)

    st = str(schedule_type).lower()
    if st in ("constant", "none"):
        return 1.0
    if st == "cosine":
        # Cosine decay from 1 -> min_factor
        return min_factor + 0.5 * (1.0 - min_factor) * (1.0 + math.cos(math.pi * t))
    if st in ("exp", "exponential"):
        # Exponential decay from 1 -> min_factor
        # factor(t)=min_factor^(t)
        return (min_factor ** t)
    if st in ("log_cosine", "log-cosine", "logcosine"):
        # Log-cosine whole-cycle schedule (U-shape in linear domain):
        #   factor(0)   = 1
        #   factor(0.5) = min_factor
        #   factor(1)   = 1
        #
        # In log space it is a cosine with phase=0:
        #   log f(t) = 0.5*log(min_factor) * (1 - cos(2*pi*t))
        # This is equivalent to:
        #   f(t) = min_factor^((1 - cos(2*pi*t))/2)
        if min_factor <= 0.0:
            raise ValueError(
                f"log_cosine schedule requires min_factor > 0, got {min_factor}"
            )
        exponent = 0.5 * (1.0 - math.cos(2.0 * math.pi * t))
        return float(min_factor ** exponent)

    raise ValueError(f"Unknown schedule type: {schedule_type}")


@dataclass
class ScheduleConfig:
    enabled: bool = False
    lr_type: str = "constant"
    lr_warmup_steps: int = 0
    lr_min_factor: float = 1.0
    damping_type: str = "constant"
    damping_warmup_steps: int = 0
    damping_min_factor: float = 1.0
    max_update_type: str = "constant"
    max_update_warmup_steps: int = 0
    max_update_min_factor: float = 1.0


class OptimizerScheduler:
    """Step-based scheduler that mutates optimizer config in-place."""

    def __init__(self, cfg: ScheduleConfig, optimizer: OptimizerBase):
        self.cfg = cfg
        self.optimizer = optimizer
        self._base_lr: Dict[str, float] = dict(getattr(optimizer.config, "lr_map", {}))
        self._base_damping: Dict[str, float] = dict(getattr(optimizer.config, "damping_map", {}))
        self._base_max_update: Dict[str, float] = dict(
            getattr(optimizer.config, "max_update_map", {})
        )

    @classmethod
    def from_hydra(cls, cfg, optimizer: OptimizerBase) -> "OptimizerScheduler":
        if isinstance(cfg, dict):
            cfg_obj = type("_Cfg", (), cfg)
        else:
            cfg_obj = cfg

        sc = ScheduleConfig(
            enabled=bool(getattr(cfg_obj, "enabled", False)),
            lr_type=str(getattr(getattr(cfg_obj, "lr", {}), "type", "constant"))
            if hasattr(cfg_obj, "lr")
            else str(getattr(cfg_obj, "lr_type", "constant")),
            lr_warmup_steps=int(getattr(getattr(cfg_obj, "lr", {}), "warmup_steps", 0))
            if hasattr(cfg_obj, "lr")
            else int(getattr(cfg_obj, "lr_warmup_steps", 0)),
            lr_min_factor=float(getattr(getattr(cfg_obj, "lr", {}), "min_factor", 1.0))
            if hasattr(cfg_obj, "lr")
            else float(getattr(cfg_obj, "lr_min_factor", 1.0)),
            damping_type=str(getattr(getattr(cfg_obj, "damping", {}), "type", "constant"))
            if hasattr(cfg_obj, "damping")
            else str(getattr(cfg_obj, "damping_type", "constant")),
            damping_warmup_steps=int(getattr(getattr(cfg_obj, "damping", {}), "warmup_steps", 0))
            if hasattr(cfg_obj, "damping")
            else int(getattr(cfg_obj, "damping_warmup_steps", 0)),
            damping_min_factor=float(getattr(getattr(cfg_obj, "damping", {}), "min_factor", 1.0))
            if hasattr(cfg_obj, "damping")
            else float(getattr(cfg_obj, "damping_min_factor", 1.0)),
            max_update_type=str(getattr(getattr(cfg_obj, "max_update", {}), "type", "constant"))
            if hasattr(cfg_obj, "max_update")
            else str(getattr(cfg_obj, "max_update_type", "constant")),
            max_update_warmup_steps=int(
                getattr(getattr(cfg_obj, "max_update", {}), "warmup_steps", 0)
            )
            if hasattr(cfg_obj, "max_update")
            else int(getattr(cfg_obj, "max_update_warmup_steps", 0)),
            max_update_min_factor=float(
                getattr(getattr(cfg_obj, "max_update", {}), "min_factor", 1.0)
            )
            if hasattr(cfg_obj, "max_update")
            else float(getattr(cfg_obj, "max_update_min_factor", 1.0)),
        )
        return cls(sc, optimizer)

    def step(self, step: int, max_steps: int) -> None:
        if not self.cfg.enabled:
            return

        lr_f = _factor(
            schedule_type=self.cfg.lr_type,
            step=step,
            max_steps=max_steps,
            warmup_steps=self.cfg.lr_warmup_steps,
            min_factor=self.cfg.lr_min_factor,
        )
        for name, base in self._base_lr.items():
            self.optimizer.config.set_lr(name, float(base) * float(lr_f))

        if self._base_damping:
            d_f = _factor(
                schedule_type=self.cfg.damping_type,
                step=step,
                max_steps=max_steps,
                warmup_steps=self.cfg.damping_warmup_steps,
                min_factor=self.cfg.damping_min_factor,
            )
            for name, base in self._base_damping.items():
                self.optimizer.config.set_damping(name, float(base) * float(d_f))

        if self._base_max_update and hasattr(self.optimizer.config, "set_max_update"):
            u_f = _factor(
                schedule_type=self.cfg.max_update_type,
                step=step,
                max_steps=max_steps,
                warmup_steps=self.cfg.max_update_warmup_steps,
                min_factor=self.cfg.max_update_min_factor,
            )
            for name, base in self._base_max_update.items():
                self.optimizer.config.set_max_update(name, float(base) * float(u_f))

