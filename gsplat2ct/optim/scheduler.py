from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

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

    denom = max(max_steps - warmup_steps, 1)
    t = _clamp((step - warmup_steps) / denom, 0.0, 1.0)
    schedule = str(schedule_type).lower()

    if schedule in ("constant", "none"):
        return 1.0
    if schedule == "cosine":
        return min_factor + 0.5 * (1.0 - min_factor) * (1.0 + math.cos(math.pi * t))
    if schedule in ("exp", "exponential"):
        return min_factor ** t

    raise ValueError(f"Unknown lr schedule type: {schedule_type}")


@dataclass
class ScheduleConfig:
    enabled: bool = False
    lr_type: str = "constant"
    lr_warmup_steps: int = 0
    lr_min_factor: float = 1.0


class OptimizerScheduler:
    """Step-based learning-rate scheduler that mutates optimizer config in-place."""

    def __init__(self, cfg: ScheduleConfig, optimizer: OptimizerBase):
        self.cfg = cfg
        self.optimizer = optimizer
        self._base_lr: Dict[str, float] = dict(getattr(optimizer.config, "lr_map", {}))

    @classmethod
    def from_hydra(cls, cfg, optimizer: OptimizerBase) -> "OptimizerScheduler":
        cfg_obj = type("_Cfg", (), cfg) if isinstance(cfg, dict) else cfg
        lr_cfg = getattr(cfg_obj, "lr", {})
        return cls(
            ScheduleConfig(
                enabled=bool(getattr(cfg_obj, "enabled", False)),
                lr_type=str(getattr(lr_cfg, "type", getattr(cfg_obj, "lr_type", "constant"))),
                lr_warmup_steps=int(
                    getattr(lr_cfg, "warmup_steps", getattr(cfg_obj, "lr_warmup_steps", 0))
                ),
                lr_min_factor=float(
                    getattr(lr_cfg, "min_factor", getattr(cfg_obj, "lr_min_factor", 1.0))
                ),
            ),
            optimizer,
        )

    def step(self, step: int, max_steps: int) -> None:
        if not self.cfg.enabled:
            return

        lr_factor = _factor(
            schedule_type=self.cfg.lr_type,
            step=step,
            max_steps=max_steps,
            warmup_steps=self.cfg.lr_warmup_steps,
            min_factor=self.cfg.lr_min_factor,
        )
        for name, base_lr in self._base_lr.items():
            self.optimizer.config.set_lr(name, float(base_lr) * float(lr_factor))
