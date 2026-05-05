"""Generic data structures for CT reconstruction operators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Protocol

import torch


class ParameterLike(Protocol):
    data: torch.Tensor


@dataclass
class Params:
    param_map: Dict[str, ParameterLike] = field(default_factory=dict)
    grad_map: Dict[str, torch.Tensor] = field(default_factory=dict)

    def items(self):
        return self.param_map.items()

    def values(self):
        return self.param_map.values()

    def keys(self):
        return self.param_map.keys()

    def get(self, name: str) -> Optional[ParameterLike]:
        return self.param_map.get(name)

    def add_param(self, name: str, param: ParameterLike) -> None:
        self.param_map[name] = param

    def __iter__(self) -> Iterable[ParameterLike]:
        return iter(self.param_map.values())

    def add_grad(self, name: str, grad: torch.Tensor) -> None:
        self.grad_map[name] = grad

    def get_grad(self, name: str) -> Optional[torch.Tensor]:
        return self.grad_map.get(name)

    def clear_grads(self) -> None:
        self.grad_map.clear()


@dataclass
class Data:
    data_map: Dict[str, Any] = field(default_factory=dict)

    def get(self, name: str, default: Any = None) -> Any:
        return self.data_map.get(name, default)

    def get_optional(self, name: str) -> Any:
        return self.data_map.get(name, None)

    def set(self, name: str = None, value: Any = None, **kwargs) -> None:
        if name is not None:
            self.data_map[name] = value
        for key, val in kwargs.items():
            self.data_map[key] = val


@dataclass
class Config:
    options: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.options.get(key, default)

    def set(self, key: str = None, value: Any = None, **kwargs) -> None:
        if key is not None:
            self.options[key] = value
        for name, val in kwargs.items():
            self.options[name] = val


def get_lambda_l2(config: Optional[Config], default: float = 1.0) -> float:
    if config is None:
        return default
    value = config.get("lambda_l2")
    if value is not None:
        return float(value)
    value = config.get("lambda_l1")
    if value is not None:
        return float(value)
    return default


@dataclass
class ForwardResult:
    outputs: Dict[str, Any] = field(default_factory=dict)
    intermediates: Dict[str, Any] = field(default_factory=dict)

    def get_loss(self) -> torch.Tensor:
        return self.outputs.get("loss")


@dataclass
class GradResult:
    param_grads: Dict[str, torch.Tensor] = field(default_factory=dict)
    aux: Dict[str, Any] = field(default_factory=dict)

    def get(self, name: str) -> Optional[torch.Tensor]:
        return self.param_grads.get(name)

    def set(self, name: str = None, grad: torch.Tensor = None, **kwargs) -> None:
        if name is not None:
            self.param_grads[name] = grad
        for key, val in kwargs.items():
            self.param_grads[key] = val

    def get_aux(self, name: str, default: Any = None) -> Any:
        return self.aux.get(name, default)

    def set_aux(self, name: str = None, value: Any = None, **kwargs) -> None:
        if name is not None:
            self.aux[name] = value
        for key, val in kwargs.items():
            self.aux[key] = val


__all__ = [
    "ParameterLike",
    "Params",
    "Data",
    "Config",
    "ForwardResult",
    "GradResult",
    "get_lambda_l2",
]
