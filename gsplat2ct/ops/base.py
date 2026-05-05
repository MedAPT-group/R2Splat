"""Abstract operator interface used by the training pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .data import Config, Data, ForwardResult, GradResult, Params


class ForwardBackwardBase(ABC):
    """Forward-only and first-order backward interface."""

    @abstractmethod
    def initialize(self, params: "Params", config: Optional["Config"] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward_only(
        self,
        params: "Params",
        data: "Data",
        config: Optional["Config"] = None,
    ) -> "ForwardResult":
        raise NotImplementedError

    @abstractmethod
    def forward_backward1(
        self,
        params: "Params",
        data: "Data",
        config: Optional["Config"] = None,
    ) -> tuple["ForwardResult", "GradResult"]:
        raise NotImplementedError


__all__ = ["ForwardBackwardBase"]
