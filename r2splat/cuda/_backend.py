"""CUDA extension loader.

The extension is loaded from the installed ``r2splat.csrc`` module. Reinstall
the package after changing CUDA/C++ sources:

    pip install -e . --no-build-isolation
"""

from __future__ import annotations

import importlib

import torch


class _LazyModule:
    def __init__(self):
        self._module = None
        self._load_attempted = False

    def _try_load(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        if not torch.cuda.is_available():
            self._module = None
            return
        try:
            self._module = importlib.import_module("r2splat.csrc")
        except ImportError:
            self._module = None

    def __getattr__(self, name: str):
        self._try_load()
        if self._module is None:
            raise RuntimeError(
                "r2splat CUDA extension is not available. Install it with "
                "`pip install -e . --no-build-isolation`, and reinstall after "
                "changing CUDA/C++ sources."
            )
        return getattr(self._module, name)

    def is_available(self) -> bool:
        self._try_load()
        return self._module is not None


_lazy_C = _LazyModule()


class _CProxy:
    def __getattr__(self, name: str):
        return getattr(_lazy_C, name)

    def __bool__(self) -> bool:
        return _lazy_C.is_available()


_C = _CProxy()


def is_cuda_available() -> bool:
    return _lazy_C.is_available()


def get_cuda_module():
    _lazy_C._try_load()
    return _lazy_C._module


__all__ = ["is_cuda_available", "_C", "get_cuda_module"]
