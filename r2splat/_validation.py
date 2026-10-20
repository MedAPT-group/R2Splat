"""Shared input validation for explicit CT operators."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Optional, Sequence, Union

import torch

Tensor = torch.Tensor


def _shape(name: str, value: Sequence[int], rank: int) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list, torch.Size)) or len(value) != rank:
        raise ValueError(f"{name} must contain {rank} positive integers")
    if any(isinstance(v, bool) or not isinstance(v, Integral) or v <= 0 for v in value):
        raise ValueError(f"{name} must contain {rank} positive integers")
    result = tuple(int(v) for v in value)
    if math.prod(result) > 2**31 - 1:
        raise ValueError(f"{name} exceeds the CUDA kernels' signed 32-bit indexing limit")
    return result


def _scalar(name: str, value: float, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        condition = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {condition}")
    return value


def _tensor(name: str, value: Tensor, shape: tuple, reference: Optional[Tensor] = None) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != len(shape) or any(s is not None and s != d for s, d in zip(shape, value.shape)):
        raise ValueError(f"{name} must have shape {shape}; got {tuple(value.shape)}")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must have dtype torch.float32 or torch.float64")
    if not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if reference is not None and (value.device != reference.device or value.dtype != reference.dtype):
        raise ValueError(f"{name} must have the same device and dtype as means")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    return value.contiguous()


def _constant(name: str, value: Tensor) -> Tensor:
    if value.requires_grad:
        raise ValueError(f"gradients with respect to {name} are not supported")
    return value


def _vector(name: str, value: Union[Tensor, Sequence[float]], reference: Tensor) -> Tensor:
    if not isinstance(value, Tensor):
        value = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    return _constant(name, _tensor(name, value, (3,), reference))


def _primitives(means, quats, scales, densities, beta, uncertainty):
    means = _tensor("means", means, (None, 3))
    n = means.shape[0]
    if n > 2**31 - 1:
        raise ValueError("primitive count exceeds the CUDA kernels' signed 32-bit indexing limit")
    quats = _tensor("quats", quats, (n, 4), means)
    scales = _tensor("scales", scales, (n, 3), means)
    densities = _tensor("densities", densities, (n,), means)
    # CUDA normalizes quaternions; only reject invalid input values here.
    if bool((quats == 0).all(dim=-1).any()):
        raise ValueError("quats must have nonzero norm")
    if bool((scales <= 0).any()):
        raise ValueError("scales must be strictly positive physical lengths")
    if bool((densities < 0).any()):
        raise ValueError("densities must be nonnegative")
    if beta is not None:
        beta = _tensor("beta", beta, (n,), means)
        if bool((beta <= 0).any()):
            raise ValueError("beta must contain positive 3D exponents")
    if uncertainty is not None:
        uncertainty = _tensor("uncertainty", uncertainty, (n,), means)
        if bool((uncertainty < 0).any()):
            raise ValueError("uncertainty must be nonnegative relative variance")
    return means, quats, scales, densities, beta, uncertainty


def _options(beta, eps, radius_clip, detach_variance):
    if not isinstance(detach_variance, bool):
        raise TypeError("detach_variance must be a bool")
    eps = _scalar("eps", eps)
    if radius_clip is None:
        radius_clip = 3.0 if beta is None else 1.0
    radius_clip = _scalar("radius_clip", radius_clip, positive=True)
    return eps, radius_clip
