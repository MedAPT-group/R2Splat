"""Opacity/density activation helpers.

R2Gaussian stores a raw density parameter and uses Softplus in rendering and
pruning paths.  This module keeps the same convention for the parameter named
``opacity`` in this codebase.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def opacity_activation(raw_opacity: torch.Tensor) -> torch.Tensor:
    """Map raw opacity/density parameters to positive CT density values."""
    return F.softplus(raw_opacity)


def opacity_activation_grad(raw_opacity: torch.Tensor) -> torch.Tensor:
    """Return d softplus(raw_opacity) / d raw_opacity."""
    return torch.sigmoid(raw_opacity)


def inverse_opacity_activation(opacity: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Inverse of Softplus for positive opacity/density values."""
    opacity = opacity.clamp_min(eps)
    return opacity + torch.log(-torch.expm1(-opacity))


__all__ = [
    "inverse_opacity_activation",
    "opacity_activation",
    "opacity_activation_grad",
]
