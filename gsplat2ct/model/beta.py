"""Beta primitive activation and CT path-factor helpers."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


def _resolve_gamma_init(
    gamma_min: Optional[float] = None,
    gamma_init: Optional[float] = None,
) -> float:
    """Resolve legacy ``gamma_min, gamma_init`` arguments.

    New code should pass ``gamma_init``.  If a single positional value is
    supplied after ``raw_beta``, treat it as the new scale so older helper-style
    calls such as ``beta_activation(raw, 3.5)`` remain intuitive.
    """
    if gamma_init is None:
        gamma_init_f = 3.5 if gamma_min is None else float(gamma_min)
    else:
        gamma_init_f = float(gamma_init)
    if gamma_init_f <= 0.0:
        raise ValueError("gamma_init must be positive")
    return gamma_init_f


def beta_activation(
    raw_beta: torch.Tensor,
    gamma_min: Optional[float] = None,
    gamma_init: Optional[float] = 3.5,
) -> torch.Tensor:
    """Activate raw beta parameters into positive 3D density exponents.

    The activation is ``gamma = gamma_init * exp(raw_beta)``.  Thus raw beta
    initialized to zero gives ``gamma_init``; with the default 3.5 the 3D beta
    density is close to a Gaussian near its core.
    """
    gamma_init_f = _resolve_gamma_init(gamma_min=gamma_min, gamma_init=gamma_init)
    return gamma_init_f * torch.exp(raw_beta)


def beta_activation_grad(
    gamma: torch.Tensor,
    gamma_min: Optional[float] = None,
) -> torch.Tensor:
    """Return d gamma / d raw_beta for ``gamma_init * exp(raw_beta)``."""
    return gamma


def beta_projected_exponent(gamma: torch.Tensor) -> torch.Tensor:
    """CT line integration maps 3D gamma to 2D exponent gamma + 1/2."""
    return gamma + 0.5


def beta_path_ratio(gamma: torch.Tensor) -> torch.Tensor:
    """Return B(1/2, gamma + 1) / sqrt(2*pi).

    Existing ``conics_mu`` already returns the Gaussian path factor
    sqrt(2*pi)/sqrt(v^T Sigma_c^-1 v). Multiplying by this ratio converts it
    to the Beta path factor B(1/2,gamma+1)/sqrt(v^T Sigma_c^-1 v).
    """
    half = torch.tensor(0.5, dtype=gamma.dtype, device=gamma.device)
    log_beta = torch.lgamma(half) + torch.lgamma(gamma + 1.0) - torch.lgamma(gamma + 1.5)
    return torch.exp(log_beta - 0.5 * math.log(2.0 * math.pi))


def beta_path_log_grad(gamma: torch.Tensor) -> torch.Tensor:
    """Return d log B(1/2, gamma + 1) / d gamma."""
    return torch.special.digamma(gamma + 1.0) - torch.special.digamma(gamma + 1.5)


def activated_beta_terms(
    raw_beta: torch.Tensor,
    gamma_min: Optional[float] = None,
    gamma_init: Optional[float] = 3.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``gamma, projected_beta, path_ratio, activation_grad``."""
    gamma = beta_activation(raw_beta, gamma_min=gamma_min, gamma_init=gamma_init)
    return (
        gamma,
        beta_projected_exponent(gamma),
        beta_path_ratio(gamma),
        beta_activation_grad(gamma),
    )


__all__ = [
    "activated_beta_terms",
    "beta_activation",
    "beta_activation_grad",
    "beta_path_log_grad",
    "beta_path_ratio",
    "beta_projected_exponent",
]
