"""CT path factors for positive physical Beta density exponents."""

from __future__ import annotations

import math

import torch


def beta_path_ratio(gamma: torch.Tensor) -> torch.Tensor:
    """Return B(1/2, gamma + 1) / sqrt(2*pi).

    Existing ``conics_mu`` already returns the Gaussian path factor
    sqrt(2*pi)/sqrt(v^T Sigma_c^-1 v). Multiplying by this ratio converts it
    to the Beta path factor B(1/2,gamma+1)/sqrt(v^T Sigma_c^-1 v).
    """
    # Promote the small-exponent branch before subtracting lgamma values.
    # For large z = gamma + 1 use log(Gamma(z) / Gamma(z + 1/2)):
    # -log(z)/2 + 1/(8z) - 1/(192z^3) + 1/(640z^5) - 17/(14336z^7).
    # At gamma >= 32 the omitted term is below double-precision rounding.
    value = gamma.to(torch.float64)
    small = value.clamp_max(32.0)
    direct = torch.lgamma(small + 1.0) - torch.lgamma(small + 1.5) - 0.5 * math.log(2.0)
    z = value.clamp_min(32.0) + 1.0
    inverse = z.reciprocal()
    inverse2 = inverse.square()
    correction = inverse * (1.0 / 8.0 + inverse2 * (
        -1.0 / 192.0 + inverse2 * (1.0 / 640.0 - inverse2 * 17.0 / 14336.0)
    ))
    asymptotic = -0.5 * (math.log(2.0) + z.log()) + correction
    # Clamp both inactive branches as well: where evaluates both expressions,
    # and an overflowing lgamma can otherwise introduce NaNs in backward.
    log_ratio = torch.where(value < 32.0, direct, asymptotic)
    return log_ratio.exp().to(gamma.dtype)


def beta_path_log_grad(gamma: torch.Tensor) -> torch.Tensor:
    """Return d log B(1/2, gamma + 1) / d gamma."""
    value = gamma.to(torch.float64)
    small = value.clamp_max(32.0)
    direct = torch.special.digamma(small + 1.0) - torch.special.digamma(small + 1.5)
    inverse = (value.clamp_min(32.0) + 1.0).reciprocal()
    inverse2 = inverse.square()
    asymptotic = inverse * (-0.5 + inverse * (
        -1.0 / 8.0 + inverse2 * (1.0 / 64.0 + inverse2 * (
            -1.0 / 128.0 + inverse2 * 17.0 / 2048.0
        ))
    ))
    return torch.where(value < 32.0, direct, asymptotic).to(gamma.dtype)


__all__ = [
    "beta_path_log_grad",
    "beta_path_ratio",
]
