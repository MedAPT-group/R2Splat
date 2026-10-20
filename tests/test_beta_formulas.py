import math

import torch

from r2splat._beta import beta_path_log_grad, beta_path_ratio


def test_beta_path_ratio_matches_integrals_for_integer_exponents():
    beta = torch.tensor([1.0, 2.0, 8.0], dtype=torch.float64)
    # Integrating (1 - z*z)**n on [-1, 1] gives I_0=2 and
    # I_n = 2*n/(2*n+1) * I_(n-1), without Gamma functions.
    expected = beta.new_tensor([
        2 * math.prod(2 * k / (2 * k + 1) for k in range(1, n + 1))
        / math.sqrt(2 * math.pi)
        for n in (1, 2, 8)
    ])
    torch.testing.assert_close(beta_path_ratio(beta), expected, atol=1e-12, rtol=1e-12)


def test_projected_profile_and_physical_beta_derivative_match_line_integral():
    beta = torch.tensor([1.25, 3.5, 8.0], dtype=torch.float64, requires_grad=True)
    remaining_radius_squared = 0.73
    projected = (
        math.sqrt(2 * math.pi) * beta_path_ratio(beta)
        * remaining_radius_squared ** (beta + 0.5)
    )
    # Independently integrate the actual compact-support 3D density along z.
    # Midpoint quadrature keeps the log derivative away from support endpoints.
    samples = (torch.arange(16384, dtype=torch.float64) + 0.5) / 8192 - 1
    radius_squared = remaining_radius_squared * (1 - samples.square())
    integral = (
        radius_squared[:, None].pow(beta).mean(dim=0)
        * 2 * math.sqrt(remaining_radius_squared)
    )
    projected_grad = torch.autograd.grad(projected.sum(), beta)[0]
    integral_grad = torch.autograd.grad(integral.sum(), beta)[0]
    torch.testing.assert_close(projected, integral, atol=2e-9, rtol=2e-9)
    torch.testing.assert_close(projected_grad, integral_grad, atol=2e-8, rtol=2e-8)


def test_beta_path_log_grad_is_derivative_with_respect_to_physical_exponent():
    gamma = torch.tensor([0.8, 3.5, 9.0], dtype=torch.float64)
    eps = 1.0e-5
    fd = (
        beta_path_ratio(gamma + eps).log()
        - beta_path_ratio(gamma - eps).log()
    ) / (2.0 * eps)
    torch.testing.assert_close(beta_path_log_grad(gamma), fd, atol=1e-8, rtol=1e-8)
