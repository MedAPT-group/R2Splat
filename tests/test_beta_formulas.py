import math

import torch

from gsplat2ct.model.beta import (
    activated_beta_terms,
    beta_activation,
    beta_path_log_grad,
    beta_path_ratio,
    beta_projected_exponent,
)


def test_beta_activation_initializes_projected_exponent_to_four():
    raw = torch.zeros(5)
    gamma = beta_activation(raw, gamma_init=3.5)
    projected = beta_projected_exponent(gamma)
    assert torch.allclose(gamma, torch.full_like(gamma, 3.5))
    assert torch.allclose(projected, torch.full_like(projected, 4.0))


def test_beta_activation_is_scaled_exponential():
    raw = torch.tensor([0.0, math.log(2.0), -math.log(2.0)])
    gamma = beta_activation(raw, gamma_init=3.5)
    assert torch.allclose(gamma, torch.tensor([3.5, 7.0, 1.75]))


def test_beta_path_ratio_matches_closed_form():
    gamma = torch.tensor([0.5, 3.5, 8.0], dtype=torch.float64)
    ratio = beta_path_ratio(gamma)
    expected = torch.exp(
        torch.lgamma(torch.tensor(0.5, dtype=torch.float64))
        + torch.lgamma(gamma + 1.0)
        - torch.lgamma(gamma + 1.5)
        - 0.5 * math.log(2.0 * math.pi)
    )
    assert torch.allclose(ratio, expected, atol=1e-12, rtol=1e-12)


def test_beta_path_log_grad_finite_difference():
    gamma = torch.tensor([3.5], dtype=torch.float64)
    eps = 1.0e-5
    fd = (
        beta_path_ratio(gamma + eps).log()
        - beta_path_ratio(gamma - eps).log()
    ) / (2.0 * eps)
    assert torch.allclose(beta_path_log_grad(gamma), fd, atol=1e-8, rtol=1e-8)


def test_activated_beta_terms_shapes():
    raw = torch.zeros(7)
    gamma, projected, path, grad = activated_beta_terms(raw)
    assert gamma.shape == projected.shape == path.shape == grad.shape == (7,)
    assert torch.allclose(grad, gamma)
