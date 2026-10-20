"""Scale-independent conics and path-factor checks against a dense reference."""

import math

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _reference(packed, eps):
    matrix = torch.stack(
        (packed[..., [0, 1, 2]], packed[..., [1, 3, 4]], packed[..., [2, 4, 5]]),
        dim=-2,
    ) + eps * torch.eye(3, dtype=packed.dtype, device=packed.device)
    inverse = torch.linalg.inv(matrix[..., :2, :2])
    conics = inverse[..., [0, 0, 1], [0, 1, 1]]
    # Independent determinant formula in float64, at scales where it is safe.
    mu = (2 * math.pi * torch.linalg.det(matrix)
          / torch.linalg.det(matrix[..., :2, :2])).sqrt()
    return conics, mu


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scale", [0.5, 0.01, 1e-5])
@pytest.mark.parametrize("eps", [0.0, 1e-4, 0.3])
def test_conics_and_mu_match_dense_reference_and_gradients(dtype, scale, eps):
    from r2splat.cuda._wrapper import conics_mu_bwd, conics_mu_fwd

    covars = torch.tensor(
        [[[1.2, 0.2, -0.1, 0.9, 0.15, 1.5],
          [0.7, -0.12, 0.16, 1.4, -0.08, 1.1]]],
        dtype=dtype, device="cuda",
    ) * scale**2
    reference_covars = covars.double().requires_grad_()
    actual_conics, actual_mu = conics_mu_fwd(covars, eps)
    expected_conics, expected_mu = _reference(reference_covars, eps)
    tolerance = 3e-5 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(actual_conics.double(), expected_conics,
                               atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual_mu.double(), expected_mu,
                               atol=0, rtol=tolerance)

    # Separate upstreams ensure a missing mu gradient cannot be hidden by the
    # much larger conic gradient at small scales. Include every packed entry.
    for source in ("conics", "mu"):
        v_conics = torch.zeros_like(actual_conics)
        v_mu = torch.zeros_like(actual_mu)
        if source == "conics":
            v_conics[:] = covars.new_tensor([0.3, -0.7, 1.1]) * (scale**2 + eps)**2
        else:
            v_mu[:] = covars.new_tensor([[0.8, -1.3]])
        actual_gradient = conics_mu_bwd(
            covars, eps, actual_conics, actual_mu, v_conics, v_mu,
        )
        expected_gradient = torch.autograd.grad(
            (expected_conics * v_conics.double()).sum()
            + (expected_mu * v_mu.double()).sum(),
            reference_covars, retain_graph=True,
        )[0]
        torch.testing.assert_close(actual_gradient.double(), expected_gradient,
                                   atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scale", [0.01, 1e-3, 1e-12])
@pytest.mark.parametrize("eps", [0.0, 1e-4])
def test_small_centered_projection_has_analytic_scale_gradient(dtype, scale, eps):
    from r2splat import rasterize_bwd, rasterize_fwd

    means = torch.tensor([[0, 0, 2]], dtype=dtype, device="cuda")
    quats = means.new_tensor([[1, 0, 0, 0]])
    scales = means.new_full((1, 3), scale)
    density = means.new_ones(1)
    viewmats = torch.eye(4, dtype=dtype, device="cuda").unsqueeze(0)
    intrinsics = means.new_tensor([[[2, 0, 0.5], [0, 2, 0.5], [0, 0, 1]]])
    output, context = rasterize_fwd(means, quats, scales, density, viewmats, intrinsics,
                                    (1, 1), eps=eps)
    expected = math.sqrt(2 * math.pi * (scale**2 + eps))
    # The public rasterizer deliberately suppresses contributions below 1e-5;
    # use the direct path-factor kernel for that extremely small case below.
    if expected < 1e-5:
        from r2splat.cuda._wrapper import conics_mu_fwd, conics_mu_bwd

        covariance = means.new_tensor([[[scale**2, 0, 0, scale**2, 0, scale**2]]])
        conics, mu = conics_mu_fwd(covariance, eps)
        gradient = conics_mu_bwd(covariance, eps, conics, mu,
                                 torch.zeros_like(conics), torch.ones_like(mu))
        actual_value = mu.squeeze()
        actual_scale_gradient = 2 * scale * gradient[..., 5].squeeze()
    else:
        actual_value = output.squeeze()
        actual_scale_gradient = rasterize_bwd(context, torch.ones_like(output)).scales[0, 2]
    expected_scale_gradient = math.sqrt(2 * math.pi) * scale / math.sqrt(scale**2 + eps)
    tolerance = 3e-5 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(actual_value, actual_value.new_tensor(expected),
                               atol=0, rtol=tolerance)
    torch.testing.assert_close(actual_scale_gradient,
                               actual_scale_gradient.new_tensor(expected_scale_gradient),
                               atol=0, rtol=tolerance)


def test_singular_projection_returns_zero_outputs_and_gradients():
    from r2splat.cuda._wrapper import conics_mu_bwd, conics_mu_fwd

    covariance = torch.tensor(
        [[[0, 0, 0, 0, 0, 0], [1, 1, 0, 1, 0, 1]]],
        dtype=torch.float64, device="cuda",
    )
    conics, mu = conics_mu_fwd(covariance, 0)
    gradient = conics_mu_bwd(covariance, 0, conics, mu,
                             torch.ones_like(conics), torch.ones_like(mu))
    torch.testing.assert_close(conics, torch.zeros_like(conics))
    torch.testing.assert_close(mu, torch.zeros_like(mu))
    torch.testing.assert_close(gradient, torch.zeros_like(gradient))
