"""Physical Beta inputs and explicit gradients against centered analytic cases."""

import math

import pytest
import torch

from r2splat import rasterize_bwd, rasterize_fwd, voxelize_bwd, voxelize_fwd


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _scene(dtype=torch.float64):
    means = torch.tensor([[0, 0, 2], [0, 0, 2]], device="cuda", dtype=dtype)
    quats = means.new_tensor([[2, 0, 0, 0], [3, 0, 0, 0]])
    scales = means.new_tensor([[0.2, 0.3, 0.4], [0.3, 0.4, 0.5]])
    densities = means.new_tensor([1.2, 0.8])
    return means, quats, scales, densities


def _forward(operation, values, beta, uncertainty=None):
    if operation == "rasterize":
        viewmats = torch.eye(4, device="cuda", dtype=values[0].dtype).repeat(2, 1, 1)
        Ks = values[0].new_tensor([[[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]]).repeat(2, 1, 1)
        return rasterize_fwd(*values, viewmats, Ks, (1, 1), beta=beta,
                             uncertainty=uncertainty, eps=0)
    return voxelize_fwd(*values, (1, 1, 1), volume_extent=(1, 1, 1),
                        volume_center=(0, 0, 2), beta=beta, uncertainty=uncertainty, eps=0)


@pytest.mark.parametrize("operation", ["rasterize", "voxelize"])
@pytest.mark.parametrize("with_uncertainty", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_physical_beta_and_uncertainty_match_analytic_values_and_vjps(operation, with_uncertainty, dtype):
    values = _scene(dtype)
    beta = values[0].new_tensor([3.5, 99, 1.25, 99])[::2]
    assert not beta.is_contiguous()
    q = values[0].new_tensor([0.2, 0.3]) if with_uncertainty else None
    output, context = _forward(operation, values, beta, q)
    mean, variance = output if with_uncertainty else (output, None)
    path = torch.ones_like(beta)
    log_derivative = torch.zeros_like(beta)
    cameras = 1
    if operation == "rasterize":
        cameras = 2
        path = values[2][:, 2] * beta.new_tensor([
            math.exp(math.lgamma(0.5) + math.lgamma(b + 1) - math.lgamma(b + 1.5))
            for b in (3.5, 1.25)
        ])
        log_derivative = torch.special.digamma(beta.double() + 1) - torch.special.digamma(beta.double() + 1.5)
        log_derivative = log_derivative.to(dtype)
    contribution = values[3] * path
    torch.testing.assert_close(mean, contribution.sum().expand_as(mean))
    factor = torch.ones_like(beta)
    if with_uncertainty:
        torch.testing.assert_close(variance, (q * contribution.square()).sum().expand_as(variance))
        factor += 0.8 * q * contribution
    backward = rasterize_bwd if operation == "rasterize" else voxelize_bwd
    grads = backward(context, torch.ones_like(mean),
                     torch.full_like(variance, 0.4) if with_uncertainty else None)
    torch.testing.assert_close(grads.densities, cameras * factor * path)
    torch.testing.assert_close(grads.beta, cameras * factor * contribution * log_derivative)
    if with_uncertainty:
        torch.testing.assert_close(grads.uncertainty, cameras * 0.4 * contribution.square())
    else:
        assert grads.uncertainty is None


@pytest.mark.parametrize("operation", ["rasterize", "voxelize"])
@pytest.mark.parametrize("bad", ["list", "rank", "count", "integer", "dtype", "zero", "negative", "nan", "inf", "device"])
def test_invalid_beta_is_rejected_before_native_dispatch(monkeypatch, operation, bad):
    from r2splat.cuda import _wrapper

    def forbidden(*args, **kwargs):
        pytest.fail("invalid beta reached native dispatch")

    monkeypatch.setattr(_wrapper, "parameter_transform_fwd", forbidden)
    values = _scene()
    beta = values[0].new_tensor([3.5, 1.25])
    invalid = {
        "list": [3.5, 1.25], "rank": beta[:, None], "count": beta[:1],
        "integer": beta.long(), "dtype": beta.float(),
        "zero": beta.new_tensor([3.5, 0]), "negative": -beta,
        "nan": beta.new_tensor([3.5, math.nan]), "inf": beta.new_tensor([3.5, math.inf]),
        "device": beta.cpu(),
    }[bad]
    with pytest.raises((TypeError, ValueError), match="beta"):
        _forward(operation, values, invalid)
