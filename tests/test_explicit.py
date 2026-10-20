"""Explicit CUDA forward/VJP contracts and independent physical finite differences."""

from dataclasses import is_dataclass

import pytest
import torch

import r2splat
from _reference import rasterize_reference, voxelize_reference


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
FIELDS = ("means", "quats", "scales", "densities", "beta", "uncertainty")


def _scene(use_beta=True, uncertainty=True, dtype=torch.float64, requires_grad=False):
    def tensor(data):
        return torch.tensor(data, device="cuda", dtype=dtype, requires_grad=requires_grad)

    values = {
        "means": tensor([[-0.12, 0.09, 1.94], [0.18, -0.13, 2.12]]),
        "quats": tensor([[1.7, 0.2, -0.15, 0.08], [0.8, -0.12, 0.16, 0.11]]),
        "scales": tensor([[0.32, 0.43, 0.37], [0.41, 0.34, 0.48]]),
        "densities": tensor([1.2, 0.8]),
        "beta": tensor([2.2, 5.5]) if use_beta else None,
        "uncertainty": tensor([0.17, 0.29]) if uncertainty else None,
    }
    viewmats = torch.eye(4, device="cuda", dtype=dtype).repeat(2, 1, 1)
    viewmats[1, :2, 3] = viewmats.new_tensor([0.12, -0.05])
    values.update(
        viewmats=viewmats,
        Ks=viewmats.new_tensor([[[12, 0, 4.3], [0, 10, 3.2], [0, 0, 1]]]).repeat(2, 1, 1),
        volume_extent=viewmats.new_tensor([1.8, 1.6, 1.4]),
        volume_center=viewmats.new_tensor([0, 0, 2]),
    )
    return values


def _clone(values, requires_grad=False):
    return {
        name: value.detach().clone().requires_grad_(requires_grad and name in FIELDS)
        if value is not None else None
        for name, value in values.items()
    }


def _forward(operation, values, explicit=True, *, mode="primitive", detach=False):
    fn = getattr(r2splat, operation + "_fwd") if explicit else {
        "rasterize": rasterize_reference, "voxelize": voxelize_reference,
    }[operation]
    options = dict(beta=values["beta"], uncertainty=values["uncertainty"], eps=0.01,
                   detach_variance=detach)
    if explicit:
        options["backward_mode"] = mode
        options["radius_clip"] = 1 if values["beta"] is not None else 3
    inputs = [values[name] for name in FIELDS[:4]]
    if operation == "rasterize":
        return fn(*inputs, values["viewmats"], values["Ks"], (7, 9), **options)
    return fn(*inputs, (5, 7, 9), volume_extent=values["volume_extent"],
              volume_center=values["volume_center"], **options)


def _outputs(output):
    return output if isinstance(output, tuple) else (output,)


def _adjoints(output, *, strided=False, requires_grad=False):
    result = []
    for index, value in enumerate(_outputs(output)):
        weights = torch.linspace(-0.3 + index, 1.1 + index, value.numel(),
                                 device=value.device, dtype=value.dtype).reshape_as(value)
        if strided:
            storage = value.new_empty((*value.shape, 2))
            storage[..., 0] = weights
            weights = storage[..., 0]
            assert not weights.is_contiguous()
        result.append(weights.requires_grad_(requires_grad))
    return result


def _reference_gradients(output, values, adjoints):
    names = [name for name in FIELDS if values[name] is not None]
    loss = sum((value * weight.detach()).sum() for value, weight in zip(_outputs(output), adjoints))
    gradients = torch.autograd.grad(loss, [values[name] for name in names])
    return dict(zip(names, gradients))


def _check_gradients(actual, expected, dtype):
    assert is_dataclass(actual)
    atol, rtol = (2e-4, 3e-4) if dtype == torch.float32 else (2e-9, 2e-9)
    for name in FIELDS:
        value = getattr(actual, name)
        if name not in expected:
            assert value is None
        else:
            assert value.grad_fn is None and not value.requires_grad
            assert bool(torch.isfinite(value).all())
            torch.testing.assert_close(value, expected[name], atol=atol, rtol=rtol)


@pytest.mark.parametrize("operation", ["rasterize", "voxelize"])
@pytest.mark.parametrize("use_beta,uncertainty,detach,dtype,mode", [
    (False, False, False, torch.float32, "primitive"),
    (True, False, False, torch.float64, "pixel"),
    (False, False, False, torch.float64, "pixel"),
    (True, False, False, torch.float32, "primitive"),
    (False, True, False, torch.float64, "pixel"),
    (True, True, False, torch.float32, "primitive"),
    (False, True, True, torch.float32, "primitive"),
    (True, True, True, torch.float64, "pixel"),
])
def test_explicit_vjp_matches_independent_math_without_edges_or_grad_side_effects(
    operation, use_beta, uncertainty, detach, dtype, mode,
):
    values = _scene(use_beta, uncertainty, dtype, requires_grad=True)
    for name in FIELDS:
        if values[name] is not None:
            values[name].grad = torch.full_like(values[name], 1.25)
    reference_values = _clone(values, requires_grad=True)
    output, context = _forward(operation, values, mode=mode, detach=detach)
    reference = _forward(operation, reference_values, explicit=False, detach=detach)
    tolerance = 2e-5 if dtype == torch.float32 else 2e-11
    for actual, expected in zip(_outputs(output), _outputs(reference)):
        assert actual.grad_fn is None and not actual.requires_grad
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    adjoints = _adjoints(output, strided=True, requires_grad=True)
    expected = _reference_gradients(reference, reference_values, adjoints)
    actual = getattr(r2splat, operation + "_bwd")(context, *adjoints)
    _check_gradients(actual, expected, dtype)
    for name in FIELDS:
        if values[name] is not None:
            torch.testing.assert_close(values[name].grad, torch.full_like(values[name], 1.25))


@pytest.mark.parametrize("operation", ["rasterize", "voxelize"])
def test_explicit_physical_vjp_matches_finite_differences(operation):
    values = _scene()
    output, context = _forward(operation, values)
    adjoints = _adjoints(output)
    gradients = getattr(r2splat, operation + "_bwd")(context, *adjoints)
    entries = {"means": (0, 0), "quats": (0, 1), "scales": (0, 2),
               "densities": (0,), "beta": (0,), "uncertainty": (0,)}
    step = 1e-4
    for name, index in entries.items():
        losses = []
        for sign in (1, -1):
            perturbed = _clone(values)
            perturbed[name][index] += sign * step
            sampled, _ = _forward(operation, perturbed)
            losses.append(sum((sample * weight).sum() for sample, weight in zip(_outputs(sampled), adjoints)))
        finite_difference = (losses[0] - losses[1]) / (2 * step)
        analytic = getattr(gradients, name)[index]
        assert abs(analytic.item()) > 1e-9, name
        torch.testing.assert_close(analytic, finite_difference, atol=2e-4, rtol=3e-3, msg=name)


@pytest.mark.parametrize("operation", ["rasterize", "voxelize"])
@pytest.mark.parametrize("adjoint_case,detach", [("zero", False), ("mean", False), ("variance", False), ("variance", True)])
def test_optional_explicit_adjoints_and_detached_variance(operation, adjoint_case, detach):
    values = _scene()
    output, context = _forward(operation, values, detach=detach)
    v_mean, v_variance = _adjoints(output)
    if adjoint_case == "zero":
        v_mean = v_variance = None
    elif adjoint_case == "mean":
        v_variance = None
    else:
        v_mean = None
    actual = getattr(r2splat, operation + "_bwd")(context, v_mean, v_variance)
    if adjoint_case == "zero":
        for name in FIELDS:
            torch.testing.assert_close(getattr(actual, name), torch.zeros_like(values[name]))
        return
    reference_values = _clone(values, requires_grad=True)
    reference = _forward(operation, reference_values, explicit=False, detach=detach)
    weights = [torch.zeros_like(output[0]) if v_mean is None else v_mean,
               torch.zeros_like(output[1]) if v_variance is None else v_variance]
    expected = _reference_gradients(reference, reference_values, weights)
    _check_gradients(actual, expected, torch.float64)
    if adjoint_case == "mean":
        torch.testing.assert_close(actual.uncertainty, torch.zeros_like(values["uncertainty"]))
    if adjoint_case == "variance" and detach:
        for name in FIELDS[:-1]:
            torch.testing.assert_close(getattr(actual, name), torch.zeros_like(values[name]))
        assert bool(torch.any(actual.uncertainty != 0))


@pytest.mark.parametrize("operation", ["rasterize", "voxelize"])
@pytest.mark.parametrize("use_beta,uncertainty", [(False, False), (True, True)])
def test_explicit_empty_primitives_return_empty_gradients(operation, use_beta, uncertainty):
    values = _scene(use_beta, uncertainty, requires_grad=True)
    for name in FIELDS:
        if values[name] is not None:
            values[name] = values[name].detach()[:0].requires_grad_()
    output, context = _forward(operation, values)
    for value in _outputs(output):
        assert value.shape == ((2, 7, 9) if operation == "rasterize" else (5, 7, 9))
        assert not value.requires_grad and value.grad_fn is None
        assert torch.count_nonzero(value) == 0
    gradients = getattr(r2splat, operation + "_bwd")(context, *_adjoints(output))
    expected = {name: torch.zeros_like(values[name]) for name in FIELDS if values[name] is not None}
    _check_gradients(gradients, expected, torch.float64)


@pytest.mark.parametrize("operation", ["rasterize", "voxelize"])
def test_explicit_context_is_consumed_after_successful_backward(operation):
    output, context = _forward(operation, _scene())
    adjoints = _adjoints(output)
    backward = getattr(r2splat, operation + "_bwd")
    backward(context, *adjoints)
    with pytest.raises(RuntimeError):
        backward(context, *adjoints)


@pytest.mark.parametrize("operation,changed", [
    (operation, name)
    for operation in ("rasterize", "voxelize")
    for name in FIELDS + (("viewmats", "Ks") if operation == "rasterize" else ("volume_extent", "volume_center"))
])
def test_explicit_context_rejects_changes_to_original_inputs(operation, changed):
    values = _scene()
    # The original strided reference must be guarded, not only its saved copy.
    backing = values["means"].new_empty(2, 6)
    backing[:, ::2] = values["means"]
    values["means"] = backing[:, ::2]
    assert not values["means"].is_contiguous()
    output, context = _forward(operation, values)
    values[changed].add_(0.01)
    with pytest.raises(RuntimeError):
        getattr(r2splat, operation + "_bwd")(context, *_adjoints(output))


@pytest.mark.parametrize("operation", ["rasterize", "voxelize"])
@pytest.mark.parametrize("channel", [0, 1])
@pytest.mark.parametrize("bad", ["shape", "dtype", "device", "nonfinite"])
def test_explicit_backward_validates_adjoints_without_consuming_context(operation, channel, bad):
    output, context = _forward(operation, _scene())
    adjoints = _adjoints(output)
    value = adjoints[channel]
    if bad == "shape":
        invalid = value.reshape(-1)
    elif bad == "dtype":
        invalid = value.float()
    elif bad == "device":
        invalid = value.cpu()
    else:
        invalid = value.clone()
        invalid.reshape(-1)[0] = float("nan")
    invalid_adjoints = list(adjoints)
    invalid_adjoints[channel] = invalid
    backward = getattr(r2splat, operation + "_bwd")
    errors = (TypeError, ValueError) if bad == "dtype" else (ValueError,)
    with pytest.raises(errors):
        backward(context, *invalid_adjoints)
    gradients = backward(context, *adjoints)
    assert bool(torch.isfinite(gradients.densities).all())


@pytest.mark.parametrize("operation,name", [
    ("rasterize", "viewmats"), ("rasterize", "Ks"),
    ("voxelize", "volume_extent"), ("voxelize", "volume_center"),
])
def test_explicit_forward_rejects_geometry_gradients_before_contiguous_copy(operation, name):
    values = _scene()
    geometry = values[name]
    storage = geometry.new_empty((*geometry.shape, 2))
    storage[..., 0] = geometry
    values[name] = storage[..., 0].requires_grad_()
    assert not values[name].is_contiguous()
    with pytest.raises(ValueError, match=f"gradients with respect to {name}"):
        _forward(operation, values)


@pytest.mark.parametrize("operation", ["rasterize", "voxelize"])
def test_explicit_backward_rejects_variance_adjoint_without_uncertainty(operation):
    output, context = _forward(operation, _scene(uncertainty=False))
    weights = _adjoints(output)[0]
    with pytest.raises(ValueError):
        getattr(r2splat, operation + "_bwd")(context, weights, weights)


@pytest.mark.parametrize("operation", ["rasterize", "voxelize"])
def test_explicit_path_does_not_call_autograd(monkeypatch, operation):
    def forbidden(*args, **kwargs):
        raise AssertionError("explicit forward/backward called an autograd API")

    values = _scene(requires_grad=True)
    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    output, context = _forward(operation, values)
    gradients = getattr(r2splat, operation + "_bwd")(context, *_adjoints(output))
    for value in _outputs(output) + tuple(getattr(gradients, name) for name in FIELDS):
        assert not value.requires_grad and value.grad_fn is None
        assert bool(torch.isfinite(value).all())
    assert all(values[name].grad is None for name in FIELDS)


def test_explicit_projection_beta_path_derivative_across_stability_branches():
    values = _scene(requires_grad=True)
    values.update(
        means=torch.tensor([[0, 0, 2]] * 3, device="cuda", dtype=torch.float64, requires_grad=True),
        quats=torch.tensor([[2, 0, 0, 0]] * 3, device="cuda", dtype=torch.float64, requires_grad=True),
        scales=torch.tensor([[0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.2, 0.3]], device="cuda", dtype=torch.float64, requires_grad=True),
        densities=torch.tensor([1.2, 0.8, 1.1], device="cuda", dtype=torch.float64, requires_grad=True),
        beta=torch.tensor([0.01, 32, 1e4], device="cuda", dtype=torch.float64, requires_grad=True),
        uncertainty=None,
    )
    viewmats = torch.eye(4, device="cuda", dtype=torch.float64).repeat(2, 1, 1)
    intrinsics = viewmats.new_tensor([[[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]]).repeat(2, 1, 1)

    def call(data, explicit):
        fn = r2splat.rasterize_fwd if explicit else rasterize_reference
        return fn(*(data[name] for name in FIELDS[:4]), viewmats, intrinsics, (1, 1),
                  beta=data["beta"], eps=0, **({"radius_clip": 8} if explicit else {}))

    output, context = call(values, True)
    weights = output.new_tensor([[[0.8]], [[1.2]]])
    actual = r2splat.rasterize_bwd(context, weights)
    reference_values = _clone(values, requires_grad=True)
    reference = call(reference_values, False)
    expected = torch.autograd.grad((reference * weights).sum(),
                                   (reference_values["beta"], reference_values["scales"]))
    torch.testing.assert_close(output, reference, atol=1e-11, rtol=1e-11)
    for gradient, target in zip((actual.beta, actual.scales), expected):
        assert bool(torch.isfinite(gradient).all())
        torch.testing.assert_close(gradient, target, atol=1e-11, rtol=1e-9)
