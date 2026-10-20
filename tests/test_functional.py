"""Public API contracts and independent numeric references for CUDA operators."""
import math

import pytest
import torch

from r2splat import rasterize_bwd, rasterize_fwd, voxelize_bwd, voxelize_fwd
from r2splat._validation import _options, _shape


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_public_api_is_small():
    import r2splat

    assert r2splat.__all__ == [
        "__version__", "rasterize_fwd", "rasterize_bwd", "voxelize_fwd",
        "voxelize_bwd", "ParameterGradients",
    ]
    assert not hasattr(r2splat, "rasterize")
    assert not hasattr(r2splat, "voxelize")


@pytest.mark.parametrize("shape", [(0, 3), (True, 3), (2.5, 3), (3,), (2**30, 4)])
def test_reject_invalid_output_shape(shape):
    with pytest.raises(ValueError, match="image_size"):
        _shape("image_size", shape, 2)


@pytest.mark.parametrize("eps,radius,detach", [(-1, None, False), (math.nan, None, False), (0, 0, False), (0, math.inf, False), (0, 1, "yes")])
def test_reject_invalid_numeric_options(eps, radius, detach):
    with pytest.raises((ValueError, TypeError)):
        _options(None, eps, radius, detach)


def test_cpu_inputs_fail_before_loading_cuda_extension():
    values = (torch.zeros(1, 3), torch.ones(1, 4), torch.ones(1, 3), torch.ones(1))
    with pytest.raises(ValueError, match="means must be a CUDA tensor"):
        voxelize_fwd(*values, (3, 4, 5), volume_extent=(1, 1, 1))


@pytest.mark.parametrize("dtype", [torch.float16, torch.int32])
def test_unsupported_precision_fails_before_loading_cuda_extension(dtype):
    means = torch.zeros(1, 3, dtype=dtype)
    with pytest.raises(TypeError, match="dtype"):
        voxelize_fwd(means, None, None, None, (3, 4, 5), volume_extent=(1, 1, 1))


def _inputs(dtype=torch.float64):
    return [
        torch.tensor([[0.12, -0.13, 2.08]], device="cuda", dtype=dtype, requires_grad=True),
        torch.tensor([[1.0, 0.1, 0.2, 0.05]], device="cuda", dtype=dtype, requires_grad=True),
        torch.tensor([[0.45, 0.55, 0.65]], device="cuda", dtype=dtype, requires_grad=True),
        torch.tensor([1.4], device="cuda", dtype=dtype, requires_grad=True),
    ]


def _camera(reference):
    return (
        torch.eye(4, device=reference.device, dtype=reference.dtype).unsqueeze(0),
        reference.new_tensor([[[12, 0, 4.5], [0, 10, 3.5], [0, 0, 1]]]),
    )


def _voxel_reference(values, beta, uncertainty):
    means, quats, scales, density = values
    q = quats[0] / torch.linalg.vector_norm(quats[0])
    w, x, y, z = q.unbind()
    rotation = torch.stack((
        1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
        2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
        2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y),
    )).reshape(3, 3)
    zz, yy, xx = torch.meshgrid(
        (torch.arange(5, device=means.device, dtype=means.dtype) + 0.5) / 5 * 2 - 1 + 2,
        (torch.arange(7, device=means.device, dtype=means.dtype) + 0.5) / 7 * 3 - 1.5,
        (torch.arange(9, device=means.device, dtype=means.dtype) + 0.5) / 9 * 4 - 2,
        indexing="ij",
    )
    delta = torch.stack((xx, yy, zz), dim=-1) - means[0]
    local = delta @ rotation / scales[0]
    distance = local.square().sum(dim=-1)
    if beta is None:
        contribution = density[0] * torch.exp(-0.5 * distance)
    else:
        contribution = density[0] * (1 - distance).clamp_min(0).pow(beta[0])
    threshold = 1e-6 if beta is None else 1e-5
    contribution = torch.where(contribution.abs() >= threshold, contribution, torch.zeros_like(contribution))
    return contribution, uncertainty[0] * contribution.square()


@cuda
@pytest.mark.parametrize("use_beta", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_non_cubic_voxelization_and_gradients_match_independent_reference(use_beta, dtype):
    values = _inputs(dtype)
    beta = values[0].new_tensor([3.5], requires_grad=True) if use_beta else None
    uncertainty = values[0].new_tensor([0.2], requires_grad=True)
    (volume, variance), context = voxelize_fwd(
        *values, (5, 7, 9), volume_extent=(4, 3, 2), volume_center=(0, 0, 2),
        beta=beta, uncertainty=uncertainty, eps=0, radius_clip=8,
    )
    reference, reference_var = _voxel_reference(values, beta, uncertainty)
    assert volume.shape == (5, 7, 9)
    torch.testing.assert_close(volume, reference, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(variance, reference_var, atol=2e-5, rtol=2e-5)
    weights = torch.linspace(0.2, 1.3, volume.numel(), device="cuda", dtype=dtype).reshape_as(volume)
    inputs = values + ([beta] if use_beta else []) + [uncertainty]
    gradients = voxelize_bwd(context, weights, weights)
    actual_grads = [gradients.means, gradients.quats, gradients.scales, gradients.densities]
    actual_grads += ([gradients.beta] if use_beta else []) + [gradients.uncertainty]
    expected_grads = torch.autograd.grad((weights * (reference + reference_var)).sum(), inputs)
    for actual, expected in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


@cuda
@pytest.mark.parametrize("use_beta", [False, True])
def test_rasterization_matches_centered_analytic_projection(use_beta):
    values = _inputs()
    values[0] = values[0].new_tensor([[0, 0, 2]], requires_grad=True)
    values[1] = values[1].new_tensor([[1, 0, 0, 0]], requires_grad=True)
    viewmats, Ks = _camera(values[0])
    beta = values[0].new_tensor([3.5], requires_grad=True) if use_beta else None
    uncertainty = values[0].new_tensor([0.2], requires_grad=True)
    (image, variance), context = rasterize_fwd(
        *values, viewmats, Ks, (7, 9), beta=beta, uncertainty=uncertainty,
        eps=0, radius_clip=8,
    )
    yy, xx = torch.meshgrid(
        torch.arange(7, device="cuda", dtype=torch.float64) + 0.5,
        torch.arange(9, device="cuda", dtype=torch.float64) + 0.5,
        indexing="ij",
    )
    scales = values[2][0]
    distance = ((xx - 4.5) / (6 * scales[0])).square() + ((yy - 3.5) / (5 * scales[1])).square()
    if use_beta:
        path = torch.exp(torch.lgamma(beta.new_tensor(0.5)) + torch.lgamma(beta[0]+1) - torch.lgamma(beta[0]+1.5)) * scales[2]
        reference = values[3][0] * path * (1-distance).clamp_min(0).pow(beta[0]+0.5)
    else:
        reference = values[3][0] * math.sqrt(2*math.pi) * scales[2] * torch.exp(-distance/2)
    reference = torch.where(reference.abs() >= 1e-5, reference, torch.zeros_like(reference)).unsqueeze(0)
    reference_var = uncertainty[0] * reference.square()
    assert image.shape == (1, 7, 9)
    torch.testing.assert_close(image, reference, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(variance, reference_var, atol=2e-5, rtol=2e-5)
    inputs = [values[2], values[3]] + ([beta] if use_beta else []) + [uncertainty]
    gradients = rasterize_bwd(context, torch.ones_like(image), torch.ones_like(variance))
    actual_grads = [gradients.scales, gradients.densities]
    actual_grads += ([gradients.beta] if use_beta else []) + [gradients.uncertainty]
    expected_grads = torch.autograd.grad((reference+reference_var).sum(), inputs)
    for actual, expected in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


@cuda
@pytest.mark.parametrize("use_beta", [False, True])
def test_raster_means_gradient_matches_finite_difference(use_beta):
    values = _inputs()
    viewmats, Ks = _camera(values[0])
    beta = values[0].new_tensor([3.5]) if use_beta else None
    weights = torch.linspace(0.2, 1.3, 63, device="cuda", dtype=torch.float64).reshape(1, 7, 9)

    def loss():
        image, _ = rasterize_fwd(*values, viewmats, Ks, (7, 9), beta=beta, eps=0, radius_clip=8)
        return (image * weights).sum()

    image, context = rasterize_fwd(*values, viewmats, Ks, (7, 9), beta=beta, eps=0, radius_clip=8)
    analytic = rasterize_bwd(context, weights).means
    step = 1e-4
    for axis in range(3):
        with torch.no_grad():
            values[0][0, axis] += step
            plus = loss()
            values[0][0, axis] -= 2*step
            minus = loss()
            values[0][0, axis] += step
        torch.testing.assert_close(analytic[0, axis], (plus-minus)/(2*step), atol=4e-4, rtol=3e-3)


@cuda
@pytest.mark.parametrize("operation", ["rasterize", "voxelize"])
def test_empty_primitives_return_zero_outputs_and_explicit_gradients(operation):
    values = [torch.empty(shape, device="cuda", requires_grad=True) for shape in [(0,3), (0,4), (0,3), (0,)]]
    beta = torch.empty(0, device="cuda", requires_grad=True)
    uncertainty = torch.empty(0, device="cuda", requires_grad=True)
    if operation == "rasterize":
        (output, variance), context = rasterize_fwd(*values, *_camera(values[0]), (7, 9), beta=beta, uncertainty=uncertainty)
        assert output.shape == (1, 7, 9)
    else:
        (output, variance), context = voxelize_fwd(*values, (5, 7, 9), volume_extent=(4, 3, 2), beta=beta, uncertainty=uncertainty)
        assert output.shape == (5, 7, 9)
    assert torch.count_nonzero(output) == torch.count_nonzero(variance) == 0
    backward = rasterize_bwd if operation == "rasterize" else voxelize_bwd
    gradients = backward(context, torch.ones_like(output), torch.ones_like(variance))
    for name, value in zip(("means", "quats", "scales", "densities", "beta", "uncertainty"), values + [beta, uncertainty]):
        gradient = getattr(gradients, name)
        assert gradient.shape == value.shape and not gradient.requires_grad
        assert value.grad is None


@cuda
@pytest.mark.parametrize("operation", ["rasterize", "voxelize"])
def test_variance_can_detach_primitive_contributions(operation):
    values = _inputs()
    uncertainty = values[0].new_tensor([0.2], requires_grad=True)
    if operation == "rasterize":
        (_, variance), context = rasterize_fwd(*values, *_camera(values[0]), (7, 9), uncertainty=uncertainty, detach_variance=True)
    else:
        (_, variance), context = voxelize_fwd(*values, (5, 7, 9), volume_extent=(4, 3, 2), volume_center=(0, 0, 2), uncertainty=uncertainty, detach_variance=True)
    backward = rasterize_bwd if operation == "rasterize" else voxelize_bwd
    gradients = backward(context, None, torch.ones_like(variance))
    for name in ("means", "quats", "scales", "densities"):
        grad = getattr(gradients, name)
        torch.testing.assert_close(grad, torch.zeros_like(grad))
    assert gradients.uncertainty.item() > 0


@cuda
@pytest.mark.parametrize("bad", ["scales", "quats", "densities", "beta", "uncertainty", "dtype", "geometry_grad"])
def test_invalid_cuda_inputs_are_rejected_before_dispatch(bad):
    values = _inputs()
    kwargs = {"volume_extent": (4, 3, 2)}
    if bad in ("scales", "quats", "densities"):
        index = {"quats": 1, "scales": 2, "densities": 3}[bad]
        values[index] = torch.zeros_like(values[index]) - (1 if bad == "densities" else 0)
    elif bad in ("beta", "uncertainty"):
        kwargs[bad] = values[0].new_tensor([-1.0])
    elif bad == "dtype":
        values[3] = values[3].float()
    else:
        kwargs["volume_extent"] = values[0].new_tensor([4, 3, 2], requires_grad=True)
    with pytest.raises(ValueError):
        voxelize_fwd(*values, (5, 7, 9), **kwargs)
