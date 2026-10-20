"""Precision and odd-tile coverage against independent sampling formulas."""

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _scene(ndim, primitive, dtype):
    def tensor(data):
        return torch.tensor(data, device="cuda", dtype=dtype, requires_grad=True)

    size = 15 if ndim == 2 else 3
    if ndim == 2:
        means = tensor([[[7.2, 7.6], [8.1, 6.8]]])
        conics = tensor([[[0.006, 0.0002, 0.007], [0.008, -0.0003, 0.005]]])
        opacity = tensor([[1.2, 0.8]])
        values = [means, conics, opacity, tensor([[0.9, 1.1]])]
    else:
        means = tensor([[1.2, 1.6, 1.4], [1.7, 1.1, 1.8]])
        conics = tensor([[0.12, 0.005, -0.002, 0.15, 0.003, 0.1],
                         [0.1, -0.004, 0.002, 0.13, 0.005, 0.14]])
        opacity = tensor([1.2, 0.8])
        values = [means, conics, opacity]
    if primitive == "beta":
        values.append(tensor([[2.8, 3.6]] if ndim == 2 else [2.3, 3.1]))
    uncertainty = tensor([0.2, 0.3])

    axis = torch.arange(size, device="cuda", dtype=dtype) + 0.5
    grid = torch.stack(torch.meshgrid(*([axis] * ndim), indexing="ij")[::-1], dim=-1)
    delta = grid.unsqueeze(0) - means.reshape(2, *([1] * ndim), ndim)
    matrix_indices = [[0, 1], [1, 2]] if ndim == 2 else [[0, 1, 2], [1, 3, 4], [2, 4, 5]]
    precision = conics.reshape(2, -1)[:, matrix_indices]
    distance = torch.einsum("n...i,nij,n...j->n...", delta, precision, delta)
    shape = (2, *([1] * ndim))
    magnitude = opacity.reshape(shape)
    if ndim == 2:
        magnitude = magnitude * values[3].reshape(shape)
    if primitive == "beta":
        contribution = magnitude * (1 - distance).clamp_min(0).pow(values[-1].reshape(shape))
    else:
        contribution = magnitude * torch.exp(-0.5 * distance)
    threshold = 1e-6 if ndim == 3 and primitive == "gaussian" else 1e-5
    contribution = torch.where(contribution.abs() >= threshold, contribution, 0)
    mean = contribution.sum(dim=0)
    variance = (uncertainty.reshape(shape) * contribution.square()).sum(dim=0)
    if ndim == 2:
        mean, variance = mean.unsqueeze(0), variance.unsqueeze(0)
    return size, values, uncertainty, mean, variance


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("primitive", ["gaussian", "beta"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_odd_tiles_forward_and_all_backwards_match_reference(ndim, primitive, dtype):
    from r2splat.cuda import _wrapper

    size, values, uncertainty, ref_mean, ref_variance = _scene(ndim, primitive, dtype)
    tile_offsets = torch.zeros((1, 1, 1), device="cuda", dtype=torch.int32)
    flatten_ids = torch.arange(2, device="cuda", dtype=torch.int32)
    isect_ids = torch.zeros(2, device="cuda", dtype=torch.int64)
    dimensions = [size] * ndim + [size]
    bwd_options = dimensions + ([False] if ndim == 2 else [])
    stem = "rasterize_to_pixels" if ndim == 2 else "voxelize_to_voxels"
    if primitive == "beta":
        stem = "beta_" + stem

    def call(suffix, *args):
        return getattr(_wrapper, stem + suffix)(*args)

    tolerance = 5e-5 if dtype == torch.float32 else 1e-10

    def close(actual, expected):
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)

    mean = call("_fwd", *values, tile_offsets, flatten_ids, *dimensions)
    close(mean, ref_mean)
    mean, variance = call(
        "_mean_var_fwd", *values, uncertainty, tile_offsets, flatten_ids, *dimensions
    )
    close(mean, ref_mean)
    close(variance, ref_variance)
    weights = torch.linspace(0.2, 1.3, mean.numel(), device="cuda", dtype=dtype).reshape_as(mean)
    variance_weights = weights.flip(-1).contiguous()
    expected = torch.autograd.grad((ref_mean * weights).sum(), values, retain_graph=True)
    # Exercise both the shared-memory legacy VJP and the public per-primitive VJP.
    legacy = call("_bwd", *values, tile_offsets, flatten_ids, weights, *bwd_options)
    per_primitive = call(
        "_bwd_per_gaussian", *values, tile_offsets, isect_ids, flatten_ids,
        weights, *bwd_options
    )
    for actual_grads in (legacy, per_primitive):
        if ndim == 2:
            actual_grads = actual_grads[1:]  # Auxiliary absolute mean gradient.
        assert len(actual_grads) == len(expected)
        for actual, reference in zip(actual_grads, expected):
            close(actual, reference)

    expected_joint = torch.autograd.grad(
        (ref_mean * weights + ref_variance * variance_weights).sum(),
        values + [uncertainty],
    )
    fused = call(
        "_mean_var_bwd_per_gaussian", *values, tile_offsets, isect_ids, flatten_ids,
        weights, variance_weights, uncertainty, False, *bwd_options
    )
    if ndim == 2:
        fused = fused[1:]
        uncertainty_grad = call(
            "_uncertainty_bwd", *values, tile_offsets, flatten_ids,
            variance_weights, *dimensions
        )
        close(uncertainty_grad, expected_joint[-1])
    assert len(fused) == len(expected_joint)
    for actual, reference in zip(fused, expected_joint):
        close(actual, reference)


@pytest.mark.parametrize("operation", ["rasterize", "voxelize"])
@pytest.mark.parametrize("branch", [0, 1], ids=["mean", "variance"])
def test_public_gaussian_double_vjp_matches_full_finite_difference_jacobian(operation, branch):
    from r2splat import rasterize_bwd, rasterize_fwd, voxelize_bwd, voxelize_fwd

    def tensor(data):
        return torch.tensor(data, device="cuda", dtype=torch.float64)

    values = [tensor([[0.12, -0.13, 2.08]]), tensor([[1.0, 0.1, 0.2, 0.05]]),
              tensor([[0.45, 0.55, 0.65]]), tensor([1.4]), tensor([0.2])]
    viewmats = torch.eye(4, device="cuda", dtype=torch.float64).unsqueeze(0)
    intrinsics = values[0].new_tensor([[[2, 0, 1.5], [0, 2, 1.5], [0, 0, 1]]])

    def evaluate(means, quats, scales, density, uncertainty):
        if operation == "rasterize":
            return rasterize_fwd(
                means, quats, scales, density, viewmats, intrinsics, (3, 3),
                uncertainty=uncertainty, eps=1e-4, radius_clip=8,
            )
        return voxelize_fwd(
            means, quats, scales, density, (2, 2, 2), volume_extent=(1, 1, 1),
            volume_center=(0, 0, 2), uncertainty=uncertainty, eps=1e-4, radius_clip=8,
        )

    outputs, _ = evaluate(*values)
    output = outputs[branch]
    fields = ("means", "quats", "scales", "densities", "uncertainty")
    analytic = [value.new_empty((output.numel(), value.numel())) for value in values]
    backward = rasterize_bwd if operation == "rasterize" else voxelize_bwd
    # A fresh context for each basis adjoint checks the entire output Jacobian.
    for row in range(output.numel()):
        _, context = evaluate(*values)
        upstream = torch.zeros_like(output)
        upstream.reshape(-1)[row] = 1
        gradients = backward(context, upstream if branch == 0 else None,
                             upstream if branch == 1 else None)
        for jacobian, name in zip(analytic, fields):
            jacobian[row] = getattr(gradients, name).reshape(-1)
    step = 1e-6
    for input_index, value in enumerate(values):
        numeric = torch.empty_like(analytic[input_index])
        for column in range(value.numel()):
            plus = [v.clone() for v in values]
            minus = [v.clone() for v in values]
            plus[input_index].reshape(-1)[column] += step
            minus[input_index].reshape(-1)[column] -= step
            plus_outputs, _ = evaluate(*plus)
            minus_outputs, _ = evaluate(*minus)
            numeric[:, column] = (plus_outputs[branch] - minus_outputs[branch]).reshape(-1) / (2 * step)
        torch.testing.assert_close(analytic[input_index], numeric, atol=1e-5, rtol=1e-3)
