import pytest
import torch
import torch.nn.functional as F


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _single_tile():
    device = "cuda"
    return (
        torch.zeros((1, 1, 1), dtype=torch.int32, device=device),
        torch.zeros((1,), dtype=torch.int32, device=device),
        torch.zeros((1,), dtype=torch.int64, device=device),
    )


def _gaussian_raster_ref(means2d, conics, opacity, mu, h=5, w=5):
    yy, xx = torch.meshgrid(
        torch.arange(h, device=means2d.device, dtype=means2d.dtype) + 0.5,
        torch.arange(w, device=means2d.device, dtype=means2d.dtype) + 0.5,
        indexing="ij",
    )
    dx = xx - means2d[0, 0, 0]
    dy = yy - means2d[0, 0, 1]
    conic = conics[0, 0]
    sigma = 0.5 * (conic[0] * dx * dx + conic[2] * dy * dy) + conic[1] * dx * dy
    alpha = opacity[0, 0] * mu[0, 0] * torch.exp(-sigma)
    valid = (sigma >= 0) & (alpha.abs() >= 1e-5)
    return torch.where(valid, alpha, torch.zeros_like(alpha)).unsqueeze(0)


def _beta_raster_ref(means2d, conics, opacity, mu, beta, h=5, w=5):
    yy, xx = torch.meshgrid(
        torch.arange(h, device=means2d.device, dtype=means2d.dtype) + 0.5,
        torch.arange(w, device=means2d.device, dtype=means2d.dtype) + 0.5,
        indexing="ij",
    )
    dx = means2d[0, 0, 0] - xx
    dy = means2d[0, 0, 1] - yy
    conic = conics[0, 0]
    d2 = conic[0] * dx * dx + 2.0 * conic[1] * dx * dy + conic[2] * dy * dy
    hval = 1.0 - d2
    alpha = opacity[0, 0] * mu[0, 0] * hval.clamp_min(0).pow(beta[0, 0])
    valid = (d2 >= 0) & (d2 < 1.0) & (alpha.abs() >= 1e-5)
    return torch.where(valid, alpha, torch.zeros_like(alpha)).unsqueeze(0)


def test_raster_uncertainty_forward_and_q_backward_match_reference():
    from r2splat.cuda._wrapper import (
        beta_rasterize_to_pixels_mean_var_bwd_per_gaussian,
        beta_rasterize_to_pixels_mean_var_fwd,
        beta_rasterize_to_pixels_uncertainty_bwd,
        rasterize_to_pixels_mean_var_bwd_per_gaussian,
        rasterize_to_pixels_mean_var_fwd,
        rasterize_to_pixels_uncertainty_bwd,
    )

    device = "cuda"
    tile_offsets, flatten_ids, _isect_ids = _single_tile()
    q = torch.tensor([0.25], device=device)
    v_var = torch.randn((1, 5, 5), device=device)

    means2d = torch.tensor([[[2.5, 2.5]]], device=device)
    conics = torch.tensor([[[1.0, 0.1, 1.2]]], device=device)
    opacity = torch.tensor([[2.0]], device=device)
    mu = torch.tensor([[3.0]], device=device)

    mean, var = rasterize_to_pixels_mean_var_fwd(
        means2d, conics, opacity, mu, q, tile_offsets, flatten_ids, 5, 5, 8
    )
    ref = _gaussian_raster_ref(means2d, conics, opacity, mu)
    assert torch.allclose(mean, ref, atol=1e-6, rtol=1e-6)
    assert torch.allclose(var, q[0] * ref.square(), atol=1e-6, rtol=1e-6)
    v_q = rasterize_to_pixels_uncertainty_bwd(
        means2d, conics, opacity, mu, tile_offsets, flatten_ids, v_var, 5, 5, 8
    )
    assert torch.allclose(v_q, (v_var * ref.square()).sum().reshape(1), atol=2e-5, rtol=2e-5)
    fused = rasterize_to_pixels_mean_var_bwd_per_gaussian(
        means2d,
        conics,
        opacity,
        mu,
        tile_offsets,
        _isect_ids,
        flatten_ids,
        torch.zeros_like(mean),
        v_var,
        5,
        5,
        8,
        False,
    )
    assert torch.allclose(fused[-1], v_q, atol=2e-5, rtol=2e-5)

    beta = torch.tensor([[4.0]], device=device)
    mean, var = beta_rasterize_to_pixels_mean_var_fwd(
        means2d, conics, opacity, mu, beta, q, tile_offsets, flatten_ids, 5, 5, 8
    )
    ref = _beta_raster_ref(means2d, conics, opacity, mu, beta)
    assert torch.allclose(mean, ref, atol=1e-6, rtol=1e-6)
    assert torch.allclose(var, q[0] * ref.square(), atol=1e-6, rtol=1e-6)
    v_q = beta_rasterize_to_pixels_uncertainty_bwd(
        means2d, conics, opacity, mu, beta, tile_offsets, flatten_ids, v_var, 5, 5, 8
    )
    assert torch.allclose(v_q, (v_var * ref.square()).sum().reshape(1), atol=2e-5, rtol=2e-5)
    fused = beta_rasterize_to_pixels_mean_var_bwd_per_gaussian(
        means2d,
        conics,
        opacity,
        mu,
        beta,
        tile_offsets,
        _isect_ids,
        flatten_ids,
        torch.zeros_like(mean),
        v_var,
        5,
        5,
        8,
        False,
    )
    assert torch.allclose(fused[-1], v_q, atol=2e-5, rtol=2e-5)


def test_voxel_uncertainty_forward_matches_reference():
    from r2splat.cuda._wrapper import (
        beta_voxelize_to_voxels_mean_var_bwd_per_gaussian,
        beta_voxelize_to_voxels_mean_var_fwd,
        voxelize_to_voxels_mean_var_bwd_per_gaussian,
        voxelize_to_voxels_mean_var_fwd,
    )

    device = "cuda"
    tile_offsets, flatten_ids, _isect_ids = _single_tile()
    q = torch.tensor([0.125], device=device)
    means3d = torch.tensor([[2.5, 2.5, 2.5]], device=device)
    conics = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 1.0]], device=device)
    opacity = torch.tensor([2.0], device=device)

    mean, var = voxelize_to_voxels_mean_var_fwd(
        means3d, conics, opacity, q, tile_offsets, flatten_ids, 5, 5, 5, 8
    )
    zz, yy, xx = torch.meshgrid(
        torch.arange(5, device=device, dtype=torch.float32) + 0.5,
        torch.arange(5, device=device, dtype=torch.float32) + 0.5,
        torch.arange(5, device=device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    dx = xx - means3d[0, 0]
    dy = yy - means3d[0, 1]
    dz = zz - means3d[0, 2]
    sigma = 0.5 * (dx * dx + dy * dy + dz * dz)
    ref = opacity[0] * torch.exp(-sigma)
    ref = torch.where(ref.abs() >= 1e-6, ref, torch.zeros_like(ref))
    assert torch.allclose(mean, ref, atol=1e-6, rtol=1e-6)
    assert torch.allclose(var, q[0] * ref.square(), atol=1e-6, rtol=1e-6)
    v_var = torch.randn_like(var)
    fused = voxelize_to_voxels_mean_var_bwd_per_gaussian(
        means3d,
        conics,
        opacity,
        tile_offsets,
        _isect_ids,
        flatten_ids,
        torch.zeros_like(mean),
        v_var,
        5,
        5,
        5,
        8,
    )
    assert torch.allclose(fused[-1], (v_var * ref.square()).sum().reshape(1), atol=2e-5, rtol=2e-5)

    gamma = torch.tensor([3.5], device=device)
    mean, var = beta_voxelize_to_voxels_mean_var_fwd(
        means3d, conics, opacity, gamma, q, tile_offsets, flatten_ids, 5, 5, 5, 8
    )
    dx = means3d[0, 0] - xx
    dy = means3d[0, 1] - yy
    dz = means3d[0, 2] - zz
    d2 = dx * dx + dy * dy + dz * dz
    ref = torch.where(
        d2 < 1.0,
        opacity[0] * (1.0 - d2).clamp_min(0).pow(gamma[0]),
        torch.zeros_like(d2),
    )
    assert torch.allclose(mean, ref, atol=1e-6, rtol=1e-6)
    assert torch.allclose(var, q[0] * ref.square(), atol=1e-6, rtol=1e-6)
    fused = beta_voxelize_to_voxels_mean_var_bwd_per_gaussian(
        means3d,
        conics,
        opacity,
        gamma,
        tile_offsets,
        _isect_ids,
        flatten_ids,
        torch.zeros_like(mean),
        v_var,
        5,
        5,
        5,
        8,
    )
    assert torch.allclose(fused[-1], (v_var * ref.square()).sum().reshape(1), atol=2e-5, rtol=2e-5)


def _voxel_reference(means, conics, opacity, beta=None):
    axis = torch.arange(5, device=means.device, dtype=means.dtype) + 0.5
    zz, yy, xx = torch.meshgrid(axis, axis, axis, indexing="ij")
    dx, dy, dz = xx - means[0, 0], yy - means[0, 1], zz - means[0, 2]
    c = conics[0]
    distance = (c[0] * dx.square() + 2 * c[1] * dx * dy + 2 * c[2] * dx * dz
                + c[3] * dy.square() + 2 * c[4] * dy * dz + c[5] * dz.square())
    if beta is None:
        contribution = opacity[0] * torch.exp(-0.5 * distance)
        valid = (distance >= 0) & (contribution.abs() >= 1e-6)
    else:
        contribution = opacity[0] * (1 - distance).clamp_min(0).pow(beta[0])
        valid = (distance >= 0) & (distance < 1) & (contribution.abs() >= 1e-5)
    return torch.where(valid, contribution, torch.zeros_like(contribution))


@pytest.mark.parametrize("operation", ["rasterize", "voxelize"])
def test_uncertainty_explicit_raw_chain_rule_matches_reference(operation):
    from r2splat.cuda import _wrapper

    tile_offsets, flatten_ids, isect_ids = _single_tile()
    raw_q = torch.tensor([-2.0], device="cuda")
    q = F.softplus(raw_q) + 1e-6
    if operation == "rasterize":
        means = q.new_tensor([[[2.5, 2.5]]])
        values = [means, q.new_tensor([[[1.0, 0.1, 1.2]]]),
                  q.new_tensor([[2.0]]), q.new_tensor([[3.0]])]
        stem, dimensions, options = "rasterize_to_pixels", [5, 5, 8], [False]
        reference = _gaussian_raster_ref(*values)
    else:
        means = q.new_tensor([[2.5, 2.5, 2.5]])
        values = [means, q.new_tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 1.0]]), q.new_tensor([2.0])]
        stem, dimensions, options = "voxelize_to_voxels", [5, 5, 5, 8], []
        reference = _voxel_reference(*values)
    forward = getattr(_wrapper, stem + "_mean_var_fwd")
    backward = getattr(_wrapper, stem + "_mean_var_bwd_per_gaussian")
    mean, variance = forward(*values, q, tile_offsets, flatten_ids, *dimensions)
    weights = torch.linspace(-0.7, 1.1, variance.numel(), device=q.device).reshape_as(variance)
    gradients = backward(*values, tile_offsets, isect_ids, flatten_ids,
                         torch.zeros_like(mean), weights, q, True, *dimensions, *options)
    actual_raw_gradient = gradients[-1] * raw_q.sigmoid()
    expected = raw_q.sigmoid() * (weights * reference.square()).sum()
    torch.testing.assert_close(mean, reference, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual_raw_gradient, expected.reshape_as(raw_q), atol=2e-5, rtol=2e-5)
    # Numerical perturbation of caller-owned raw uncertainty checks the full chain.
    step = 1e-3
    _, plus = forward(*values, F.softplus(raw_q + step) + 1e-6,
                      tile_offsets, flatten_ids, *dimensions)
    _, minus = forward(*values, F.softplus(raw_q - step) + 1e-6,
                       tile_offsets, flatten_ids, *dimensions)
    numeric = ((plus - minus) * weights).sum().reshape_as(raw_q) / (2 * step)
    torch.testing.assert_close(actual_raw_gradient, numeric, atol=2e-4, rtol=2e-3)


@pytest.mark.parametrize("operation", ["rasterize", "voxelize"])
@pytest.mark.parametrize("primitive", ["gaussian", "beta"])
@pytest.mark.parametrize("detach", [False, True])
def test_fused_uncertainty_vjp_matches_independent_geometry_reference(operation, primitive, detach):
    from r2splat.cuda import _wrapper

    tile_offsets, flatten_ids, isect_ids = _single_tile()
    q = torch.tensor([0.25], device="cuda")
    if operation == "rasterize":
        values = [q.new_tensor([[[2.3, 2.6]]]), q.new_tensor([[[0.22, 0.015, 0.18]]]),
                  q.new_tensor([[2.0]]), q.new_tensor([[3.0]])]
        stem, dimensions, options = "rasterize_to_pixels", [5, 5, 8], [False]
        if primitive == "beta":
            values.append(q.new_tensor([[3.7]]))
        reference_fn = _beta_raster_ref if primitive == "beta" else _gaussian_raster_ref
    else:
        values = [q.new_tensor([[2.3, 2.6, 2.4]]),
                  q.new_tensor([[0.3, 0.01, -0.005, 0.25, 0.008, 0.22]]), q.new_tensor([2.0])]
        stem, dimensions, options = "voxelize_to_voxels", [5, 5, 5, 8], []
        if primitive == "beta":
            values.append(q.new_tensor([3.5]))
        reference_fn = _voxel_reference
    if primitive == "beta":
        stem = "beta_" + stem
    mean, variance = getattr(_wrapper, stem + "_mean_var_fwd")(
        *values, q, tile_offsets, flatten_ids, *dimensions
    )
    weights = torch.linspace(-0.3, 0.9, mean.numel(), device=q.device).reshape_as(mean)
    variance_weights = weights.flip(-1).contiguous()
    actual = getattr(_wrapper, stem + "_mean_var_bwd_per_gaussian")(
        *values, tile_offsets, isect_ids, flatten_ids, weights, variance_weights,
        q, detach, *dimensions, *options
    )
    if operation == "rasterize":
        actual = actual[1:]  # The first output is optional absgrad metadata.
    # Only the independent closed-form reference participates in autograd.
    reference_inputs = [value.detach().clone().requires_grad_() for value in values]
    q_reference = q.detach().clone().requires_grad_()
    reference = reference_fn(*reference_inputs)
    contribution = reference.detach() if detach else reference
    reference_variance = q_reference[0] * contribution.square()
    expected = torch.autograd.grad(
        (reference * weights + reference_variance * variance_weights).sum(),
        reference_inputs + [q_reference],
    )
    torch.testing.assert_close(mean, reference, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(variance, q[0] * reference.square(), atol=1e-6, rtol=1e-6)
    assert len(actual) == len(expected)
    for actual_gradient, expected_gradient in zip(actual, expected):
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=2e-4, rtol=2e-4)
