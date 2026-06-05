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
    from gsplat2ct.cuda._wrapper import (
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
    from gsplat2ct.cuda._wrapper import (
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


def test_raster_uncertainty_autograd_raw_gradient_matches_reference():
    from gsplat2ct.ops.torch_autograd import _RasterizeMeanVarFunction

    device = "cuda"
    tile_offsets, flatten_ids, isect_ids = _single_tile()
    means2d = torch.tensor([[[2.5, 2.5]]], device=device)
    conics = torch.tensor([[[1.0, 0.1, 1.2]]], device=device)
    opacity = torch.tensor([[2.0]], device=device)
    mu = torch.tensor([[3.0]], device=device)
    raw_q = torch.tensor([-2.0], device=device, requires_grad=True)
    q_min = 1.0e-6
    q = F.softplus(raw_q) + q_min

    image, variance = _RasterizeMeanVarFunction.apply(
        means2d,
        conics,
        opacity,
        mu,
        torch.empty(0, device=device),
        q,
        tile_offsets,
        flatten_ids,
        isect_ids,
        5,
        5,
        8,
        "gaussian",
        False,
        False,
        True,
    )
    v_var = torch.randn_like(variance)
    loss = (variance * v_var).sum()
    loss.backward()

    ref = _gaussian_raster_ref(means2d, conics, opacity, mu)
    expected = torch.sigmoid(raw_q.detach()) * (v_var * ref.square()).sum()
    assert torch.allclose(image, ref, atol=1e-6, rtol=1e-6)
    assert torch.allclose(raw_q.grad, expected.reshape_as(raw_q), atol=2e-5, rtol=2e-5)


def test_voxel_uncertainty_autograd_raw_gradient_matches_reference():
    from gsplat2ct.ops.torch_autograd import _VoxelizeMeanVarFunction

    device = "cuda"
    tile_offsets, flatten_ids, isect_ids = _single_tile()
    means3d = torch.tensor([[2.5, 2.5, 2.5]], device=device)
    conics = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 1.0]], device=device)
    opacity = torch.tensor([2.0], device=device)
    raw_q = torch.tensor([-2.0], device=device, requires_grad=True)
    q_min = 1.0e-6
    q = F.softplus(raw_q) + q_min

    volume, variance = _VoxelizeMeanVarFunction.apply(
        means3d,
        conics,
        opacity,
        torch.empty(0, device=device),
        q,
        tile_offsets,
        flatten_ids,
        isect_ids,
        5,
        5,
        5,
        8,
        "gaussian",
        False,
        True,
    )
    v_var = torch.randn_like(variance)
    loss = (variance * v_var).sum()
    loss.backward()

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
    expected = torch.sigmoid(raw_q.detach()) * (v_var * ref.square()).sum()
    assert torch.allclose(volume, ref, atol=1e-6, rtol=1e-6)
    assert torch.allclose(raw_q.grad, expected.reshape_as(raw_q), atol=2e-5, rtol=2e-5)


def test_raster_uncertainty_no_detach_geometry_gradient_matches_reference():
    from gsplat2ct.ops.torch_autograd import _RasterizeMeanVarFunction

    device = "cuda"
    tile_offsets, flatten_ids, isect_ids = _single_tile()
    v_img = torch.randn((1, 5, 5), device=device)
    v_var = torch.randn((1, 5, 5), device=device)

    means2d = torch.tensor([[[2.5, 2.5]]], device=device, requires_grad=True)
    conics = torch.tensor([[[1.0, 0.1, 1.2]]], device=device, requires_grad=True)
    opacity = torch.tensor([[2.0]], device=device, requires_grad=True)
    mu = torch.tensor([[3.0]], device=device, requires_grad=True)
    q = torch.tensor([0.25], device=device, requires_grad=True)
    image, variance = _RasterizeMeanVarFunction.apply(
        means2d,
        conics,
        opacity,
        mu,
        torch.empty(0, device=device),
        q,
        tile_offsets,
        flatten_ids,
        isect_ids,
        5,
        5,
        8,
        "gaussian",
        False,
        True,
        False,
    )
    (image * v_img + variance * v_var).sum().backward()

    means_ref = means2d.detach().clone().requires_grad_(True)
    conics_ref = conics.detach().clone().requires_grad_(True)
    opacity_ref = opacity.detach().clone().requires_grad_(True)
    mu_ref = mu.detach().clone().requires_grad_(True)
    q_ref = q.detach().clone().requires_grad_(True)
    ref = _gaussian_raster_ref(means_ref, conics_ref, opacity_ref, mu_ref)
    ref_var = q_ref[0] * ref.square()
    (ref * v_img + ref_var * v_var).sum().backward()

    assert torch.allclose(means2d.grad, means_ref.grad, atol=2e-4, rtol=2e-4)
    assert torch.allclose(conics.grad, conics_ref.grad, atol=2e-4, rtol=2e-4)
    assert torch.allclose(opacity.grad, opacity_ref.grad, atol=2e-4, rtol=2e-4)
    assert torch.allclose(mu.grad, mu_ref.grad, atol=2e-4, rtol=2e-4)
    assert torch.allclose(q.grad, q_ref.grad, atol=2e-5, rtol=2e-5)

    beta = torch.tensor([[4.0]], device=device, requires_grad=True)
    means2d_b = means2d.detach().clone().requires_grad_(True)
    conics_b = conics.detach().clone().requires_grad_(True)
    opacity_b = opacity.detach().clone().requires_grad_(True)
    mu_b = mu.detach().clone().requires_grad_(True)
    q_b = q.detach().clone().requires_grad_(True)
    image, variance = _RasterizeMeanVarFunction.apply(
        means2d_b,
        conics_b,
        opacity_b,
        mu_b,
        beta,
        q_b,
        tile_offsets,
        flatten_ids,
        isect_ids,
        5,
        5,
        8,
        "beta",
        False,
        True,
        False,
    )
    (image * v_img + variance * v_var).sum().backward()

    means_ref = means2d.detach().clone().requires_grad_(True)
    conics_ref = conics.detach().clone().requires_grad_(True)
    opacity_ref = opacity.detach().clone().requires_grad_(True)
    mu_ref = mu.detach().clone().requires_grad_(True)
    beta_ref = beta.detach().clone().requires_grad_(True)
    q_ref = q.detach().clone().requires_grad_(True)
    ref = _beta_raster_ref(means_ref, conics_ref, opacity_ref, mu_ref, beta_ref)
    ref_var = q_ref[0] * ref.square()
    (ref * v_img + ref_var * v_var).sum().backward()

    assert torch.allclose(means2d_b.grad, means_ref.grad, atol=2e-4, rtol=2e-4)
    assert torch.allclose(conics_b.grad, conics_ref.grad, atol=2e-4, rtol=2e-4)
    assert torch.allclose(opacity_b.grad, opacity_ref.grad, atol=2e-4, rtol=2e-4)
    assert torch.allclose(mu_b.grad, mu_ref.grad, atol=2e-4, rtol=2e-4)
    assert torch.allclose(beta.grad, beta_ref.grad, atol=2e-4, rtol=2e-4)
    assert torch.allclose(q_b.grad, q_ref.grad, atol=2e-5, rtol=2e-5)


def test_voxel_uncertainty_no_detach_geometry_gradient_matches_reference():
    from gsplat2ct.ops.torch_autograd import _VoxelizeMeanVarFunction

    device = "cuda"
    tile_offsets, flatten_ids, isect_ids = _single_tile()
    v_vol = torch.randn((5, 5, 5), device=device)
    v_var = torch.randn((5, 5, 5), device=device)
    zz, yy, xx = torch.meshgrid(
        torch.arange(5, device=device, dtype=torch.float32) + 0.5,
        torch.arange(5, device=device, dtype=torch.float32) + 0.5,
        torch.arange(5, device=device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )

    means3d = torch.tensor([[2.5, 2.5, 2.5]], device=device, requires_grad=True)
    conics = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 1.0]], device=device, requires_grad=True)
    opacity = torch.tensor([2.0], device=device, requires_grad=True)
    q = torch.tensor([0.125], device=device, requires_grad=True)
    volume, variance = _VoxelizeMeanVarFunction.apply(
        means3d,
        conics,
        opacity,
        torch.empty(0, device=device),
        q,
        tile_offsets,
        flatten_ids,
        isect_ids,
        5,
        5,
        5,
        8,
        "gaussian",
        True,
        False,
    )
    (volume * v_vol + variance * v_var).sum().backward()

    means_ref = means3d.detach().clone().requires_grad_(True)
    conics_ref = conics.detach().clone().requires_grad_(True)
    opacity_ref = opacity.detach().clone().requires_grad_(True)
    q_ref = q.detach().clone().requires_grad_(True)
    dx = xx - means_ref[0, 0]
    dy = yy - means_ref[0, 1]
    dz = zz - means_ref[0, 2]
    c = conics_ref[0]
    sigma = (
        0.5 * (c[0] * dx * dx + c[3] * dy * dy + c[5] * dz * dz)
        + c[1] * dx * dy
        + c[2] * dx * dz
        + c[4] * dy * dz
    )
    ref = opacity_ref[0] * torch.exp(-sigma)
    ref = torch.where((sigma >= 0) & (ref.abs() >= 1e-6), ref, torch.zeros_like(ref))
    ref_var = q_ref[0] * ref.square()
    (ref * v_vol + ref_var * v_var).sum().backward()

    assert torch.allclose(means3d.grad, means_ref.grad, atol=2e-4, rtol=2e-4)
    assert torch.allclose(conics.grad, conics_ref.grad, atol=2e-4, rtol=2e-4)
    assert torch.allclose(opacity.grad, opacity_ref.grad, atol=2e-4, rtol=2e-4)
    assert torch.allclose(q.grad, q_ref.grad, atol=2e-5, rtol=2e-5)

    gamma = torch.tensor([3.5], device=device, requires_grad=True)
    means_b = means3d.detach().clone().requires_grad_(True)
    conics_b = conics.detach().clone().requires_grad_(True)
    opacity_b = opacity.detach().clone().requires_grad_(True)
    q_b = q.detach().clone().requires_grad_(True)
    volume, variance = _VoxelizeMeanVarFunction.apply(
        means_b,
        conics_b,
        opacity_b,
        gamma,
        q_b,
        tile_offsets,
        flatten_ids,
        isect_ids,
        5,
        5,
        5,
        8,
        "beta",
        True,
        False,
    )
    (volume * v_vol + variance * v_var).sum().backward()

    means_ref = means3d.detach().clone().requires_grad_(True)
    conics_ref = conics.detach().clone().requires_grad_(True)
    opacity_ref = opacity.detach().clone().requires_grad_(True)
    gamma_ref = gamma.detach().clone().requires_grad_(True)
    q_ref = q.detach().clone().requires_grad_(True)
    dx = means_ref[0, 0] - xx
    dy = means_ref[0, 1] - yy
    dz = means_ref[0, 2] - zz
    c = conics_ref[0]
    d2 = (
        c[0] * dx * dx
        + 2.0 * c[1] * dx * dy
        + 2.0 * c[2] * dx * dz
        + c[3] * dy * dy
        + 2.0 * c[4] * dy * dz
        + c[5] * dz * dz
    )
    h = 1.0 - d2
    ref = opacity_ref[0] * h.clamp_min(0).pow(gamma_ref[0])
    ref = torch.where((d2 >= 0) & (h > 0) & (ref.abs() >= 1e-5), ref, torch.zeros_like(ref))
    ref_var = q_ref[0] * ref.square()
    (ref * v_vol + ref_var * v_var).sum().backward()

    assert torch.allclose(means_b.grad, means_ref.grad, atol=2e-4, rtol=2e-4)
    assert torch.allclose(conics_b.grad, conics_ref.grad, atol=2e-4, rtol=2e-4)
    assert torch.allclose(opacity_b.grad, opacity_ref.grad, atol=2e-4, rtol=2e-4)
    assert torch.allclose(gamma.grad, gamma_ref.grad, atol=2e-4, rtol=2e-4)
    assert torch.allclose(q_b.grad, q_ref.grad, atol=2e-5, rtol=2e-5)
