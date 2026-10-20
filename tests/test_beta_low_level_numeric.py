import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_beta_rasterize_forward_matches_reference():
    from r2splat.cuda._wrapper import beta_rasterize_to_pixels_fwd

    device = "cuda"
    means2d = torch.tensor([[[2.5, 2.5]]], device=device)
    conics = torch.tensor([[[1.0, 0.0, 1.0]]], device=device)
    opacity = torch.tensor([[2.0]], device=device)
    mu = torch.tensor([[3.0]], device=device)
    beta = torch.tensor([[4.0]], device=device)
    tile_offsets = torch.zeros((1, 1, 1), dtype=torch.int32, device=device)
    flatten_ids = torch.zeros((1,), dtype=torch.int32, device=device)

    out = beta_rasterize_to_pixels_fwd(
        means2d, conics, opacity, mu, beta, tile_offsets, flatten_ids, 5, 5, 8
    )

    yy, xx = torch.meshgrid(
        torch.arange(5, device=device, dtype=torch.float32) + 0.5,
        torch.arange(5, device=device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    dx = means2d[0, 0, 0] - xx
    dy = means2d[0, 0, 1] - yy
    d2 = dx * dx + dy * dy
    ref = torch.where(
        d2 < 1.0,
        opacity[0, 0] * mu[0, 0] * (1.0 - d2).pow(beta[0, 0]),
        torch.zeros_like(d2),
    ).unsqueeze(0)

    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-6)


def test_beta_voxelize_forward_matches_reference_and_backward_shapes():
    from r2splat.cuda._wrapper import (
        beta_voxelize_to_voxels_bwd,
        beta_voxelize_to_voxels_fwd,
    )

    device = "cuda"
    means3d = torch.tensor([[2.5, 2.5, 2.5]], device=device)
    conics = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 1.0]], device=device)
    opacity = torch.tensor([2.0], device=device)
    gamma = torch.tensor([3.5], device=device)
    tile_offsets = torch.zeros((1, 1, 1), dtype=torch.int32, device=device)
    flatten_ids = torch.zeros((1,), dtype=torch.int32, device=device)

    out = beta_voxelize_to_voxels_fwd(
        means3d, conics, opacity, gamma, tile_offsets, flatten_ids, 5, 5, 5, 8
    )

    zz, yy, xx = torch.meshgrid(
        torch.arange(5, device=device, dtype=torch.float32) + 0.5,
        torch.arange(5, device=device, dtype=torch.float32) + 0.5,
        torch.arange(5, device=device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    dx = means3d[0, 0] - xx
    dy = means3d[0, 1] - yy
    dz = means3d[0, 2] - zz
    q = dx * dx + dy * dy + dz * dz
    ref = torch.where(
        q < 1.0,
        opacity[0] * (1.0 - q).pow(gamma[0]),
        torch.zeros_like(q),
    )
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-6)

    grads = beta_voxelize_to_voxels_bwd(
        means3d,
        conics,
        opacity,
        gamma,
        tile_offsets,
        flatten_ids,
        torch.ones_like(out),
        5,
        5,
        5,
        8,
    )
    assert [tuple(g.shape) for g in grads] == [(1, 3), (1, 6), (1,), (1,)]


def test_per_primitive_backward_matches_pixel_backward():
    from r2splat.cuda._wrapper import (
        beta_rasterize_to_pixels_bwd,
        beta_rasterize_to_pixels_bwd_per_gaussian,
        beta_voxelize_to_voxels_bwd,
        beta_voxelize_to_voxels_bwd_per_gaussian,
        rasterize_to_pixels_bwd,
        rasterize_to_pixels_bwd_per_gaussian,
        voxelize_to_voxels_bwd,
        voxelize_to_voxels_bwd_per_gaussian,
    )

    def assert_grads_close(pixel_grads, primitive_grads):
        for pixel_grad, primitive_grad in zip(pixel_grads, primitive_grads):
            if pixel_grad is None or pixel_grad.numel() == 0:
                assert primitive_grad is None or primitive_grad.numel() == 0
                continue
            assert torch.allclose(
                pixel_grad, primitive_grad, atol=2e-5, rtol=2e-5
            )

    def gaussian_absgrad_reference(means, conics, opacity, mu, v_img):
        yy, xx = torch.meshgrid(
            torch.arange(v_img.shape[1], device=v_img.device, dtype=v_img.dtype) + 0.5,
            torch.arange(v_img.shape[2], device=v_img.device, dtype=v_img.dtype) + 0.5,
            indexing="ij",
        )
        dx = xx - means[0, 0, 0]
        dy = yy - means[0, 0, 1]
        conic = conics[0, 0]
        sigma = 0.5 * (conic[0] * dx * dx + conic[2] * dy * dy) + conic[1] * dx * dy
        vis = torch.exp(-sigma)
        alpha = opacity[0, 0] * mu[0, 0] * vis
        valid = (sigma >= 0) & (alpha.abs() >= 1e-5)
        v_sigma = -v_img[0] * alpha
        vx = -v_sigma * (conic[0] * dx + conic[1] * dy)
        vy = -v_sigma * (conic[2] * dy + conic[1] * dx)
        return torch.stack((vx[valid].abs().sum(), vy[valid].abs().sum())).reshape(1, 1, 2)

    def beta_absgrad_reference(means, conics, opacity, mu, beta, v_img):
        yy, xx = torch.meshgrid(
            torch.arange(v_img.shape[1], device=v_img.device, dtype=v_img.dtype) + 0.5,
            torch.arange(v_img.shape[2], device=v_img.device, dtype=v_img.dtype) + 0.5,
            indexing="ij",
        )
        dx = means[0, 0, 0] - xx
        dy = means[0, 0, 1] - yy
        conic = conics[0, 0]
        d2 = conic[0] * dx * dx + 2.0 * conic[1] * dx * dy + conic[2] * dy * dy
        h = 1.0 - d2
        valid = (d2 >= 0) & (h > 0)
        vis = torch.where(valid, h.clamp_min(0).pow(beta[0, 0]), torch.zeros_like(h))
        alpha = opacity[0, 0] * mu[0, 0] * vis
        valid = valid & (alpha.abs() >= 1e-5)
        v_d2 = -opacity[0, 0] * mu[0, 0] * beta[0, 0] * h.clamp_min(0).pow(beta[0, 0] - 1.0) * v_img[0]
        vx = 2.0 * v_d2 * (conic[0] * dx + conic[1] * dy)
        vy = 2.0 * v_d2 * (conic[1] * dx + conic[2] * dy)
        return torch.stack((vx[valid].abs().sum(), vy[valid].abs().sum())).reshape(1, 1, 2)

    device = "cuda"
    torch.manual_seed(0)
    isect_ids = torch.zeros((1,), dtype=torch.int64, device=device)
    flatten_ids = torch.zeros((1,), dtype=torch.int32, device=device)
    tile_offsets_2d = torch.zeros((1, 1, 1), dtype=torch.int32, device=device)
    tile_offsets_3d = torch.zeros((1, 1, 1), dtype=torch.int32, device=device)

    means2d = torch.tensor([[[2.5, 2.5]]], device=device)
    conics2d = torch.tensor([[[1.0, 0.1, 1.2]]], device=device)
    opacity2d = torch.tensor([[2.0]], device=device)
    mu = torch.tensor([[3.0]], device=device)
    v_img = torch.randn((1, 5, 5), device=device)
    assert_grads_close(
        rasterize_to_pixels_bwd(
            means2d, conics2d, opacity2d, mu, tile_offsets_2d, flatten_ids,
            v_img, 5, 5, 8, False
        ),
        rasterize_to_pixels_bwd_per_gaussian(
            means2d, conics2d, opacity2d, mu, tile_offsets_2d, isect_ids,
            flatten_ids, v_img, 5, 5, 8, False
        ),
    )
    pixel_grads = rasterize_to_pixels_bwd(
        means2d, conics2d, opacity2d, mu, tile_offsets_2d, flatten_ids,
        v_img, 5, 5, 8, True
    )
    primitive_grads = rasterize_to_pixels_bwd_per_gaussian(
        means2d, conics2d, opacity2d, mu, tile_offsets_2d, isect_ids,
        flatten_ids, v_img, 5, 5, 8, True
    )
    assert_grads_close(pixel_grads, primitive_grads)
    assert torch.allclose(
        pixel_grads[0],
        gaussian_absgrad_reference(means2d, conics2d, opacity2d, mu, v_img),
        atol=2e-5,
        rtol=2e-5,
    )

    beta2d = torch.tensor([[4.0]], device=device)
    assert_grads_close(
        beta_rasterize_to_pixels_bwd(
            means2d, conics2d, opacity2d, mu, beta2d, tile_offsets_2d,
            flatten_ids, v_img, 5, 5, 8, False
        ),
        beta_rasterize_to_pixels_bwd_per_gaussian(
            means2d, conics2d, opacity2d, mu, beta2d, tile_offsets_2d,
            isect_ids, flatten_ids, v_img, 5, 5, 8, False
        ),
    )
    pixel_grads = beta_rasterize_to_pixels_bwd(
        means2d, conics2d, opacity2d, mu, beta2d, tile_offsets_2d,
        flatten_ids, v_img, 5, 5, 8, True
    )
    primitive_grads = beta_rasterize_to_pixels_bwd_per_gaussian(
        means2d, conics2d, opacity2d, mu, beta2d, tile_offsets_2d,
        isect_ids, flatten_ids, v_img, 5, 5, 8, True
    )
    assert_grads_close(pixel_grads, primitive_grads)
    assert torch.allclose(
        pixel_grads[0],
        beta_absgrad_reference(means2d, conics2d, opacity2d, mu, beta2d, v_img),
        atol=2e-5,
        rtol=2e-5,
    )

    means3d = torch.tensor([[2.5, 2.5, 2.5]], device=device)
    conics3d = torch.tensor([[1.0, 0.1, 0.0, 1.2, -0.05, 1.1]], device=device)
    opacity3d = torch.tensor([2.0], device=device)
    v_vol = torch.randn((5, 5, 5), device=device)
    assert_grads_close(
        voxelize_to_voxels_bwd(
            means3d, conics3d, opacity3d, tile_offsets_3d, flatten_ids,
            v_vol, 5, 5, 5, 8
        ),
        voxelize_to_voxels_bwd_per_gaussian(
            means3d, conics3d, opacity3d, tile_offsets_3d, isect_ids,
            flatten_ids, v_vol, 5, 5, 5, 8
        ),
    )

    gamma = torch.tensor([3.5], device=device)
    assert_grads_close(
        beta_voxelize_to_voxels_bwd(
            means3d, conics3d, opacity3d, gamma, tile_offsets_3d, flatten_ids,
            v_vol, 5, 5, 5, 8
        ),
        beta_voxelize_to_voxels_bwd_per_gaussian(
            means3d, conics3d, opacity3d, gamma, tile_offsets_3d, isect_ids,
            flatten_ids, v_vol, 5, 5, 5, 8
        ),
    )
