import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_beta_rasterize_forward_matches_reference():
    from gsplat2ct.cuda._wrapper import beta_rasterize_to_pixels_fwd

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
    from gsplat2ct.cuda._wrapper import (
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
