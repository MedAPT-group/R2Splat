import pytest
import torch

from gsplat2ct.model.scale import resolve_scale_bounds, scanner_volume_to_world
from gsplat2ct.strategy.default import DefaultStrategy


def test_scanner_volume_to_world_uses_largest_volume_side():
    assert scanner_volume_to_world({"sVoxel": [1.0, 2.5, 2.0]}) == 2.5


def test_resolve_scale_bounds_matches_r2gaussian_relative_semantics():
    scale_min_world, scale_max_world, volume_to_world = resolve_scale_bounds(
        0.0005,
        0.5,
        {"sVoxel": [2.0, 1.0, 1.5]},
    )

    assert volume_to_world == 2.0
    assert scale_min_world == pytest.approx(0.001)
    assert scale_max_world == pytest.approx(1.0)


def test_densification_stats_apply_r2gaussian_screen_gradient_scaling():
    strategy = DefaultStrategy(
        {
            "enabled": True,
            "absgrad": False,
            "scale_grad_by_image": True,
            "refine_stop_iter": 10,
        }
    )

    v_means2d = torch.tensor([[[1.0, 2.0]]])
    radii_2d = torch.ones((1, 1), dtype=torch.int32)
    strategy.accumulate_grad_stats(
        step=0,
        v_means2d=v_means2d,
        v_means2d_abs=None,
        radii_2d=radii_2d,
        img_w=100,
        img_h=50,
        n_cameras=1,
    )

    # r2_gaussian multiplies dL/d(pixel center) by W/2 and H/2 in CUDA before
    # Python densification sees grad_means2D.
    assert strategy._grad2d[0].item() == pytest.approx((50.0**2 + 50.0**2) ** 0.5)
