import pytest
import torch
import torch.nn.functional as F


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_case(primitive_type):
    from gsplat2ct.model.gaussian_params import GaussianParams
    from gsplat2ct.ops.data import Config, Data

    points = torch.tensor(
        [
            [0.0, 0.0, 2.0, 0.5],
            [0.15, 0.1, 2.2, 0.4],
        ],
        device="cuda",
    )
    gp = GaussianParams.from_points(
        points,
        scale_min=0.05,
        scale_max=0.5,
        primitive_type=primitive_type,
        device="cuda",
    )

    data = Data()
    data.set(
        viewmats=torch.eye(4, device="cuda").unsqueeze(0),
        Ks=torch.tensor(
            [[[20.0, 0.0, 4.0], [0.0, 20.0, 4.0], [0.0, 0.0, 1.0]]],
            device="cuda",
        ),
        targets=torch.zeros((1, 8, 8), device="cuda"),
        vol_ranges=torch.ones(3, device="cuda"),
        vol_centers=torch.tensor([0.0, 0.0, 2.0], device="cuda"),
    )
    cfg = Config()
    cfg.set(
        img_dim1=8,
        img_dim0=8,
        img_tile_size=8,
        eps_blur=1.0e-8,
        lambda_l1=1.0,
        lambda_dssim=0.0,
        ssim_window=11,
        vol_dim2=4,
        vol_dim1=4,
        vol_dim0=4,
        vol_tile_size=4,
        eps3d=1.0e-8,
        radius_clip=1.0 if primitive_type == "beta" else 3.0,
        img_radius_clip=1.0 if primitive_type == "beta" else 3.0,
        lambda_tv=0.01,
        scale_min=0.05,
        scale_max=0.5,
        primitive_type=primitive_type,
        beta_gamma_init=3.5,
        absgrad=False,
        backward_mode="pixel",
    )
    return gp, data, cfg


@pytest.mark.parametrize("primitive_type", ["gaussian", "beta"])
def test_torch_autograd_path_matches_legacy_training_operator(primitive_type):
    from gsplat2ct.ops.gs3d_ct_train import GS3DCTLossCUDA
    from gsplat2ct.ops.torch_autograd import (
        GaussianMappingVoxelizer,
        GaussianProjectionRasterizer,
        activate_gaussian_params,
    )

    gp, data, cfg = _make_case(primitive_type)
    params = gp.to_params()
    legacy = GS3DCTLossCUDA()
    legacy.initialize(params, cfg)
    fwd, legacy_grad = legacy.forward_backward1(params, data, cfg)

    means = gp.means.detach().clone().requires_grad_(True)
    quats = gp.quats.detach().clone().requires_grad_(True)
    scale_ori = gp.scale_ori.detach().clone().requires_grad_(True)
    opacity_raw = gp.opacity.detach().clone().requires_grad_(True)
    beta = None
    if primitive_type == "beta":
        beta = gp.beta.detach().clone().requires_grad_(True)

    _quats, _scales, opacity, covars, max_scales = activate_gaussian_params(
        quats,
        scale_ori,
        opacity_raw,
        scale_min=cfg.get("scale_min"),
        scale_max=cfg.get("scale_max"),
    )
    rasterizer = GaussianProjectionRasterizer(
        img_w=cfg.get("img_dim1"),
        img_h=cfg.get("img_dim0"),
        tile_size=cfg.get("img_tile_size"),
        eps_blur=cfg.get("eps_blur"),
        radius_clip=cfg.get("img_radius_clip"),
        primitive_type=primitive_type,
        absgrad=False,
        backward_mode="pixel",
    )
    voxelizer = GaussianMappingVoxelizer(
        vol_dim2=cfg.get("vol_dim2"),
        vol_dim1=cfg.get("vol_dim1"),
        vol_dim0=cfg.get("vol_dim0"),
        tile_size=cfg.get("vol_tile_size"),
        eps3d=cfg.get("eps3d"),
        radius_clip=cfg.get("radius_clip"),
        primitive_type=primitive_type,
        backward_mode="pixel",
    )
    image = rasterizer(
        means,
        covars,
        opacity,
        data.get("viewmats"),
        data.get("Ks"),
        beta=beta,
        beta_gamma_init=cfg.get("beta_gamma_init"),
    ).image
    volume = voxelizer(
        means,
        covars,
        max_scales,
        opacity,
        data.get("vol_ranges"),
        data.get("vol_centers"),
        beta=beta,
        beta_gamma_init=cfg.get("beta_gamma_init"),
    ).volume
    loss = F.l1_loss(image, data.get("targets"), reduction="mean")
    edge_count = (
        (volume.shape[0] - 1) * volume.shape[1] * volume.shape[2]
        + volume.shape[0] * (volume.shape[1] - 1) * volume.shape[2]
        + volume.shape[0] * volume.shape[1] * (volume.shape[2] - 1)
    )
    loss = loss + cfg.get("lambda_tv") * (
        torch.diff(volume, dim=0).abs().sum()
        + torch.diff(volume, dim=1).abs().sum()
        + torch.diff(volume, dim=2).abs().sum()
    ) / max(edge_count, 1)
    loss.backward()

    assert torch.allclose(loss.detach(), fwd.outputs["loss"], atol=2e-5, rtol=2e-5)
    assert torch.allclose(means.grad, legacy_grad.get("means"), atol=3e-4, rtol=3e-4)
    assert torch.allclose(quats.grad, legacy_grad.get("quats"), atol=3e-4, rtol=3e-4)
    assert torch.allclose(scale_ori.grad, legacy_grad.get("scale_ori"), atol=3e-4, rtol=3e-4)
    assert torch.allclose(opacity_raw.grad, legacy_grad.get("opacity"), atol=3e-4, rtol=3e-4)
    if primitive_type == "beta":
        assert torch.allclose(beta.grad, legacy_grad.get("beta"), atol=3e-4, rtol=3e-4)
