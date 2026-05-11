import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_beta_training_operator_returns_beta_gradient():
    from gsplat2ct.model.gaussian_params import GaussianParams
    from gsplat2ct.ops.data import Config, Data
    from gsplat2ct.ops.gs3d_ct_train import GS3DCTLossCUDA

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
        primitive_type="beta",
        device="cuda",
    )
    params = gp.to_params()

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
        lambda_dssim=0.25,
        ssim_window=11,
        vol_dim2=4,
        vol_dim1=4,
        vol_dim0=4,
        vol_tile_size=4,
        eps3d=1.0e-8,
        radius_clip=3.0,
        lambda_tv=0.01,
        scale_min=0.05,
        scale_max=0.5,
        primitive_type="beta",
        beta_gamma_init=3.5,
        absgrad=True,
    )

    op = GS3DCTLossCUDA()
    op.initialize(params, cfg)
    fwd, grad = op.forward_backward1(params, data, cfg)

    assert torch.isfinite(fwd.outputs["loss"])
    assert torch.isfinite(fwd.outputs["loss_l1"])
    assert torch.isfinite(fwd.outputs["loss_dssim"])
    assert set(grad.param_grads) == {"means", "quats", "scale_ori", "opacity", "beta"}
    assert grad.param_grads["beta"].shape == gp.beta.shape


@pytest.mark.parametrize("primitive_type", ["gaussian", "beta"])
@pytest.mark.parametrize("backward_mode", ["pixel", "primitive"])
def test_training_operator_supports_backward_modes_for_primitives(primitive_type, backward_mode):
    from gsplat2ct.model.gaussian_params import GaussianParams
    from gsplat2ct.ops.data import Config, Data
    from gsplat2ct.ops.gs3d_ct_train import GS3DCTLossCUDA

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
    params = gp.to_params()

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
        lambda_dssim=0.25,
        ssim_window=11,
        vol_dim2=4,
        vol_dim1=4,
        vol_dim0=4,
        vol_tile_size=4,
        eps3d=1.0e-8,
        radius_clip=1.0 if primitive_type == "beta" else 3.0,
        lambda_tv=0.01,
        scale_min=0.05,
        scale_max=0.5,
        primitive_type=primitive_type,
        beta_gamma_init=3.5,
        absgrad=True,
        backward_mode=backward_mode,
    )

    op = GS3DCTLossCUDA()
    op.initialize(params, cfg)
    fwd, grad = op.forward_backward1(params, data, cfg)

    expected = {"means", "quats", "scale_ori", "opacity"}
    if primitive_type == "beta":
        expected.add("beta")
    assert torch.isfinite(fwd.outputs["loss"])
    assert set(grad.param_grads) == expected
