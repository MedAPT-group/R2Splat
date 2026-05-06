import torch
import pytest

from gsplat2ct.model.gaussian_params import GaussianParams
from gsplat2ct.model.opacity import (
    inverse_opacity_activation,
    opacity_activation,
    opacity_activation_grad,
)


def test_opacity_activation_roundtrip_matches_r2gaussian_density_semantics():
    density = torch.tensor([1.0e-6, 1.0e-4, 0.01, 0.5, 10.0])
    raw = inverse_opacity_activation(density)

    assert torch.allclose(opacity_activation(raw), density, rtol=1e-5, atol=1e-8)


def test_opacity_activation_gradient_is_sigmoid():
    raw = torch.tensor([-8.0, -1.0, 0.0, 2.0], dtype=torch.float64)

    assert torch.allclose(opacity_activation_grad(raw), torch.sigmoid(raw))


def test_softplus_checkpoint_loads_raw_opacity(tmp_path):
    density = torch.tensor([0.02, 0.5], dtype=torch.float32)
    raw_density = inverse_opacity_activation(density)
    ckpt_path = tmp_path / "softplus_raw.pt"
    torch.save(
        {
            "step": 7,
            "means": torch.zeros((2, 3), dtype=torch.float32),
            "quats": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
                dtype=torch.float32,
            ),
            "scale_ori": torch.zeros((2, 3), dtype=torch.float32),
            "opacity": raw_density,
            "opacity_activation": "softplus",
            "beta": None,
            "primitive_type": "gaussian",
        },
        ckpt_path,
    )

    gp, step = GaussianParams.load(str(ckpt_path), device="cpu")

    assert step == 7
    assert torch.allclose(opacity_activation(gp.opacity), density, rtol=1e-5, atol=1e-8)


def test_checkpoint_without_softplus_marker_is_rejected(tmp_path):
    ckpt_path = tmp_path / "legacy_linear.pt"
    torch.save(
        {
            "step": 7,
            "means": torch.zeros((1, 3), dtype=torch.float32),
            "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            "scale_ori": torch.zeros((1, 3), dtype=torch.float32),
            "opacity": torch.tensor([0.02], dtype=torch.float32),
            "beta": None,
            "primitive_type": "gaussian",
        },
        ckpt_path,
    )

    with pytest.raises(ValueError, match="opacity_activation='softplus'"):
        GaussianParams.load(str(ckpt_path), device="cpu")
