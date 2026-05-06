from omegaconf import OmegaConf

from scripts.train import make_train_config


def _cfg(primitive_type: str):
    return OmegaConf.create(
        {
            "model": {
                "primitive_type": primitive_type,
                "beta_gamma_init": 3.5,
            },
            "trainer": {
                "img_tile_size": 16,
                "eps_blur": 1.0e-8,
                "lambda_l1": 1.0,
                "lambda_dssim": 0.25,
                "ssim_window": 11,
                "vol_tile_size": 8,
                "eps3d": 1.0e-8,
                "radius_clip": 3.0,
                "beta_radius_clip": 1.0,
                "img_radius_clip": 3.0,
                "beta_img_radius_clip": 1.0,
                "lambda_tv": 0.05,
            },
            "strategy": {"absgrad": False},
        }
    )


def test_beta_train_config_uses_compact_support_radius_clip():
    cfg = _cfg("beta")

    op_cfg = make_train_config(
        cfg,
        img_shape=(32, 32),
        vol_dims=(8, 8, 8),
        scale_min_world=0.001,
        scale_max_world=1.0,
    )

    assert op_cfg.get("radius_clip") == 1.0
    assert op_cfg.get("img_radius_clip") == 1.0


def test_gaussian_train_config_keeps_gaussian_radius_clip():
    cfg = _cfg("gaussian")

    op_cfg = make_train_config(
        cfg,
        img_shape=(32, 32),
        vol_dims=(8, 8, 8),
        scale_min_world=0.001,
        scale_max_world=1.0,
    )

    assert op_cfg.get("radius_clip") == 3.0
    assert op_cfg.get("img_radius_clip") == 3.0
