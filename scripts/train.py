"""Training entry point for 3DGS-based CT reconstruction.

Usage::

    python -m scripts.train                        # defaults
    python -m scripts.train exp=15_views           # override experiment
    python -m scripts.train optim.use_second_order=true  # enable Newton
    python -m scripts.train exp.data_dir=/abs/path exp.result_dir=/abs/out
"""

import os
import random
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from gsplat2ct.data.ctproj import CTImageDataset, CTImageParser
from gsplat2ct.model.gaussian_params import GaussianParams
from gsplat2ct.model.ct_initializer import initialize_from_metadata
from gsplat2ct.ops.data import Params, Data, Config, ForwardResult, GradResult, HessResult
from gsplat2ct.ops.gs3d_ct_train import GS3DCTLossCUDA
from gsplat2ct.optim.base import HessianMode, SignalStrategy
from gsplat2ct.eval.evaluator import ProjectionEvaluator, VolumeEvaluator
from gsplat2ct.vis.plotting import (
    save_projection_comparison,
    save_volume_comparison,
    save_init_coarse_volume_debug,
    save_loss_curve,
    save_projection_triplet,
    save_residual_slices_with_points,
)
from gsplat2ct.vis.video import (
    GaussianStateVideoConfig,
    GaussianStateVideoRecorder,
    ProjectionVideoRecorder,
    VideoVisConfig,
)
from gsplat2ct.ops.gs3d_ct_rasterize import GS3DCTRasterizeCUDA
from gsplat2ct.ops.gs3d_ct_voxelize import GS3DCTVoxelizeCUDA


def _env_truthy(name: str, default: str = "") -> bool:
    val = os.environ.get(name, default)
    return val.lower() in {"1", "true", "yes", "on"}


def _maybe_init_wandb(cfg: DictConfig) -> None:
    """Initialize W&B run for both sweep and non-sweep launches.

    Non-sweep runs can be enabled by setting `WANDB_ENABLE=1` (or by setting
    `WANDB_PROJECT`). Sweep runs keep the existing `WANDB_SWEEP_ID` behavior.
    """
    try:
        import wandb  # type: ignore
    except ImportError:
        return

    if wandb.run is not None:
        return

    in_sweep = os.environ.get("WANDB_SWEEP_ID") is not None
    enable_regular = _env_truthy("WANDB_ENABLE") or bool(os.environ.get("WANDB_PROJECT"))
    if not in_sweep and not enable_regular:
        return

    init_kwargs: Dict[str, Any] = {}
    if os.environ.get("WANDB_PROJECT"):
        init_kwargs["project"] = os.environ["WANDB_PROJECT"]
    if os.environ.get("WANDB_ENTITY"):
        init_kwargs["entity"] = os.environ["WANDB_ENTITY"]
    if os.environ.get("WANDB_GROUP"):
        init_kwargs["group"] = os.environ["WANDB_GROUP"]
    if os.environ.get("WANDB_RUN_NAME"):
        init_kwargs["name"] = os.environ["WANDB_RUN_NAME"]
    if os.environ.get("WANDB_TAGS"):
        init_kwargs["tags"] = [t.strip() for t in os.environ["WANDB_TAGS"].split(",") if t.strip()]
    if os.environ.get("WANDB_JOB_TYPE"):
        init_kwargs["job_type"] = os.environ["WANDB_JOB_TYPE"]

    # Store resolved config for easier run-to-run comparisons in W&B.
    init_kwargs["config"] = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(**init_kwargs)


# =====================================================================
# Seeding
# =====================================================================

def seed_everything(seed: int, deterministic: bool = True,
                    cudnn_benchmark: bool = False,
                    deterministic_warn_only: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=deterministic_warn_only)
    torch.backends.cudnn.benchmark = cudnn_benchmark


# =====================================================================
# Data helpers
# =====================================================================

def make_train_data(
    batch: Dict[str, torch.Tensor],
    *,
    vol_ranges: torch.Tensor,
    vol_centers: torch.Tensor,
    device: torch.device,
) -> Data:
    """Convert a dataloader batch into an operator ``Data`` container."""
    c2w = batch["camtoworld"].to(device).contiguous()
    # CUDA kernels require contiguous row-major camera matrices.
    viewmats = torch.linalg.inv(c2w).contiguous()
    Ks = batch["K"].to(device).contiguous()
    targets = batch["image"].to(device).squeeze(-1).contiguous()  # (B, H, W)

    data_obj = Data()
    data_obj.set(
        viewmats=viewmats,
        Ks=Ks,
        targets=targets,
        vol_ranges=vol_ranges.contiguous(),
        vol_centers=vol_centers.contiguous(),
    )
    return data_obj


def make_train_config(
    cfg: DictConfig,
    img_shape: Tuple[int, int],
    vol_dims: Tuple[int, int, int],
) -> Config:
    """Build the operator ``Config`` from Hydra config and batch geometry."""
    img_h, img_w = img_shape
    vol_w, vol_d, vol_h = vol_dims

    tr = cfg.trainer
    if "lambda_l2" in tr:
        lambda_l2 = float(tr.lambda_l2)
    elif "lambda_l1" in tr:
        lambda_l2 = float(tr.lambda_l1)
    else:
        lambda_l2 = 1.0

    config = Config()
    config.set(
        img_dim1=img_w,
        img_dim0=img_h,
        img_tile_size=cfg.trainer.img_tile_size,
        eps_blur=cfg.trainer.eps_blur,
        lambda_l2=lambda_l2,
        lambda_ssim=cfg.trainer.lambda_ssim,
        window_size=cfg.trainer.window_size,
        vol_dim2=vol_w,
        vol_dim1=vol_d,
        vol_dim0=vol_h,
        vol_tile_size=cfg.trainer.vol_tile_size,
        eps3d=cfg.trainer.eps3d,
        radius_clip=cfg.trainer.radius_clip,
        vol_loss_eps=cfg.trainer.vol_loss_eps,
        lambda_tv=cfg.trainer.lambda_tv,
        scale_min=cfg.model.scale_min,
        scale_max=cfg.model.scale_max,
        image_loss_type=str(getattr(cfg.trainer, "image_loss_type", "l2")),
        absgrad=bool(getattr(cfg, "strategy", None) is not None and getattr(cfg.strategy, "absgrad", False)),
    )
    return config


def _extract_loss_triplet(fwd: ForwardResult) -> Tuple[float, float, float]:
    """Read total / image / TV loss scalars from a forward result."""
    return (
        float(fwd.outputs["loss"].item()),
        float(fwd.outputs["loss_image"].item()),
        float(fwd.outputs["loss_tv"].item()),
    )


# =====================================================================
# TV sub-volume sampling
# =====================================================================

def sample_tv_subvolume(
    bbox: torch.Tensor,
    tv_vol_size: int,
    volume_shape_wdh: Tuple[int, int, int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int]]:
    """Sample a random sub-volume inside the bounding box for TV loss.

    Returns:
        (vol_ranges, vol_centers, vol_dims) where vol_dims is a tuple
        ``(W_sub, D_sub, H_sub)`` matching the operator convention.
    """
    bbox_dev = bbox.to(device)
    full_ranges = bbox_dev[1] - bbox_dev[0]
    full_center = (bbox_dev[0] + bbox_dev[1]) / 2.0

    vol_shape_tensor = torch.tensor(
        volume_shape_wdh, device=device, dtype=torch.float32
    )
    voxel_size = full_ranges / vol_shape_tensor

    sub_n = torch.tensor([tv_vol_size] * 3, device=device, dtype=torch.float32)
    sub_ranges = sub_n * voxel_size

    half = sub_ranges / 2.0
    valid_min = bbox_dev[0] + half
    valid_max = bbox_dev[1] - half

    center = valid_min + (valid_max - valid_min) * torch.rand(3, device=device)

    return sub_ranges, center, (tv_vol_size, tv_vol_size, tv_vol_size)


# =====================================================================
# View batch sampling
# =====================================================================

def _angles_to_radians(
    angles: torch.Tensor,
    angle_unit: str,
) -> torch.Tensor:
    """Convert angle values to radians and fold to [0, 2pi)."""
    unit = angle_unit.lower()
    if unit not in {"auto", "radian", "degree"}:
        raise ValueError(f"Unsupported trainer.angle_unit={angle_unit}")

    if unit == "auto":
        max_abs = float(angles.abs().max().item()) if angles.numel() > 0 else 0.0
        # Most CT metadata in this codebase stores radians in [0, 2pi).
        # If values are larger than 2pi but within ~360, treat as degrees.
        unit = "radian" if max_abs <= (2.0 * math.pi + 1e-3) else "degree"

    if unit == "degree":
        angles = torch.deg2rad(angles)

    two_pi = 2.0 * math.pi
    return torch.remainder(angles, two_pi)


def _sample_random_batch_indices(
    *,
    num_views: int,
    batch_size: int,
    shuffle_perm: torch.Tensor,
    perm_cursor: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Existing random-without-replacement cycling behavior."""
    if perm_cursor + batch_size <= num_views:
        batch_indices = shuffle_perm[perm_cursor:perm_cursor + batch_size]
        perm_cursor += batch_size
    else:
        remain = num_views - perm_cursor
        head = shuffle_perm[perm_cursor:] if remain > 0 else torch.empty(0, dtype=torch.long, device=device)
        shuffle_perm = torch.randperm(num_views, device=device)
        take = batch_size - remain
        tail = shuffle_perm[:take]
        perm_cursor = take
        batch_indices = torch.cat([head, tail], dim=0)
    return batch_indices, shuffle_perm, perm_cursor


def _sample_evenly_spaced_angle_batch_indices(
    angles_rad: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Sample views whose angles are as evenly spread over [0, 2pi) as possible.

    Strategy:
    1. Draw a random phase offset.
    2. Split the circle into ``batch_size`` equal angular bins.
    3. Pick one view per bin (random in-bin if available; otherwise nearest to bin center).
    """
    num_views = int(angles_rad.numel())
    device = angles_rad.device
    if num_views == 0:
        raise RuntimeError("No views available for angle-based sampling")

    if batch_size <= 1:
        return torch.randint(0, num_views, (1,), device=device)

    if batch_size > num_views:
        core = _sample_evenly_spaced_angle_batch_indices(angles_rad, num_views)
        extra_idx = torch.randint(0, num_views, (batch_size - num_views,), device=device)
        return torch.cat([core, core.index_select(0, extra_idx)], dim=0)

    two_pi = 2.0 * math.pi
    step = two_pi / float(batch_size)
    phase = float(torch.rand(1, device=device).item() * two_pi)

    selected: List[int] = []
    selected_mask = torch.zeros(num_views, dtype=torch.bool, device=device)

    for i in range(batch_size):
        start = (phase + i * step) % two_pi
        end = (start + step) % two_pi
        if start <= end:
            in_bin = (angles_rad >= start) & (angles_rad < end)
        else:
            in_bin = (angles_rad >= start) | (angles_rad < end)

        cand = torch.nonzero(in_bin & (~selected_mask), as_tuple=False).flatten()
        if cand.numel() > 0:
            pick = int(cand[torch.randint(0, cand.numel(), (1,), device=device)].item())
            selected.append(pick)
            selected_mask[pick] = True
            continue

        center = (start + 0.5 * step) % two_pi
        diff = torch.abs(angles_rad - center)
        circ_dist = torch.minimum(diff, two_pi - diff)
        penalty = selected_mask.to(circ_dist.dtype) * 1e6
        pick = int(torch.argmin(circ_dist + penalty).item())
        selected.append(pick)
        selected_mask[pick] = True

    return torch.tensor(selected, dtype=torch.long, device=device)


# =====================================================================
# Optimizer construction
# =====================================================================

def build_optimizer(cfg: DictConfig, params: Params):
    """Build either Adam or Newton optimizer from Hydra config."""
    from gsplat2ct.optim.base import (
        FirstOrderConfig,
        SecondOrderConfig,
    )
    from gsplat2ct.optim.gs3d_ct import (
        GS3DCTAdamOptimizer,
        GS3DCTNewtonOptimizer,
    )
    from gsplat2ct.optim.gs3d_ct_diag_clamp import (
        DiagonalClampSecondOrderConfig,
        GS3DCTNewtonDiagClampOptimizer,
    )
    from gsplat2ct.optim.gs3d_ct_kfac import (
        KFACSecondOrderConfig,
        GS3DCTKFACOptimizer,
    )
    from gsplat2ct.optim.gs3d_ct_reduced import (
        ReducedNewtonConfig,
        GS3DCTReducedNewtonOptimizer,
    )

    lr_map = {
        "means": cfg.optim.lr_means,
        "quats": cfg.optim.lr_quats,
        "scale_ori": cfg.optim.lr_scale_ori,
        "opacity": cfg.optim.lr_opacity,
    }

    if cfg.optim.use_second_order:
        damping_map = {
            "means": cfg.optim.damping_means,
            "quats": cfg.optim.damping_quats,
            "scale_ori": cfg.optim.damping_scale_ori,
            "opacity": cfg.optim.damping_opacity,
        }

        # Signal-processing hyper-parameters for Newton-family optimizers.
        # For EMA strategy, only beta1_* is used.
        signal_kwargs = dict(
            beta1_g=float(getattr(cfg.optim, "beta1_g", 0.9)),
            beta2_g=float(getattr(cfg.optim, "beta2_g", 0.999)),
            eps_g=float(getattr(cfg.optim, "eps_g", 1e-8)),
            beta1_h=float(getattr(cfg.optim, "beta1_h", 0.9)),
            beta2_h=float(getattr(cfg.optim, "beta2_h", 0.999)),
            eps_h=float(getattr(cfg.optim, "eps_h", 1e-8)),
            warmup_beta_enabled=bool(getattr(cfg.optim, "warmup_beta_enabled", False)),
            warmup_beta_schedule=str(getattr(cfg.optim, "warmup_beta_schedule", "COSINE")),
            warmup_beta_steps=max(int(getattr(cfg.optim, "warmup_beta_steps", 0)), 0),
            warmup_beta1_g_start=float(getattr(cfg.optim, "warmup_beta1_g_start", 0.1)),
            warmup_beta1_g_end=float(getattr(cfg.optim, "warmup_beta1_g_end", 0.9)),
            warmup_beta1_h_start=float(getattr(cfg.optim, "warmup_beta1_h_start", 0.1)),
            warmup_beta1_h_end=float(getattr(cfg.optim, "warmup_beta1_h_end", 0.9)),
            warmup_beta_bias_correction=bool(
                getattr(cfg.optim, "warmup_beta_bias_correction", False)
            ),
            warmup_beta_verbose=bool(getattr(cfg.optim, "warmup_beta_verbose", False)),
            topology_damping_reanneal_enabled=bool(
                getattr(cfg.optim, "topology_damping_reanneal_enabled", False)
            ),
            topology_damping_reanneal_factor=float(
                getattr(cfg.optim, "topology_damping_reanneal_factor", 100.0)
            ),
            topology_damping_reanneal_steps=max(
                int(getattr(cfg.optim, "topology_damping_reanneal_steps", 50)),
                0,
            ),
            topology_damping_reanneal_schedule=str(
                getattr(cfg.optim, "topology_damping_reanneal_schedule", "COSINE")
            ),
            topology_damping_reanneal_verbose=bool(
                getattr(cfg.optim, "topology_damping_reanneal_verbose", False)
            ),
        )

        second_order_optimizer = str(
            getattr(cfg.optim, "second_order_optimizer", "newton_diag_clamp")
        ).lower()

        if second_order_optimizer in {"newton_diag_clamp", "diag_clamp", "clamped_diag"}:
            clamp_map = {
                "means": float(getattr(cfg.optim, "clamp_means", 1e-5)),
                "quats": float(getattr(cfg.optim, "clamp_quats", 1e-5)),
                "scale_ori": float(getattr(cfg.optim, "clamp_scale_ori", 1e-5)),
                "opacity": float(getattr(cfg.optim, "clamp_opacity", 1e-5)),
            }
            max_update_map = {
                "means": float(getattr(cfg.optim, "max_update_means", float("inf"))),
                "quats": float(getattr(cfg.optim, "max_update_quats", float("inf"))),
                "scale_ori": float(getattr(cfg.optim, "max_update_scale_ori", float("inf"))),
                "opacity": float(getattr(cfg.optim, "max_update_opacity", float("inf"))),
            }
            config = DiagonalClampSecondOrderConfig(
                lr_map=lr_map,
                damping_map=damping_map,
                clamp_map=clamp_map,
                max_update_map=max_update_map,
                hess_mode=HessianMode.DIAGONAL,
                grad_strategy=SignalStrategy[cfg.optim.grad_strategy],
                hess_strategy=SignalStrategy[cfg.optim.hess_strategy],
                verbose=bool(getattr(cfg.optim, "verbose", False)),
                **signal_kwargs,
            )
            config.allow_negative_opacity = bool(getattr(cfg.optim, "allow_negative_opacity", False))
            return GS3DCTNewtonDiagClampOptimizer(params, config)

        if second_order_optimizer in {"newton", "legacy_newton"}:
            config = SecondOrderConfig(
                lr_map=lr_map,
                damping_map=damping_map,
                hess_mode=HessianMode[cfg.optim.hess_mode],
                grad_strategy=SignalStrategy[cfg.optim.grad_strategy],
                hess_strategy=SignalStrategy[cfg.optim.hess_strategy],
                verbose=bool(getattr(cfg.optim, "verbose", False)),
                **signal_kwargs,
            )
            config.allow_negative_opacity = bool(getattr(cfg.optim, "allow_negative_opacity", False))
            return GS3DCTNewtonOptimizer(params, config)

        if second_order_optimizer in {"reduced_newton", "newton_reduced", "gs3d_reduced"}:
            clamp_map = {
                "means": float(getattr(cfg.optim, "clamp_means", 1e-8)),
                "quats": float(getattr(cfg.optim, "clamp_quats", 1e-8)),
                "scale_ori": float(getattr(cfg.optim, "clamp_scale_ori", 1e-8)),
                "opacity": float(getattr(cfg.optim, "clamp_opacity", 1e-8)),
            }
            max_update_map = {
                "means": float(getattr(cfg.optim, "max_update_means", float("inf"))),
                "quats": float(getattr(cfg.optim, "max_update_quats", float("inf"))),
                "scale_ori": float(getattr(cfg.optim, "max_update_scale_ori", float("inf"))),
                "opacity": float(getattr(cfg.optim, "max_update_opacity", float("inf"))),
            }
            config = ReducedNewtonConfig(
                lr_map=lr_map,
                damping_map=damping_map,
                clamp_map=clamp_map,
                max_update_map=max_update_map,
                svd_threshold=float(getattr(cfg.optim, "reduced_svd_threshold", 1e-8)),
                max_scale_rank=int(getattr(cfg.optim, "reduced_max_scale_rank", 2)),
                max_rotation_angle=float(getattr(cfg.optim, "max_rotation_angle", 5.0e-2)),
                allow_negative_opacity=bool(getattr(cfg.optim, "allow_negative_opacity", False)),
                min_opacity=float(getattr(cfg.optim, "min_opacity", 1.0e-4)),
                means_bound=float(getattr(cfg.optim, "means_bound", 1.0)),
                verbose=bool(getattr(cfg.optim, "verbose", False)),
            )
            return GS3DCTReducedNewtonOptimizer(params, config)

        if second_order_optimizer in {"kfac", "empirical_fisher", "ef"}:
            # K-FAC / Empirical-Fisher optimizer. Does not consume the true
            # Hessian, so its ``needs_hessian`` attribute is False and the
            # training loop below falls back to forward_backward1.
            ema_decay_map = {
                "means": float(getattr(cfg.optim, "ema_decay_means", 0.95)),
                "quats": float(getattr(cfg.optim, "ema_decay_quats", 0.95)),
                "scale_ori": float(getattr(cfg.optim, "ema_decay_scale_ori", 0.95)),
                "opacity": float(getattr(cfg.optim, "ema_decay_opacity", 0.95)),
            }
            grad_ema_decay_map = {
                "means": float(getattr(cfg.optim, "grad_ema_decay_means", 0.9)),
                "quats": float(getattr(cfg.optim, "grad_ema_decay_quats", 0.9)),
                "scale_ori": float(getattr(cfg.optim, "grad_ema_decay_scale_ori", 0.9)),
                "opacity": float(getattr(cfg.optim, "grad_ema_decay_opacity", 0.9)),
            }
            eig_clip_map = {
                "means": float(getattr(cfg.optim, "eig_clip_means", 0.0)),
                "quats": float(getattr(cfg.optim, "eig_clip_quats", 0.0)),
                "scale_ori": float(getattr(cfg.optim, "eig_clip_scale_ori", 0.0)),
                "opacity": float(getattr(cfg.optim, "eig_clip_opacity", 0.0)),
            }
            max_update_map = {
                "means": float(getattr(cfg.optim, "max_update_means", float("inf"))),
                "quats": float(getattr(cfg.optim, "max_update_quats", float("inf"))),
                "scale_ori": float(getattr(cfg.optim, "max_update_scale_ori", float("inf"))),
                "opacity": float(getattr(cfg.optim, "max_update_opacity", float("inf"))),
            }
            damping_rel_map = {
                "means": float(getattr(cfg.optim, "damping_rel_means", 0.1)),
                "quats": float(getattr(cfg.optim, "damping_rel_quats", 0.1)),
                "scale_ori": float(getattr(cfg.optim, "damping_rel_scale_ori", 0.1)),
                "opacity": float(getattr(cfg.optim, "damping_rel_opacity", 0.1)),
            }
            kfac_mode = str(getattr(cfg.optim, "kfac_mode", "per_param_group")).lower()
            if kfac_mode not in ("per_gaussian", "per_param_group"):
                raise ValueError(
                    f"Unknown optim.kfac_mode={kfac_mode}. "
                    "Use 'per_gaussian' or 'per_param_group'."
                )
            # KFAC does not consume the signal_kwargs beta1_g / beta2_g etc.
            # (it uses its own ema_decay_map), but we accept them so that the
            # shared SecondOrderConfig machinery (warmup-beta logging, damping
            # reanneal, etc.) continues to function.
            config = KFACSecondOrderConfig(
                lr_map=lr_map,
                damping_map=damping_map,
                damping_mode=str(
                    getattr(cfg.optim, "damping_mode", "absolute")
                ).lower(),
                damping_rel_map=damping_rel_map,
                warmup_damping_floor=float(
                    getattr(cfg.optim, "warmup_damping_floor", 0.0)
                ),
                warmup_match_fisher_scale=bool(
                    getattr(cfg.optim, "warmup_match_fisher_scale", False)
                ),
                ema_decay_map=ema_decay_map,
                grad_ema_decay_map=grad_ema_decay_map,
                eig_clip_map=eig_clip_map,
                max_update_map=max_update_map,
                mode=kfac_mode,
                bias_correction=bool(getattr(cfg.optim, "bias_correction", True)),
                min_step_for_fisher=int(getattr(cfg.optim, "min_step_for_fisher", 0)),
                hess_mode=HessianMode.BLOCK,
                grad_strategy=SignalStrategy.RAW,
                hess_strategy=SignalStrategy.RAW,
                verbose=bool(getattr(cfg.optim, "verbose", False)),
                **signal_kwargs,
            )
            config.allow_negative_opacity = bool(
                getattr(cfg.optim, "allow_negative_opacity", False)
            )
            return GS3DCTKFACOptimizer(params, config)

        raise ValueError(
            f"Unknown optim.second_order_optimizer={second_order_optimizer}. "
            "Use one of: newton_diag_clamp, newton, reduced_newton, kfac."
        )

    config = FirstOrderConfig(
        lr_map=lr_map,
        beta1=cfg.optim.beta1,
        beta2=cfg.optim.beta2,
        eps=cfg.optim.eps,
        verbose=bool(getattr(cfg.optim, "verbose", False)),
    )
    config.allow_negative_opacity = bool(getattr(cfg.optim, "allow_negative_opacity", False))
    return GS3DCTAdamOptimizer(params, config)


# =====================================================================
# Main training function
# =====================================================================

def train(cfg: DictConfig) -> None:
    # Weights & Biases is optional. It initializes automatically for sweeps,
    # and can be enabled for normal runs through environment variables.
    _maybe_init_wandb(cfg)

    device = torch.device(cfg.trainer.device if torch.cuda.is_available() else "cpu")
    # Always prepend a timestamp to output directory name.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_result_dir = cfg.exp.result_dir
    parent_dir = os.path.dirname(base_result_dir)
    run_name = os.path.basename(base_result_dir.rstrip("/"))
    if run_name == "":
        run_name = "run"
    result_dir = os.path.join(parent_dir, f"{timestamp}_{run_name}")
    cfg.exp.result_dir = result_dir
    os.makedirs(result_dir, exist_ok=True)

    ckpt_dir = os.path.join(result_dir, "ckpts")
    vis_dir = os.path.join(result_dir, "vis")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    OmegaConf.save(cfg, os.path.join(result_dir, "config_used.yaml"))

    # ---- Data ----
    print("[1/5] Loading dataset...")
    parser = CTImageParser(cfg.exp.data_dir)
    trainset = CTImageDataset(parser, split="train")
    try:
        testset = CTImageDataset(parser, split="test")
    except Exception:
        testset = None
    evalset = testset if testset is not None else trainset

    trainset.preload_to(device)
    if evalset is not trainset:
        evalset.preload_to(device)

    # ---- Ground truth volume ----
    vol_gt: Optional[torch.Tensor] = None
    volume_shape_wdh: Optional[Tuple[int, int, int]] = None
    try:
        vol_gt = trainset.get_vol_gt().to(device)
        volume_shape_wdh = tuple(vol_gt.shape)
        print(f"   GT volume loaded: {volume_shape_wdh}")
    except Exception as e:
        print(f"   GT volume not available: {e}")
        nv = parser.meta.get("nVoxel")
        if nv is not None:
            volume_shape_wdh = tuple(nv)

    bbox: Optional[torch.Tensor] = None
    if parser.bbox is not None:
        bbox = torch.tensor(parser.bbox, dtype=torch.float32)
        print(f"   BBox: {bbox[0].tolist()} -> {bbox[1].tolist()}")

    # ---- Parameters ----
    print("[2/5] Initializing Gaussian parameters...")
    init_source = str(getattr(cfg.init, "source", "online")).lower()
    if init_source not in {"online", "file"}:
        raise ValueError("init.source must be one of [online, file]")

    coarse_volume_wdh: Optional[torch.Tensor] = None
    if init_source == "online":
        init_result = initialize_from_metadata(
            data_dir=cfg.exp.data_dir,
            scanner_cfg=parser.get_scanner_params(),
            proj_train=parser.get_proj_info("train"),
            recon_method=str(cfg.init.recon_method),
            n_points=int(cfg.init.n_points),
            density_thresh=float(cfg.init.density_thresh),
            density_rescale=float(cfg.init.density_rescale),
            random_density_max=float(cfg.init.random_density_max),
            # Keep compatibility with init profiles that omit sart_n_iter.
            sart_n_iter=int(getattr(cfg.init, "sart_n_iter", 20)),
        )
        if init_result.coarse_volume_wdh is not None:
            coarse_volume_wdh = torch.from_numpy(init_result.coarse_volume_wdh).float().to(device)
        gp = GaussianParams.from_points(
            init_result.sampled_points_n4,
            scale_min=cfg.model.scale_min,
            scale_max=cfg.model.scale_max,
            global_scale=cfg.model.global_scale,
            init_opa_scale=cfg.model.init_opa_scale,
            init_scale_multiplier=cfg.model.init_scale_multiplier,
            bbox=bbox,
            device=str(device),
        )
        if init_result.coarse_volume_wdh is not None:
            print(f"   Coarse volume reconstructed: {tuple(init_result.coarse_volume_wdh.shape)}")
    else:
        gp = GaussianParams.from_data_dir(
            cfg.exp.data_dir,
            scale_min=cfg.model.scale_min,
            scale_max=cfg.model.scale_max,
            global_scale=cfg.model.global_scale,
            init_opa_scale=cfg.model.init_opa_scale,
            init_scale_multiplier=cfg.model.init_scale_multiplier,
            bbox=bbox,
            device=str(device),
        )
    params = gp.to_params()
    print(f"   {gp.num_gaussians} Gaussians initialized")

    # Debug visualization for initialization coarse volume (disabled).
    # if coarse_volume_wdh is not None and vol_gt is not None:
    #     cw = coarse_volume_wdh.detach().cpu()
    #     vg = vol_gt.detach().cpu()
    #     w = min(int(cw.shape[0]), int(vg.shape[0]))
    #     d = min(int(cw.shape[1]), int(vg.shape[1]))
    #     h = min(int(cw.shape[2]), int(vg.shape[2]))
    #     cw = cw[:w, :d, :h]
    #     vg = vg[:w, :d, :h]
    #     slice_indices = (w // 2, d // 2, h // 2)
    #     init_debug_dir = os.path.join(vis_dir, "init_debug")
    #     os.makedirs(init_debug_dir, exist_ok=True)
    #     save_init_coarse_volume_debug(
    #         coarse_vol=cw,
    #         gt_vol=vg,
    #         path=os.path.join(init_debug_dir, "init_coarse_vs_gt.png"),
    #         psnr_val=_compute_psnr(cw, vg),
    #         slice_indices=slice_indices,
    #     )

    # ---- Operator ----
    print("[3/5] Creating operator...")
    operator = GS3DCTLossCUDA()
    operator.initialize(params)

    # ---- Optimizer ----
    print("[4/5] Creating optimizer...")
    optimizer = build_optimizer(cfg, params)
    use_second_order = cfg.optim.use_second_order
    # Some second-order optimizers (e.g. KFAC / Empirical Fisher) only consume
    # first-order gradients. They advertise this via ``needs_hessian = False``
    # so that the training loop can skip the expensive forward_backward2.
    needs_hessian = bool(use_second_order) and bool(
        getattr(optimizer, "needs_hessian", True)
    )
    groupwise_step = bool(getattr(optimizer, "groupwise_step", False))
    if use_second_order:
        if groupwise_step:
            label = "Reduced Newton (groupwise second-order)"
        else:
            label = "Newton (second-order)" if needs_hessian else "KFAC (empirical-Fisher, first-order backward)"
    else:
        label = "Adam (first-order)"
    print(f"   Mode: {label}")

    # ---- Scheduler ----
    from gsplat2ct.optim.scheduler import OptimizerScheduler
    scheduler = OptimizerScheduler.from_hydra(getattr(cfg, "schedule", None) or {}, optimizer)

    # ---- Strategy ----
    from gsplat2ct.strategy import DefaultStrategy
    from gsplat2ct.strategy import CompensationStrategy
    strategy = DefaultStrategy(getattr(cfg, "strategy", None) or {})
    compensate_strategy = CompensationStrategy(getattr(cfg, "compensate", None) or {})
    topology_reanneal_enabled = bool(
        getattr(cfg.optim, "topology_reanneal_enabled", False)
    )
    topology_reanneal_steps = max(
        int(getattr(cfg.optim, "topology_reanneal_steps", 50)),
        0,
    )
    topology_reanneal_reset_moments = bool(
        getattr(cfg.optim, "topology_reanneal_reset_moments", False)
    )
    topology_damping_reanneal_enabled = bool(
        getattr(cfg.optim, "topology_damping_reanneal_enabled", False)
    )
    topology_damping_reanneal_factor = float(
        getattr(cfg.optim, "topology_damping_reanneal_factor", 100.0)
    )
    topology_damping_reanneal_steps = max(
        int(getattr(cfg.optim, "topology_damping_reanneal_steps", 50)),
        0,
    )
    topology_damping_reanneal_schedule = str(
        getattr(cfg.optim, "topology_damping_reanneal_schedule", "COSINE")
    )

    # ---- Visualization manager ----
    vis_cfg = getattr(cfg, "vis", None)
    video_cfg_node = getattr(vis_cfg, "video", None)
    video_vis_cfg = VideoVisConfig(
        enabled=bool(getattr(video_cfg_node, "enabled", False)) if video_cfg_node is not None else False,
        interval=int(getattr(video_cfg_node, "interval", 5)) if video_cfg_node is not None else 5,
        view_index=int(getattr(video_cfg_node, "view_index", 0)) if video_cfg_node is not None else 0,
        fps=int(getattr(video_cfg_node, "fps", 10)) if video_cfg_node is not None else 10,
        value_min=float(getattr(video_cfg_node, "value_min", 0.0)) if video_cfg_node is not None else 0.0,
        value_max=float(getattr(video_cfg_node, "value_max", 1.0)) if video_cfg_node is not None else 1.0,
        frame_subdir=str(getattr(video_cfg_node, "frame_subdir", "video_frames")) if video_cfg_node is not None else "video_frames",
        output_name=str(getattr(video_cfg_node, "output_name", "projection_progress.mp4")) if video_cfg_node is not None else "projection_progress.mp4",
    )
    gaussian_video_cfg_node = getattr(vis_cfg, "gaussian_state_video", None)
    gaussian_opacity_vmin = None
    gaussian_opacity_vmax = None
    if gaussian_video_cfg_node is not None and getattr(gaussian_video_cfg_node, "opacity_vmin", None) is not None:
        gaussian_opacity_vmin = float(getattr(gaussian_video_cfg_node, "opacity_vmin"))
    if gaussian_video_cfg_node is not None and getattr(gaussian_video_cfg_node, "opacity_vmax", None) is not None:
        gaussian_opacity_vmax = float(getattr(gaussian_video_cfg_node, "opacity_vmax"))

    gaussian_video_vis_cfg = GaussianStateVideoConfig(
        enabled=bool(getattr(gaussian_video_cfg_node, "enabled", False)) if gaussian_video_cfg_node is not None else False,
        interval=int(getattr(gaussian_video_cfg_node, "interval", 5)) if gaussian_video_cfg_node is not None else 5,
        fps=int(getattr(gaussian_video_cfg_node, "fps", 10)) if gaussian_video_cfg_node is not None else 10,
        frame_subdir=str(getattr(gaussian_video_cfg_node, "frame_subdir", "gaussian_state_frames")) if gaussian_video_cfg_node is not None else "gaussian_state_frames",
        output_name=str(getattr(gaussian_video_cfg_node, "output_name", "gaussian_state_progress.mp4")) if gaussian_video_cfg_node is not None else "gaussian_state_progress.mp4",
        max_points=int(getattr(gaussian_video_cfg_node, "max_points", 3000)) if gaussian_video_cfg_node is not None else 3000,
        point_size_min=float(getattr(gaussian_video_cfg_node, "point_size_min", 8.0)) if gaussian_video_cfg_node is not None else 8.0,
        point_size_max=float(getattr(gaussian_video_cfg_node, "point_size_max", 120.0)) if gaussian_video_cfg_node is not None else 120.0,
        use_log_scale_size=bool(getattr(gaussian_video_cfg_node, "use_log_scale_size", True)) if gaussian_video_cfg_node is not None else True,
        axis_length_factor=float(getattr(gaussian_video_cfg_node, "axis_length_factor", 0.25)) if gaussian_video_cfg_node is not None else 0.25,
        axis_length_min_ratio=float(getattr(gaussian_video_cfg_node, "axis_length_min_ratio", 0.002)) if gaussian_video_cfg_node is not None else 0.002,
        axis_length_max_ratio=float(getattr(gaussian_video_cfg_node, "axis_length_max_ratio", 0.05)) if gaussian_video_cfg_node is not None else 0.05,
        axis_alpha=float(getattr(gaussian_video_cfg_node, "axis_alpha", 0.45)) if gaussian_video_cfg_node is not None else 0.45,
        axis_linewidth=float(getattr(gaussian_video_cfg_node, "axis_linewidth", 0.6)) if gaussian_video_cfg_node is not None else 0.6,
        elev=float(getattr(gaussian_video_cfg_node, "elev", 20.0)) if gaussian_video_cfg_node is not None else 20.0,
        azim=float(getattr(gaussian_video_cfg_node, "azim", 35.0)) if gaussian_video_cfg_node is not None else 35.0,
        colormap=str(getattr(gaussian_video_cfg_node, "colormap", "viridis")) if gaussian_video_cfg_node is not None else "viridis",
        opacity_activation=str(getattr(gaussian_video_cfg_node, "opacity_activation", "raw")) if gaussian_video_cfg_node is not None else "raw",
        opacity_vmin=gaussian_opacity_vmin,
        opacity_vmax=gaussian_opacity_vmax,
    )

    video_recorder = ProjectionVideoRecorder(
        vis_cfg=video_vis_cfg,
        dataset=evalset,
        device=device,
        out_dir=vis_dir,
        scale_min=cfg.model.scale_min,
        scale_max=cfg.model.scale_max,
        eps_blur=cfg.trainer.eps_blur,
    )
    gaussian_video_recorder = GaussianStateVideoRecorder(
        vis_cfg=gaussian_video_vis_cfg,
        out_dir=vis_dir,
        scale_min=cfg.model.scale_min,
        scale_max=cfg.model.scale_max,
        bbox=bbox,
    )
    video_recorder.set_params(params)
    gaussian_video_recorder.set_params(params)
    strategy_debug_vis_dir = os.path.join(vis_dir, "strategy_debug")
    reduced_step_vis_cfg = getattr(vis_cfg, "reduced_step_projection", None)

    # ---- Training loop ----
    print("[5/5] Starting training...")
    loss_history: Dict[str, List[float]] = {"total": [], "image": [], "tv": []}
    max_steps = cfg.trainer.max_steps
    batch_size = int(cfg.trainer.batch_size)
    num_views = len(trainset)

    angle_strategy_raw = str(getattr(cfg.trainer, "angle_sampling_strategy", "random")).lower()
    angle_strategy_alias = {
        "random": "random",
        "rand": "random",
        "evenly_spaced": "evenly_spaced",
        "even": "evenly_spaced",
        "stratified": "evenly_spaced",
    }
    if angle_strategy_raw not in angle_strategy_alias:
        raise ValueError(
            f"Unsupported trainer.angle_sampling_strategy={angle_strategy_raw}. "
            "Use one of: random, evenly_spaced."
        )
    angle_strategy = angle_strategy_alias[angle_strategy_raw]
    angle_unit = str(getattr(cfg.trainer, "angle_unit", "auto")).lower()

    angle_values = torch.tensor(
        [float(info["angle"]) for info in trainset.proj_info],
        dtype=torch.float32,
        device=device,
    )
    angle_values_rad = _angles_to_radians(angle_values, angle_unit)

    print(
        "   View sampling: "
        f"strategy={angle_strategy}, "
        f"batch_size={batch_size}, "
        f"angle_unit={angle_unit}"
    )

    shuffle_perm = torch.randperm(num_views, device=device)
    perm_cursor = 0
    progress = tqdm(range(max_steps), desc="Training")

    # Save baseline visualization right after initialization (step 0).
    _run_eval(
        params, cfg, evalset, vol_gt, bbox, volume_shape_wdh,
        device, vis_dir, step=0,
    )

    for step in progress:
        scheduler.step(step, max_steps)

        # Keep the existing random path unchanged for batch_size==1 training.
        if angle_strategy == "random" or batch_size <= 1:
            batch_indices, shuffle_perm, perm_cursor = _sample_random_batch_indices(
                num_views=num_views,
                batch_size=batch_size,
                shuffle_perm=shuffle_perm,
                perm_cursor=perm_cursor,
                device=device,
            )
        else:
            batch_indices = _sample_evenly_spaced_angle_batch_indices(
                angle_values_rad,
                batch_size,
            )

        raw_batch = trainset.get_batch(batch_indices)

        # Image dimensions from the batch
        img_tensor = raw_batch["image"]
        img_h, img_w = img_tensor.shape[1], img_tensor.shape[2]

        # Sample TV sub-volume
        if cfg.trainer.lambda_tv > 0 and bbox is not None and volume_shape_wdh is not None:
            vol_ranges, vol_centers, vol_dims = sample_tv_subvolume(
                bbox, cfg.trainer.tv_vol_size, volume_shape_wdh, device
            )
        else:
            vol_ranges = torch.tensor([1.0, 1.0, 1.0], device=device)
            vol_centers = torch.zeros(3, device=device)
            vol_dims = (2, 2, 2)
            # Set lambda_tv to 0 effectively when no bbox
            if bbox is None:
                pass  # lambda_tv from config will still be used, set it to 0 in config

        data_obj = make_train_data(
            raw_batch,
            vol_ranges=vol_ranges,
            vol_centers=vol_centers,
            device=device,
        )
        op_config = make_train_config(cfg, (img_h, img_w), vol_dims)

        # Forward + backward
        groupwise_initial = None
        skip_optimizer_update = False
        if needs_hessian:
            fwd, grad, hess = operator.forward_backward2(params, data_obj, op_config)
            if groupwise_step:
                groupwise_initial = (fwd, grad, hess)
            else:
                optimizer.add_grad_hess(grad, hess)
        else:
            fwd, grad = operator.forward_backward1(params, data_obj, op_config)
            optimizer.add_grad(grad)

        loss_total, loss_img, loss_tv = _extract_loss_triplet(fwd)

        # For densification stats, the true-Hessian second-order path can
        # optionally query first-order abs-grad aux tensors without changing
        # the second-order optimization signals. Not needed when the
        # backward was already first-order (Adam or KFAC).
        strategy_grad_aux = grad
        if (
            needs_hessian
            and strategy.should_accumulate_grad_stats(step)
            and bool(getattr(strategy, "absgrad", False))
        ):
            _, grad_first_order_aux = operator.forward_backward1(params, data_obj, op_config)
            strategy_grad_aux = grad_first_order_aux

        # Strategy: gsplat-style grad2d/count accumulation + periodic refine
        v_means2d = strategy_grad_aux.get_aux("v_means2d")
        v_means2d_abs = strategy_grad_aux.get_aux("v_means2d_abs")
        radii_2d = strategy_grad_aux.get_aux("radii_2d")
        n_cameras = int(raw_batch["camtoworld"].shape[0])
        if strategy.should_accumulate_grad_stats(step):
            if v_means2d is not None and radii_2d is not None:
                strategy.accumulate_grad_stats(
                    step=step,
                    v_means2d=v_means2d,
                    v_means2d_abs=v_means2d_abs,
                    radii_2d=radii_2d,
                    img_w=img_w,
                    img_h=img_h,
                    n_cameras=n_cameras,
                )

        if strategy.want_refine(step):
            strategy_verbose = bool(getattr(strategy, "verbose", False))
            n_before_refine = int(gp.num_gaussians)
            _save_strategy_debug_visualization(
                params=params,
                cfg=cfg,
                evalset=evalset,
                vol_gt=vol_gt,
                bbox=bbox,
                volume_shape_wdh=volume_shape_wdh,
                device=device,
                out_dir=strategy_debug_vis_dir,
                step=step,
                phase="before",
            )
            if strategy_verbose:
                print(
                    f"[Strategy step {step}] loss_before_refine: "
                    f"total={loss_total:.6f}, image={loss_img:.6f}, tv={loss_tv:.6f}, "
                    f"N={n_before_refine}"
                )

            grow_scale3d_abs = None
            if bbox is not None:
                world_scale = (bbox[1] - bbox[0]).max().item()
                grow_scale3d_abs = float(strategy.grow_scale3d) * float(world_scale)
            out = strategy.decide(
                gp,
                scale_min=cfg.model.scale_min,
                scale_max=cfg.model.scale_max,
                grow_scale3d_abs=grow_scale3d_abs,
            )
            changed, transition = strategy.apply(
                gp,
                out,
                scale_min=cfg.model.scale_min,
                scale_max=cfg.model.scale_max,
                bbox=bbox,
                step=step,
            )
            if changed:
                skip_optimizer_update = True
                # Clear only DefaultStrategy grad2d/count history and keep
                # optimizer history through topology transition.
                strategy.reset_grad_accumulator()
                if hasattr(optimizer, "apply_topology_transition"):
                    optimizer.apply_topology_transition(
                        duplicate_sel=transition.duplicate_sel,
                        split_keep_parent=transition.split_keep_parent,
                        split_children=transition.split_children,
                        prune_keep=transition.prune_keep,
                    )
                params = gp.to_params()
                operator.initialize(params)
                optimizer.params = params
                video_recorder.set_params(params)
                gaussian_video_recorder.set_params(params)
                optimizer.zero_grad()
                if (
                    use_second_order
                    and topology_reanneal_enabled
                    and hasattr(optimizer, "restart_beta_warmup")
                ):
                    optimizer.restart_beta_warmup(
                        reset_moments=topology_reanneal_reset_moments,
                        warmup_steps=topology_reanneal_steps,
                    )
                if (
                    use_second_order
                    and topology_damping_reanneal_enabled
                    and hasattr(optimizer, "restart_damping_reanneal")
                ):
                    optimizer.restart_damping_reanneal(
                        factor=topology_damping_reanneal_factor,
                        steps=topology_damping_reanneal_steps,
                        schedule=topology_damping_reanneal_schedule,
                    )

            _save_strategy_debug_visualization(
                params=params,
                cfg=cfg,
                evalset=evalset,
                vol_gt=vol_gt,
                bbox=bbox,
                volume_shape_wdh=volume_shape_wdh,
                device=device,
                out_dir=strategy_debug_vis_dir,
                step=step,
                phase="after",
            )

            if strategy_verbose:
                if changed:
                    fwd_after_refine = operator.forward_only(params, data_obj, op_config)
                    post_total, post_img, post_tv = _extract_loss_triplet(fwd_after_refine)
                else:
                    post_total, post_img, post_tv = loss_total, loss_img, loss_tv
                next_opt_step = int(getattr(optimizer, "_step_count", 0)) + 1
                print(
                    f"[Strategy step {step}] loss_before_optimizer_step: "
                    f"total={post_total:.6f}, image={post_img:.6f}, tv={post_tv:.6f}, "
                    f"N={gp.num_gaussians}, changed={bool(changed)}, "
                    f"optimizer_step={next_opt_step}"
                )

        if (
            compensate_strategy.should_resample(step)
            and coarse_volume_wdh is not None
            and bbox is not None
            and volume_shape_wdh is not None
        ):
            # Debug visualization before compensation (disabled).
            # before_pred_proj, before_gt_proj = _render_eval_projection_single(
            #     params=params,
            #     dataset=evalset,
            #     device=device,
            #     scale_min=cfg.model.scale_min,
            #     scale_max=cfg.model.scale_max,
            #     eps_blur=cfg.trainer.eps_blur,
            #     view_index=0,
            # )
            # before_pred_vol = _render_eval_volume(
            #     params=params,
            #     bbox=bbox,
            #     volume_shape_wdh=volume_shape_wdh,
            #     device=device,
            #     scale_min=cfg.model.scale_min,
            #     scale_max=cfg.model.scale_max,
            #     eps3d=cfg.trainer.eps3d,
            # )

            n_before_comp = gp.num_gaussians
            changed_comp, comp_stats = compensate_strategy.apply(
                gp,
                coarse_volume_wdh=coarse_volume_wdh,
                bbox=bbox,
                volume_shape_wdh=volume_shape_wdh,
                scale_min=cfg.model.scale_min,
                scale_max=cfg.model.scale_max,
                step=step,
            )
            if changed_comp:
                skip_optimizer_update = True
                strategy.reset_grad_accumulator()
                applied_topology_transition = False
                if hasattr(optimizer, "apply_topology_transition"):
                    n_added = max(int(gp.num_gaussians) - int(n_before_comp), 0)
                    optimizer.apply_topology_transition(append_new=n_added)
                    applied_topology_transition = True
                params = gp.to_params()
                operator.initialize(params)
                optimizer.params = params
                if use_second_order and not applied_topology_transition and hasattr(
                    optimizer, "reset_moments"
                ):
                    optimizer.reset_moments()
                video_recorder.set_params(params)
                gaussian_video_recorder.set_params(params)
                optimizer.zero_grad()
                if (
                    use_second_order
                    and topology_reanneal_enabled
                    and hasattr(optimizer, "restart_beta_warmup")
                ):
                    optimizer.restart_beta_warmup(
                        reset_moments=topology_reanneal_reset_moments,
                        warmup_steps=topology_reanneal_steps,
                    )
                if (
                    use_second_order
                    and topology_damping_reanneal_enabled
                    and hasattr(optimizer, "restart_damping_reanneal")
                ):
                    optimizer.restart_damping_reanneal(
                        factor=topology_damping_reanneal_factor,
                        steps=topology_damping_reanneal_steps,
                        schedule=topology_damping_reanneal_schedule,
                    )

                # Debug visualization after compensation (disabled).
                # after_pred_proj, after_gt_proj = _render_eval_projection_single(
                #     params=params,
                #     dataset=evalset,
                #     device=device,
                #     scale_min=cfg.model.scale_min,
                #     scale_max=cfg.model.scale_max,
                #     eps_blur=cfg.trainer.eps_blur,
                #     view_index=0,
                # )
                # after_pred_vol = _render_eval_volume(
                #     params=params,
                #     bbox=bbox,
                #     volume_shape_wdh=volume_shape_wdh,
                #     device=device,
                #     scale_min=cfg.model.scale_min,
                #     scale_max=cfg.model.scale_max,
                #     eps3d=cfg.trainer.eps3d,
                # )
                #
                # comp_vis_dir = os.path.join(vis_dir, "compensate_debug")
                # os.makedirs(comp_vis_dir, exist_ok=True)
                #
                # save_projection_triplet(
                #     pred=before_pred_proj,
                #     gt=before_gt_proj,
                #     path=os.path.join(comp_vis_dir, f"step_{step:06d}_proj_before.png"),
                #     title=(
                #         f"Compensation BEFORE (step {step}) "
                #         f"PSNR={_compute_psnr(before_pred_proj, before_gt_proj):.2f} dB"
                #     ),
                # )
                # save_projection_triplet(
                #     pred=after_pred_proj,
                #     gt=after_gt_proj,
                #     path=os.path.join(comp_vis_dir, f"step_{step:06d}_proj_after.png"),
                #     title=(
                #         f"Compensation AFTER (step {step}) "
                #         f"PSNR={_compute_psnr(after_pred_proj, after_gt_proj):.2f} dB"
                #     ),
                # )
                #
                # if vol_gt is not None:
                #     slice_indices = (
                #         int(before_pred_vol.shape[0] // 2),
                #         int(before_pred_vol.shape[1] // 2),
                #         int(before_pred_vol.shape[2] // 2),
                #     )
                #     save_volume_comparison(
                #         before_pred_vol,
                #         vol_gt,
                #         os.path.join(comp_vis_dir, f"step_{step:06d}_vol_before.png"),
                #         psnr_val=_compute_psnr(before_pred_vol, vol_gt.detach().cpu()),
                #         slice_indices=slice_indices,
                #     )
                #     save_volume_comparison(
                #         after_pred_vol,
                #         vol_gt,
                #         os.path.join(comp_vis_dir, f"step_{step:06d}_vol_after.png"),
                #         psnr_val=_compute_psnr(after_pred_vol, vol_gt.detach().cpu()),
                #         slice_indices=slice_indices,
                #     )
                # else:
                #     slice_indices = (
                #         int(before_pred_vol.shape[0] // 2),
                #         int(before_pred_vol.shape[1] // 2),
                #         int(before_pred_vol.shape[2] // 2),
                #     )
                #
                # if comp_stats.residual_volume_wdh is not None and comp_stats.sampled_means_world is not None:
                #     save_residual_slices_with_points(
                #         residual_vol=comp_stats.residual_volume_wdh,
                #         sampled_points_world=comp_stats.sampled_means_world,
                #         sampled_values=comp_stats.sampled_opacity,
                #         bbox=bbox,
                #         path=os.path.join(comp_vis_dir, f"step_{step:06d}_residual_slices_with_points.png"),
                #         slice_indices=slice_indices,
                #     )

        # Align with r2_gaussian loop semantics: always advance one optimizer
        # step per training iteration, even if topology changed this round.
        # When topology changed, optimizer buffers were cleared above, so this
        # step is intentionally a no-op update but still advances step_count.
        if groupwise_step:
            if skip_optimizer_update:
                optimizer.step()
            else:
                capture_reduced_step_vis = (
                    reduced_step_vis_cfg is not None
                    and bool(getattr(reduced_step_vis_cfg, "enabled", False))
                    and int(getattr(reduced_step_vis_cfg, "interval", 1)) > 0
                    and step % int(getattr(reduced_step_vis_cfg, "interval", 1)) == 0
                )
                optimizer.step_groupwise(
                    operator,
                    data_obj,
                    op_config,
                    initial=groupwise_initial,
                    capture_stages=capture_reduced_step_vis,
                )
                if capture_reduced_step_vis and hasattr(optimizer, "pop_stage_snapshots"):
                    snapshots = optimizer.pop_stage_snapshots()
                    if snapshots:
                        _save_reduced_step_projection_visualization(
                            snapshots=snapshots,
                            cfg=cfg,
                            raw_batch=raw_batch,
                            data_obj=data_obj,
                            vol_gt=vol_gt,
                            bbox=bbox,
                            device=device,
                            vis_dir=vis_dir,
                            step=step,
                        )
        else:
            optimizer.step()
        optimizer.zero_grad()

        # Logging
        loss_history["total"].append(loss_total)
        loss_history["image"].append(loss_img)
        loss_history["tv"].append(loss_tv)

        progress.set_postfix(
            total=f"{loss_total:.4f}",
            img=f"{loss_img:.4f}",
            tv=f"{loss_tv:.4f}",
            N=gp.num_gaussians,
        )

        try:
            import wandb  # type: ignore

            if wandb.run is not None:
                wandb.log(
                    {
                        "train/loss_total": float(loss_total),
                        "train/loss_image": float(loss_img),
                        "train/loss_tv": float(loss_tv),
                        "train/num_gaussians": float(gp.num_gaussians),
                    },
                    step=step,
                )
        except ImportError:
            pass

        video_recorder.maybe_record_step(step)
        gaussian_video_recorder.maybe_record_step(step)

        # Periodic evaluation
        if step > 0 and step % cfg.trainer.eval_interval == 0:
            _run_eval(
                params, cfg, evalset, vol_gt, bbox, volume_shape_wdh,
                device, vis_dir, step,
            )

        # Save checkpoint
        if step in cfg.trainer.save_steps:
            gp.update_from_params(params)
            gp.save(os.path.join(ckpt_dir, f"ckpt_{step}.pt"), step=step)
            print(f"\n   Checkpoint saved at step {step}")

    # Final save
    gp.update_from_params(params)
    gp.save(os.path.join(ckpt_dir, "ckpt_final.pt"), step=max_steps - 1)
    if bool(getattr(vis_cfg, "save_loss_curve", False)):
        save_loss_curve(loss_history, os.path.join(vis_dir, "loss_curve.png"))
    video_path = video_recorder.finalize()
    if video_path is not None:
        print(f"   [VideoVis] Saved mp4: {video_path}")
    gaussian_video_path = gaussian_video_recorder.finalize()
    if gaussian_video_path is not None:
        print(f"   [GaussianStateVideo] Saved mp4: {gaussian_video_path}")
    print(f"\nTraining complete. Results in: {result_dir}")


# =====================================================================
# Evaluation helper
# =====================================================================

@torch.no_grad()
def _run_eval(
    params: Params,
    cfg: DictConfig,
    evalset: CTImageDataset,
    vol_gt: Optional[torch.Tensor],
    bbox: Optional[torch.Tensor],
    volume_shape_wdh: Optional[Tuple[int, int, int]],
    device: torch.device,
    vis_dir: str,
    step: int,
) -> None:
    """Run 2-D and 3-D evaluation and save visualizations."""
    log_dict: Dict[str, float] = {}

    proj_eval = ProjectionEvaluator(
        params,
        scale_min=cfg.model.scale_min,
        scale_max=cfg.model.scale_max,
        eps_blur=cfg.trainer.eps_blur,
        ssim_window=cfg.trainer.window_size,
    )
    proj_metrics, vis_pairs = proj_eval.evaluate(evalset, device)
    print(
        f"\n   [Eval step {step}] "
        f"PSNR={proj_metrics['psnr']:.2f} dB, "
        f"SSIM={proj_metrics['ssim']:.4f}"
    )
    log_dict.update({
        "img_metrics/l1": float(proj_metrics["l1"]),
        "img_metrics/mse": float(proj_metrics["mse"]),
        "img_metrics/psnr": float(proj_metrics["psnr"]),
        "img_metrics/ssim": float(proj_metrics["ssim"]),
    })

    if vis_pairs:
        vis_cfg = getattr(cfg, "vis", None)
        if bool(getattr(vis_cfg, "save_projection_comparison", False)):
            save_projection_comparison(
                vis_pairs,
                os.path.join(vis_dir, f"proj_step_{step}.png"),
                step=step,
            )

    if vol_gt is not None and bbox is not None and volume_shape_wdh is not None:
        vol_eval = VolumeEvaluator(
            params,
            scale_min=cfg.model.scale_min,
            scale_max=cfg.model.scale_max,
            eps3d=cfg.trainer.eps3d,
            ssim_window=cfg.trainer.window_size,
        )
        vol_metrics, pred_vol = vol_eval.evaluate(
            vol_gt, bbox, volume_shape_wdh, device
        )
        print(
            f"   [Eval step {step}] "
            f"Vol PSNR={vol_metrics['psnr']:.2f} dB, "
            f"Vol SSIM={vol_metrics['ssim']:.4f}"
        )
        log_dict.update({
            "vol_metrics/l1": float(vol_metrics["l1"]),
            "vol_metrics/mse": float(vol_metrics["mse"]),
            "vol_metrics/psnr": float(vol_metrics["psnr"]),
            "vol_metrics/ssim": float(vol_metrics["ssim"]),
            # Backward compatibility with existing sweep configs.
            "vol/psnr": float(vol_metrics["psnr"]),
        })
        vis_cfg = getattr(cfg, "vis", None)
        if bool(getattr(vis_cfg, "save_volume_comparison", False)):
            save_volume_comparison(
                pred_vol, vol_gt,
                os.path.join(vis_dir, f"vol_step_{step}.png"),
                psnr_val=vol_metrics["psnr"],
            )

    try:
        import wandb  # type: ignore

        if wandb.run is not None and log_dict:
            wandb.log(log_dict, step=step)
    except ImportError:
        pass


@torch.no_grad()
def _save_strategy_debug_visualization(
    *,
    params: Params,
    cfg: DictConfig,
    evalset: CTImageDataset,
    vol_gt: Optional[torch.Tensor],
    bbox: Optional[torch.Tensor],
    volume_shape_wdh: Optional[Tuple[int, int, int]],
    device: torch.device,
    out_dir: str,
    step: int,
    phase: str,
) -> None:
    """Save projection/volume snapshots for strategy before/after debugging."""
    os.makedirs(out_dir, exist_ok=True)

    proj_eval = ProjectionEvaluator(
        params,
        scale_min=cfg.model.scale_min,
        scale_max=cfg.model.scale_max,
        eps_blur=cfg.trainer.eps_blur,
        ssim_window=cfg.trainer.window_size,
    )
    _proj_metrics, vis_pairs = proj_eval.evaluate(evalset, device)
    if vis_pairs:
        save_projection_comparison(
            vis_pairs,
            os.path.join(out_dir, f"proj_step_{step:06d}_{phase}.png"),
            step=step,
        )

    if vol_gt is None or bbox is None or volume_shape_wdh is None:
        return

    vol_eval = VolumeEvaluator(
        params,
        scale_min=cfg.model.scale_min,
        scale_max=cfg.model.scale_max,
        eps3d=cfg.trainer.eps3d,
        ssim_window=cfg.trainer.window_size,
    )
    vol_metrics, pred_vol = vol_eval.evaluate(vol_gt, bbox, volume_shape_wdh, device)
    save_volume_comparison(
        pred_vol,
        vol_gt,
        os.path.join(out_dir, f"vol_step_{step:06d}_{phase}.png"),
        psnr_val=vol_metrics["psnr"],
    )


@torch.no_grad()
def _render_eval_projection_single(
    *,
    params: Params,
    dataset: CTImageDataset,
    device: torch.device,
    scale_min: float,
    scale_max: float,
    eps_blur: float,
    view_index: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Render a single evaluation projection (pred, gt), both clamped to [0,1]."""
    class _ParamWrapper:
        def __init__(self, data: torch.Tensor):
            self.data = data

    idx = int(max(0, min(view_index, len(dataset) - 1)))
    dataset.preload_to(device)
    batch = dataset.get_batch(idx)

    c2w = batch["camtoworld"].contiguous()
    K = batch["K"].contiguous()
    gt = batch["image"].contiguous().squeeze(-1)
    if gt.ndim == 2:
        gt = gt.unsqueeze(0)
    # gt = gt.clamp(0.0, 1.0)

    viewmats = torch.linalg.inv(c2w).contiguous()
    C = viewmats.shape[0]
    opacity_raw = params.get("opacity").data
    opacity_2d = opacity_raw.unsqueeze(0).expand(C, -1).contiguous()

    rast_params = Params()
    rast_params.add_param("means", params.get("means"))
    rast_params.add_param("quats", params.get("quats"))
    rast_params.add_param("scale_ori", params.get("scale_ori"))
    rast_params.add_param("opacity", _ParamWrapper(opacity_2d))

    data_obj = Data()
    data_obj.set(viewmats=viewmats, Ks=K)

    img_h, img_w = int(gt.shape[1]), int(gt.shape[2])
    op_cfg = Config()
    op_cfg.set(
        img_dim1=img_w,
        img_dim0=img_h,
        tile_size=16,
        eps_blur=eps_blur,
        scale_min=scale_min,
        scale_max=scale_max,
    )

    op = GS3DCTRasterizeCUDA()
    op.initialize(rast_params)
    pred = op.forward_only(rast_params, data_obj, op_cfg).outputs["render_output"]
    pred = pred.clamp(0.0, 1.0)
    return pred[0].detach().cpu(), gt[0].detach().cpu()


@torch.no_grad()
def _render_eval_volume(
    *,
    params: Params,
    bbox: torch.Tensor,
    volume_shape_wdh: Tuple[int, int, int],
    device: torch.device,
    scale_min: float,
    scale_max: float,
    eps3d: float,
) -> torch.Tensor:
    """Render current full volume for debug visualization in (W,D,H), clamped to [0,1]."""
    bbox_dev = bbox.to(device)
    ranges = bbox_dev[1] - bbox_dev[0]
    centers = (bbox_dev[0] + bbox_dev[1]) / 2.0
    W, D, H = volume_shape_wdh

    data_obj = Data()
    data_obj.set(ranges=ranges, centers=centers)

    cfg = Config()
    cfg.set(
        vol_dim2=W,
        vol_dim1=D,
        vol_dim0=H,
        vol_tile_size=8,
        eps3d=eps3d,
        radius_clip=3.0,
        scale_min=scale_min,
        scale_max=scale_max,
    )

    op = GS3DCTVoxelizeCUDA()
    op.initialize(params)
    pred_vol = op.forward_only(params, data_obj, cfg).outputs["render_output"]
    if pred_vol.ndim >= 3:
        pred_vol = pred_vol.squeeze()
    if pred_vol.ndim == 3:
        pred_vol = pred_vol.permute(2, 1, 0).contiguous()
    pred_vol = pred_vol.clamp(0.0, 1.0)
    return pred_vol.detach().cpu()


class _TensorParamWrapper:
    def __init__(self, data: torch.Tensor):
        self.data = data


def _scale_intrinsics(K: torch.Tensor, sx: float, sy: float) -> torch.Tensor:
    K_scaled = K.clone()
    K_scaled[:, 0, 0] *= float(sx)
    K_scaled[:, 0, 2] *= float(sx)
    K_scaled[:, 1, 1] *= float(sy)
    K_scaled[:, 1, 2] *= float(sy)
    return K_scaled.contiguous()


def _rotate_camtoworld_around_z(c2w: torch.Tensor, angle_rad: float) -> torch.Tensor:
    ca = math.cos(float(angle_rad))
    sa = math.sin(float(angle_rad))
    Rz = torch.tensor(
        [[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]],
        device=c2w.device,
        dtype=c2w.dtype,
    )
    out = c2w.clone()
    out[:, :3, :3] = torch.einsum("ij,bjk->bik", Rz, c2w[:, :3, :3])
    out[:, :3, 3] = torch.einsum("ij,bj->bi", Rz, c2w[:, :3, 3])
    return out.contiguous()


@torch.no_grad()
def _resize_projection_for_vis(
    image: torch.Tensor,
    img_h: int,
    img_w: int,
) -> torch.Tensor:
    import torch.nn.functional as F

    x = image.detach().float()
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x.squeeze(-1)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x.squeeze(0)
    x4 = x.unsqueeze(0).unsqueeze(0)
    y = F.interpolate(x4, size=(int(img_h), int(img_w)), mode="bilinear", align_corners=False)
    return y[0, 0].detach().cpu()


@torch.no_grad()
def _render_projection_from_snapshot(
    *,
    snapshot: Dict[str, torch.Tensor],
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    img_h: int,
    img_w: int,
    device: torch.device,
    scale_min: float,
    scale_max: float,
    eps_blur: float,
    tile_size: int,
) -> torch.Tensor:
    C = int(viewmats.shape[0])
    opacity_raw = snapshot["opacity"].to(device=device).contiguous()
    if opacity_raw.ndim == 1:
        opacity = opacity_raw.unsqueeze(0).expand(C, -1).contiguous()
    else:
        opacity = opacity_raw[:C].contiguous()

    rast_params = Params()
    rast_params.add_param(
        "means",
        _TensorParamWrapper(snapshot["means"].to(device=device).contiguous()),
    )
    rast_params.add_param(
        "quats",
        _TensorParamWrapper(snapshot["quats"].to(device=device).contiguous()),
    )
    rast_params.add_param(
        "scale_ori",
        _TensorParamWrapper(snapshot["scale_ori"].to(device=device).contiguous()),
    )
    rast_params.add_param("opacity", _TensorParamWrapper(opacity))

    data_obj = Data()
    data_obj.set(viewmats=viewmats.contiguous(), Ks=Ks.contiguous())

    op_cfg = Config()
    op_cfg.set(
        img_dim1=int(img_w),
        img_dim0=int(img_h),
        tile_size=int(tile_size),
        eps_blur=float(eps_blur),
        scale_min=float(scale_min),
        scale_max=float(scale_max),
    )

    op = GS3DCTRasterizeCUDA()
    op.initialize(rast_params)
    pred = op.forward_only(rast_params, data_obj, op_cfg).outputs["render_output"]
    return pred[0].detach().clamp(0.0, 1.0).cpu()


@torch.no_grad()
def _project_volume_gt_for_view(
    *,
    vol_gt: torch.Tensor,
    bbox: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    img_h: int,
    img_w: int,
    ray_samples: int,
) -> torch.Tensor:
    """Ray-march the GT volume for qualitative projection debugging."""
    import torch.nn.functional as F

    device = viewmats.device
    dtype = viewmats.dtype
    vol = vol_gt.to(device=device, dtype=dtype)
    bbox_dev = bbox.to(device=device, dtype=dtype)
    bmin, bmax = bbox_dev[0], bbox_dev[1]

    c2w = torch.linalg.inv(viewmats[:1])[0]
    K = Ks[0].to(device=device, dtype=dtype)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    yy, xx = torch.meshgrid(
        torch.arange(int(img_h), device=device, dtype=dtype),
        torch.arange(int(img_w), device=device, dtype=dtype),
        indexing="ij",
    )
    dirs_cam = torch.stack(
        [
            (xx + 0.5 - cx) / fx.clamp_min(1e-12),
            (yy + 0.5 - cy) / fy.clamp_min(1e-12),
            torch.ones_like(xx),
        ],
        dim=-1,
    )
    dirs_world = torch.einsum("ij,hwj->hwi", c2w[:3, :3], dirs_cam)
    dirs_world = dirs_world / dirs_world.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    origin = c2w[:3, 3]

    eps = torch.as_tensor(1e-12, device=device, dtype=dtype)
    parallel = dirs_world.abs() < eps
    safe_dirs = torch.where(
        parallel,
        torch.where(dirs_world >= 0.0, eps, -eps),
        dirs_world,
    )
    inv_dirs = 1.0 / safe_dirs
    t0 = (bmin.view(1, 1, 3) - origin.view(1, 1, 3)) * inv_dirs
    t1 = (bmax.view(1, 1, 3) - origin.view(1, 1, 3)) * inv_dirs
    t_axis_min = torch.minimum(t0, t1)
    t_axis_max = torch.maximum(t0, t1)

    outside_parallel = parallel & (
        (origin.view(1, 1, 3) < bmin.view(1, 1, 3))
        | (origin.view(1, 1, 3) > bmax.view(1, 1, 3))
    )
    valid = ~outside_parallel.any(dim=-1)
    t_near = t_axis_min.max(dim=-1).values
    t_far = t_axis_max.min(dim=-1).values
    t_start = torch.clamp(t_near, min=0.0)
    valid = valid & (t_far > t_start)
    length = (t_far - t_start).clamp_min(0.0)

    n_samples = max(int(ray_samples), 2)
    alpha = torch.linspace(0.0, 1.0, n_samples, device=device, dtype=dtype)
    t = t_start.unsqueeze(0) + alpha.view(n_samples, 1, 1) * length.unsqueeze(0)
    points = origin.view(1, 1, 1, 3) + t.unsqueeze(-1) * dirs_world.unsqueeze(0)
    grid = 2.0 * (points - bmin.view(1, 1, 1, 3)) / (bmax - bmin).view(1, 1, 1, 3) - 1.0

    # vol_gt is stored as (W, D, H) for x/y/z. grid_sample expects
    # input shape (N, C, D, H, W) and grid coordinates ordered as x/y/z.
    volume = vol.permute(2, 1, 0).contiguous().unsqueeze(0).unsqueeze(0)
    samples = F.grid_sample(
        volume,
        grid.unsqueeze(0),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0, 0]
    proj = samples.mean(dim=0) * length
    proj = torch.where(valid, proj, torch.zeros_like(proj))
    return proj.detach().cpu()


def _finite_concat_for_vis(
    tensors: List[torch.Tensor],
    *,
    abs_value: bool = False,
) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    for tensor in tensors:
        x = tensor.detach().float().reshape(-1).cpu()
        x = x[torch.isfinite(x)]
        if x.numel() == 0:
            continue
        chunks.append(x.abs() if abs_value else x)
    if not chunks:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(chunks)


def _robust_percentile_for_vis(
    tensors: List[torch.Tensor],
    *,
    percentile: float,
    fallback: float,
    abs_value: bool = False,
) -> float:
    vals = _finite_concat_for_vis(tensors, abs_value=abs_value)
    if vals.numel() == 0:
        return float(fallback)
    q = min(max(float(percentile) / 100.0, 0.0), 1.0)
    value = float(torch.quantile(vals, q).item())
    return max(value, float(fallback), 1e-12)


def _stage_slug(title: str) -> str:
    return title.lower().replace(" ", "_")


def _save_single_reduced_stage_figure(
    *,
    out_dir: str,
    step: int,
    title: str,
    current_before: torch.Tensor,
    current_after: torch.Tensor,
    current_gt: torch.Tensor,
    perp_before: torch.Tensor,
    perp_after: torch.Tensor,
    perp_gt: torch.Tensor,
    vmin: float,
    value_vmax: float,
    diff_limit: float,
    improvement_limit: float,
    gamma: float,
    value_cmap: str,
    diff_cmap: str,
    improvement_cmap: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm, TwoSlopeNorm

    current_diff = current_after - current_before
    perp_diff = perp_after - perp_before
    current_improvement = (current_before - current_gt).abs() - (current_after - current_gt).abs()
    perp_improvement = (perp_before - perp_gt).abs() - (perp_after - perp_gt).abs()

    value_norm = PowerNorm(
        gamma=max(float(gamma), 1e-6),
        vmin=float(vmin),
        vmax=max(float(value_vmax), float(vmin) + 1e-12),
    )
    diff_norm = TwoSlopeNorm(vcenter=0.0, vmin=-float(diff_limit), vmax=float(diff_limit))
    improvement_norm = TwoSlopeNorm(
        vcenter=0.0,
        vmin=-float(improvement_limit),
        vmax=float(improvement_limit),
    )

    rows = [
        (
            "Current view",
            [current_before, current_after, current_gt, current_diff, current_improvement],
        ),
        (
            "Perpendicular view",
            [perp_before, perp_after, perp_gt, perp_diff, perp_improvement],
        ),
    ]
    col_titles = ["before", "after", "GT", "after - before", "|before-GT|-|after-GT|"]

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    fig.subplots_adjust(left=0.075, right=0.98, top=0.86, bottom=0.18, wspace=0.04, hspace=0.10)
    value_im = diff_im = improvement_im = None

    for row_idx, (_row_label, images) in enumerate(rows):
        for col_idx, image in enumerate(images):
            ax = axes[row_idx, col_idx]
            if col_idx < 3:
                value_im = ax.imshow(image.numpy(), cmap=value_cmap, norm=value_norm)
            elif col_idx == 3:
                diff_im = ax.imshow(image.numpy(), cmap=diff_cmap, norm=diff_norm)
            else:
                improvement_im = ax.imshow(image.numpy(), cmap=improvement_cmap, norm=improvement_norm)
            if row_idx == 0:
                ax.set_title(col_titles[col_idx])
            ax.axis("off")

    fig.text(0.025, 0.62, "Current view", rotation=90, va="center", ha="center", fontsize=12)
    fig.text(0.025, 0.33, "Perpendicular view", rotation=90, va="center", ha="center", fontsize=12)

    if value_im is not None:
        cax = fig.add_axes([0.12, 0.085, 0.28, 0.025])
        cbar = fig.colorbar(value_im, cax=cax, orientation="horizontal")
        cbar.set_label("projection value")
    if diff_im is not None:
        cax = fig.add_axes([0.47, 0.085, 0.20, 0.025])
        cbar = fig.colorbar(diff_im, cax=cax, orientation="horizontal")
        cbar.set_label("after - before")
    if improvement_im is not None:
        cax = fig.add_axes([0.73, 0.085, 0.20, 0.025])
        cbar = fig.colorbar(improvement_im, cax=cax, orientation="horizontal")
        cbar.set_label("positive = closer to GT")

    fig.suptitle(
        f"Reduced Newton {title} update - step {step} "
        f"(value p-range <= {value_vmax:.3g}, diff +/-{diff_limit:.3g}, improvement +/-{improvement_limit:.3g})",
        y=0.97,
    )
    fig.savefig(os.path.join(out_dir, f"step_{step:06d}_{_stage_slug(title)}.png"), dpi=150)
    plt.close(fig)


@torch.no_grad()
def _save_reduced_step_projection_visualization(
    *,
    snapshots: List[Tuple[str, Dict[str, torch.Tensor]]],
    cfg: DictConfig,
    raw_batch: Dict[str, torch.Tensor],
    data_obj: Data,
    vol_gt: Optional[torch.Tensor],
    bbox: Optional[torch.Tensor],
    device: torch.device,
    vis_dir: str,
    step: int,
) -> None:
    reduced_cfg = getattr(getattr(cfg, "vis", None), "reduced_step_projection", None)
    if reduced_cfg is None:
        return

    label_to_snapshot = {label: snap for label, snap in snapshots}
    stage_pairs = [
        ("before", "after_means", "Means"),
        ("after_means", "after_scales", "Scales"),
        ("after_scales", "after_quats", "Quats"),
        ("after_quats", "after_opacity", "Opacity"),
    ]
    if any(
        before_label not in label_to_snapshot or after_label not in label_to_snapshot
        for before_label, after_label, _title in stage_pairs
    ):
        return

    img_tensor = raw_batch["image"]
    src_h, src_w = int(img_tensor.shape[1]), int(img_tensor.shape[2])
    max_image_size = max(int(getattr(reduced_cfg, "max_image_size", 256)), 1)
    scale = min(1.0, float(max_image_size) / float(max(src_h, src_w)))
    img_h = max(int(round(src_h * scale)), 1)
    img_w = max(int(round(src_w * scale)), 1)
    sy = float(img_h) / float(src_h)
    sx = float(img_w) / float(src_w)

    view_current = data_obj.get("viewmats")[:1].to(device=device).contiguous()
    K_current = _scale_intrinsics(data_obj.get("Ks")[:1].to(device=device), sx, sy)

    angle_offset = math.radians(float(getattr(reduced_cfg, "perpendicular_angle_offset_deg", 90.0)))
    c2w_current = raw_batch["camtoworld"][:1].to(device=device).contiguous()
    c2w_perp = _rotate_camtoworld_around_z(c2w_current, angle_offset)
    view_perp = torch.linalg.inv(c2w_perp).contiguous()
    K_perp = K_current.clone()

    tile_size = int(getattr(cfg.trainer, "img_tile_size", 16))
    current_preds: Dict[str, torch.Tensor] = {}
    perp_preds: Dict[str, torch.Tensor] = {}
    for label, snap in label_to_snapshot.items():
        current_preds[label] = (
            _render_projection_from_snapshot(
                snapshot=snap,
                viewmats=view_current,
                Ks=K_current,
                img_h=img_h,
                img_w=img_w,
                device=device,
                scale_min=cfg.model.scale_min,
                scale_max=cfg.model.scale_max,
                eps_blur=cfg.trainer.eps_blur,
                tile_size=tile_size,
            )
        )
        perp_preds[label] = (
            _render_projection_from_snapshot(
                snapshot=snap,
                viewmats=view_perp,
                Ks=K_perp,
                img_h=img_h,
                img_w=img_w,
                device=device,
                scale_min=cfg.model.scale_min,
                scale_max=cfg.model.scale_max,
                eps_blur=cfg.trainer.eps_blur,
                tile_size=tile_size,
            )
        )

    use_volume_gt = bool(getattr(reduced_cfg, "use_volume_gt", True))
    if use_volume_gt and vol_gt is not None and bbox is not None:
        ray_samples = int(getattr(reduced_cfg, "ray_samples", 128))
        gt_current = _project_volume_gt_for_view(
            vol_gt=vol_gt,
            bbox=bbox,
            viewmats=view_current,
            Ks=K_current,
            img_h=img_h,
            img_w=img_w,
            ray_samples=ray_samples,
        )
        gt_perp = _project_volume_gt_for_view(
            vol_gt=vol_gt,
            bbox=bbox,
            viewmats=view_perp,
            Ks=K_perp,
            img_h=img_h,
            img_w=img_w,
            ray_samples=ray_samples,
        )
    else:
        gt_current = _resize_projection_for_vis(img_tensor[0], img_h, img_w)
        gt_perp = torch.zeros_like(gt_current)

    frame_subdir = str(getattr(reduced_cfg, "frame_subdir", "reduced_step_projection"))
    out_dir = os.path.join(vis_dir, frame_subdir)
    os.makedirs(out_dir, exist_ok=True)

    vmin = float(getattr(reduced_cfg, "value_min", 0.0))
    value_vmax = float(getattr(reduced_cfg, "value_max", 1.0))
    value_percentile = float(getattr(reduced_cfg, "value_percentile", 99.5))
    diff_percentile = float(getattr(reduced_cfg, "diff_percentile", 99.0))
    improvement_percentile = float(getattr(reduced_cfg, "improvement_percentile", 99.0))
    gamma = float(getattr(reduced_cfg, "gamma", 0.6))
    value_cmap = str(getattr(reduced_cfg, "value_cmap", "magma"))
    diff_cmap = str(getattr(reduced_cfg, "diff_cmap", "coolwarm"))
    improvement_cmap = str(getattr(reduced_cfg, "improvement_cmap", "RdYlGn"))

    all_value_tensors: List[torch.Tensor] = list(current_preds.values()) + list(perp_preds.values())
    all_value_tensors.extend([gt_current, gt_perp])
    if bool(getattr(reduced_cfg, "auto_value_range", True)):
        value_vmax = _robust_percentile_for_vis(
            all_value_tensors,
            percentile=value_percentile,
            fallback=max(value_vmax, vmin + 1e-12),
        )

    for before_label, after_label, title in stage_pairs:
        current_before = current_preds[before_label]
        current_after = current_preds[after_label]
        perp_before = perp_preds[before_label]
        perp_after = perp_preds[after_label]
        current_diff = current_after - current_before
        perp_diff = perp_after - perp_before
        current_improvement = (current_before - gt_current).abs() - (current_after - gt_current).abs()
        perp_improvement = (perp_before - gt_perp).abs() - (perp_after - gt_perp).abs()
        diff_limit = _robust_percentile_for_vis(
            [current_diff, perp_diff],
            percentile=diff_percentile,
            fallback=1e-6,
            abs_value=True,
        )
        improvement_limit = _robust_percentile_for_vis(
            [current_improvement, perp_improvement],
            percentile=improvement_percentile,
            fallback=1e-6,
            abs_value=True,
        )
        _save_single_reduced_stage_figure(
            out_dir=out_dir,
            step=step,
            title=title,
            current_before=current_before,
            current_after=current_after,
            current_gt=gt_current,
            perp_before=perp_before,
            perp_after=perp_after,
            perp_gt=gt_perp,
            vmin=vmin,
            value_vmax=value_vmax,
            diff_limit=diff_limit,
            improvement_limit=improvement_limit,
            gamma=gamma,
            value_cmap=value_cmap,
            diff_cmap=diff_cmap,
            improvement_cmap=improvement_cmap,
        )


def _compute_psnr(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-12) -> float:
    """Compute PSNR for tensors expected to be in [0, 1]."""
    p = pred.detach().float()
    g = gt.detach().float()
    if p.shape != g.shape:
        slices = tuple(slice(0, min(a, b)) for a, b in zip(p.shape, g.shape))
        p = p[slices]
        g = g[slices]
    mse = torch.mean((p - g) ** 2).item()
    mse = max(float(mse), eps)
    return float(-10.0 * np.log10(mse))


# =====================================================================
# Hydra entry point
# =====================================================================

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    cfg.exp.data_dir = to_absolute_path(cfg.exp.data_dir)
    cfg.exp.result_dir = to_absolute_path(cfg.exp.result_dir)

    seed_everything(
        seed=cfg.random.seed,
        deterministic=cfg.random.deterministic,
        cudnn_benchmark=cfg.random.cudnn_benchmark,
        deterministic_warn_only=cfg.random.deterministic_warn_only,
    )

    train(cfg)


if __name__ == "__main__":
    main()
