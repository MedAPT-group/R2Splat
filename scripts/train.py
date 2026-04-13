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
from gsplat2ct.vis.video import ProjectionVideoRecorder, VideoVisConfig
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
            )
            config.allow_negative_opacity = bool(getattr(cfg.optim, "allow_negative_opacity", False))
            return GS3DCTNewtonOptimizer(params, config)

        raise ValueError(
            f"Unknown optim.second_order_optimizer={second_order_optimizer}. "
            "Use one of: newton_diag_clamp, newton."
        )

    config = FirstOrderConfig(
        lr_map=lr_map,
        beta1=cfg.optim.beta1,
        beta2=cfg.optim.beta2,
        eps=cfg.optim.eps,
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
    print(f"   Mode: {'Newton (second-order)' if use_second_order else 'Adam (first-order)'}")

    # ---- Scheduler ----
    from gsplat2ct.optim.scheduler import OptimizerScheduler
    scheduler = OptimizerScheduler.from_hydra(getattr(cfg, "schedule", None) or {}, optimizer)

    # ---- Strategy ----
    from gsplat2ct.strategy import DefaultStrategy
    from gsplat2ct.strategy import CompensationStrategy
    strategy = DefaultStrategy(getattr(cfg, "strategy", None) or {})
    compensate_strategy = CompensationStrategy(getattr(cfg, "compensate", None) or {})

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
    video_recorder = ProjectionVideoRecorder(
        vis_cfg=video_vis_cfg,
        dataset=evalset,
        device=device,
        out_dir=vis_dir,
        scale_min=cfg.model.scale_min,
        scale_max=cfg.model.scale_max,
        eps_blur=cfg.trainer.eps_blur,
    )
    video_recorder.set_params(params)

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
        if use_second_order:
            fwd, grad, hess = operator.forward_backward2(params, data_obj, op_config)
            optimizer.add_grad_hess(grad, hess)
        else:
            fwd, grad = operator.forward_backward1(params, data_obj, op_config)
            optimizer.add_grad(grad)

        # Strategy: gsplat-style grad2d/count accumulation + periodic refine
        topology_changed = False
        v_means2d = grad.get_aux("v_means2d")
        v_means2d_abs = grad.get_aux("v_means2d_abs")
        radii_2d = grad.get_aux("radii_2d")
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
                if not use_second_order and hasattr(optimizer, "apply_topology_transition"):
                    optimizer.apply_topology_transition(
                        duplicate_sel=transition.duplicate_sel,
                        split_keep_parent=transition.split_keep_parent,
                        split_children=transition.split_children,
                        prune_keep=transition.prune_keep,
                    )
                params = gp.to_params()
                operator.initialize(params)
                optimizer.params = params
                if use_second_order:
                    optimizer.reset_moments()
                video_recorder.set_params(params)
                optimizer.zero_grad()
                topology_changed = True

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
                strategy.reset_grad_accumulator()
                if not use_second_order and hasattr(optimizer, "apply_topology_transition"):
                    n_added = max(int(gp.num_gaussians) - int(n_before_comp), 0)
                    optimizer.apply_topology_transition(append_new=n_added)
                params = gp.to_params()
                operator.initialize(params)
                optimizer.params = params
                if use_second_order:
                    optimizer.reset_moments()
                video_recorder.set_params(params)
                optimizer.zero_grad()
                topology_changed = True

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

        if not topology_changed:
            optimizer.step()
            optimizer.zero_grad()

        # Logging
        loss_total = fwd.outputs["loss"].item()
        loss_img = fwd.outputs["loss_image"].item()
        loss_tv = fwd.outputs["loss_tv"].item()
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
                wandb.log({"train/num_gaussians": float(gp.num_gaussians)}, step=step)
        except ImportError:
            pass

        video_recorder.maybe_record_step(step)

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
