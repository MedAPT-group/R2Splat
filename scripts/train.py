"""Training entry point for 3DGS-based CT reconstruction.

Usage::

    python -m scripts.train                        # defaults
    python -m scripts.train exp=15_views           # override experiment
    python -m scripts.train optim.use_second_order=true  # enable Newton
    python -m scripts.train exp.data_dir=/abs/path exp.result_dir=/abs/out
"""

import os
import random
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
from gsplat2ct.ops.data import Params, Data, Config, ForwardResult, GradResult, HessResult
from gsplat2ct.ops.gs3d_ct_train import GS3DCTLossCUDA
from gsplat2ct.optim.base import HessianMode, SignalStrategy
from gsplat2ct.eval.evaluator import ProjectionEvaluator, VolumeEvaluator
from gsplat2ct.vis.plotting import (
    save_projection_comparison,
    save_volume_comparison,
    save_loss_curve,
)
from gsplat2ct.vis.video import ProjectionVideoRecorder, VideoVisConfig


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
        PARAM_NAMES,
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
        config = SecondOrderConfig(
            lr_map=lr_map,
            damping_map=damping_map,
            hess_mode=HessianMode[cfg.optim.hess_mode],
            grad_strategy=SignalStrategy[cfg.optim.grad_strategy],
            hess_strategy=SignalStrategy[cfg.optim.hess_strategy],
        )
        return GS3DCTNewtonOptimizer(params, config)

    config = FirstOrderConfig(
        lr_map=lr_map,
        beta1=cfg.optim.beta1,
        beta2=cfg.optim.beta2,
        eps=cfg.optim.eps,
    )
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
    strategy = DefaultStrategy(getattr(cfg, "strategy", None) or {})

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
            changed = strategy.apply(
                gp,
                out,
                scale_min=cfg.model.scale_min,
                scale_max=cfg.model.scale_max,
                step=step,
            )
            strategy.reset_grad_accumulator()
            if changed:
                params = gp.to_params()
                operator.initialize(params)
                optimizer.params = params
                optimizer.reset_moments()
                video_recorder.set_params(params)
                optimizer.zero_grad()
                topology_changed = True

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
