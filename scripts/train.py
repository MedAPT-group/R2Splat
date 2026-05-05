"""Training entry point for first-order 3DGS CT reconstruction.

Usage:
    python -m scripts.train
    python -m scripts.train exp=25_views strategy=default
"""

from __future__ import annotations

import math
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
from gsplat2ct.eval.evaluator import ProjectionEvaluator, VolumeEvaluator
from gsplat2ct.model.ct_initializer import initialize_from_metadata
from gsplat2ct.model.gaussian_params import GaussianParams
from gsplat2ct.ops.data import Config, Data, ForwardResult, Params
from gsplat2ct.ops.gs3d_ct_train import GS3DCTLossCUDA
from gsplat2ct.optim.base import FirstOrderConfig
from gsplat2ct.optim.gs3d_ct import GS3DCTAdamOptimizer
from gsplat2ct.vis.plotting import save_projection_comparison, save_volume_comparison


def _env_truthy(name: str, default: str = "") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "on"}


def _maybe_init_wandb(cfg: DictConfig) -> None:
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

    init_kwargs: Dict[str, Any] = {
        "config": OmegaConf.to_container(cfg, resolve=True)
    }
    for env_name, key in (
        ("WANDB_PROJECT", "project"),
        ("WANDB_ENTITY", "entity"),
        ("WANDB_GROUP", "group"),
        ("WANDB_RUN_NAME", "name"),
        ("WANDB_JOB_TYPE", "job_type"),
    ):
        if os.environ.get(env_name):
            init_kwargs[key] = os.environ[env_name]
    if os.environ.get("WANDB_TAGS"):
        init_kwargs["tags"] = [
            tag.strip() for tag in os.environ["WANDB_TAGS"].split(",") if tag.strip()
        ]
    wandb.init(**init_kwargs)


def seed_everything(
    seed: int,
    deterministic: bool = True,
    cudnn_benchmark: bool = False,
    deterministic_warn_only: bool = True,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=deterministic_warn_only)
    torch.backends.cudnn.benchmark = cudnn_benchmark


def make_train_data(
    batch: Dict[str, torch.Tensor],
    *,
    vol_ranges: torch.Tensor,
    vol_centers: torch.Tensor,
    device: torch.device,
) -> Data:
    c2w = batch["camtoworld"].to(device).contiguous()
    data = Data()
    data.set(
        viewmats=torch.linalg.inv(c2w).contiguous(),
        Ks=batch["K"].to(device).contiguous(),
        targets=batch["image"].to(device).squeeze(-1).contiguous(),
        vol_ranges=vol_ranges.contiguous(),
        vol_centers=vol_centers.contiguous(),
    )
    return data


def make_train_config(
    cfg: DictConfig,
    img_shape: Tuple[int, int],
    vol_dims: Tuple[int, int, int],
    *,
    lambda_tv: Optional[float] = None,
) -> Config:
    img_h, img_w = img_shape
    vol_w, vol_d, vol_h = vol_dims
    trainer = cfg.trainer

    op_cfg = Config()
    op_cfg.set(
        img_dim1=img_w,
        img_dim0=img_h,
        img_tile_size=trainer.img_tile_size,
        eps_blur=trainer.eps_blur,
        lambda_l1=float(getattr(trainer, "lambda_l1", 1.0)),
        lambda_dssim=float(getattr(trainer, "lambda_dssim", 0.0)),
        ssim_window=int(getattr(trainer, "ssim_window", 11)),
        vol_dim2=vol_w,
        vol_dim1=vol_d,
        vol_dim0=vol_h,
        vol_tile_size=trainer.vol_tile_size,
        eps3d=trainer.eps3d,
        radius_clip=trainer.radius_clip,
        lambda_tv=float(trainer.lambda_tv if lambda_tv is None else lambda_tv),
        scale_min=cfg.model.scale_min,
        scale_max=cfg.model.scale_max,
        primitive_type=str(getattr(cfg.model, "primitive_type", "gaussian")),
        beta_gamma_init=float(getattr(cfg.model, "beta_gamma_init", 3.5)),
        absgrad=bool(getattr(getattr(cfg, "strategy", None), "absgrad", False)),
    )
    return op_cfg


def _extract_loss_values(fwd: ForwardResult) -> Tuple[float, float, float, float, float]:
    return (
        float(fwd.outputs["loss"].item()),
        float(fwd.outputs["loss_image"].item()),
        float(fwd.outputs.get("loss_l1", fwd.outputs["loss_image"]).item()),
        float(fwd.outputs.get("loss_dssim", torch.zeros(())).item()),
        float(fwd.outputs["loss_tv"].item()),
    )


def sample_tv_subvolume(
    bbox: torch.Tensor,
    tv_vol_size: int,
    volume_shape_wdh: Tuple[int, int, int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int]]:
    bbox_dev = bbox.to(device)
    full_ranges = bbox_dev[1] - bbox_dev[0]
    volume_shape = torch.tensor(volume_shape_wdh, device=device, dtype=torch.float32)
    voxel_size = full_ranges / volume_shape
    sub_ranges = torch.tensor([tv_vol_size] * 3, device=device, dtype=torch.float32) * voxel_size
    half = sub_ranges / 2.0
    valid_min = bbox_dev[0] + half
    valid_max = bbox_dev[1] - half
    center = valid_min + (valid_max - valid_min) * torch.rand(3, device=device)
    return sub_ranges, center, (tv_vol_size, tv_vol_size, tv_vol_size)


def _angles_to_radians(angles: torch.Tensor, angle_unit: str) -> torch.Tensor:
    unit = angle_unit.lower()
    if unit not in {"auto", "radian", "degree"}:
        raise ValueError(f"Unsupported trainer.angle_unit={angle_unit}")
    if unit == "auto":
        max_abs = float(angles.abs().max().item()) if angles.numel() else 0.0
        unit = "radian" if max_abs <= (2.0 * math.pi + 1e-3) else "degree"
    if unit == "degree":
        angles = torch.deg2rad(angles)
    return torch.remainder(angles, 2.0 * math.pi)


def _sample_random_batch_indices(
    *,
    num_views: int,
    batch_size: int,
    shuffle_perm: torch.Tensor,
    perm_cursor: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    if perm_cursor + batch_size <= num_views:
        batch_indices = shuffle_perm[perm_cursor : perm_cursor + batch_size]
        perm_cursor += batch_size
        return batch_indices, shuffle_perm, perm_cursor

    remain = num_views - perm_cursor
    head = shuffle_perm[perm_cursor:] if remain > 0 else torch.empty(0, dtype=torch.long, device=device)
    shuffle_perm = torch.randperm(num_views, device=device)
    take = batch_size - remain
    tail = shuffle_perm[:take]
    return torch.cat([head, tail], dim=0), shuffle_perm, take


def _sample_evenly_spaced_angle_batch_indices(
    angles_rad: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    num_views = int(angles_rad.numel())
    device = angles_rad.device
    if num_views == 0:
        raise RuntimeError("No views available for angle-based sampling")
    if batch_size <= 1:
        return torch.randint(0, num_views, (1,), device=device)
    if batch_size > num_views:
        core = _sample_evenly_spaced_angle_batch_indices(angles_rad, num_views)
        extra = torch.randint(0, num_views, (batch_size - num_views,), device=device)
        return torch.cat([core, core.index_select(0, extra)], dim=0)

    two_pi = 2.0 * math.pi
    step = two_pi / float(batch_size)
    phase = float(torch.rand(1, device=device).item() * two_pi)
    selected: List[int] = []
    selected_mask = torch.zeros(num_views, dtype=torch.bool, device=device)

    for i in range(batch_size):
        start = (phase + i * step) % two_pi
        end = (start + step) % two_pi
        in_bin = (angles_rad >= start) & (angles_rad < end) if start <= end else (
            (angles_rad >= start) | (angles_rad < end)
        )
        candidates = torch.nonzero(in_bin & (~selected_mask), as_tuple=False).flatten()
        if candidates.numel() > 0:
            pick = int(candidates[torch.randint(0, candidates.numel(), (1,), device=device)].item())
        else:
            center = (start + 0.5 * step) % two_pi
            diff = torch.abs(angles_rad - center)
            circ_dist = torch.minimum(diff, two_pi - diff)
            pick = int(torch.argmin(circ_dist + selected_mask.to(circ_dist.dtype) * 1e6).item())
        selected.append(pick)
        selected_mask[pick] = True
    return torch.tensor(selected, dtype=torch.long, device=device)


def build_optimizer(cfg: DictConfig, params: Params) -> GS3DCTAdamOptimizer:
    lr_map = {
        "means": float(cfg.optim.lr_means),
        "quats": float(cfg.optim.lr_quats),
        "scale_ori": float(cfg.optim.lr_scale_ori),
        "opacity": float(cfg.optim.lr_opacity),
    }
    if params.get("beta") is not None:
        lr_map["beta"] = float(getattr(cfg.optim, "lr_beta", cfg.optim.lr_opacity))
    config = FirstOrderConfig(
        lr_map=lr_map,
        beta1=float(cfg.optim.beta1),
        beta2=float(cfg.optim.beta2),
        eps=float(cfg.optim.eps),
    )
    config.allow_negative_opacity = bool(getattr(cfg.optim, "allow_negative_opacity", False))
    return GS3DCTAdamOptimizer(params, config)


def train(cfg: DictConfig) -> None:
    _maybe_init_wandb(cfg)

    device = torch.device(cfg.trainer.device if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_result_dir = cfg.exp.result_dir
    parent_dir = os.path.dirname(base_result_dir)
    run_name = os.path.basename(base_result_dir.rstrip("/")) or "run"
    result_dir = os.path.join(parent_dir, f"{timestamp}_{run_name}")
    cfg.exp.result_dir = result_dir
    ckpt_dir = os.path.join(result_dir, "ckpts")
    vis_dir = os.path.join(result_dir, "vis")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(result_dir, "config_used.yaml"))

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

    vol_gt: Optional[torch.Tensor] = None
    volume_shape_wdh: Optional[Tuple[int, int, int]] = None
    try:
        vol_gt = trainset.get_vol_gt().to(device)
        volume_shape_wdh = tuple(vol_gt.shape)
        print(f"   GT volume loaded: {volume_shape_wdh}")
    except Exception as exc:
        print(f"   GT volume not available: {exc}")
        nv = parser.meta.get("nVoxel")
        if nv is not None:
            volume_shape_wdh = tuple(nv)

    bbox: Optional[torch.Tensor] = None
    if parser.bbox is not None:
        bbox = torch.tensor(parser.bbox, dtype=torch.float32)
        print(f"   BBox: {bbox[0].tolist()} -> {bbox[1].tolist()}")

    print("[2/5] Initializing Gaussian parameters...")
    init_source = str(getattr(cfg.init, "source", "online")).lower()
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
            sart_n_iter=int(getattr(cfg.init, "sart_n_iter", 20)),
        )
        if init_result.coarse_volume_wdh is not None:
            print(f"   Coarse volume reconstructed: {tuple(init_result.coarse_volume_wdh.shape)}")
        gp = GaussianParams.from_points(
            init_result.sampled_points_n4,
            scale_min=cfg.model.scale_min,
            scale_max=cfg.model.scale_max,
            global_scale=cfg.model.global_scale,
            init_opa_scale=cfg.model.init_opa_scale,
            init_scale_multiplier=cfg.model.init_scale_multiplier,
            bbox=bbox,
            primitive_type=str(getattr(cfg.model, "primitive_type", "gaussian")),
            beta_gamma_init=float(getattr(cfg.model, "beta_gamma_init", 3.5)),
            device=str(device),
        )
    elif init_source == "file":
        gp = GaussianParams.from_data_dir(
            cfg.exp.data_dir,
            scale_min=cfg.model.scale_min,
            scale_max=cfg.model.scale_max,
            global_scale=cfg.model.global_scale,
            init_opa_scale=cfg.model.init_opa_scale,
            init_scale_multiplier=cfg.model.init_scale_multiplier,
            bbox=bbox,
            primitive_type=str(getattr(cfg.model, "primitive_type", "gaussian")),
            beta_gamma_init=float(getattr(cfg.model, "beta_gamma_init", 3.5)),
            device=str(device),
        )
    else:
        raise ValueError("init.source must be one of [online, file]")

    params = gp.to_params()
    primitive_name = "Beta primitives" if gp.primitive_type == "beta" else "Gaussians"
    print(f"   {gp.num_gaussians} {primitive_name} initialized")

    print("[3/5] Creating operator...")
    operator = GS3DCTLossCUDA()
    operator.initialize(params, make_train_config(cfg, (2, 2), (2, 2, 2), lambda_tv=0.0))

    print("[4/5] Creating Adam optimizer...")
    optimizer = build_optimizer(cfg, params)

    from gsplat2ct.optim.scheduler import OptimizerScheduler
    from gsplat2ct.strategy import build_strategy

    scheduler = OptimizerScheduler.from_hydra(getattr(cfg, "schedule", None) or {}, optimizer)
    strategy = build_strategy(getattr(cfg, "strategy", None) or {})

    print("[5/5] Starting training...")
    max_steps = int(cfg.trainer.max_steps)
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
        raise ValueError("trainer.angle_sampling_strategy must be random or evenly_spaced")
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
        f"strategy={angle_strategy}, batch_size={batch_size}, angle_unit={angle_unit}"
    )

    shuffle_perm = torch.randperm(num_views, device=device)
    perm_cursor = 0
    progress = tqdm(range(max_steps), desc="Training")

    _run_eval(params, cfg, evalset, vol_gt, bbox, volume_shape_wdh, device, vis_dir, step=0)

    for step in progress:
        scheduler.step(step, max_steps)

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
                angle_values_rad, batch_size
            )

        raw_batch = trainset.get_batch(batch_indices)
        img_tensor = raw_batch["image"]
        img_h, img_w = int(img_tensor.shape[1]), int(img_tensor.shape[2])

        tv_enabled = (
            float(cfg.trainer.lambda_tv) > 0.0
            and bbox is not None
            and volume_shape_wdh is not None
        )
        if tv_enabled:
            vol_ranges, vol_centers, vol_dims = sample_tv_subvolume(
                bbox, int(cfg.trainer.tv_vol_size), volume_shape_wdh, device
            )
            lambda_tv = float(cfg.trainer.lambda_tv)
        else:
            vol_ranges = torch.ones(3, device=device)
            vol_centers = torch.zeros(3, device=device)
            vol_dims = (2, 2, 2)
            lambda_tv = 0.0

        data_obj = make_train_data(
            raw_batch,
            vol_ranges=vol_ranges,
            vol_centers=vol_centers,
            device=device,
        )
        op_config = make_train_config(cfg, (img_h, img_w), vol_dims, lambda_tv=lambda_tv)

        fwd, grad = operator.forward_backward1(params, data_obj, op_config)
        optimizer.add_grad(grad)
        loss_total, loss_img, loss_l1, loss_dssim, loss_tv = _extract_loss_values(fwd)

        v_means2d = grad.get_aux("v_means2d")
        v_means2d_abs = grad.get_aux("v_means2d_abs")
        radii_2d = grad.get_aux("radii_2d")
        n_cameras = int(raw_batch["camtoworld"].shape[0])
        if strategy.should_accumulate_grad_stats(step) and v_means2d is not None and radii_2d is not None:
            strategy.accumulate_grad_stats(
                step=step,
                v_means2d=v_means2d,
                v_means2d_abs=v_means2d_abs,
                radii_2d=radii_2d,
                img_w=img_w,
                img_h=img_h,
                n_cameras=n_cameras,
            )

        skip_optimizer_update = False
        if strategy.want_refine(step):
            grow_scale3d_abs = None
            if bbox is not None:
                grow_scale3d_abs = float(strategy.grow_scale3d) * float((bbox[1] - bbox[0]).max().item())
            decision = strategy.decide(
                gp,
                scale_min=cfg.model.scale_min,
                scale_max=cfg.model.scale_max,
                grow_scale3d_abs=grow_scale3d_abs,
            )
            changed, transition = strategy.apply(
                gp,
                decision,
                scale_min=cfg.model.scale_min,
                scale_max=cfg.model.scale_max,
                bbox=bbox,
                step=step,
            )
            if changed:
                skip_optimizer_update = True
                strategy.reset_grad_accumulator()
                optimizer.apply_topology_transition(
                    duplicate_sel=transition.duplicate_sel,
                    split_keep_parent=transition.split_keep_parent,
                    split_children=transition.split_children,
                    prune_keep=transition.prune_keep,
                    append_new=getattr(transition, "append_new", 0),
                )
                params = gp.to_params()
                operator.initialize(params)
                optimizer.params = params
                optimizer.zero_grad()

        optimizer.step()
        optimizer.zero_grad()

        progress.set_postfix(
            total=f"{loss_total:.4f}",
            img=f"{loss_img:.4f}",
            l1=f"{loss_l1:.4f}",
            dssim=f"{loss_dssim:.4f}",
            tv=f"{loss_tv:.4f}",
            N=gp.num_gaussians,
            skip=int(skip_optimizer_update),
        )

        try:
            import wandb  # type: ignore

            if wandb.run is not None:
                wandb.log(
                    {
                        "train/loss_total": loss_total,
                        "train/loss_image": loss_img,
                        "train/loss_l1": loss_l1,
                        "train/loss_dssim": loss_dssim,
                        "train/loss_tv": loss_tv,
                        "train/num_gaussians": float(gp.num_gaussians),
                    },
                    step=step,
                )
        except ImportError:
            pass

        if step > 0 and step % int(cfg.trainer.eval_interval) == 0:
            _run_eval(params, cfg, evalset, vol_gt, bbox, volume_shape_wdh, device, vis_dir, step)

        if step in cfg.trainer.save_steps:
            gp.update_from_params(params)
            gp.save(os.path.join(ckpt_dir, f"ckpt_{step}.pt"), step=step)
            print(f"\n   Checkpoint saved at step {step}")

    gp.update_from_params(params)
    gp.save(os.path.join(ckpt_dir, "ckpt_final.pt"), step=max_steps - 1)
    print(f"\nTraining complete. Results in: {result_dir}")


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
    log_dict: Dict[str, float] = {}

    proj_eval = ProjectionEvaluator(
        params,
        scale_min=cfg.model.scale_min,
        scale_max=cfg.model.scale_max,
        eps_blur=cfg.trainer.eps_blur,
        ssim_window=int(getattr(cfg.trainer, "ssim_window", 11)),
        primitive_type=str(getattr(cfg.model, "primitive_type", "gaussian")),
        beta_gamma_init=float(getattr(cfg.model, "beta_gamma_init", 3.5)),
    )
    proj_metrics, vis_pairs = proj_eval.evaluate(evalset, device)
    print(
        f"\n   [Eval step {step}] "
        f"PSNR={proj_metrics['psnr']:.2f} dB, SSIM={proj_metrics['ssim']:.4f}"
    )
    log_dict.update({
        "img_metrics/l1": float(proj_metrics["l1"]),
        "img_metrics/mse": float(proj_metrics["mse"]),
        "img_metrics/psnr": float(proj_metrics["psnr"]),
        "img_metrics/ssim": float(proj_metrics["ssim"]),
    })

    vis_cfg = getattr(cfg, "vis", None)
    if vis_pairs and bool(getattr(vis_cfg, "save_projection_comparison", False)):
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
            ssim_window=int(getattr(cfg.trainer, "ssim_window", 11)),
            primitive_type=str(getattr(cfg.model, "primitive_type", "gaussian")),
            beta_gamma_init=float(getattr(cfg.model, "beta_gamma_init", 3.5)),
        )
        vol_metrics, pred_vol = vol_eval.evaluate(vol_gt, bbox, volume_shape_wdh, device)
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
            "vol/psnr": float(vol_metrics["psnr"]),
        })
        if bool(getattr(vis_cfg, "save_volume_comparison", False)):
            save_volume_comparison(
                pred_vol,
                vol_gt,
                os.path.join(vis_dir, f"vol_step_{step}.png"),
                psnr_val=vol_metrics["psnr"],
            )

    try:
        import wandb  # type: ignore

        if wandb.run is not None and log_dict:
            wandb.log(log_dict, step=step)
    except ImportError:
        pass


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
