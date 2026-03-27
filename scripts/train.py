"""Training entry point for 3DGS-based CT reconstruction.

Usage::

    python -m scripts.train                        # defaults
    python -m scripts.train exp=15_views           # override experiment
    python -m scripts.train optim.use_second_order=true  # enable Newton
    python -m scripts.train exp.data_dir=/abs/path exp.result_dir=/abs/out
"""

import os
import random
from typing import Any, Dict, Iterator, List, Optional, Tuple

import hydra
import numpy as np
import torch
import torch.utils.data
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from gsplat2ct.data.ctproj import CTImageDataset, CTImageParser
from gsplat2ct.model.gaussian_params import GaussianParams
from gsplat2ct.ops.data import Params, Data, Config, ForwardResult, GradResult, HessResult
from gsplat2ct.ops.gs3d_ct_train import GS3DCTTrainCUDA
from gsplat2ct.optim.base import HessianMode, SignalStrategy
from gsplat2ct.eval.evaluator import ProjectionEvaluator, VolumeEvaluator
from gsplat2ct.vis.plotting import (
    save_projection_comparison,
    save_volume_comparison,
    save_loss_curve,
)


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

    config = Config()
    config.set(
        img_dim1=img_w,
        img_dim0=img_h,
        img_tile_size=cfg.trainer.img_tile_size,
        eps_blur=cfg.trainer.eps_blur,
        lambda_l1=cfg.trainer.lambda_l1,
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
# Dataloader helpers
# =====================================================================

def fetch_next_batch(
    loader_iter: Iterator,
    loader: torch.utils.data.DataLoader,
) -> Tuple[Dict[str, torch.Tensor], Iterator]:
    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        batch = next(loader_iter)
    return batch, loader_iter


# =====================================================================
# Main training function
# =====================================================================

def train(cfg: DictConfig) -> None:
    device = torch.device(cfg.trainer.device if torch.cuda.is_available() else "cpu")
    result_dir = cfg.exp.result_dir
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

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=cfg.trainer.batch_size,
        shuffle=True,
        num_workers=cfg.trainer.num_workers,
        pin_memory=cfg.trainer.pin_memory,
        persistent_workers=(cfg.trainer.num_workers > 0 and cfg.trainer.persistent_workers),
    )

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
    operator = GS3DCTTrainCUDA()
    operator.initialize(params)

    # ---- Optimizer ----
    print("[4/5] Creating optimizer...")
    optimizer = build_optimizer(cfg, params)
    use_second_order = cfg.optim.use_second_order
    print(f"   Mode: {'Newton (second-order)' if use_second_order else 'Adam (first-order)'}")

    # ---- Training loop ----
    print("[5/5] Starting training...")
    loss_history: Dict[str, List[float]] = {"total": [], "image": [], "tv": []}
    train_iter = iter(train_loader)
    max_steps = cfg.trainer.max_steps
    progress = tqdm(range(max_steps), desc="Training")

    # Save baseline visualization right after initialization (step 0).
    _run_eval(
        params, cfg, evalset, vol_gt, bbox, volume_shape_wdh,
        device, vis_dir, step=0,
    )

    for step in progress:
        raw_batch, train_iter = fetch_next_batch(train_iter, train_loader)

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
    save_loss_curve(loss_history, os.path.join(vis_dir, "loss_curve.png"))
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

    if vis_pairs:
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
        save_volume_comparison(
            pred_vol, vol_gt,
            os.path.join(vis_dir, f"vol_step_{step}.png"),
            psnr_val=vol_metrics["psnr"],
        )


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
