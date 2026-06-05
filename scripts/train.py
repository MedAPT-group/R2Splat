"""Training entry point for first-order 3DGS CT reconstruction.

Usage:
    python -m scripts.train
    python -m scripts.train --config-name gaussian exp=0_chest_cone_25_views
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import torch
import torch.nn as nn
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from gsplat2ct.data.ctproj import CTImageDataset, CTImageParser
from gsplat2ct.eval.evaluator import ProjectionEvaluator, VolumeEvaluator
from gsplat2ct.eval.metric import ssim2d
from gsplat2ct.model.ct_initializer import initialize_from_metadata
from gsplat2ct.model.gaussian_params import GaussianParams
from gsplat2ct.model.scale import resolve_scale_bounds
from gsplat2ct.ops.data import Config, Data, ForwardResult, Params
from gsplat2ct.ops.gs3d_ct_train import GS3DCTLossCUDA
from gsplat2ct.ops.torch_autograd import (
    GaussianMappingVoxelizer,
    GaussianProjectionRasterizer,
    activate_gaussian_params,
    activate_uncertainty,
    pop_last_raster_absgrad,
)
from gsplat2ct.optim.base import FirstOrderConfig
from gsplat2ct.optim.gs3d_ct import GS3DCTAdamOptimizer
from gsplat2ct.optim.scheduler import _factor as _lr_schedule_factor
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


def set_cuda_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)


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
    scale_min_world: Optional[float] = None,
    scale_max_world: Optional[float] = None,
) -> Config:
    if scale_min_world is None or scale_max_world is None:
        raise ValueError("make_train_config expects world-space scale bounds")

    def _backward_mode(name: str, value: object) -> str:
        mode = str(value).lower()
        if mode not in {"pixel", "primitive"}:
            raise ValueError(f"trainer.{name} must be 'pixel' or 'primitive', got {mode!r}")
        return mode

    img_h, img_w = img_shape
    vol_w, vol_d, vol_h = vol_dims
    trainer = cfg.trainer
    primitive_type = str(getattr(cfg.model, "primitive_type", "gaussian")).lower()
    radius_clip = float(getattr(trainer, "radius_clip", 3.0))
    img_radius_clip = float(getattr(trainer, "img_radius_clip", 3.0))
    if primitive_type == "beta":
        radius_clip = float(getattr(trainer, "beta_radius_clip", 1.0))
        img_radius_clip = float(getattr(trainer, "beta_img_radius_clip", 1.0))
    backward_mode = _backward_mode(
        "backward_mode", getattr(trainer, "backward_mode", "pixel")
    )
    img_backward_mode = _backward_mode(
        "img_backward_mode", getattr(trainer, "img_backward_mode", backward_mode)
    )
    vol_backward_mode = _backward_mode(
        "vol_backward_mode", getattr(trainer, "vol_backward_mode", backward_mode)
    )

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
        radius_clip=radius_clip,
        img_radius_clip=img_radius_clip,
        lambda_tv=float(trainer.lambda_tv if lambda_tv is None else lambda_tv),
        scale_min=float(scale_min_world),
        scale_max=float(scale_max_world),
        primitive_type=primitive_type,
        beta_gamma_init=float(getattr(cfg.model, "beta_gamma_init", 3.5)),
        uncertainty_enabled=bool(getattr(trainer, "uncertainty_enabled", False)),
        uncertainty_q_min=float(getattr(cfg.model, "uncertainty_q_min", 1.0e-6)),
        uncertainty_q_max=(
            None
            if getattr(cfg.model, "uncertainty_q_max", None) is None
            else float(getattr(cfg.model, "uncertainty_q_max"))
        ),
        uncertainty_init=float(getattr(cfg.model, "init_uncertainty", 0.0025)),
        uncertainty_var_floor=float(getattr(trainer, "uncertainty_var_floor", 1.0e-6)),
        uncertainty_var_max=float(getattr(trainer, "uncertainty_var_max", 0.0)),
        lambda_nll=float(getattr(trainer, "lambda_nll", 0.0)),
        uncertainty_nll_auto_weight=bool(
            getattr(trainer, "uncertainty_nll_auto_weight", True)
        ),
        uncertainty_nll_auto_min=float(
            getattr(trainer, "uncertainty_nll_auto_min", 20.0)
        ),
        uncertainty_nll_auto_max=float(
            getattr(trainer, "uncertainty_nll_auto_max", 1000.0)
        ),
        lambda_uncertainty_prior=float(
            getattr(trainer, "lambda_uncertainty_prior", 0.0)
        ),
        uncertainty_detach_contribution=bool(
            getattr(trainer, "uncertainty_detach_contribution", True)
        ),
        uncertainty_variance_scale=float(
            getattr(trainer, "uncertainty_variance_scale", 1.0)
        ),
        detector_uncertainty_var=float(
            getattr(trainer, "detector_uncertainty_var", 0.0)
        ),
        absgrad=bool(getattr(getattr(cfg, "strategy", None), "absgrad", False)),
        backward_mode=backward_mode,
        img_backward_mode=img_backward_mode,
        vol_backward_mode=vol_backward_mode,
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
        "raw_uncertainty": float(getattr(cfg.optim, "lr_uncertainty", cfg.optim.lr_opacity)),
    }
    if params.get("beta") is not None:
        lr_map["beta"] = float(getattr(cfg.optim, "lr_beta", cfg.optim.lr_opacity))
    config = FirstOrderConfig(
        lr_map=lr_map,
        beta1=float(cfg.optim.beta1),
        beta2=float(cfg.optim.beta2),
        eps=float(cfg.optim.eps),
    )
    return GS3DCTAdamOptimizer(params, config)


def _wrap_gp_parameters(gp: GaussianParams) -> None:
    """Make Gaussian tensors leaf ``nn.Parameter`` objects for torch optimizers."""
    for name in ("means", "quats", "scale_ori", "opacity", "raw_uncertainty", "beta"):
        tensor = getattr(gp, name, None)
        if tensor is None:
            continue
        if isinstance(tensor, nn.Parameter):
            if not tensor.requires_grad:
                tensor.requires_grad_(True)
            continue
        setattr(gp, name, nn.Parameter(tensor.detach().clone().contiguous()))


def _torch_param_groups(cfg: DictConfig, gp: GaussianParams) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = [
        {"params": [gp.means], "lr": float(cfg.optim.lr_means), "name": "means"},
        {"params": [gp.quats], "lr": float(cfg.optim.lr_quats), "name": "quats"},
        {"params": [gp.scale_ori], "lr": float(cfg.optim.lr_scale_ori), "name": "scale_ori"},
        {"params": [gp.opacity], "lr": float(cfg.optim.lr_opacity), "name": "opacity"},
        {
            "params": [gp.raw_uncertainty],
            "lr": float(getattr(cfg.optim, "lr_uncertainty", cfg.optim.lr_opacity)),
            "name": "raw_uncertainty",
        },
    ]
    if gp.beta is not None:
        groups.append(
            {
                "params": [gp.beta],
                "lr": float(getattr(cfg.optim, "lr_beta", cfg.optim.lr_opacity)),
                "name": "beta",
            }
        )
    return groups


def build_torch_optimizer(cfg: DictConfig, gp: GaussianParams) -> torch.optim.Adam:
    _wrap_gp_parameters(gp)
    return torch.optim.Adam(
        _torch_param_groups(cfg, gp),
        lr=0.0,
        betas=(float(cfg.optim.beta1), float(cfg.optim.beta2)),
        eps=float(cfg.optim.eps),
    )


def _set_torch_optimizer_lrs(
    optimizer: torch.optim.Optimizer,
    base_lrs: Dict[str, float],
    cfg: DictConfig,
    step: int,
    max_steps: int,
) -> None:
    sched = getattr(cfg, "schedule", None)
    if sched is None or not bool(getattr(sched, "enabled", False)):
        factor = 1.0
    else:
        lr_cfg = getattr(sched, "lr", {})
        factor = _lr_schedule_factor(
            schedule_type=str(getattr(lr_cfg, "type", getattr(sched, "lr_type", "constant"))),
            step=step,
            max_steps=max_steps,
            warmup_steps=int(getattr(lr_cfg, "warmup_steps", getattr(sched, "lr_warmup_steps", 0))),
            min_factor=float(getattr(lr_cfg, "min_factor", getattr(sched, "lr_min_factor", 1.0))),
        )
    for group in optimizer.param_groups:
        name = str(group.get("name", ""))
        if name in base_lrs:
            group["lr"] = float(base_lrs[name]) * float(factor)


def _transition_state_tensor(
    state: Optional[torch.Tensor],
    *,
    duplicate_sel: Optional[torch.Tensor] = None,
    split_keep_parent: Optional[torch.Tensor] = None,
    split_children: int = 0,
    prune_keep: Optional[torch.Tensor] = None,
    append_new: int = 0,
) -> Optional[torch.Tensor]:
    if state is None:
        return None

    def append_zeros(x: torch.Tensor, n_new: int) -> torch.Tensor:
        if n_new <= 0:
            return x
        pad = torch.zeros((int(n_new),) + tuple(x.shape[1:]), device=x.device, dtype=x.dtype)
        return torch.cat([x, pad], dim=0).contiguous()

    out = state
    if duplicate_sel is not None and duplicate_sel.numel() > 0:
        out = append_zeros(out, int(duplicate_sel.numel()))
    if split_keep_parent is not None:
        keep = split_keep_parent.to(device=out.device, dtype=torch.bool)
        if keep.numel() != out.shape[0]:
            return None
        out = out[keep].contiguous()
        out = append_zeros(out, int(split_children))
    if prune_keep is not None:
        keep = prune_keep.to(device=out.device, dtype=torch.bool)
        if keep.numel() != out.shape[0]:
            return None
        out = out[keep].contiguous()
    if append_new > 0:
        out = append_zeros(out, int(append_new))
    return out


def _rewrap_gp_after_topology(
    gp: GaussianParams,
    optimizer: torch.optim.Optimizer,
    transition,
) -> None:
    old_by_name = {
        str(group.get("name")): group["params"][0]
        for group in optimizer.param_groups
        if group.get("params")
    }
    old_state = {
        name: optimizer.state.get(param, {}).copy()
        for name, param in old_by_name.items()
    }
    _wrap_gp_parameters(gp)
    new_by_name = {
        "means": gp.means,
        "quats": gp.quats,
        "scale_ori": gp.scale_ori,
        "opacity": gp.opacity,
        "raw_uncertainty": gp.raw_uncertainty,
    }
    if gp.beta is not None:
        new_by_name["beta"] = gp.beta

    optimizer.state.clear()
    new_groups = []
    for group in optimizer.param_groups:
        name = str(group.get("name"))
        if name not in new_by_name:
            continue
        group["params"] = [new_by_name[name]]
        new_groups.append(group)
        state_in = old_state.get(name, {})
        migrated: Dict[str, Any] = {}
        for key, value in state_in.items():
            if torch.is_tensor(value) and value.dim() > 0 and value.shape[:1] == old_by_name[name].shape[:1]:
                value = _transition_state_tensor(
                    value,
                    duplicate_sel=transition.duplicate_sel,
                    split_keep_parent=transition.split_keep_parent,
                    split_children=transition.split_children,
                    prune_keep=transition.prune_keep,
                    append_new=getattr(transition, "append_new", 0),
                )
                if value is None:
                    continue
            migrated[key] = value
        if migrated:
            optimizer.state[new_by_name[name]] = migrated
    optimizer.param_groups[:] = new_groups


def _torch_tv_loss(volume: torch.Tensor) -> torch.Tensor:
    edge_count = (
        (volume.shape[0] - 1) * volume.shape[1] * volume.shape[2]
        + volume.shape[0] * (volume.shape[1] - 1) * volume.shape[2]
        + volume.shape[0] * volume.shape[1] * (volume.shape[2] - 1)
    )
    return (
        torch.diff(volume, dim=0).abs().sum()
        + torch.diff(volume, dim=1).abs().sum()
        + torch.diff(volume, dim=2).abs().sum()
    ) / max(int(edge_count), 1)


def _projection_laplace_nll(
    prediction: torch.Tensor,
    target: torch.Tensor,
    variance: torch.Tensor,
    *,
    var_floor: float,
    var_max: float,
    eps: float = 1.0e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    var = variance + float(var_floor)
    if float(var_max) > 0.0:
        var = var.clamp(max=float(var_max))
    var = var.clamp_min(eps)
    scale = torch.sqrt(var)
    residual = target - prediction
    nll_map = math.sqrt(2.0) * residual.abs() / scale + 0.5 * torch.log(var)
    weights = math.sqrt(2.0) / scale
    return nll_map.mean(), weights, var


def _variance_scale_tensor(
    *,
    base_scale: float,
    log_std_scale: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    detach_scale: bool,
) -> torch.Tensor:
    scale = torch.as_tensor(float(base_scale), device=device, dtype=dtype)
    if log_std_scale is not None:
        dyn = torch.exp(2.0 * log_std_scale.to(device=device, dtype=dtype))
        if detach_scale:
            dyn = dyn.detach()
        scale = scale * dyn
    return scale


def _variance_scale_float(cfg: DictConfig, log_std_scale: Optional[torch.Tensor]) -> float:
    scale = float(getattr(cfg.trainer, "uncertainty_variance_scale", 1.0))
    if log_std_scale is not None:
        tau = float(torch.exp(log_std_scale.detach()).item())
        scale *= tau * tau
    return scale


def _select_evenly_spaced_view_indices(
    angles_rad: torch.Tensor,
    count: int,
) -> torch.Tensor:
    num_views = int(angles_rad.numel())
    count = min(max(int(count), 0), max(num_views - 1, 0))
    if count <= 0:
        return torch.empty(0, device=angles_rad.device, dtype=torch.long)

    order = torch.argsort(torch.remainder(angles_rad, 2.0 * math.pi))
    pos = (
        (torch.arange(count, device=angles_rad.device, dtype=torch.float32) + 0.5)
        * float(num_views)
        / float(count)
    )
    pos = pos.floor().to(torch.long).clamp(max=num_views - 1)
    return order.index_select(0, pos).unique(sorted=True)


def _build_calibration_optimizer(
    cfg: DictConfig,
    gp: GaussianParams,
    mode: str,
    log_std_scale: Optional[nn.Parameter],
) -> Optional[torch.optim.Optimizer]:
    if mode == "uncertainty":
        return torch.optim.Adam(
            [gp.raw_uncertainty],
            lr=float(getattr(cfg.trainer, "calibration_lr_uncertainty", 1.0e-4)),
            betas=(float(cfg.optim.beta1), float(cfg.optim.beta2)),
            eps=float(cfg.optim.eps),
        )
    if mode == "global_scale":
        if log_std_scale is None:
            raise ValueError("global_scale calibration requires log_std_scale")
        return torch.optim.Adam(
            [log_std_scale],
            lr=float(getattr(cfg.trainer, "calibration_lr_scale", 1.0e-2)),
            betas=(float(cfg.optim.beta1), float(cfg.optim.beta2)),
            eps=float(cfg.optim.eps),
        )
    return None


def _run_projection_calibration_step(
    *,
    cfg: DictConfig,
    gp: GaussianParams,
    raw_batch: Dict[str, torch.Tensor],
    op_config: Config,
    scale_min_world: float,
    scale_max_world: float,
    calibration_mode: str,
    calibration_optimizer: torch.optim.Optimizer,
    calibration_log_std_scale: Optional[nn.Parameter],
    device: torch.device,
) -> Dict[str, float]:
    calibration_optimizer.zero_grad(set_to_none=True)

    img_tensor = raw_batch["image"]
    img_h, img_w = int(img_tensor.shape[1]), int(img_tensor.shape[2])
    data_obj = make_train_data(
        raw_batch,
        vol_ranges=torch.ones(3, device=device),
        vol_centers=torch.zeros(3, device=device),
        device=device,
    )

    with torch.set_grad_enabled(calibration_mode == "uncertainty"):
        _quats, _scales, opacity, covars, _max_scales = activate_gaussian_params(
            gp.quats,
            gp.scale_ori,
            gp.opacity,
            scale_min=scale_min_world,
            scale_max=scale_max_world,
        )
        q = activate_uncertainty(
            gp.raw_uncertainty,
            q_min=float(op_config.get("uncertainty_q_min", 1.0e-6)),
            q_max=op_config.get("uncertainty_q_max", None),
        )

        if calibration_mode == "global_scale":
            q = q.detach()
        primitive_type = str(getattr(cfg.model, "primitive_type", "gaussian")).lower()
        rasterizer = GaussianProjectionRasterizer(
            img_w=img_w,
            img_h=img_h,
            tile_size=int(op_config.get("img_tile_size", 16)),
            eps_blur=float(op_config.get("eps_blur", 0.3)),
            radius_clip=float(op_config.get("img_radius_clip", 3.0)),
            primitive_type=primitive_type,
            absgrad=False,
            backward_mode=str(op_config.get("img_backward_mode", op_config.get("backward_mode", "primitive"))),
            uncertainty_detach_contribution=True,
        )
        fwd = rasterizer(
            gp.means.detach(),
            covars.detach(),
            opacity.detach(),
            data_obj.get("viewmats"),
            data_obj.get("Ks"),
            beta=gp.beta.detach() if gp.beta is not None else None,
            beta_gamma_init=float(op_config.get("beta_gamma_init", 3.5)),
            uncertainty=q,
        )

    if fwd.variance is None:
        raise RuntimeError("Calibration requires uncertainty variance")

    variance_scale = _variance_scale_tensor(
        base_scale=float(op_config.get("uncertainty_variance_scale", 1.0)),
        log_std_scale=calibration_log_std_scale,
        device=device,
        dtype=fwd.variance.dtype,
        detach_scale=(calibration_mode != "global_scale"),
    )
    projection_variance = fwd.variance.detach() if calibration_mode == "global_scale" else fwd.variance
    projection_variance = projection_variance * variance_scale
    detector_var = float(op_config.get("detector_uncertainty_var", 0.0))
    if detector_var != 0.0:
        projection_variance = projection_variance + detector_var
    nll_loss, _nll_weights, _var_used = _projection_laplace_nll(
        fwd.image.detach(),
        data_obj.get("targets"),
        projection_variance,
        var_floor=float(op_config.get("uncertainty_var_floor", 1.0e-6)),
        var_max=float(op_config.get("uncertainty_var_max", 0.0)),
    )
    nll_loss.backward()
    calibration_optimizer.step()

    if calibration_log_std_scale is not None:
        min_scale = float(getattr(cfg.trainer, "calibration_scale_min", 0.05))
        max_scale = float(getattr(cfg.trainer, "calibration_scale_max", 20.0))
        with torch.no_grad():
            calibration_log_std_scale.clamp_(
                min=math.log(max(min_scale, 1.0e-12)),
                max=math.log(max(max_scale, min_scale + 1.0e-12)),
            )

    calibration_optimizer.zero_grad(set_to_none=True)
    if gp.raw_uncertainty.grad is not None:
        gp.raw_uncertainty.grad = None

    out = {
        "calibration/loss_nll": float(nll_loss.detach().item()),
        "calibration/num_views": float(raw_batch["image_id"].numel()),
    }
    if calibration_log_std_scale is not None:
        tau = float(torch.exp(calibration_log_std_scale.detach()).item())
        out["calibration/std_scale"] = tau
        out["calibration/variance_scale"] = tau * tau * float(
            op_config.get("uncertainty_variance_scale", 1.0)
        )
    return out


def _uncertainty_log_prior(
    q: torch.Tensor,
    *,
    q0: float,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    target = math.log(max(float(q0), eps))
    return (torch.log(q.clamp_min(eps)) - target).square().mean()


def train(cfg: DictConfig) -> None:
    train_start_time = time.time()
    _maybe_init_wandb(cfg)

    device = torch.device(cfg.trainer.device if torch.cuda.is_available() else "cpu")
    set_cuda_device(device)
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
    scale_min_world, scale_max_world, volume_to_world = resolve_scale_bounds(
        cfg.model.scale_min_relative,
        cfg.model.scale_max_relative,
        parser.get_scanner_params(),
    )
    print(
        "   Scale bounds: "
        f"relative=[{float(cfg.model.scale_min_relative):.6g}, "
        f"{float(cfg.model.scale_max_relative):.6g}] "
        f"* max(sVoxel)={volume_to_world:.6g} -> "
        f"world=[{scale_min_world:.6g}, {scale_max_world:.6g}]"
    )
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
        set_cuda_device(device)
        gp = GaussianParams.from_points(
            init_result.sampled_points_n4,
            scale_min=scale_min_world,
            scale_max=scale_max_world,
            global_scale=cfg.model.global_scale,
            init_opa_scale=cfg.model.init_opa_scale,
            init_scale_multiplier=cfg.model.init_scale_multiplier,
            bbox=bbox,
            primitive_type=str(getattr(cfg.model, "primitive_type", "gaussian")),
            beta_gamma_init=float(getattr(cfg.model, "beta_gamma_init", 3.5)),
            init_uncertainty=float(getattr(cfg.model, "init_uncertainty", 0.0025)),
            uncertainty_q_min=float(getattr(cfg.model, "uncertainty_q_min", 1.0e-6)),
            device=str(device),
        )
    elif init_source == "file":
        gp = GaussianParams.from_data_dir(
            cfg.exp.data_dir,
            scale_min=scale_min_world,
            scale_max=scale_max_world,
            global_scale=cfg.model.global_scale,
            init_opa_scale=cfg.model.init_opa_scale,
            init_scale_multiplier=cfg.model.init_scale_multiplier,
            bbox=bbox,
            primitive_type=str(getattr(cfg.model, "primitive_type", "gaussian")),
            beta_gamma_init=float(getattr(cfg.model, "beta_gamma_init", 3.5)),
            init_uncertainty=float(getattr(cfg.model, "init_uncertainty", 0.0025)),
            uncertainty_q_min=float(getattr(cfg.model, "uncertainty_q_min", 1.0e-6)),
            device=str(device),
        )
    else:
        raise ValueError("init.source must be one of [online, file]")

    params = gp.to_params()
    primitive_name = "Beta primitives" if gp.primitive_type == "beta" else "Gaussians"
    print(f"   {gp.num_gaussians} {primitive_name} initialized")

    print("[3/5] Preparing autograd splatting modules...")
    _wrap_gp_parameters(gp)
    params = gp.to_params()

    print("[4/5] Creating torch Adam optimizer...")
    optimizer = build_torch_optimizer(cfg, gp)
    base_lrs = {str(group["name"]): float(group["lr"]) for group in optimizer.param_groups}

    calibration_log_std_scale: Optional[nn.Parameter] = None
    if str(getattr(cfg.trainer, "calibration_mode", "none")).lower() == "global_scale":
        calibration_log_std_scale = nn.Parameter(
            torch.tensor(
                math.log(max(float(getattr(cfg.trainer, "calibration_scale_init", 1.0)), 1.0e-12)),
                device=device,
                dtype=gp.means.dtype,
            )
        )

    from gsplat2ct.strategy import build_strategy

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
    calibration_mode = str(getattr(cfg.trainer, "calibration_mode", "none")).lower()
    if calibration_mode not in {"none", "uncertainty", "global_scale"}:
        raise ValueError(
            "trainer.calibration_mode must be one of "
            "['none', 'uncertainty', 'global_scale']"
        )
    calibration_enabled = (
        calibration_mode != "none"
        and bool(getattr(cfg.trainer, "uncertainty_enabled", False))
        and int(getattr(cfg.trainer, "calibration_num_views", 0)) > 0
        and int(getattr(cfg.trainer, "calibration_interval", 0)) > 0
    )
    all_train_indices = torch.arange(num_views, device=device, dtype=torch.long)
    calibration_indices = torch.empty(0, device=device, dtype=torch.long)
    if calibration_enabled:
        calibration_indices = _select_evenly_spaced_view_indices(
            angle_values_rad,
            int(getattr(cfg.trainer, "calibration_num_views", 0)),
        )
    train_mask = torch.ones(num_views, device=device, dtype=torch.bool)
    if calibration_indices.numel() > 0:
        train_mask[calibration_indices] = False
    train_candidate_indices = all_train_indices[train_mask]
    if train_candidate_indices.numel() == 0:
        raise RuntimeError("Calibration split left no reconstruction training views")
    train_angle_values_rad = angle_values_rad.index_select(0, train_candidate_indices)
    print(
        "   View sampling: "
        f"strategy={angle_strategy}, batch_size={batch_size}, angle_unit={angle_unit}, "
        f"train_views={int(train_candidate_indices.numel())}, "
        f"calibration_views={int(calibration_indices.numel())}"
    )
    if calibration_indices.numel() > 0:
        print(f"   Calibration indices: {calibration_indices.detach().cpu().tolist()}")

    shuffle_perm = torch.randperm(int(train_candidate_indices.numel()), device=device)
    perm_cursor = 0
    calibration_optimizer = (
        _build_calibration_optimizer(
            cfg, gp, calibration_mode, calibration_log_std_scale
        )
        if calibration_enabled
        else None
    )
    progress = tqdm(range(max_steps), desc="Training")

    if bool(getattr(cfg.trainer, "eval_at_start", True)):
        _run_eval(
            params,
            cfg,
            evalset,
            vol_gt,
            bbox,
            volume_shape_wdh,
            device,
            vis_dir,
            step=0,
            scale_min_world=scale_min_world,
            scale_max_world=scale_max_world,
            uncertainty_variance_scale=_variance_scale_float(
                cfg, calibration_log_std_scale
            ),
        )

    train_loop_start_time = time.time()
    last_train_metrics: Dict[str, float] = {}
    loss_log_interval = int(getattr(cfg.trainer, "loss_log_interval", 50))
    loss_log_path = os.path.join(result_dir, "train_loss_history.jsonl")
    if loss_log_interval > 0:
        with open(loss_log_path, "w", encoding="utf-8"):
            pass
    for step in progress:
        _set_torch_optimizer_lrs(optimizer, base_lrs, cfg, step, max_steps)

        if angle_strategy == "random" or batch_size <= 1:
            batch_positions, shuffle_perm, perm_cursor = _sample_random_batch_indices(
                num_views=int(train_candidate_indices.numel()),
                batch_size=batch_size,
                shuffle_perm=shuffle_perm,
                perm_cursor=perm_cursor,
                device=device,
            )
            batch_indices = train_candidate_indices.index_select(0, batch_positions)
        else:
            batch_positions = _sample_evenly_spaced_angle_batch_indices(
                train_angle_values_rad, batch_size
            )
            batch_indices = train_candidate_indices.index_select(0, batch_positions)

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
        op_config = make_train_config(
            cfg,
            (img_h, img_w),
            vol_dims,
            lambda_tv=lambda_tv,
            scale_min_world=scale_min_world,
            scale_max_world=scale_max_world,
        )

        optimizer.zero_grad(set_to_none=True)
        _quats, _scales, opacity, covars, max_scales = activate_gaussian_params(
            gp.quats,
            gp.scale_ori,
            gp.opacity,
            scale_min=scale_min_world,
            scale_max=scale_max_world,
        )
        uncertainty_enabled = bool(op_config.get("uncertainty_enabled", False))
        q = None
        if uncertainty_enabled:
            q = activate_uncertainty(
                gp.raw_uncertainty,
                q_min=float(op_config.get("uncertainty_q_min", 1.0e-6)),
                q_max=op_config.get("uncertainty_q_max", None),
            )
        primitive_type = str(getattr(cfg.model, "primitive_type", "gaussian")).lower()
        img_backward_mode = str(
            op_config.get("img_backward_mode", op_config.get("backward_mode", "pixel"))
        )
        vol_backward_mode = str(
            op_config.get("vol_backward_mode", op_config.get("backward_mode", "pixel"))
        )
        rasterizer = GaussianProjectionRasterizer(
            img_w=img_w,
            img_h=img_h,
            tile_size=int(op_config.get("img_tile_size", 16)),
            eps_blur=float(op_config.get("eps_blur", 0.3)),
            radius_clip=float(op_config.get("img_radius_clip", 3.0)),
            primitive_type=primitive_type,
            absgrad=bool(op_config.get("absgrad", False)),
            backward_mode=img_backward_mode,
            uncertainty_detach_contribution=bool(
                op_config.get("uncertainty_detach_contribution", True)
            ),
        )
        render_pkg = rasterizer(
            gp.means,
            covars,
            opacity,
            data_obj.get("viewmats"),
            data_obj.get("Ks"),
            beta=gp.beta,
            beta_gamma_init=float(op_config.get("beta_gamma_init", 3.5)),
            uncertainty=q,
        )
        if render_pkg.means2d.requires_grad:
            render_pkg.means2d.retain_grad()

        l1_loss = (render_pkg.image - data_obj.get("targets")).abs().mean()
        lambda_l1_loss = float(op_config.get("lambda_l1", 1.0))
        lambda_dssim_loss = float(op_config.get("lambda_dssim", 0.0))
        if lambda_dssim_loss > 0.0:
            dssim_loss = 1.0 - ssim2d(
                render_pkg.image.unsqueeze(1),
                data_obj.get("targets").unsqueeze(1),
                window_size=int(op_config.get("ssim_window", 11)),
            )
        else:
            dssim_loss = torch.zeros((), device=device, dtype=render_pkg.image.dtype)
        nll_loss = torch.zeros((), device=device, dtype=render_pkg.image.dtype)
        q_prior_loss = torch.zeros((), device=device, dtype=render_pkg.image.dtype)
        lambda_nll_eff = 0.0
        if uncertainty_enabled:
            if render_pkg.variance is None:
                raise RuntimeError("uncertainty_enabled=True but rasterizer did not return variance")
            detector_var = float(op_config.get("detector_uncertainty_var", 0.0))
            projection_variance = render_pkg.variance * _variance_scale_tensor(
                base_scale=float(op_config.get("uncertainty_variance_scale", 1.0)),
                log_std_scale=calibration_log_std_scale,
                device=device,
                dtype=render_pkg.variance.dtype,
                detach_scale=True,
            )
            if detector_var != 0.0:
                projection_variance = projection_variance + detector_var
            nll_loss, nll_weights, _var_used = _projection_laplace_nll(
                render_pkg.image,
                data_obj.get("targets"),
                projection_variance,
                var_floor=float(op_config.get("uncertainty_var_floor", 1.0e-6)),
                var_max=float(op_config.get("uncertainty_var_max", 0.0)),
            )
            lambda_nll = float(op_config.get("lambda_nll", 0.0))
            if bool(op_config.get("uncertainty_nll_auto_weight", True)) and lambda_nll != 0.0:
                denom = nll_weights.detach().mean().clamp(
                    min=float(op_config.get("uncertainty_nll_auto_min", 20.0)),
                    max=float(op_config.get("uncertainty_nll_auto_max", 1000.0)),
                )
                lambda_nll_eff = float(lambda_nll / denom.item())
            else:
                lambda_nll_eff = lambda_nll
            if q is None:
                raise RuntimeError("Internal error: q is None while uncertainty is enabled")
            if q is not None:
                q_prior_loss = _uncertainty_log_prior(
                    q,
                    q0=float(op_config.get("uncertainty_init", 0.0025)),
                )

        img_loss = (
            lambda_l1_loss * l1_loss
            + lambda_dssim_loss * dssim_loss
            + float(lambda_nll_eff) * nll_loss
            + float(op_config.get("lambda_uncertainty_prior", 0.0)) * q_prior_loss
        )

        if lambda_tv > 0.0:
            voxelizer = GaussianMappingVoxelizer(
                vol_dim2=int(op_config.get("vol_dim2")),
                vol_dim1=int(op_config.get("vol_dim1")),
                vol_dim0=int(op_config.get("vol_dim0")),
                tile_size=int(op_config.get("vol_tile_size", 4)),
                eps3d=float(op_config.get("eps3d", 1.0e-4)),
                radius_clip=float(op_config.get("radius_clip", 3.0)),
                primitive_type=primitive_type,
                backward_mode=vol_backward_mode,
                uncertainty_detach_contribution=bool(
                    op_config.get("uncertainty_detach_contribution", True)
                ),
            )
            volume_pkg = voxelizer(
                gp.means,
                covars,
                max_scales,
                opacity,
                data_obj.get("vol_ranges"),
                data_obj.get("vol_centers"),
                beta=gp.beta,
                beta_gamma_init=float(op_config.get("beta_gamma_init", 3.5)),
            )
            tv_loss = _torch_tv_loss(volume_pkg.volume)
            vol_loss_total = float(lambda_tv) * tv_loss
            render_volume = volume_pkg.volume
        else:
            tv_loss = torch.zeros((), device=device, dtype=render_pkg.image.dtype)
            vol_loss_total = tv_loss
            render_volume = torch.empty(0, device=device, dtype=render_pkg.image.dtype)
        loss = img_loss + vol_loss_total
        loss.backward()
        v_means2d_abs_aux = pop_last_raster_absgrad()

        loss_total = float(loss.detach().item())
        loss_img = float(img_loss.detach().item())
        loss_l1 = float(l1_loss.detach().item())
        loss_dssim = float(dssim_loss.detach().item())
        loss_tv = float(vol_loss_total.detach().item())
        last_train_metrics = {
            "train/loss_total": loss_total,
            "train/loss_image": loss_img,
            "train/loss_l1": loss_l1,
            "train/loss_dssim": loss_dssim,
            "train/loss_nll": float(nll_loss.detach().item()),
            "train/loss_uncertainty_prior": float(q_prior_loss.detach().item()),
            "train/lambda_nll_eff": float(lambda_nll_eff),
            "train/loss_tv": loss_tv,
            "train/num_gaussians": float(gp.num_gaussians),
        }
        if q is not None:
            q_det = q.detach()
            last_train_metrics.update(
                {
                    "train/q_min": float(q_det.min().item()),
                    "train/q_median": float(q_det.median().item()),
                    "train/q_mean": float(q_det.mean().item()),
                    "train/q_max": float(q_det.max().item()),
                }
            )
        if uncertainty_enabled:
            last_train_metrics["train/detector_uncertainty_var"] = float(
                op_config.get("detector_uncertainty_var", 0.0)
            )
            variance_scale_log = float(op_config.get("uncertainty_variance_scale", 1.0))
            if calibration_log_std_scale is not None:
                tau_log = float(torch.exp(calibration_log_std_scale.detach()).item())
                last_train_metrics["train/uncertainty_std_scale"] = tau_log
                variance_scale_log *= tau_log * tau_log
            last_train_metrics["train/uncertainty_variance_scale"] = variance_scale_log

        v_means2d = render_pkg.means2d.grad
        v_means2d_abs = v_means2d_abs_aux
        radii_2d = render_pkg.radii
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
            grow_scale3d_abs = float(strategy.grow_scale3d) * float(volume_to_world)
            decision = strategy.decide(
                gp,
                scale_min=scale_min_world,
                scale_max=scale_max_world,
                grow_scale3d_abs=grow_scale3d_abs,
                uncertainty=q.detach() if q is not None else None,
            )
            changed, transition = strategy.apply(
                gp,
                decision,
                scale_min=scale_min_world,
                scale_max=scale_max_world,
                bbox=bbox,
                step=step,
            )
            if changed:
                skip_optimizer_update = True
                strategy.reset_grad_accumulator()
                _rewrap_gp_after_topology(gp, optimizer, transition)
                if calibration_enabled and calibration_mode == "uncertainty":
                    calibration_optimizer = _build_calibration_optimizer(
                        cfg, gp, calibration_mode, calibration_log_std_scale
                    )
                params = gp.to_params()
                optimizer.zero_grad(set_to_none=True)

        if not skip_optimizer_update:
            optimizer.step()
            with torch.no_grad():
                gp.quats.div_(gp.quats.norm(dim=-1, keepdim=True).clamp_min(1e-12))
        optimizer.zero_grad(set_to_none=True)

        calibration_interval = int(getattr(cfg.trainer, "calibration_interval", 0))
        calibration_start_iter = int(getattr(cfg.trainer, "calibration_start_iter", 0))
        if (
            calibration_enabled
            and calibration_optimizer is not None
            and calibration_interval > 0
            and step >= calibration_start_iter
            and step % calibration_interval == 0
        ):
            calibration_batch_size = int(getattr(cfg.trainer, "calibration_batch_size", 0))
            if calibration_batch_size > 0 and calibration_batch_size < int(calibration_indices.numel()):
                perm = torch.randperm(int(calibration_indices.numel()), device=device)
                cal_indices = calibration_indices.index_select(0, perm[:calibration_batch_size])
            else:
                cal_indices = calibration_indices
            calibration_metrics = _run_projection_calibration_step(
                cfg=cfg,
                gp=gp,
                raw_batch=trainset.get_batch(cal_indices),
                op_config=op_config,
                scale_min_world=scale_min_world,
                scale_max_world=scale_max_world,
                calibration_mode=calibration_mode,
                calibration_optimizer=calibration_optimizer,
                calibration_log_std_scale=calibration_log_std_scale,
                device=device,
            )
            last_train_metrics.update(calibration_metrics)
            if q is not None and calibration_mode == "uncertainty":
                q_cal = activate_uncertainty(
                    gp.raw_uncertainty,
                    q_min=float(op_config.get("uncertainty_q_min", 1.0e-6)),
                    q_max=op_config.get("uncertainty_q_max", None),
                ).detach()
                last_train_metrics.update(
                    {
                        "train/q_min": float(q_cal.min().item()),
                        "train/q_median": float(q_cal.median().item()),
                        "train/q_mean": float(q_cal.mean().item()),
                        "train/q_max": float(q_cal.max().item()),
                    }
                )
            if calibration_log_std_scale is not None:
                tau = float(torch.exp(calibration_log_std_scale.detach()).item())
                last_train_metrics["train/uncertainty_std_scale"] = tau
                last_train_metrics["train/uncertainty_variance_scale"] = (
                    tau * tau * float(op_config.get("uncertainty_variance_scale", 1.0))
                )

        last_train_metrics["train/num_gaussians"] = float(gp.num_gaussians)
        last_train_metrics["train/skipped_optimizer_update"] = float(skip_optimizer_update)

        if loss_log_interval > 0 and (
            step % loss_log_interval == 0 or step == max_steps - 1
        ):
            loss_record = {"step": step, **last_train_metrics}
            with open(loss_log_path, "a", encoding="utf-8") as fout:
                fout.write(json.dumps(loss_record) + "\n")

        progress.set_postfix(
            total=f"{loss_total:.4f}",
            img=f"{loss_img:.4f}",
            l1=f"{loss_l1:.4f}",
            nll=f"{float(nll_loss.detach().item()):.4f}",
            dssim=f"{loss_dssim:.4f}",
            tv=f"{loss_tv:.4f}",
            N=gp.num_gaussians,
            skip=int(skip_optimizer_update),
        )

        try:
            import wandb  # type: ignore

            if wandb.run is not None:
                wandb.log(last_train_metrics, step=step)
        except ImportError:
            pass

        eval_interval = int(getattr(cfg.trainer, "eval_interval", 0))
        if eval_interval > 0 and step > 0 and step % eval_interval == 0:
            _run_eval(
                params,
                cfg,
                evalset,
                vol_gt,
                bbox,
                volume_shape_wdh,
                device,
                vis_dir,
                step,
                scale_min_world=scale_min_world,
                scale_max_world=scale_max_world,
                uncertainty_variance_scale=_variance_scale_float(
                    cfg, calibration_log_std_scale
                ),
            )

        if step in cfg.trainer.save_steps:
            gp.update_from_params(params)
            gp.save(os.path.join(ckpt_dir, f"ckpt_{step}.pt"), step=step)
            print(f"\n   Checkpoint saved at step {step}")

    train_loop_elapsed_sec = time.time() - train_loop_start_time
    final_step = max(0, max_steps - 1)
    gp.update_from_params(params)
    gp.save(os.path.join(ckpt_dir, "ckpt_final.pt"), step=final_step)
    train_to_checkpoint_elapsed_sec = time.time() - train_start_time

    final_metrics: Dict[str, float] = {}
    final_eval_elapsed_sec = 0.0
    if bool(getattr(cfg.trainer, "eval_at_end", False)):
        eval_start_time = time.time()
        final_metrics = _run_eval(
            params,
            cfg,
            evalset,
            vol_gt,
            bbox,
            volume_shape_wdh,
            device,
            vis_dir,
            step=final_step,
            scale_min_world=scale_min_world,
            scale_max_world=scale_max_world,
            uncertainty_variance_scale=_variance_scale_float(
                cfg, calibration_log_std_scale
            ),
        )
        final_eval_elapsed_sec = time.time() - eval_start_time

    total_elapsed_sec = time.time() - train_start_time
    metrics_report = {
        "result_dir": result_dir,
        "data_dir": cfg.exp.data_dir,
        "checkpoint": os.path.join(ckpt_dir, "ckpt_final.pt"),
        "max_steps": max_steps,
        "final_step": final_step,
        "num_primitives": gp.num_gaussians,
        "loss_log": loss_log_path if loss_log_interval > 0 else None,
        "uncertainty_variance_scale": _variance_scale_float(
            cfg, calibration_log_std_scale
        ),
        "uncertainty_std_scale": (
            float(torch.exp(calibration_log_std_scale.detach()).item())
            if calibration_log_std_scale is not None
            else math.sqrt(float(getattr(cfg.trainer, "uncertainty_variance_scale", 1.0)))
        ),
        "timing": {
            "train_to_checkpoint_sec": train_to_checkpoint_elapsed_sec,
            "setup_train_eval_sec": total_elapsed_sec,
            "train_loop_sec": train_loop_elapsed_sec,
            "final_eval_sec": final_eval_elapsed_sec,
        },
        "last_train_metrics": last_train_metrics,
        "final_metrics": final_metrics,
    }
    metrics_path = os.path.join(result_dir, "metrics_final.json")
    with open(metrics_path, "w", encoding="utf-8") as fout:
        json.dump(metrics_report, fout, indent=2)

    try:
        import wandb  # type: ignore

        if wandb.run is not None:
            wandb_payload = {
                "time/setup_train_eval_sec": total_elapsed_sec,
                "time/train_loop_sec": train_loop_elapsed_sec,
                "time/final_eval_sec": final_eval_elapsed_sec,
                "final/num_primitives": float(gp.num_gaussians),
            }
            wandb_payload.update({f"final/{k}": v for k, v in final_metrics.items()})
            wandb.log(wandb_payload, step=final_step)
            for key, value in wandb_payload.items():
                wandb.run.summary[key] = value
    except ImportError:
        pass

    print(f"\nFinal metrics saved to: {metrics_path}")
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
    *,
    scale_min_world: float,
    scale_max_world: float,
    uncertainty_variance_scale: Optional[float] = None,
) -> Dict[str, float]:
    log_dict: Dict[str, float] = {}
    if uncertainty_variance_scale is None:
        uncertainty_variance_scale = float(
            getattr(cfg.trainer, "uncertainty_variance_scale", 1.0)
        )

    proj_eval = ProjectionEvaluator(
        params,
        scale_min=scale_min_world,
        scale_max=scale_max_world,
        eps_blur=cfg.trainer.eps_blur,
        ssim_window=int(getattr(cfg.trainer, "ssim_window", 11)),
        primitive_type=str(getattr(cfg.model, "primitive_type", "gaussian")),
        beta_gamma_init=float(getattr(cfg.model, "beta_gamma_init", 3.5)),
        uncertainty_enabled=bool(getattr(cfg.trainer, "uncertainty_enabled", False)),
        uncertainty_q_min=float(getattr(cfg.model, "uncertainty_q_min", 1.0e-6)),
        uncertainty_q_max=(
            None
            if getattr(cfg.model, "uncertainty_q_max", None) is None
            else float(getattr(cfg.model, "uncertainty_q_max"))
        ),
        uncertainty_var_floor=float(
            getattr(cfg.trainer, "uncertainty_var_floor", 1.0e-6)
        ),
        uncertainty_variance_scale=float(uncertainty_variance_scale),
        detector_uncertainty_var=float(
            getattr(cfg.trainer, "detector_uncertainty_var", 0.0)
        ),
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
    if "uncertainty_error_corr" in proj_metrics:
        log_dict["img_uncertainty/error_corr"] = float(
            proj_metrics["uncertainty_error_corr"]
        )
    if "uncertainty_std_mean" in proj_metrics:
        log_dict["img_uncertainty/std_mean"] = float(
            proj_metrics["uncertainty_std_mean"]
        )

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
            scale_min=scale_min_world,
            scale_max=scale_max_world,
            eps3d=cfg.trainer.eps3d,
            ssim_window=int(getattr(cfg.trainer, "ssim_window", 11)),
            primitive_type=str(getattr(cfg.model, "primitive_type", "gaussian")),
            beta_gamma_init=float(getattr(cfg.model, "beta_gamma_init", 3.5)),
            uncertainty_enabled=bool(getattr(cfg.trainer, "uncertainty_enabled", False)),
            uncertainty_q_min=float(getattr(cfg.model, "uncertainty_q_min", 1.0e-6)),
            uncertainty_q_max=(
                None
                if getattr(cfg.model, "uncertainty_q_max", None) is None
                else float(getattr(cfg.model, "uncertainty_q_max"))
            ),
            uncertainty_var_floor=float(
                getattr(cfg.trainer, "uncertainty_var_floor", 1.0e-6)
            ),
            uncertainty_variance_scale=float(uncertainty_variance_scale),
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
        if "uncertainty_error_corr" in vol_metrics:
            log_dict["vol_uncertainty/error_corr"] = float(
                vol_metrics["uncertainty_error_corr"]
            )
        if "uncertainty_std_mean" in vol_metrics:
            log_dict["vol_uncertainty/std_mean"] = float(
                vol_metrics["uncertainty_std_mean"]
            )
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

    return log_dict


@hydra.main(version_base=None, config_path="../configs", config_name="gaussian")
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
