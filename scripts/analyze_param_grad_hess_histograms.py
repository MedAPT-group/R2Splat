"""Analyze parameter values, gradients, and second-order terms for GS3D-CT.

Loads three parameter groups:
1) init parameters from data_dir using GaussianParams.from_data_dir
2) checkpoint at step 999
3) checkpoint at step 29999

For each group, runs one forward_backward2 pass on the same sampled batch and
plots normalized histograms (y-axis = frequency, i.e., count / total elements).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from gsplat2ct.data.ctproj import CTImageDataset, CTImageParser
from gsplat2ct.model.gaussian_params import GaussianParams
from gsplat2ct.ops.data import Config, Data
from gsplat2ct.ops.gs3d_ct_train import GS3DCTLossCUDA


PARAM_ORDER = ["means", "scale_ori", "quats", "opacity"]
INTER_GROUP_CROSS_KEYS = [
    "quats_scale_ori",
    "quats_means",
    "quats_opacity",
    "scale_ori_means",
    "scale_ori_opacity",
    "means_opacity",
]


@dataclass
class AnalysisResult:
    values: Dict[str, torch.Tensor]
    grads: Dict[str, torch.Tensor]
    hess_self: Dict[str, torch.Tensor]
    hess_cross: Dict[str, torch.Tensor]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Load init/ckpt_999/ckpt_29999 parameters, run second-order backward, "
            "and save frequency histograms for values, gradients, and Hessian terms."
        )
    )
    p.add_argument("--data-dir", type=str, required=True, help="Sample data directory.")
    p.add_argument("--ckpt-dir", type=str, required=True, help="Checkpoint directory containing ckpt_*.pt.")
    p.add_argument(
        "--config-used",
        type=str,
        default=None,
        help="Path to config_used.yaml. Defaults to <ckpt_dir>/../config_used.yaml.",
    )
    p.add_argument(
        "--ckpt-999",
        type=str,
        default=None,
        help="Override checkpoint path for step 999. Defaults to <ckpt_dir>/ckpt_999.pt.",
    )
    p.add_argument(
        "--ckpt-29999",
        type=str,
        default=None,
        help="Override checkpoint path for step 29999. Defaults to <ckpt_dir>/ckpt_29999.pt.",
    )
    p.add_argument("--split", type=str, default="train", choices=["train", "test"])
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Number of projections per batch for forward_backward2 analysis. "
            "Useful for studying gradient/Hessian distributions at batch_size=3/4/5."
        ),
    )
    p.add_argument("--tv-vol-size", type=int, default=None, help="Override tv subvolume size.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bins", type=int, default=120)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help=(
            "Output directory. Defaults to "
            "<ckpt_dir>/../analysis_second_order_hist_bs<BATCH_SIZE>."
        ),
    )
    return p.parse_args()


def _resolve_ckpt_path(path_override: Optional[str], ckpt_dir: Path, step: int) -> Path:
    if path_override:
        p = Path(path_override).expanduser().resolve()
        if p.suffix == ".py":
            alt = p.with_suffix(".pt")
            if alt.is_file():
                print(f"[info] checkpoint suffix corrected: {p} -> {alt}")
                p = alt
        if not p.is_file():
            raise FileNotFoundError(f"checkpoint not found: {p}")
        return p

    candidate = ckpt_dir / f"ckpt_{step}.pt"
    if candidate.is_file():
        return candidate

    wrong_suffix = ckpt_dir / f"ckpt_{step}.py"
    if wrong_suffix.is_file():
        print(f"[warn] found .py checkpoint, using it as provided: {wrong_suffix}")
        return wrong_suffix

    raise FileNotFoundError(f"checkpoint not found: {candidate}")


def _sample_tv_subvolume(
    bbox: torch.Tensor,
    tv_vol_size: int,
    volume_shape_wdh: Tuple[int, int, int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int]]:
    bbox_dev = bbox.to(device)
    full_ranges = bbox_dev[1] - bbox_dev[0]
    vol_shape_tensor = torch.tensor(volume_shape_wdh, device=device, dtype=torch.float32)
    voxel_size = full_ranges / vol_shape_tensor

    sub_n = torch.tensor([tv_vol_size] * 3, device=device, dtype=torch.float32)
    sub_ranges = sub_n * voxel_size
    half = sub_ranges / 2.0
    valid_min = bbox_dev[0] + half
    valid_max = bbox_dev[1] - half
    center = valid_min + (valid_max - valid_min) * torch.rand(3, device=device)
    return sub_ranges, center, (tv_vol_size, tv_vol_size, tv_vol_size)


def _make_data(batch: Dict[str, torch.Tensor], vol_ranges: torch.Tensor, vol_centers: torch.Tensor, device: torch.device) -> Data:
    c2w = batch["camtoworld"].to(device).contiguous()
    viewmats = torch.linalg.inv(c2w).contiguous()
    Ks = batch["K"].to(device).contiguous()
    targets = batch["image"].to(device).squeeze(-1).contiguous()

    data_obj = Data()
    data_obj.set(
        viewmats=viewmats,
        Ks=Ks,
        targets=targets,
        vol_ranges=vol_ranges.contiguous(),
        vol_centers=vol_centers.contiguous(),
    )
    return data_obj


def _make_config(cfg, img_shape: Tuple[int, int], vol_dims: Tuple[int, int, int]) -> Config:
    img_h, img_w = img_shape
    vol_w, vol_d, vol_h = vol_dims

    tr = cfg.trainer
    if "lambda_l2" in tr:
        lambda_l2 = float(tr.lambda_l2)
    elif "lambda_l1" in tr:
        lambda_l2 = float(tr.lambda_l1)
    else:
        lambda_l2 = 1.0

    c = Config()
    c.set(
        img_dim1=img_w,
        img_dim0=img_h,
        img_tile_size=int(tr.img_tile_size),
        eps_blur=float(tr.eps_blur),
        lambda_l2=lambda_l2,
        lambda_ssim=float(tr.lambda_ssim),
        window_size=int(tr.window_size),
        vol_dim2=vol_w,
        vol_dim1=vol_d,
        vol_dim0=vol_h,
        vol_tile_size=int(tr.vol_tile_size),
        eps3d=float(tr.eps3d),
        radius_clip=float(tr.radius_clip),
        vol_loss_eps=float(tr.vol_loss_eps),
        lambda_tv=float(tr.lambda_tv),
        scale_min=float(cfg.model.scale_min),
        scale_max=float(cfg.model.scale_max),
        image_loss_type=str(getattr(tr, "image_loss_type", "l2")),
        absgrad=bool(getattr(getattr(cfg, "strategy", None), "absgrad", False)),
    )
    return c


def _flatten(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    return x.detach().float().reshape(-1).cpu()


def _extract_self_diag(name: str, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if name == "means":  # (N, 6): xx, xy, xz, yy, yz, zz
        idx = [0, 3, 5]
        return tensor[:, idx]
    if name == "quats":  # (N, 10): ww, wx, wy, wz, xx, xy, xz, yy, yz, zz
        idx = [0, 4, 7, 9]
        return tensor[:, idx]
    if name == "scale_ori":  # (N, 6): ss00, ss01, ss02, ss11, ss12, ss22
        idx = [0, 3, 5]
        return tensor[:, idx]
    return tensor


def _extract_self_cross(name: str, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if name == "means":
        idx = [1, 2, 4]  # xy, xz, yz
        return tensor[:, idx]
    if name == "quats":
        idx = [1, 2, 3, 5, 6, 8]  # non-diagonal terms
        return tensor[:, idx]
    if name == "scale_ori":
        idx = [1, 2, 4]  # ss01, ss02, ss12
        return tensor[:, idx]
    # opacity (N,) contains only a scalar self term.
    return None


def _hist_frequency(ax, arrays: Iterable[Tuple[str, Optional[torch.Tensor]]], bins: int, title: str) -> None:
    valid = [(label, _flatten(arr)) for label, arr in arrays]
    valid = [(label, arr) for label, arr in valid if arr is not None and arr.numel() > 0]
    if not valid:
        ax.set_title(f"{title} (N/A)")
        ax.set_ylabel("frequency")
        ax.grid(alpha=0.2)
        return

    lo = min(float(arr.min()) for _, arr in valid)
    hi = max(float(arr.max()) for _, arr in valid)
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = -1.0, 1.0
    if hi <= lo:
        delta = 1e-6 if hi == 0 else max(abs(hi) * 1e-3, 1e-6)
        lo, hi = lo - delta, hi + delta

    colors = {
        "init": "#1f77b4",
        "step999": "#ff7f0e",
        "step29999": "#2ca02c",
    }

    for label, arr in valid:
        n = arr.numel()
        weights = np.ones(n, dtype=np.float64) / float(n)
        ax.hist(
            arr.numpy(),
            bins=bins,
            range=(lo, hi),
            weights=weights,
            alpha=0.35,
            label=label,
            color=colors.get(label, None),
            edgecolor="none",
        )

    ax.set_title(title)
    ax.set_ylabel("frequency")
    ax.grid(alpha=0.2)


def _plot_values(results: Dict[str, AnalysisResult], bins: int, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Parameter values distribution (frequency)", fontsize=12)

    for ax, name in zip(axes.reshape(-1), PARAM_ORDER):
        _hist_frequency(
            ax,
            [(stage, res.values.get(name)) for stage, res in results.items()],
            bins=bins,
            title=name,
        )
        ax.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_grads(results: Dict[str, AnalysisResult], bins: int, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("First-order gradients distribution (frequency)", fontsize=12)

    for ax, name in zip(axes.reshape(-1), PARAM_ORDER):
        _hist_frequency(
            ax,
            [(stage, res.grads.get(name)) for stage, res in results.items()],
            bins=bins,
            title=f"grad {name}",
        )
        ax.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_hess(results: Dict[str, AnalysisResult], bins: int, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 4, hspace=0.45, wspace=0.3)
    fig.suptitle("Second-order terms distribution (frequency)", fontsize=12)

    # Row 0: self diagonal terms.
    for col, name in enumerate(PARAM_ORDER):
        ax = fig.add_subplot(gs[0, col])
        _hist_frequency(
            ax,
            [(stage, _extract_self_diag(name, res.hess_self.get(name))) for stage, res in results.items()],
            bins=bins,
            title=f"self diag {name}",
        )
        ax.legend()

    # Row 1: self cross terms.
    for col, name in enumerate(PARAM_ORDER):
        ax = fig.add_subplot(gs[1, col])
        _hist_frequency(
            ax,
            [(stage, _extract_self_cross(name, res.hess_self.get(name))) for stage, res in results.items()],
            bins=bins,
            title=f"self cross {name}",
        )
        ax.legend()

    # Rows 2-3: inter-group cross terms.
    for i, key in enumerate(INTER_GROUP_CROSS_KEYS):
        row = 2 + i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        _hist_frequency(
            ax,
            [(stage, res.hess_cross.get(key)) for stage, res in results.items()],
            bins=bins,
            title=f"inter-group {key}",
        )
        ax.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _stats(x: Optional[torch.Tensor]) -> Optional[Dict[str, float]]:
    xx = _flatten(x)
    if xx is None or xx.numel() == 0:
        return None
    q = torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99])
    qq = torch.quantile(xx, q)
    return {
        "numel": int(xx.numel()),
        "min": float(xx.min()),
        "max": float(xx.max()),
        "mean": float(xx.mean()),
        "std": float(xx.std(unbiased=False)),
        "q01": float(qq[0]),
        "q05": float(qq[1]),
        "q50": float(qq[2]),
        "q95": float(qq[3]),
        "q99": float(qq[4]),
    }


def _analyze_one(
    operator: GS3DCTLossCUDA,
    gp: GaussianParams,
    data_obj: Data,
    op_config: Config,
) -> AnalysisResult:
    params = gp.to_params()
    operator.initialize(params)
    _fwd, grad, hess = operator.forward_backward2(params, data_obj, op_config)

    values = {
        "means": params.get("means").data.detach(),
        "scale_ori": params.get("scale_ori").data.detach(),
        "quats": params.get("quats").data.detach(),
        "opacity": params.get("opacity").data.detach(),
    }

    return AnalysisResult(
        values=values,
        grads={k: v.detach() for k, v in grad.param_grads.items()},
        hess_self={k: v.detach() for k, v in hess.param_hess.items()},
        hess_cross={k: v.detach() for k, v in hess.cross_hess.items()},
    )


def main() -> None:
    args = parse_args()
    if int(args.batch_size) < 1:
        raise ValueError(f"--batch-size must be >= 1, got {args.batch_size}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This script requires CUDA for GS3DCTLossCUDA forward_backward2.")

    data_dir = Path(args.data_dir).expanduser().resolve()
    ckpt_dir = Path(args.ckpt_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data dir not found: {data_dir}")
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"ckpt dir not found: {ckpt_dir}")

    run_dir = ckpt_dir.parent
    if args.config_used is not None:
        config_used = Path(args.config_used).expanduser().resolve()
    else:
        config_used = run_dir / "config_used.yaml"
    if not config_used.is_file():
        raise FileNotFoundError(f"config_used.yaml not found: {config_used}")

    ckpt_999 = _resolve_ckpt_path(args.ckpt_999, ckpt_dir, step=999)
    ckpt_29999 = _resolve_ckpt_path(args.ckpt_29999, ckpt_dir, step=29999)

    cfg = OmegaConf.load(str(config_used))

    parser = CTImageParser(str(data_dir))
    dataset = CTImageDataset(parser, split=args.split)
    if len(dataset) == 0:
        raise RuntimeError(f"empty dataset split: {args.split}")
    dataset.preload_to(device)

    requested_batch_size = int(args.batch_size)
    batch_size = max(1, min(requested_batch_size, len(dataset)))
    if batch_size != requested_batch_size:
        print(
            f"[warn] Requested batch_size={requested_batch_size}, "
            f"but split '{args.split}' has only {len(dataset)} views; using {batch_size}."
        )

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else (run_dir / f"analysis_second_order_hist_bs{batch_size}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] Using batch_size={batch_size}, split={args.split}")
    indices = torch.arange(batch_size, device=device)
    batch = dataset.get_batch(indices)

    img_tensor = batch["image"]
    img_h, img_w = int(img_tensor.shape[1]), int(img_tensor.shape[2])

    bbox = torch.tensor(parser.bbox, dtype=torch.float32, device=device) if parser.bbox is not None else None
    volume_shape_wdh: Optional[Tuple[int, int, int]] = None
    try:
        vol_gt = dataset.get_vol_gt().to(device)
        volume_shape_wdh = tuple(int(x) for x in vol_gt.shape)
    except Exception:
        nv = parser.meta.get("nVoxel")
        if nv is not None:
            volume_shape_wdh = tuple(int(x) for x in nv)

    lambda_tv = float(cfg.trainer.lambda_tv)
    tv_vol_size = int(args.tv_vol_size) if args.tv_vol_size is not None else int(cfg.trainer.tv_vol_size)
    if lambda_tv > 0 and bbox is not None and volume_shape_wdh is not None:
        vol_ranges, vol_centers, vol_dims = _sample_tv_subvolume(bbox, tv_vol_size, volume_shape_wdh, device)
    else:
        vol_ranges = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
        vol_centers = torch.zeros(3, dtype=torch.float32, device=device)
        vol_dims = (2, 2, 2)

    data_obj = _make_data(batch, vol_ranges, vol_centers, device)
    op_config = _make_config(cfg, (img_h, img_w), vol_dims)

    init_gp = GaussianParams.from_data_dir(
        str(data_dir),
        scale_min=float(cfg.model.scale_min),
        scale_max=float(cfg.model.scale_max),
        global_scale=float(cfg.model.global_scale),
        init_opa_scale=float(cfg.model.init_opa_scale),
        init_scale_multiplier=float(cfg.model.init_scale_multiplier),
        bbox=bbox,
        device=str(device),
    )

    ckpt999_gp, _ = GaussianParams.load(str(ckpt_999), device=str(device))
    ckpt29999_gp, _ = GaussianParams.load(str(ckpt_29999), device=str(device))
    if bbox is not None:
        ckpt999_gp.crop_to_bbox(bbox)
        ckpt29999_gp.crop_to_bbox(bbox)

    operator = GS3DCTLossCUDA()

    print("[1/3] Running second-order analysis for init...")
    result_init = _analyze_one(operator, init_gp, data_obj, op_config)
    print("[2/3] Running second-order analysis for ckpt_999...")
    result_999 = _analyze_one(operator, ckpt999_gp, data_obj, op_config)
    print("[3/3] Running second-order analysis for ckpt_29999...")
    result_29999 = _analyze_one(operator, ckpt29999_gp, data_obj, op_config)

    results = {
        "init": result_init,
        "step999": result_999,
        "step29999": result_29999,
    }

    values_png = out_dir / "values_hist.png"
    grads_png = out_dir / "grads_hist.png"
    hess_png = out_dir / "hess_hist.png"

    _plot_values(results, bins=int(args.bins), out_path=values_png)
    _plot_grads(results, bins=int(args.bins), out_path=grads_png)
    _plot_hess(results, bins=int(args.bins), out_path=hess_png)

    summary = {
        "data_dir": str(data_dir),
        "ckpt_999": str(ckpt_999),
        "ckpt_29999": str(ckpt_29999),
        "config_used": str(config_used),
        "device": str(device),
        "split": args.split,
        "batch_size": int(batch_size),
        "bins": int(args.bins),
        "groups": {},
    }

    for stage, res in results.items():
        summary["groups"][stage] = {
            "values": {k: _stats(v) for k, v in res.values.items()},
            "grads": {k: _stats(v) for k, v in res.grads.items()},
            "hess_self": {k: _stats(v) for k, v in res.hess_self.items()},
            "hess_self_diag": {
                k: _stats(_extract_self_diag(k, res.hess_self.get(k))) for k in PARAM_ORDER
            },
            "hess_self_cross": {
                k: _stats(_extract_self_cross(k, res.hess_self.get(k))) for k in PARAM_ORDER
            },
            "hess_cross_groups": {k: _stats(res.hess_cross.get(k)) for k in INTER_GROUP_CROSS_KEYS},
        }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved: {values_png}")
    print(f"Saved: {grads_png}")
    print(f"Saved: {hess_png}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
