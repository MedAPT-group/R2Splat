"""Histograms of Gaussian parameters from a checkpoint (.pt).

Supports:
  - ``gsplat2ct``: flat keys ``means``, ``quats``, ``scale_ori``, ``opacity`` (this repo).
  - ``binary_tree``: nested ``model`` with ``means``, ``scales`` (log-space; exp for
    activated scale), ``quats``, ``opacities`` (pre-softplus raw).

Y-axis: frequency = (count in bin) / N.
X-axis: fixed ranges as requested; activated scales use log-scale x with limits (1e-5, 0.2).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


# Fixed display ranges (user request)
XLIM_MEAN = (-1.1, 1.1)
XLIM_SCALE_LOG = (1e-5, 0.2)  # log10 x; left bound > 0 for log scale
XLIM_QUAT_W = (0.0, 2.0)
XLIM_QUAT_XYZ = (-0.6, 0.6)
XLIM_OPACITY_STANDARD = (0.0, 0.25)
# Pre-softplus logits can be negative; wide range for binary_tree opacity
XLIM_OPACITY_PRE_SOFTPLUS = (-10.0, 2.0)


def _load_scale_bounds_from_run_dir(run_dir: str) -> Tuple[float, float]:
    path = os.path.join(run_dir, "config_used.yaml")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    m1 = re.search(r"^\s*scale_min:\s*([0-9.eE+-]+)\s*$", text, re.MULTILINE)
    m2 = re.search(r"^\s*scale_max:\s*([0-9.eE+-]+)\s*$", text, re.MULTILINE)
    if not m1 or not m2:
        raise ValueError(f"Could not parse scale_min/scale_max from {path}")
    return float(m1.group(1)), float(m2.group(1))


def _hist_frequency(
    ax,
    data_1d: torch.Tensor,
    *,
    bins: Any,
    range_: Optional[Tuple[float, float]] = None,
    xscale: str = "linear",
    **kw,
) -> None:
    """Histogram with bar height = fraction of samples in bin (count / N)."""
    x = data_1d.detach().float().cpu().numpy().ravel()
    n = x.size
    if n == 0:
        return
    w = np.ones(n, dtype=np.float64) / float(n)
    ax.hist(x, bins=bins, range=range_, weights=w, **kw)
    ax.set_xscale(xscale)
    ax.set_ylabel("frequency")


def _plot_figure(
    *,
    means: torch.Tensor,
    scale_act: torch.Tensor,
    quats: torch.Tensor,
    opacity: torch.Tensor,
    out_path: str,
    title_extra: str,
    opacity_title: str,
    opacity_xlim: Tuple[float, float],
) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = int(means.shape[0])
    log_bins = np.logspace(
        np.log10(XLIM_SCALE_LOG[0]),
        np.log10(XLIM_SCALE_LOG[1]),
        81,
    )

    fig = plt.figure(figsize=(12, 14))
    gs = fig.add_gridspec(5, 3, hspace=0.38, wspace=0.32)
    fig.suptitle(
        f"Gaussian parameter distributions (N={n})\n{title_extra}",
        fontsize=11,
    )

    # Row 0: means
    for i, title in enumerate(["mean x", "mean y", "mean z"]):
        ax = fig.add_subplot(gs[0, i])
        _hist_frequency(
            ax,
            means[:, i],
            bins=80,
            range_=XLIM_MEAN,
            xscale="linear",
            color="steelblue",
            edgecolor="white",
            linewidth=0.3,
        )
        ax.set_xlim(XLIM_MEAN[0], XLIM_MEAN[1])
        ax.set_title(title)

    # Row 1: activated scales (log x)
    for i, title in enumerate(["activated scale 0", "activated scale 1", "activated scale 2"]):
        ax = fig.add_subplot(gs[1, i])
        s = torch.clamp(scale_act[:, i], min=XLIM_SCALE_LOG[0], max=XLIM_SCALE_LOG[1])
        _hist_frequency(
            ax,
            s,
            bins=log_bins,
            range_=None,
            xscale="log",
            color="seagreen",
            edgecolor="white",
            linewidth=0.3,
        )
        ax.set_xlim(XLIM_SCALE_LOG[0], XLIM_SCALE_LOG[1])
        ax.set_title(title + " (log x)")

    # Rows 2–3: quaternions
    q_titles = ["quat w", "quat x", "quat y", "quat z"]
    xlims = [XLIM_QUAT_W, XLIM_QUAT_XYZ, XLIM_QUAT_XYZ, XLIM_QUAT_XYZ]
    for i in range(4):
        ax = fig.add_subplot(gs[2 + i // 3, i % 3])
        _hist_frequency(
            ax,
            quats[:, i],
            bins=80,
            range_=xlims[i],
            xscale="linear",
            color="coral",
            edgecolor="white",
            linewidth=0.3,
        )
        ax.set_xlim(xlims[i][0], xlims[i][1])
        ax.set_title(q_titles[i])

    ax_op = fig.add_subplot(gs[4, :])
    _hist_frequency(
        ax_op,
        opacity.reshape(-1),
        bins=80,
        range_=opacity_xlim,
        xscale="linear",
        color="mediumpurple",
        edgecolor="white",
        linewidth=0.3,
    )
    ax_op.set_xlim(opacity_xlim[0], opacity_xlim[1])
    ax_op.set_title(opacity_title)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return n


def _summary(
    means: torch.Tensor,
    scale_act: torch.Tensor,
    quats: torch.Tensor,
    opacity: torch.Tensor,
) -> Dict[str, Any]:
    def _q(x: torch.Tensor) -> dict:
        x = x.reshape(-1).float()
        qs = [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0]
        return {
            "min": float(x.min()),
            "max": float(x.max()),
            "mean": float(x.mean()),
            "std": float(x.std(unbiased=False)),
            "quantiles": {str(q): float(torch.quantile(x, q)) for q in qs},
        }

    return {
        "means_xyz": [_q(means[:, i]) for i in range(3)],
        "activated_scale": [_q(scale_act[:, i]) for i in range(3)],
        "quats_wxyz": [_q(quats[:, i]) for i in range(4)],
        "opacity": _q(opacity),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--scale-min", type=float, default=None)
    p.add_argument("--scale-max", type=float, default=None)
    p.add_argument(
        "--format",
        type=str,
        choices=("gsplat2ct", "binary_tree"),
        default="gsplat2ct",
    )
    args = p.parse_args()

    raw = torch.load(args.ckpt, map_location="cpu", weights_only=True)

    if args.format == "gsplat2ct":
        if args.scale_min is not None and args.scale_max is not None:
            scale_min, scale_max = args.scale_min, args.scale_max
        elif args.run_dir is not None:
            scale_min, scale_max = _load_scale_bounds_from_run_dir(args.run_dir)
        else:
            raise SystemExit(
                "gsplat2ct: provide --scale-min and --scale-max, or --run-dir"
            )
        means = raw["means"].float()
        quats = raw["quats"].float()
        scale_ori = raw["scale_ori"].float()
        opacity = raw["opacity"].float()
        scale_act = scale_min + (scale_max - scale_min) * torch.sigmoid(scale_ori)
        title_extra = (
            f"activated scale = scale_min + (scale_max - scale_min) * sigmoid(scale_ori); "
            f"scale_min={scale_min}, scale_max={scale_max}"
        )
        opacity_title = "opacity (raw stored in checkpoint)"
        opacity_xlim = XLIM_OPACITY_STANDARD
    else:
        m = raw["model"]
        means = m["means"].float()
        quats = m["quats"].float()
        scales_log = m["scales"].float()
        opacity = m["opacities"].float()
        # Physical scale (log-space storage, same spirit as exp in 3DGS)
        scale_act = torch.exp(torch.clamp(scales_log, max=20.0))
        title_extra = (
            "activated scale = exp(scales) with scales clamped to 20 before exp "
            "(binary_tree checkpoint; log-space scales)"
        )
        opacity_title = "opacity (pre-softplus raw)"
        opacity_xlim = XLIM_OPACITY_PRE_SOFTPLUS

    out = args.out
    if out is None:
        out = os.path.join(os.path.dirname(args.ckpt), "gaussian_distributions.png")

    n = _plot_figure(
        means=means,
        scale_act=scale_act,
        quats=quats,
        opacity=opacity,
        out_path=out,
        title_extra=title_extra,
        opacity_title=opacity_title,
        opacity_xlim=opacity_xlim,
    )

    summary: Dict[str, Any] = {
        "ckpt": os.path.abspath(args.ckpt),
        "format": args.format,
        "N": n,
        **_summary(means, scale_act, quats, opacity),
    }
    if args.format == "gsplat2ct":
        summary["scale_min"] = scale_min
        summary["scale_max"] = scale_max
    summary_path = os.path.splitext(out)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
