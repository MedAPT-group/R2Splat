"""Evaluation visualization helpers."""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch


def save_projection_comparison(
    pairs: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    path: str,
    step: Optional[int] = None,
) -> None:
    """Save a grid of prediction / ground-truth / difference triplets."""
    if not pairs:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    n_rows = len(pairs)
    fig = plt.figure(figsize=(14, 3 * n_rows + 0.8))
    grid = GridSpec(
        n_rows,
        4,
        figure=fig,
        width_ratios=[1, 1, 1, 0.04],
        wspace=0.1,
        hspace=0.3,
    )
    col_titles = ["Prediction", "Ground Truth", "Difference"]

    for row, (pred_t, gt_t) in enumerate(pairs):
        pred = pred_t.detach().cpu().numpy()
        gt = gt_t.detach().cpu().numpy()
        diff = pred - gt
        vmin = float(min(pred.min(), gt.min()))
        vmax = float(max(pred.max(), gt.max()))

        axes = [fig.add_subplot(grid[row, col]) for col in range(3)]
        cbar_ax = fig.add_subplot(grid[row, 3])
        axes[0].imshow(pred, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
        axes[1].imshow(gt, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
        im = axes[2].imshow(diff, cmap="seismic", vmin=-0.1, vmax=0.1, aspect="equal")
        fig.colorbar(im, cax=cbar_ax)

        axes[0].set_ylabel(f"View {row}", fontsize=10)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        if row == 0:
            for ax, title in zip(axes, col_titles):
                ax.set_title(title, fontsize=11)

    title = "Projection Comparison"
    if step is not None:
        title += f" (Step {step})"
    fig.suptitle(title, fontsize=13, y=1.01)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_volume_comparison(
    pred_vol: torch.Tensor,
    gt_vol: torch.Tensor,
    path: str,
    psnr_val: Optional[float] = None,
    slice_indices: Optional[Tuple[int, int, int]] = None,
    title: str = "Volume Reconstruction",
) -> None:
    """Save mid-plane GT / prediction / difference slices for a volume."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    pred = pred_vol.detach().cpu().numpy()
    gt = gt_vol.detach().cpu().numpy()
    w, d, h = pred.shape
    if slice_indices is None:
        mid = (w // 2, d // 2, h // 2)
    else:
        mid = (
            int(max(0, min(slice_indices[0], w - 1))),
            int(max(0, min(slice_indices[1], d - 1))),
            int(max(0, min(slice_indices[2], h - 1))),
        )

    def _slice(arr, axis, idx):
        if axis == 0:
            return arr[idx, :, :]
        if axis == 1:
            return arr[:, idx, :]
        return arr[:, :, idx]

    gt_slices = [_slice(gt, axis, mid[axis]) for axis in range(3)]
    pred_slices = [_slice(pred, axis, mid[axis]) for axis in range(3)]
    diff_slices = [p - g for p, g in zip(pred_slices, gt_slices)]

    vmin = float(min(gt.min(), pred.min()))
    vmax = float(max(gt.max(), pred.max()))
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    axis_names = ["W-axis (Sagittal)", "D-axis (Coronal)", "H-axis (Axial)"]
    row_names = ["Ground Truth", "Prediction", "Difference"]

    for col in range(3):
        im0 = axes[0, col].imshow(gt_slices[col], cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0, col].set_title(axis_names[col], fontsize=12, fontweight="bold")
        axes[0, col].axis("off")
        plt.colorbar(im0, ax=axes[0, col], fraction=0.046, pad=0.04)

        im1 = axes[1, col].imshow(pred_slices[col], cmap="viridis", vmin=vmin, vmax=vmax)
        axes[1, col].axis("off")
        plt.colorbar(im1, ax=axes[1, col], fraction=0.046, pad=0.04)

        im2 = axes[2, col].imshow(diff_slices[col], cmap="seismic", vmin=-0.1, vmax=0.1)
        axes[2, col].axis("off")
        plt.colorbar(im2, ax=axes[2, col], fraction=0.046, pad=0.04)

    for row, name in enumerate(row_names):
        axes[row, 0].set_ylabel(name, fontsize=14, fontweight="bold", rotation=90, labelpad=20)

    fig_title = title
    if psnr_val is not None:
        fig_title += f" (PSNR: {psnr_val:.2f} dB)"
    fig.suptitle(fig_title, fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


__all__ = ["save_projection_comparison", "save_volume_comparison"]
