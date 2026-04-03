"""Pure-function visualization helpers.

All functions take tensors / numpy arrays and write image files to disk.
They have no dependency on training state or Hydra config.
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch

def save_projection_triplet(
    pred: torch.Tensor,
    gt: torch.Tensor,
    path: str,
    *,
    value_range: Tuple[float, float] = (0.0, 1.0),
    title: Optional[str] = None,
) -> None:
    """Save a single (GT, Pred, |Pred-GT|) visualization.

    Args:
        pred: Predicted projection image (H, W).
        gt: Ground-truth projection image (H, W).
        path: Output image path.
        value_range: Shared fixed range for all three panels.
        title: Optional figure title.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pred_np = pred.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()
    diff_np = np.abs(pred_np - gt_np)
    vmin, vmax = float(value_range[0]), float(value_range[1])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    panels = [
        (gt_np, "Ground Truth"),
        (pred_np, "Prediction"),
        (diff_np, "Abs Difference"),
    ]
    for ax, (img, panel_title) in zip(axes, panels):
        im = ax.imshow(img, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(panel_title)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_projection_comparison(
    pairs: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    path: str,
    step: Optional[int] = None,
) -> None:
    """Save a grid of (prediction, ground-truth, difference) triplets.

    Args:
        pairs: List of ``(pred, gt)`` 2-D tensors (each ``H x W``).
        path: Output image path.
        step: Optional step number for the title.
    """
    n = len(pairs)
    if n == 0:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    fig = plt.figure(figsize=(14, 3 * n + 0.8))
    grid = GridSpec(n, 4, figure=fig, width_ratios=[1, 1, 1, 0.04],
                    wspace=0.1, hspace=0.3)
    col_titles = ["Prediction", "Ground Truth", "Difference"]

    for i, (pred_t, gt_t) in enumerate(pairs):
        pred = pred_t.detach().cpu().numpy()
        gt = gt_t.detach().cpu().numpy()
        diff = np.abs(pred - gt)

        vmin = float(min(pred.min(), gt.min()))
        vmax = float(max(pred.max(), gt.max()))

        axes = [fig.add_subplot(grid[i, c]) for c in range(3)]
        cbar_ax = fig.add_subplot(grid[i, 3])

        axes[0].imshow(pred, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
        axes[1].imshow(gt, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
        im = axes[2].imshow(diff, cmap="viridis", vmin=diff.min(), vmax=diff.max(), aspect="equal")
        fig.colorbar(im, cax=cbar_ax)

        axes[0].set_ylabel(f"View {i}", fontsize=10)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        if i == 0:
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
) -> None:
    """Save a 3x3 grid of mid-plane slices (GT / Pred / Diff) x (W / D / H).

    Args:
        pred_vol: Predicted volume ``(W, D, H)``.
        gt_vol: Ground-truth volume ``(W, D, H)``.
        path: Output image path.
        psnr_val: Optional PSNR annotation for the title.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    pv = pred_vol.detach().cpu().numpy()
    gv = gt_vol.detach().cpu().numpy()

    W, D, H = pv.shape
    mid = (W // 2, D // 2, H // 2)

    def _slice(arr, axis, idx):
        if axis == 0:
            return arr[idx, :, :]
        if axis == 1:
            return arr[:, idx, :]
        return arr[:, :, idx]

    slices_gt = [_slice(gv, a, mid[a]) for a in range(3)]
    slices_pred = [_slice(pv, a, mid[a]) for a in range(3)]
    slices_diff = [sp - sg for sp, sg in zip(slices_pred, slices_gt)]

    vmin = min(gv.min(), pv.min())
    vmax = max(gv.max(), pv.max())
    diff_absmax = max(
        max(abs(s.min()) for s in slices_diff),
        max(abs(s.max()) for s in slices_diff),
    )

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    axis_names = ["W-axis (Sagittal)", "D-axis (Coronal)", "H-axis (Axial)"]
    row_names = ["Ground Truth", "Prediction", "Difference"]

    for col in range(3):
        im0 = axes[0, col].imshow(slices_gt[col], cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0, col].set_title(axis_names[col], fontsize=12, fontweight="bold")
        axes[0, col].axis("off")
        plt.colorbar(im0, ax=axes[0, col], fraction=0.046, pad=0.04)

        im1 = axes[1, col].imshow(slices_pred[col], cmap="viridis", vmin=vmin, vmax=vmax)
        axes[1, col].axis("off")
        plt.colorbar(im1, ax=axes[1, col], fraction=0.046, pad=0.04)

        im2 = axes[2, col].imshow(slices_diff[col], cmap="viridis",
                                   vmin=-diff_absmax, vmax=diff_absmax)
        axes[2, col].axis("off")
        plt.colorbar(im2, ax=axes[2, col], fraction=0.046, pad=0.04)

    for row, name in enumerate(row_names):
        axes[row, 0].set_ylabel(name, fontsize=14, fontweight="bold",
                                rotation=90, labelpad=20)

    title = "Volume Reconstruction"
    if psnr_val is not None:
        title += f" (PSNR: {psnr_val:.2f} dB)"
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_loss_curve(
    loss_history: Dict[str, List[float]],
    path: str,
) -> None:
    """Save a training loss curve plot.

    Args:
        loss_history: ``{name: [values]}`` where each list is indexed by step.
        path: Output image path.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, values in loss_history.items():
        ax.plot(values, label=name, alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
