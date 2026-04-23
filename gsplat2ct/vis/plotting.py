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
from mpl_toolkits.mplot3d.art3d import Line3DCollection
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
    """Save a single (GT, Pred, Pred-GT) visualization.

    Args:
        pred: Predicted projection image (H, W).
        gt: Ground-truth projection image (H, W).
        path: Output image path.
        value_range: Fixed range used by GT and Prediction panels.
        title: Optional figure title.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pred_np = pred.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()
    diff_np = pred_np - gt_np
    vmin, vmax = float(value_range[0]), float(value_range[1])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    panels = [
        (gt_np, "Ground Truth"),
        (pred_np, "Prediction"),
        (diff_np, "Difference (Pred-GT)"),
    ]
    for idx, (ax, (img, panel_title)) in enumerate(zip(axes, panels)):
        if idx < 2:
            im = ax.imshow(img, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
        else:
            im = ax.imshow(img, cmap="seismic", vmin=-0.1, vmax=0.1, aspect="equal")
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
        diff = pred - gt

        vmin = float(min(pred.min(), gt.min()))
        vmax = float(max(pred.max(), gt.max()))

        axes = [fig.add_subplot(grid[i, c]) for c in range(3)]
        cbar_ax = fig.add_subplot(grid[i, 3])

        axes[0].imshow(pred, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
        axes[1].imshow(gt, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
        im = axes[2].imshow(diff, cmap="seismic", vmin=-0.1, vmax=0.1, aspect="equal")
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
    slice_indices: Optional[Tuple[int, int, int]] = None,
    title: str = "Volume Reconstruction",
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
    if slice_indices is None:
        mid = (W // 2, D // 2, H // 2)
    else:
        mid = (
            int(max(0, min(slice_indices[0], W - 1))),
            int(max(0, min(slice_indices[1], D - 1))),
            int(max(0, min(slice_indices[2], H - 1))),
        )

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

        im2 = axes[2, col].imshow(slices_diff[col], cmap="seismic",
                       vmin=-0.1, vmax=0.1)
        axes[2, col].axis("off")
        plt.colorbar(im2, ax=axes[2, col], fraction=0.046, pad=0.04)

    for row, name in enumerate(row_names):
        axes[row, 0].set_ylabel(name, fontsize=14, fontweight="bold",
                                rotation=90, labelpad=20)

    fig_title = title
    if psnr_val is not None:
        fig_title += f" (PSNR: {psnr_val:.2f} dB)"
    fig.suptitle(fig_title, fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_init_coarse_volume_debug(
    coarse_vol: torch.Tensor,
    gt_vol: torch.Tensor,
    path: str,
    *,
    psnr_val: Optional[float] = None,
    slice_indices: Optional[Tuple[int, int, int]] = None,
) -> None:
    """Save initialization coarse-volume debug visualization.

    The layout, direction order, and slice selection follow
    ``save_volume_comparison`` exactly.
    """
    save_volume_comparison(
        pred_vol=coarse_vol,
        gt_vol=gt_vol,
        path=path,
        psnr_val=psnr_val,
        slice_indices=slice_indices,
        title="Initialization Coarse vs Ground Truth",
    )


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
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.0)
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_residual_slices_with_points(
    residual_vol: torch.Tensor,
    sampled_points_world: torch.Tensor,
    sampled_values: Optional[torch.Tensor],
    bbox: torch.Tensor,
    path: str,
    slice_indices: Optional[Tuple[int, int, int]] = None,
) -> None:
    """Save residual slices with sampled-point overlays.

    Panel order and slice directions are aligned with ``save_volume_comparison``:
    W-axis (Sagittal), D-axis (Coronal), H-axis (Axial).

    Point styles:
        - positive sampled values: black circle
        - negative sampled values: black x
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    rv = residual_vol.detach().cpu().float()
    if rv.ndim != 3:
        raise ValueError(f"residual_vol must be 3-D (W,D,H), got {tuple(rv.shape)}")
    W, D, H = int(rv.shape[0]), int(rv.shape[1]), int(rv.shape[2])

    pts = sampled_points_world.detach().cpu().float()
    bb = bbox.detach().cpu().float()
    mins = bb[0]
    ranges = (bb[1] - bb[0]).clamp_min(1e-8)

    scale = torch.tensor([W, D, H], dtype=torch.float32)
    idx = (pts - mins.unsqueeze(0)) / ranges.unsqueeze(0) * scale.unsqueeze(0)

    if slice_indices is None:
        mid_w, mid_d, mid_h = W // 2, D // 2, H // 2
    else:
        mid_w = int(max(0, min(slice_indices[0], W - 1)))
        mid_d = int(max(0, min(slice_indices[1], D - 1)))
        mid_h = int(max(0, min(slice_indices[2], H - 1)))

    # Keep same direction order as volume comparison: W, D, H.
    # Keep the same slice tensor ordering as save_volume_comparison (no transpose).
    # W-axis slice: (D, H), D-axis slice: (W, H), H-axis slice: (W, D)
    sag_w = rv[mid_w, :, :].numpy()
    cor_d = rv[:, mid_d, :].numpy()
    axi_h = rv[:, :, mid_h].numpy()

    vmax_abs = float(rv.abs().max().item())
    if vmax_abs <= 0:
        vmax_abs = 1e-6

    # Keep a narrow slab around each mid-slice for clearer overlays.
    tol = 0.75
    mask_w = (idx[:, 0] - float(mid_w)).abs() <= tol
    mask_d = (idx[:, 1] - float(mid_d)).abs() <= tol
    mask_h = (idx[:, 2] - float(mid_h)).abs() <= tol

    if sampled_values is not None:
        vals = sampled_values.detach().cpu().float().reshape(-1)
        pos = vals > 0
        neg = vals < 0
    else:
        vals = None
        pos = torch.zeros(idx.shape[0], dtype=torch.bool)
        neg = torch.ones(idx.shape[0], dtype=torch.bool)

    def _scatter_signed(ax, x: torch.Tensor, y: torch.Tensor, base_mask: torch.Tensor) -> None:
        if vals is None:
            ax.scatter(
                x[base_mask].numpy(),
                y[base_mask].numpy(),
                s=30,
                c="black",
                marker="x",
                alpha=0.95,
                linewidths=1.6,
            )
            return

        pos_mask = base_mask & pos
        neg_mask = base_mask & neg
        if pos_mask.any():
            ax.scatter(
                x[pos_mask].numpy(),
                y[pos_mask].numpy(),
                s=18,
                c="black",
                marker="o",
                alpha=0.95,
                linewidths=0.0,
            )
        if neg_mask.any():
            ax.scatter(
                x[neg_mask].numpy(),
                y[neg_mask].numpy(),
                s=30,
                c="black",
                marker="x",
                alpha=0.95,
                linewidths=1.6,
            )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # imshow uses x=column, y=row. For (D,H): x<-H, y<-D.
    im0 = axes[0].imshow(sag_w, cmap="seismic", vmin=-vmax_abs, vmax=vmax_abs, aspect="equal")
    _scatter_signed(axes[0], idx[:, 2], idx[:, 1], mask_w)
    axes[0].set_title(f"Residual W-axis (w={mid_w})")
    axes[0].set_xlabel("D")
    axes[0].set_ylabel("H")

    # For (W,H): x<-H, y<-W.
    im1 = axes[1].imshow(cor_d, cmap="seismic", vmin=-vmax_abs, vmax=vmax_abs, aspect="equal")
    _scatter_signed(axes[1], idx[:, 2], idx[:, 0], mask_d)
    axes[1].set_title(f"Residual D-axis (d={mid_d})")
    axes[1].set_xlabel("W")
    axes[1].set_ylabel("H")

    # For (W,D): x<-D, y<-W.
    im2 = axes[2].imshow(axi_h, cmap="seismic", vmin=-vmax_abs, vmax=vmax_abs, aspect="equal")
    _scatter_signed(axes[2], idx[:, 1], idx[:, 0], mask_h)
    axes[2].set_title(f"Residual H-axis (h={mid_h})")
    axes[2].set_xlabel("W")
    axes[2].set_ylabel("D")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    cbar = fig.colorbar(im2, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("coarse - pred")
    fig.suptitle("Pre-compensation Residual with Sampled Points", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _quats_to_rotmats(quats: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Convert unit quaternions (w, x, y, z) to rotation matrices (N, 3, 3)."""
    if quats.ndim != 2 or quats.shape[1] != 4:
        raise ValueError(f"quats must have shape (N, 4), got {tuple(quats.shape)}")

    q = quats / quats.norm(dim=-1, keepdim=True).clamp_min(eps)
    w, x, y, z = q.unbind(dim=-1)

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)
    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)

    row0 = torch.stack([r00, r01, r02], dim=-1)
    row1 = torch.stack([r10, r11, r12], dim=-1)
    row2 = torch.stack([r20, r21, r22], dim=-1)
    return torch.stack([row0, row1, row2], dim=1)


def _activate_opacity_values(opacity_raw: torch.Tensor, mode: str) -> torch.Tensor:
    """Apply a configurable opacity activation for visualization only."""
    mode_l = mode.lower()
    if mode_l == "raw":
        return opacity_raw
    if mode_l == "sigmoid":
        return torch.sigmoid(opacity_raw)
    if mode_l == "softplus":
        return torch.nn.functional.softplus(opacity_raw)
    raise ValueError(f"Unsupported opacity activation mode: {mode}")


def save_gaussian_state_3d_frame(
    means: torch.Tensor,
    quats: torch.Tensor,
    scale_ori: torch.Tensor,
    opacity_raw: torch.Tensor,
    path: str,
    *,
    bbox: Optional[torch.Tensor] = None,
    scale_min: float = 0.001,
    scale_max: float = 1.0,
    title: Optional[str] = None,
    max_points: int = 3000,
    point_size_min: float = 8.0,
    point_size_max: float = 120.0,
    use_log_scale_size: bool = True,
    axis_length_factor: float = 0.25,
    axis_length_min_ratio: float = 0.002,
    axis_length_max_ratio: float = 0.05,
    axis_alpha: float = 0.45,
    axis_linewidth: float = 0.6,
    elev: float = 20.0,
    azim: float = 35.0,
    colormap: str = "viridis",
    opacity_activation: str = "raw",
    opacity_vmin: Optional[float] = None,
    opacity_vmax: Optional[float] = None,
    dpi: int = 140,
) -> None:
    """Save one 3-D frame of Gaussian states for optimization stability tracking.

    Visual channels:
    - point position: Gaussian means in world-space
    - point size: mean activated scale (optionally logarithmic)
    - point color: activated opacity
    - tiny local XYZ axes: quaternion orientation

    Axes limits are fixed by ``bbox`` to keep temporal consistency in videos.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    means_t = means.detach().cpu().float()
    quats_t = quats.detach().cpu().float()
    scale_t = scale_ori.detach().cpu().float()
    opacity_t = opacity_raw.detach().cpu().float().reshape(-1)

    n_total = int(means_t.shape[0])
    if n_total == 0:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        if bbox is not None:
            bb = bbox.detach().cpu().float().numpy()
            mins = bb[0]
            maxs = bb[1]
        else:
            mins = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
            maxs = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        ax.set_xlim(float(mins[0]), float(maxs[0]))
        ax.set_ylim(float(mins[1]), float(maxs[1]))
        ax.set_zlim(float(mins[2]), float(maxs[2]))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=float(elev), azim=float(azim))
        ax.set_title(title if title is not None else "Gaussian State (empty)")
        plt.tight_layout()
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return

    if max_points > 0 and n_total > max_points:
        keep_idx = torch.linspace(0, n_total - 1, steps=max_points).long()
        means_t = means_t[keep_idx]
        quats_t = quats_t[keep_idx]
        scale_t = scale_t[keep_idx]
        opacity_t = opacity_t[keep_idx]

    n_vis = int(means_t.shape[0])

    if bbox is not None:
        bb = bbox.detach().cpu().float().numpy()
        mins = bb[0].astype(np.float64)
        maxs = bb[1].astype(np.float64)
    else:
        mins = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
        maxs = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    ranges = maxs - mins
    for i in range(3):
        if (not np.isfinite(ranges[i])) or float(ranges[i]) <= 1e-8:
            center = 0.5 * (float(mins[i]) + float(maxs[i]))
            mins[i] = center - 0.5
            maxs[i] = center + 0.5
            ranges[i] = 1.0
    scene_extent = float(np.max(ranges))

    scale_activated = float(scale_min) + (float(scale_max) - float(scale_min)) * torch.sigmoid(scale_t)
    mean_scale = scale_activated.mean(dim=-1).clamp_min(1e-12)

    if use_log_scale_size:
        log_min = float(np.log10(max(float(scale_min), 1e-12)))
        log_max = float(np.log10(max(float(scale_max), float(scale_min) + 1e-12)))
        denom = max(log_max - log_min, 1e-12)
        norm = (torch.log10(mean_scale) - log_min) / denom
    else:
        denom = max(float(scale_max) - float(scale_min), 1e-12)
        norm = (mean_scale - float(scale_min)) / denom
    norm = norm.clamp(0.0, 1.0)
    marker_sizes = float(point_size_min) + norm * (float(point_size_max) - float(point_size_min))

    rotmats = _quats_to_rotmats(quats_t)
    # Columns of R are local basis vectors transformed to world-space.
    axis_dirs = rotmats.transpose(1, 2)

    axis_lengths = mean_scale * float(axis_length_factor)
    axis_lengths = axis_lengths.clamp(
        min=float(axis_length_min_ratio) * scene_extent,
        max=float(axis_length_max_ratio) * scene_extent,
    )

    endpoints_x = means_t + axis_dirs[:, 0, :] * axis_lengths.unsqueeze(-1)
    endpoints_y = means_t + axis_dirs[:, 1, :] * axis_lengths.unsqueeze(-1)
    endpoints_z = means_t + axis_dirs[:, 2, :] * axis_lengths.unsqueeze(-1)
    seg_x = torch.stack([means_t, endpoints_x], dim=1).numpy()
    seg_y = torch.stack([means_t, endpoints_y], dim=1).numpy()
    seg_z = torch.stack([means_t, endpoints_z], dim=1).numpy()

    opacity_activated = _activate_opacity_values(opacity_t, opacity_activation)
    opacity_np = opacity_activated.detach().cpu().numpy()

    if opacity_vmin is None:
        cmin = float(np.percentile(opacity_np, 1.0))
    else:
        cmin = float(opacity_vmin)
    if opacity_vmax is None:
        cmax = float(np.percentile(opacity_np, 99.0))
    else:
        cmax = float(opacity_vmax)

    if (not np.isfinite(cmin)) or (not np.isfinite(cmax)) or cmax <= cmin:
        cmin = float(np.nanmin(opacity_np)) if np.isfinite(np.nanmin(opacity_np)) else 0.0
        cmax = float(np.nanmax(opacity_np)) if np.isfinite(np.nanmax(opacity_np)) else 1.0
        if cmax <= cmin:
            cmax = cmin + 1e-6

    means_np = means_t.numpy()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        means_np[:, 0],
        means_np[:, 1],
        means_np[:, 2],
        s=marker_sizes.numpy(),
        c=opacity_np,
        cmap=colormap,
        vmin=cmin,
        vmax=cmax,
        alpha=0.95,
        linewidths=0.0,
        depthshade=False,
    )

    ax.add_collection3d(
        Line3DCollection(
            seg_x,
            colors="#ef4444",
            linewidths=float(axis_linewidth),
            alpha=float(axis_alpha),
        )
    )
    ax.add_collection3d(
        Line3DCollection(
            seg_y,
            colors="#22c55e",
            linewidths=float(axis_linewidth),
            alpha=float(axis_alpha),
        )
    )
    ax.add_collection3d(
        Line3DCollection(
            seg_z,
            colors="#3b82f6",
            linewidths=float(axis_linewidth),
            alpha=float(axis_alpha),
        )
    )

    ax.set_xlim(float(mins[0]), float(maxs[0]))
    ax.set_ylim(float(mins[1]), float(maxs[1]))
    ax.set_zlim(float(mins[2]), float(maxs[2]))
    ax.set_box_aspect((float(ranges[0]), float(ranges[1]), float(ranges[2])))
    ax.view_init(elev=float(elev), azim=float(azim))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title(title if title is not None else "Gaussian State")
    ax.text2D(
        0.02,
        0.98,
        f"visible={n_vis}/{n_total}",
        transform=ax.transAxes,
        va="top",
    )
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label(f"Opacity ({opacity_activation})")

    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
