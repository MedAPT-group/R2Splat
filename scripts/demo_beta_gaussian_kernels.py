"""Plot Gaussian and Beta primitive kernels in 1D, 2D, and 3D.

The comparison uses the same center opacity, mean, and rotation in every panel.
Gaussian uses scale 1 and exp(-q / 2).  Beta uses scale 3 and
(1 - q)_+^beta, matching the compact-support kernel shape used by the CUDA
primitive after beta activation.
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def gaussian_kernel(q: np.ndarray, opacity: float = 1.0) -> np.ndarray:
    return opacity * np.exp(-0.5 * q)


def beta_kernel(q: np.ndarray, beta: float, opacity: float = 1.0) -> np.ndarray:
    h = np.maximum(1.0 - q, 0.0)
    return opacity * np.power(h, beta)


def make_figure(output_dir: str, beta_scale: float = 3.0) -> str:
    os.makedirs(output_dir, exist_ok=True)

    betas = np.arange(1.0, 6.0 + 1.0e-9, 0.5)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, betas.size))

    x = np.linspace(-6.0, 6.0, 1600)
    q1_gauss = x * x
    q1_beta = (x / beta_scale) ** 2
    r = np.linspace(0.0, 6.0, 1600)
    qr_gauss = r * r
    qr_beta = (r / beta_scale) ** 2

    grid = np.linspace(-4.0, 4.0, 420)
    xx, yy = np.meshgrid(grid, grid, indexing="xy")
    q2_gauss = xx * xx + yy * yy
    q2_beta = (xx / beta_scale) ** 2 + (yy / beta_scale) ** 2

    fig = plt.figure(figsize=(18, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.05, 1.0])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax_legend = fig.add_subplot(gs[0, 3])

    ax1.plot(x, gaussian_kernel(q1_gauss), color="black", lw=2.2, label="Gaussian")
    for beta, color in zip(betas, colors):
        ax1.plot(x, beta_kernel(q1_beta, beta), color=color, lw=1.5)
    ax1.set_title("1D kernel value")
    ax1.set_xlabel("x")
    ax1.set_ylabel("density")
    ax1.set_xlim(-6.0, 6.0)
    ax1.set_ylim(-0.03, 1.05)
    ax1.grid(alpha=0.25)

    ax2.plot(r, 2.0 * np.pi * r * gaussian_kernel(qr_gauss), color="black", lw=2.2)
    for beta, color in zip(betas, colors):
        ax2.plot(r, 2.0 * np.pi * r * beta_kernel(qr_beta, beta), color=color, lw=1.5)
    ax2.set_title("2D radial shell density")
    ax2.set_xlabel("radius")
    ax2.set_ylabel("2*pi*r*f(r)")
    ax2.set_xlim(0.0, 6.0)
    ax2.grid(alpha=0.25)

    ax3.plot(r, 4.0 * np.pi * r * r * gaussian_kernel(qr_gauss), color="black", lw=2.2)
    for beta, color in zip(betas, colors):
        ax3.plot(r, 4.0 * np.pi * r * r * beta_kernel(qr_beta, beta), color=color, lw=1.5)
    ax3.set_title("3D radial shell density")
    ax3.set_xlabel("radius")
    ax3.set_ylabel("4*pi*r^2*f(r)")
    ax3.set_xlim(0.0, 6.0)
    ax3.grid(alpha=0.25)

    ax_legend.axis("off")
    handles = [plt.Line2D([0], [0], color="black", lw=2.2, label="Gaussian")]
    handles.extend(
        plt.Line2D([0], [0], color=color, lw=1.8, label=f"beta={beta:.1f}")
        for beta, color in zip(betas, colors)
    )
    ax_legend.legend(handles=handles, loc="center left", frameon=False, ncol=1)

    heatmaps = [
        ("2D Gaussian slice, scale=1", gaussian_kernel(q2_gauss)),
        (f"2D Beta slice, beta=1.0, scale={beta_scale:g}", beta_kernel(q2_beta, 1.0)),
        (f"2D Beta slice, beta=3.5, scale={beta_scale:g}", beta_kernel(q2_beta, 3.5)),
        (f"2D Beta slice, beta=6.0, scale={beta_scale:g}", beta_kernel(q2_beta, 6.0)),
    ]
    for idx, (title, image) in enumerate(heatmaps):
        ax = fig.add_subplot(gs[1, idx])
        im = ax.imshow(
            image,
            extent=[grid.min(), grid.max(), grid.min(), grid.max()],
            origin="lower",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Same opacity=1, mean=0, rotation=identity: "
        f"Gaussian scale=1 vs Beta scale={beta_scale:g}",
        fontsize=14,
    )

    out_path = os.path.join(output_dir, "beta_gaussian_kernel_comparison.png")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=os.path.join("output", "demo_beta_gaussian_kernels"),
        help="Directory where the comparison image will be saved.",
    )
    parser.add_argument(
        "--beta-scale",
        type=float,
        default=3.0,
        help="Beta scale relative to the Gaussian scale.",
    )
    args = parser.parse_args()
    out_path = make_figure(args.output_dir, beta_scale=args.beta_scale)
    print(out_path)


if __name__ == "__main__":
    main()
