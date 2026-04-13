"""Standalone validation script for trained CT reconstruction models.

Usage::

    python -m scripts.validate \\
        data_dir=/path/to/scene \\
        checkpoint_path=/path/to/ckpt.pt \\
        result_dir=/path/to/output
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm

from gsplat2ct.data.ctproj import CTImageDataset, CTImageParser
from gsplat2ct.model.gaussian_params import GaussianParams
from gsplat2ct.eval.evaluator import VolumeEvaluator
from gsplat2ct.eval.metric import metric_proj
from gsplat2ct.vis.plotting import save_volume_comparison


def _evaluate_projections(
    params,
    dataset: CTImageDataset,
    cfg: DictConfig,
    device: torch.device,
    split_name: str,
    output_dir: str,
) -> dict:
    """Render every view in *dataset*, compute metrics, and save comparison images."""
    from gsplat2ct.ops.data import Params, Data, Config
    from gsplat2ct.ops.gs3d_ct_rasterize import GS3DCTRasterizeCUDA

    class _PW:
        def __init__(self, data):
            self.data = data

    proj_dir = os.path.join(output_dir, split_name, "projections")
    os.makedirs(proj_dir, exist_ok=True)

    op = GS3DCTRasterizeCUDA()
    op.initialize(params)

    dataset.preload_to(device)
    psnr_list, ssim_list = [], []
    pred_proj_all: List[np.ndarray] = []
    gt_proj_all: List[np.ndarray] = []

    for idx in tqdm(range(len(dataset)), desc=f"Evaluating {split_name}"):
        batch = dataset.get_batch(idx)
        c2w = batch["camtoworld"].contiguous()
        K = batch["K"].contiguous()
        gt = batch["image"].contiguous()

        viewmats = torch.linalg.inv(c2w).contiguous()
        C = viewmats.shape[0]

        opacity_raw = params.get("opacity").data
        opacity_2d = opacity_raw.unsqueeze(0).expand(C, -1).contiguous()

        rast_params = Params()
        rast_params.add_param("means", params.get("means"))
        rast_params.add_param("quats", params.get("quats"))
        rast_params.add_param("scale_ori", params.get("scale_ori"))
        rast_params.add_param("opacity", _PW(opacity_2d))

        data_obj = Data()
        data_obj.set(viewmats=viewmats, Ks=K)

        img_h, img_w = gt.shape[1], gt.shape[2]
        op_cfg = Config()
        op_cfg.set(
            img_dim1=img_w, img_dim0=img_h, tile_size=16,
            eps_blur=0.3,
            scale_min=cfg.scale_min, scale_max=cfg.scale_max,
        )

        with torch.no_grad():
            fwd = op.forward_only(rast_params, data_obj, op_cfg)
        pred = fwd.outputs["render_output"]
        gt_sq = gt.squeeze(-1)

        m = metric_proj(pred, gt_sq, window_size=cfg.ssim_window)
        psnr_list.append(float(m["psnr"].item()))
        ssim_list.append(float(m["ssim"].item()))

        if bool(getattr(cfg, "save_projection_npy", True)) and split_name == "test":
            pred_proj_all.append(pred[0].detach().cpu().numpy())
            gt_proj_all.append(gt_sq[0].detach().cpu().numpy())

        if idx < cfg.eval_save_images_n:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            pred_np = pred[0].cpu().numpy()
            gt_np = gt_sq[0].cpu().numpy()
            diff_np = pred_np - gt_np

            vmin = min(gt_np.min(), pred_np.min())
            vmax = max(gt_np.max(), pred_np.max())

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(gt_np, cmap="viridis", vmin=vmin, vmax=vmax)
            axes[0].set_title("Ground Truth")
            axes[0].axis("off")
            axes[1].imshow(pred_np, cmap="viridis", vmin=vmin, vmax=vmax)
            axes[1].set_title("Prediction")
            axes[1].axis("off")
            d_abs = max(abs(diff_np.min()), abs(diff_np.max()))
            axes[2].imshow(diff_np, cmap="viridis", vmin=-d_abs, vmax=d_abs)
            axes[2].set_title("Difference")
            axes[2].axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(proj_dir, f"proj_{idx:04d}.png"), dpi=100)
            plt.close()

    metrics = {
        "split": split_name,
        "psnr_mean": float(np.mean(psnr_list)) if psnr_list else 0.0,
        "ssim_mean": float(np.mean(ssim_list)) if ssim_list else 0.0,
        "num_projections": len(psnr_list),
    }
    if pred_proj_all and gt_proj_all:
        pred_proj_path = os.path.join(output_dir, split_name, "pred_projections.npy")
        gt_proj_path = os.path.join(output_dir, split_name, "gt_projections.npy")
        np.save(pred_proj_path, np.stack(pred_proj_all, axis=0))
        np.save(gt_proj_path, np.stack(gt_proj_all, axis=0))
        metrics["pred_projection_npy"] = pred_proj_path
        metrics["gt_projection_npy"] = gt_proj_path

    with open(os.path.join(output_dir, split_name, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  {split_name}: PSNR={metrics['psnr_mean']:.2f} dB, SSIM={metrics['ssim_mean']:.4f}")
    return metrics


@hydra.main(version_base=None, config_path="../configs", config_name="validate")
def main(cfg: DictConfig) -> None:
    cfg.data_dir = to_absolute_path(cfg.data_dir)
    cfg.checkpoint_path = to_absolute_path(cfg.checkpoint_path)
    if cfg.result_dir is None:
        cfg.result_dir = os.path.join(os.path.dirname(cfg.checkpoint_path), "validation")
    cfg.result_dir = to_absolute_path(cfg.result_dir)
    os.makedirs(cfg.result_dir, exist_ok=True)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Load data
    print("[1/4] Loading dataset...")
    parser = CTImageParser(cfg.data_dir)
    trainset = CTImageDataset(parser, split="train")
    testset = None
    try:
        testset = CTImageDataset(parser, split="test")
    except Exception:
        pass

    # Load checkpoint
    print("[2/4] Loading checkpoint...")
    gp, step = GaussianParams.load(cfg.checkpoint_path, device=str(device))
    params = gp.to_params()
    print(f"   Loaded {gp.num_gaussians} Gaussians from step {step}")

    # Projection evaluation
    print("[3/4] Evaluating projections...")
    saved_arrays: Dict[str, str] = {}
    train_metrics = _evaluate_projections(
        params, trainset, cfg, device, "train", cfg.result_dir
    )
    test_metrics = {}
    if testset is not None:
        test_metrics = _evaluate_projections(
            params, testset, cfg, device, "test", cfg.result_dir
        )
        if "pred_projection_npy" in test_metrics:
            saved_arrays["pred_projection_npy"] = test_metrics["pred_projection_npy"]
        if "gt_projection_npy" in test_metrics:
            saved_arrays["gt_projection_npy"] = test_metrics["gt_projection_npy"]

    # Volume evaluation
    print("[4/4] Volume evaluation...")
    vol_metrics = {}
    vol_gt = None
    try:
        vol_gt = trainset.get_vol_gt().to(device)
    except Exception:
        pass

    if vol_gt is not None and parser.bbox is not None:
        bbox = torch.tensor(parser.bbox, dtype=torch.float32)
        vol_eval = VolumeEvaluator(
            params,
            scale_min=cfg.scale_min,
            scale_max=cfg.scale_max,
            eps3d=float(getattr(cfg, "eps3d", 1e-4)),
            ssim_window=cfg.ssim_window,
        )
        vol_metrics, pred_vol = vol_eval.evaluate(
            vol_gt, bbox, tuple(vol_gt.shape), device
        )
        print(f"   Volume PSNR={vol_metrics['psnr']:.2f} dB, SSIM={vol_metrics['ssim']:.4f}")

        vol_dir = os.path.join(cfg.result_dir, "volume")
        os.makedirs(vol_dir, exist_ok=True)
        save_volume_comparison(
            pred_vol, vol_gt,
            os.path.join(vol_dir, "volume_comparison.png"),
            psnr_val=vol_metrics.get("psnr"),
        )

        if bool(getattr(cfg, "save_volume_npy", True)):
            pred_vol_path = os.path.join(vol_dir, "pred_volume.npy")
            gt_vol_path = os.path.join(vol_dir, "gt_volume.npy")
            np.save(pred_vol_path, pred_vol.detach().cpu().numpy())
            np.save(gt_vol_path, vol_gt.detach().cpu().numpy())
            saved_arrays["pred_volume_npy"] = pred_vol_path
            saved_arrays["gt_volume_npy"] = gt_vol_path
    else:
        print("   GT volume or bbox not available, skipping.")

    # Summary
    report = {
        "checkpoint": cfg.checkpoint_path,
        "data_dir": cfg.data_dir,
        "num_gaussians": gp.num_gaussians,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "volume_metrics": vol_metrics,
        "saved_arrays": saved_arrays,
    }
    with open(os.path.join(cfg.result_dir, "validation_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nValidation complete. Results in: {cfg.result_dir}")


if __name__ == "__main__":
    main()
