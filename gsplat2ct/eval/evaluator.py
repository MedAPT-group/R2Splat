"""Evaluators for CT reconstruction quality.

``ProjectionEvaluator`` renders all test views and computes 2-D metrics.
``VolumeEvaluator`` voxelizes the full bounding box and computes 3-D metrics.

Both use the forward-only operators (``GS3DCTRasterizeCUDA``,
``GS3DCTVoxelizeCUDA``) so they never allocate backward buffers.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch

from gsplat2ct.ops.data import Params
from gsplat2ct.ops.torch_autograd import (
    GaussianMappingVoxelizer,
    GaussianProjectionRasterizer,
    activate_gaussian_params,
    activate_uncertainty,
)
from gsplat2ct.eval.metric import metric_proj, metric_vol
from gsplat2ct.model.opacity import opacity_activation


def _pearson_corr(a: torch.Tensor, b: torch.Tensor) -> Optional[float]:
    a = a.reshape(-1).to(torch.float32)
    b = b.reshape(-1).to(torch.float32)
    finite = torch.isfinite(a) & torch.isfinite(b)
    if int(finite.sum().item()) < 2:
        return None
    a = a[finite]
    b = b[finite]
    a = a - a.mean()
    b = b - b.mean()
    denom = a.square().mean().sqrt() * b.square().mean().sqrt()
    if float(denom.item()) <= 0.0:
        return None
    return float((a * b).mean().div(denom).item())


class _ParamWrapper:
    def __init__(self, data: torch.Tensor):
        self.data = data


class ProjectionEvaluator:
    """Evaluate 2-D projection quality over a dataset split.

    Args:
        params: Current Gaussian parameters (``Params`` container).
        scale_min: World-space ParameterTransform lower bound.
        scale_max: World-space ParameterTransform upper bound.
        eps_blur: Conic blur epsilon.
        ssim_window: SSIM kernel size.
    """

    def __init__(
        self,
        params: Params,
        *,
        scale_min: float = 0.001,
        scale_max: float = 1.0,
        eps_blur: float = 0.3,
        ssim_window: int = 11,
        primitive_type: str = "gaussian",
        beta_gamma_init: float = 3.5,
        uncertainty_enabled: bool = False,
        uncertainty_q_min: float = 1.0e-6,
        uncertainty_q_max: Optional[float] = None,
        uncertainty_var_floor: float = 1.0e-6,
        uncertainty_variance_scale: float = 1.0,
        detector_uncertainty_var: float = 0.0,
    ):
        self.params = params
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.eps_blur = eps_blur
        self.ssim_window = ssim_window
        self.primitive_type = str(primitive_type).lower()
        self.beta_gamma_init = float(beta_gamma_init)
        self.uncertainty_enabled = bool(uncertainty_enabled)
        self.uncertainty_q_min = float(uncertainty_q_min)
        self.uncertainty_q_max = uncertainty_q_max
        self.uncertainty_var_floor = float(uncertainty_var_floor)
        self.uncertainty_variance_scale = float(uncertainty_variance_scale)
        self.detector_uncertainty_var = float(detector_uncertainty_var)

    @torch.no_grad()
    def evaluate(
        self,
        dataset: Any,
        device: torch.device,
        max_vis: int = 5,
    ) -> Tuple[Dict[str, float], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Run evaluation on *dataset* and return aggregate metrics plus
        up to *max_vis* (pred, target) image pairs for visualization."""
        dataset.preload_to(device)

        metric_sums: Dict[str, float] = {
            "l1": 0.0,
            "mse": 0.0,
            "psnr": 0.0,
            "ssim": 0.0,
        }
        metric_count = 0
        uncertainty_corr_sum = 0.0
        uncertainty_corr_count = 0
        uncertainty_std_sum = 0.0
        uncertainty_std_count = 0
        vis_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        import numpy as np
        vis_indices = set(
            np.linspace(0, len(dataset) - 1, max_vis, dtype=int).tolist()
        )

        for idx in range(len(dataset)):
            batch = dataset.get_batch(idx)
            c2w = batch["camtoworld"].contiguous()
            K = batch["K"].contiguous()
            gt = batch["image"].contiguous()

            viewmats = torch.linalg.inv(c2w).contiguous()
            img_h, img_w = gt.shape[1], gt.shape[2]
            _quats, _scales, opacity, covars, _max_scales = activate_gaussian_params(
                self.params.get("quats").data,
                self.params.get("scale_ori").data,
                self.params.get("opacity").data,
                scale_min=self.scale_min,
                scale_max=self.scale_max,
            )
            rasterizer = GaussianProjectionRasterizer(
                img_w=img_w,
                img_h=img_h,
                tile_size=16,
                eps_blur=self.eps_blur,
                radius_clip=1.0 if self.primitive_type == "beta" else 3.0,
                primitive_type=self.primitive_type,
            )
            q = None
            if self.uncertainty_enabled and self.params.get("raw_uncertainty") is not None:
                q = activate_uncertainty(
                    self.params.get("raw_uncertainty").data,
                    q_min=self.uncertainty_q_min,
                    q_max=self.uncertainty_q_max,
                )
            fwd = rasterizer(
                self.params.get("means").data,
                covars,
                opacity,
                viewmats,
                K,
                beta=(
                    self.params.get("beta").data
                    if self.params.get("beta") is not None
                    else None
                ),
                beta_gamma_init=self.beta_gamma_init,
                uncertainty=q,
            )
            # Signed kernels can produce values outside [0, 1]; clamp for evaluation.
            pred = fwd.image.clamp(0.0, 1.0)

            gt_squeezed = gt.squeeze(-1)
            m = metric_proj(
                pred, gt_squeezed, window_size=self.ssim_window
            )
            metric_sums["l1"] += float(m["l1"].item())
            metric_sums["mse"] += float(m["mse"].item())
            metric_sums["psnr"] += float(m["psnr"].item())
            metric_sums["ssim"] += float(m["ssim"].item())
            metric_count += 1

            if fwd.variance is not None:
                pred_std = (
                    fwd.variance * self.uncertainty_variance_scale
                    + self.detector_uncertainty_var
                    + self.uncertainty_var_floor
                ).clamp_min(0).sqrt()
                corr = _pearson_corr((fwd.image - gt_squeezed).abs(), pred_std)
                if corr is not None:
                    uncertainty_corr_sum += corr
                    uncertainty_corr_count += 1
                uncertainty_std_sum += float(pred_std.mean().item())
                uncertainty_std_count += 1

            if idx in vis_indices:
                vis_pairs.append(
                    (pred[0].cpu(), gt_squeezed[0].cpu())
                )

        denom = max(metric_count, 1)
        metrics = {
            k: float(v / denom) for k, v in metric_sums.items()
        }
        if uncertainty_corr_count > 0:
            metrics["uncertainty_error_corr"] = float(
                uncertainty_corr_sum / uncertainty_corr_count
            )
        if uncertainty_std_count > 0:
            metrics["uncertainty_std_mean"] = float(
                uncertainty_std_sum / uncertainty_std_count
            )
        return metrics, vis_pairs


class VolumeEvaluator:
    """Evaluate 3-D volume reconstruction quality.

    Args:
        params: Current Gaussian parameters (``Params`` container).
        scale_min: World-space ParameterTransform lower bound.
        scale_max: World-space ParameterTransform upper bound.
        eps3d: World-to-volume mapping epsilon.
        ssim_window: SSIM kernel size for 3-D SSIM.
    """

    def __init__(
        self,
        params: Params,
        *,
        scale_min: float = 0.001,
        scale_max: float = 1.0,
        eps3d: float = 1e-4,
        ssim_window: int = 11,
        primitive_type: str = "gaussian",
        beta_gamma_init: float = 3.5,
        uncertainty_enabled: bool = False,
        uncertainty_q_min: float = 1.0e-6,
        uncertainty_q_max: Optional[float] = None,
        uncertainty_var_floor: float = 1.0e-6,
        uncertainty_variance_scale: float = 1.0,
    ):
        self.params = params
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.eps3d = eps3d
        self.ssim_window = ssim_window
        self.primitive_type = str(primitive_type).lower()
        self.beta_gamma_init = float(beta_gamma_init)
        self.uncertainty_enabled = bool(uncertainty_enabled)
        self.uncertainty_q_min = float(uncertainty_q_min)
        self.uncertainty_q_max = uncertainty_q_max
        self.uncertainty_var_floor = float(uncertainty_var_floor)
        self.uncertainty_variance_scale = float(uncertainty_variance_scale)

    @torch.no_grad()
    def evaluate(
        self,
        vol_gt: torch.Tensor,
        bbox: torch.Tensor,
        volume_shape_wdh: Tuple[int, int, int],
        device: torch.device,
    ) -> Tuple[Dict[str, float], torch.Tensor]:
        """Voxelize the current Gaussians and compare with ground truth.

        Returns:
            (metrics_dict, predicted_volume) where predicted volume shape
            is ``(W, D, H)`` matching the GT convention.
        """
        bbox_dev = bbox.to(device)
        ranges = bbox_dev[1] - bbox_dev[0]
        centers = (bbox_dev[0] + bbox_dev[1]) / 2.0

        W, D, H = volume_shape_wdh

        _quats, _scales, opacity, covars, max_scales = activate_gaussian_params(
            self.params.get("quats").data,
            self.params.get("scale_ori").data,
            self.params.get("opacity").data,
            scale_min=self.scale_min,
            scale_max=self.scale_max,
        )
        voxelizer = GaussianMappingVoxelizer(
            vol_dim2=W,
            vol_dim1=D,
            vol_dim0=H,
            tile_size=4,
            eps3d=self.eps3d,
            radius_clip=1.0 if self.primitive_type == "beta" else 3.0,
            primitive_type=self.primitive_type,
        )
        q = None
        if self.uncertainty_enabled and self.params.get("raw_uncertainty") is not None:
            q = activate_uncertainty(
                self.params.get("raw_uncertainty").data,
                q_min=self.uncertainty_q_min,
                q_max=self.uncertainty_q_max,
            )
        fwd = voxelizer(
            self.params.get("means").data,
            covars,
            max_scales,
            opacity,
            ranges,
            centers,
            beta=(
                self.params.get("beta").data
                if self.params.get("beta") is not None
                else None
            ),
            beta_gamma_init=self.beta_gamma_init,
            uncertainty=q,
        )
        pred_vol = fwd.volume
        pred_var = fwd.variance

        if pred_vol.ndim >= 3:
            pred_vol = pred_vol.squeeze()

        # Align dimension order to match GT (W, D, H).
        # The voxelization kernel outputs axes in reversed order relative to GT.
        if pred_vol.ndim == 3:
            pred_vol = pred_vol.permute(2, 1, 0).contiguous()
        if pred_var is not None and pred_var.ndim == 3:
            pred_var = pred_var.permute(2, 1, 0).contiguous()

        gt = vol_gt.to(device)
        if pred_vol.shape != gt.shape:
            slices = tuple(
                slice(0, min(p, g))
                for p, g in zip(pred_vol.shape, gt.shape)
            )
            pred_vol = pred_vol[slices]
            gt = gt[slices]
            if pred_var is not None:
                pred_var = pred_var[slices]

        # Signed kernels can produce values outside [0, 1]; clamp for evaluation.
        pred_vol = pred_vol.clamp(0.0, 1.0)

        m = metric_vol(
            pred_vol, gt, window_size=self.ssim_window
        )
        metrics = {k: float(v.item()) for k, v in m.items()}
        if pred_var is not None:
            pred_std = (
                pred_var * self.uncertainty_variance_scale
                + self.uncertainty_var_floor
            ).clamp_min(0).sqrt()
            corr = _pearson_corr((pred_vol - gt).abs(), pred_std)
            if corr is not None:
                metrics["uncertainty_error_corr"] = corr
            metrics["uncertainty_std_mean"] = float(pred_std.mean().item())
        return metrics, pred_vol
