"""Evaluators for CT reconstruction quality.

``ProjectionEvaluator`` renders all test views and computes 2-D metrics.
``VolumeEvaluator`` voxelizes the full bounding box and computes 3-D metrics.

Both use the forward-only operators (``GS3DCTRasterizeCUDA``,
``GS3DCTVoxelizeCUDA``) so they never allocate backward buffers.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.utils.data

from gsplat2ct.ops.data import Params, Data, Config, ForwardResult
from gsplat2ct.ops.gs3d_ct_rasterize import GS3DCTRasterizeCUDA
from gsplat2ct.ops.gs3d_ct_voxelize import GS3DCTVoxelizeCUDA
from gsplat2ct.eval.metric import metric_proj, metric_vol


class _ParamWrapper:
    def __init__(self, data: torch.Tensor):
        self.data = data


class ProjectionEvaluator:
    """Evaluate 2-D projection quality over a dataset split.

    Args:
        params: Current Gaussian parameters (``Params`` container).
        scale_min: ParameterTransform lower bound (must match training).
        scale_max: ParameterTransform upper bound.
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
    ):
        self.params = params
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.eps_blur = eps_blur
        self.ssim_window = ssim_window
        self._op = GS3DCTRasterizeCUDA()
        self._op.initialize(params)

    @torch.no_grad()
    def evaluate(
        self,
        dataset: torch.utils.data.Dataset,
        device: torch.device,
        max_vis: int = 5,
    ) -> Tuple[Dict[str, float], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Run evaluation on *dataset* and return aggregate metrics plus
        up to *max_vis* (pred, target) image pairs for visualization."""
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0
        )

        psnr_vals, ssim_vals = [], []
        vis_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        import numpy as np
        vis_indices = set(
            np.linspace(0, len(dataset) - 1, max_vis, dtype=int).tolist()
        )

        for idx, batch in enumerate(loader):
            c2w = batch["camtoworld"].to(device).contiguous()
            K = batch["K"].to(device).contiguous()
            gt = batch["image"].to(device).contiguous()

            viewmats = torch.linalg.inv(c2w).contiguous()
            C = viewmats.shape[0]
            N = self.params.get("means").data.shape[0]

            opacity_raw = self.params.get("opacity").data
            opacity_2d = opacity_raw.unsqueeze(0).expand(C, -1).contiguous()

            rast_params = Params()
            rast_params.add_param("means", self.params.get("means"))
            rast_params.add_param("quats", self.params.get("quats"))
            rast_params.add_param("scale_ori", self.params.get("scale_ori"))
            rast_params.add_param("opacity", _ParamWrapper(opacity_2d))

            data_obj = Data()
            data_obj.set(viewmats=viewmats, Ks=K)

            img_h, img_w = gt.shape[1], gt.shape[2]
            cfg = Config()
            cfg.set(
                img_dim1=img_w,
                img_dim0=img_h,
                tile_size=16,
                eps_blur=self.eps_blur,
                scale_min=self.scale_min,
                scale_max=self.scale_max,
            )

            fwd = self._op.forward_only(rast_params, data_obj, cfg)
            pred = fwd.outputs["render_output"]

            gt_squeezed = gt.squeeze(-1)
            m = metric_proj(
                pred, gt_squeezed, window_size=self.ssim_window
            )
            psnr_vals.append(float(m["psnr"].item()))
            ssim_vals.append(float(m["ssim"].item()))

            if idx in vis_indices:
                vis_pairs.append(
                    (pred[0].cpu(), gt_squeezed[0].cpu())
                )

        metrics = {
            "psnr": float(sum(psnr_vals) / max(len(psnr_vals), 1)),
            "ssim": float(sum(ssim_vals) / max(len(ssim_vals), 1)),
        }
        return metrics, vis_pairs


class VolumeEvaluator:
    """Evaluate 3-D volume reconstruction quality.

    Args:
        params: Current Gaussian parameters (``Params`` container).
        scale_min: ParameterTransform lower bound.
        scale_max: ParameterTransform upper bound.
        ssim_window: SSIM kernel size for 3-D SSIM.
    """

    def __init__(
        self,
        params: Params,
        *,
        scale_min: float = 0.001,
        scale_max: float = 1.0,
        ssim_window: int = 11,
    ):
        self.params = params
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.ssim_window = ssim_window
        self._op = GS3DCTVoxelizeCUDA()
        self._op.initialize(params)

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

        data_obj = Data()
        data_obj.set(ranges=ranges, centers=centers)

        cfg = Config()
        cfg.set(
            vol_dim2=W,
            vol_dim1=D,
            vol_dim0=H,
            vol_tile_size=4,
            eps3d=1e-4,
            radius_clip=3.0,
            scale_min=self.scale_min,
            scale_max=self.scale_max,
        )

        fwd = self._op.forward_only(self.params, data_obj, cfg)
        pred_vol = fwd.outputs["render_output"]

        if pred_vol.ndim >= 3:
            pred_vol = pred_vol.squeeze()

        # Align dimension order to match GT (W, D, H).
        # The voxelization kernel outputs axes in reversed order relative to GT.
        if pred_vol.ndim == 3:
            pred_vol = pred_vol.permute(2, 1, 0).contiguous()

        gt = vol_gt.to(device)
        if pred_vol.shape != gt.shape:
            slices = tuple(
                slice(0, min(p, g))
                for p, g in zip(pred_vol.shape, gt.shape)
            )
            pred_vol = pred_vol[slices]
            gt = gt[slices]

        m = metric_vol(
            pred_vol, gt, window_size=self.ssim_window
        )
        metrics = {k: float(v.item()) for k, v in m.items()}
        return metrics, pred_vol
