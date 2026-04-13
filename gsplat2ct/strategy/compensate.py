from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

from gsplat2ct.model.gaussian_params import GaussianParams
from gsplat2ct.ops.data import Config, Data
from gsplat2ct.ops.gs3d_ct_voxelize import GS3DCTVoxelizeCUDA


@dataclass
class CompensateStats:
    sampled_total: int = 0
    sampled_pos: int = 0
    sampled_neg: int = 0
    candidate_total: int = 0
    residual_volume_wdh: Optional[torch.Tensor] = None
    sampled_means_world: Optional[torch.Tensor] = None
    sampled_opacity: Optional[torch.Tensor] = None


class CompensationStrategy:
    """Residual-driven Gaussian compensation strategy.

    This strategy is independent from the existing gsplat-style strategy and can
    run in parallel: at configured steps it voxelizes current Gaussians,
    computes residual against a fixed coarse volume, and injects new signed
    Gaussians at sampled residual voxels.
    """

    def __init__(self, cfg):
        if isinstance(cfg, dict):
            cfg_obj = type("_Cfg", (), cfg)
        else:
            cfg_obj = cfg

        self.enabled = bool(getattr(cfg_obj, "enabled", False))
        self.verbose = bool(getattr(cfg_obj, "verbose", False))
        self.steps = tuple(int(x) for x in getattr(cfg_obj, "steps", []))
        self.diff_threshold = float(getattr(cfg_obj, "diff_threshold", 0.05))
        self.candidate_ratio = float(getattr(cfg_obj, "candidate_ratio", 0.2))
        self.max_new_ratio = float(getattr(cfg_obj, "max_new_ratio", 0.1))
        self.init_scale_multiplier = float(getattr(cfg_obj, "init_scale_multiplier", 0.5))
        self.opacity_scale = float(getattr(cfg_obj, "opacity_scale", 0.3))
        # Sampling weight: p(i) ∝ exp(|residual_i| / T).
        # T=0.1082 gives ~4x odds between diff=0.20 and diff=0.05.
        self.sample_temperature = float(getattr(cfg_obj, "sample_temperature", 0.1082))

        self._voxel_op = GS3DCTVoxelizeCUDA()

    def should_resample(self, step: int) -> bool:
        if not self.enabled:
            return False
        return step in self.steps

    @torch.no_grad()
    def apply(
        self,
        gp: GaussianParams,
        *,
        coarse_volume_wdh: torch.Tensor,
        bbox: torch.Tensor,
        volume_shape_wdh: Tuple[int, int, int],
        scale_min: float,
        scale_max: float,
        step: Optional[int] = None,
    ) -> Tuple[bool, CompensateStats]:
        stats = CompensateStats()
        if not self.enabled:
            return False, stats

        if coarse_volume_wdh is None:
            return False, stats

        device = gp.means.device
        bbox_dev = bbox.to(device)
        ranges = (bbox_dev[1] - bbox_dev[0]).contiguous()
        centers = ((bbox_dev[0] + bbox_dev[1]) / 2.0).contiguous()

        W, D, H = (int(volume_shape_wdh[0]), int(volume_shape_wdh[1]), int(volume_shape_wdh[2]))
        params = gp.to_params()
        self._voxel_op.initialize(params)

        data_obj = Data()
        data_obj.set(ranges=ranges, centers=centers)
        config = Config()
        config.set(
            vol_dim2=W,
            vol_dim1=D,
            vol_dim0=H,
            vol_tile_size=4,
            eps3d=1e-4,
            radius_clip=3.0,
            scale_min=scale_min,
            scale_max=scale_max,
        )
        fwd = self._voxel_op.forward_only(params, data_obj, config)
        pred_vol = fwd.outputs["render_output"]
        if pred_vol.ndim >= 3:
            pred_vol = pred_vol.squeeze()
        if pred_vol.ndim == 3:
            pred_vol = pred_vol.permute(2, 1, 0).contiguous()

        coarse = coarse_volume_wdh.to(device)
        w = min(int(coarse.shape[0]), int(pred_vol.shape[0]))
        d = min(int(coarse.shape[1]), int(pred_vol.shape[1]))
        h = min(int(coarse.shape[2]), int(pred_vol.shape[2]))
        coarse = coarse[:w, :d, :h]
        pred = pred_vol[:w, :d, :h]

        residual = coarse - pred
        stats.residual_volume_wdh = residual.detach()
        candidate_mask = residual.abs() >= self.diff_threshold
        candidate_idx = torch.nonzero(candidate_mask, as_tuple=False)
        stats.candidate_total = int(candidate_idx.shape[0])
        if stats.candidate_total == 0:
            return False, stats

        n_cur = gp.num_gaussians
        n_from_candidates = int(stats.candidate_total * self.candidate_ratio)
        n_from_current = int(n_cur * self.max_new_ratio)
        n_add = min(n_from_candidates, n_from_current)
        if n_add <= 0:
            return False, stats

        candidate_residual = residual[
            candidate_idx[:, 0],
            candidate_idx[:, 1],
            candidate_idx[:, 2],
        ]
        candidate_abs = candidate_residual.abs()

        if self.sample_temperature <= 1e-8:
            # Deterministic fallback: prioritize largest residual magnitudes.
            pick_local = torch.topk(candidate_abs, k=n_add, largest=True).indices
        else:
            # Numerically stable softmax-like sampling weights.
            logits = candidate_abs / float(self.sample_temperature)
            logits = logits - logits.max()
            weights = torch.exp(logits).clamp_min(1e-20)
            pick_local = torch.multinomial(weights, num_samples=n_add, replacement=False)

        picked_idx = candidate_idx.index_select(0, pick_local)

        picked_residual = residual[picked_idx[:, 0], picked_idx[:, 1], picked_idx[:, 2]]
        pos_mask = picked_residual > 0
        neg_mask = picked_residual < 0
        stats.sampled_total = int(picked_idx.shape[0])
        stats.sampled_pos = int(pos_mask.sum().item())
        stats.sampled_neg = int(neg_mask.sum().item())

        if stats.sampled_total == 0:
            return False, stats

        grid_size = torch.tensor([w, d, h], dtype=gp.means.dtype, device=device)
        voxel_size = (ranges / grid_size).contiguous()
        means_new = bbox_dev[0] + (picked_idx.to(gp.means.dtype) + 0.5) * voxel_size
        stats.sampled_means_world = means_new.detach()

        opacity_new = (picked_residual * self.opacity_scale).to(gp.opacity.dtype)
        stats.sampled_opacity = opacity_new.detach()

        n_new = int(means_new.shape[0])
        quats_new = torch.zeros((n_new, 4), dtype=gp.quats.dtype, device=device)
        quats_new[:, 0] = 1.0

        scales_act = torch.sigmoid(gp.scale_ori) * (scale_max - scale_min) + scale_min
        target_scale = scales_act.mean(dim=0) * self.init_scale_multiplier
        target_scale = target_scale.clamp(min=scale_min + 1e-6, max=scale_max - 1e-6)
        scale_ori_new = GaussianParams._inverse_scale_activation(
            target_scale.unsqueeze(0).expand(n_new, 3), scale_min, scale_max
        ).to(gp.scale_ori.dtype)

        gp.means = torch.cat([gp.means, means_new], dim=0).contiguous()
        gp.quats = torch.cat([gp.quats, quats_new], dim=0).contiguous()
        gp.scale_ori = torch.cat([gp.scale_ori, scale_ori_new], dim=0).contiguous()
        gp.opacity = torch.cat([gp.opacity, opacity_new], dim=0).contiguous()

        if self.verbose:
            prefix = f"[Compensate step {step}] " if step is not None else "[Compensate] "
            print(
                f"{prefix}added={stats.sampled_total} "
                f"(pos={stats.sampled_pos}, neg={stats.sampled_neg}), "
                f"candidates={stats.candidate_total}, N={n_cur}->{gp.num_gaussians}"
            )

        return True, stats
