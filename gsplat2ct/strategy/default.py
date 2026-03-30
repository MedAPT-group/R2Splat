from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from gsplat2ct.model.gaussian_params import GaussianParams


def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """q: (N,4) wxyz -> (N,3,3). Assumes unit quats."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = torch.stack(
        [
            1 - 2 * (y2 + z2),
            2 * (xy - wz),
            2 * (xz + wy),
            2 * (xy + wz),
            1 - 2 * (x2 + z2),
            2 * (yz - wx),
            2 * (xz - wy),
            2 * (yz + wx),
            1 - 2 * (x2 + y2),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)
    return R


@dataclass
class StrategyOutput:
    prune_mask: torch.Tensor  # (N,) bool
    dupli_mask: torch.Tensor  # (N,) bool
    split_mask: torch.Tensor  # (N,) bool


class DefaultStrategy:
    def __init__(self, cfg):
        if isinstance(cfg, dict):
            cfg_obj = type("_Cfg", (), cfg)
        else:
            cfg_obj = cfg

        self.enabled = bool(getattr(cfg_obj, "enabled", False))
        self.absgrad = bool(getattr(cfg_obj, "absgrad", False))
        self.refine_start_iter = int(getattr(cfg_obj, "refine_start_iter", 0))
        self.refine_stop_iter = int(getattr(cfg_obj, "refine_stop_iter", 1 << 30))
        self.refine_every = int(getattr(cfg_obj, "refine_every", 100))

        self.prune_opa_abs = float(getattr(cfg_obj, "prune_opa_abs", 0.0))
        self.grow_grad2d = float(getattr(cfg_obj, "grow_grad2d", 0.0))
        self.grow_scale3d = float(getattr(cfg_obj, "grow_scale3d", 0.0))

        self.min_remaining = int(getattr(cfg_obj, "min_remaining", 0))
        self.max_gaussians = int(getattr(cfg_obj, "max_gaussians", 0))

        self.split_scales_div = float(getattr(cfg_obj, "split_scales_div", 1.6))

    def want_refine(self, step: int) -> bool:
        if not self.enabled:
            return False
        if step < self.refine_start_iter or step >= self.refine_stop_iter:
            return False
        return (step % max(self.refine_every, 1)) == 0

    @torch.no_grad()
    def decide(
        self,
        gp: GaussianParams,
        *,
        v_means2d: torch.Tensor,
        v_means2d_abs: Optional[torch.Tensor],
        scale_min: float,
        scale_max: float,
    ) -> StrategyOutput:
        N = gp.num_gaussians
        device = gp.means.device

        # --- prune by opacity ---
        opacity = gp.opacity
        prune = opacity.abs() < self.prune_opa_abs

        # --- grad magnitude signal ---
        if self.absgrad and v_means2d_abs is not None and v_means2d_abs.numel() > 0:
            g = v_means2d_abs
        else:
            g = v_means2d.abs()
        # g: (C,N,2) -> (N,)
        grad2d = g.norm(dim=-1).mean(dim=0)
        is_grad_high = grad2d > self.grow_grad2d

        # --- scale_3d = max activated scale ---
        scales = torch.sigmoid(gp.scale_ori) * (scale_max - scale_min) + scale_min
        scale3d = scales.max(dim=-1).values
        is_large = scale3d > self.grow_scale3d

        dupli = is_grad_high & ~is_large
        split = is_grad_high & is_large

        # Enforce limits: min_remaining / max_gaussians (simple, conservative)
        if self.min_remaining > 0:
            # Never prune below min_remaining
            if (N - prune.sum().item()) < self.min_remaining:
                # Keep the highest-opacity ones
                keep_n = self.min_remaining
                topk = torch.topk(opacity.abs(), k=keep_n, largest=True).indices
                new_prune = torch.ones(N, dtype=torch.bool, device=device)
                new_prune[topk] = False
                prune = new_prune

        if self.max_gaussians > 0:
            current = N - prune.sum().item()
            if current >= self.max_gaussians:
                dupli.zero_()
                split.zero_()

        return StrategyOutput(prune_mask=prune, dupli_mask=dupli, split_mask=split)

    @torch.no_grad()
    def apply(
        self,
        gp: GaussianParams,
        out: StrategyOutput,
        *,
        scale_min: float,
        scale_max: float,
    ) -> bool:
        """Apply prune/duplicate/split to GaussianParams. Returns True if topology changed."""
        if not self.enabled:
            return False

        changed = False
        device = gp.means.device

        # ---- prune ----
        if out.prune_mask.any():
            keep = ~out.prune_mask
            gp.means = gp.means[keep].contiguous()
            gp.quats = gp.quats[keep].contiguous()
            gp.scale_ori = gp.scale_ori[keep].contiguous()
            gp.opacity = gp.opacity[keep].contiguous()
            changed = True

        N = gp.num_gaussians
        if N == 0:
            return changed

        # ---- duplicate ----
        if out.dupli_mask.any():
            sel = torch.where(out.dupli_mask[:N])[0]
            gp.means = torch.cat([gp.means, gp.means[sel]], dim=0).contiguous()
            gp.quats = torch.cat([gp.quats, gp.quats[sel]], dim=0).contiguous()
            gp.scale_ori = torch.cat([gp.scale_ori, gp.scale_ori[sel]], dim=0).contiguous()
            gp.opacity = torch.cat([gp.opacity, gp.opacity[sel]], dim=0).contiguous()
            changed = True

        # ---- split ----
        if out.split_mask.any():
            sel = torch.where(out.split_mask[: gp.num_gaussians])[0]
            if sel.numel() > 0:
                quats = gp.quats[sel]
                quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                scales = torch.sigmoid(gp.scale_ori[sel]) * (scale_max - scale_min) + scale_min
                R = _quat_to_rotmat(quats)
                noise = torch.randn((2, sel.numel(), 3), device=device, dtype=scales.dtype)
                samples = torch.einsum("nij,nj,bnj->bni", R, scales, noise)
                means_split = (gp.means[sel].unsqueeze(0) + samples).reshape(-1, 3)

                target_scales = (scales / self.split_scales_div).clamp_min(scale_min + 1e-6)
                scale_ori_split = GaussianParams._inverse_scale_activation(
                    target_scales.repeat(2, 1), scale_min, scale_max
                )

                quats_split = gp.quats[sel].repeat(2, 1)
                opa_split = gp.opacity[sel].repeat(2)

                gp.means = torch.cat([gp.means, means_split], dim=0).contiguous()
                gp.quats = torch.cat([gp.quats, quats_split], dim=0).contiguous()
                gp.scale_ori = torch.cat([gp.scale_ori, scale_ori_split], dim=0).contiguous()
                gp.opacity = torch.cat([gp.opacity, opa_split], dim=0).contiguous()
                changed = True

        return changed

