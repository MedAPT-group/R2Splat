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
    dupli_mask: torch.Tensor  # (N,) bool
    split_mask: torch.Tensor  # (N,) bool


@dataclass
class StrategyTransition:
    """Topology transition metadata for optimizer-state migration.

    Fields represent sequential edits applied in ``apply``:
    1) duplicate selected parents (append new children)
    2) split selected parents (remove parents, append 2x children)
    3) prune low-opacity points
    """

    duplicate_sel: Optional[torch.Tensor] = None
    split_keep_parent: Optional[torch.Tensor] = None
    split_children: int = 0
    prune_keep: Optional[torch.Tensor] = None


class DefaultStrategy:
    """Densification aligned with gsplat ``DefaultStrategy`` / 3DGS.

    While ``enabled`` and ``step < refine_stop_iter``, each backward adds to
    ``grad2d`` / ``count`` like gsplat ``_update_state``: only (camera, Gaussian)
    pairs with ``radii_2d > 0``, gradient scaled by
    ``(width/2)*n_cameras`` and ``(height/2)*n_cameras``, norm accumulated.

    Refine when ``step > refine_start_iter`` and ``step % refine_every == 0``
    (strict ``>``, same as gsplat).

    ``absgrad``: use ``v_means2d_abs`` when non-empty; else ``v_means2d``.
    """

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

        self._grad2d: Optional[torch.Tensor] = None
        self._count: Optional[torch.Tensor] = None

    def should_accumulate_grad_stats(self, step: int) -> bool:
        if not self.enabled:
            return False
        return step < self.refine_stop_iter

    def want_refine(self, step: int) -> bool:
        if not self.enabled:
            return False
        if step >= self.refine_stop_iter:
            return False
        if step <= self.refine_start_iter:
            return False
        return (step % max(self.refine_every, 1)) == 0

    def reset_grad_accumulator(self) -> None:
        if self._grad2d is not None:
            self._grad2d.zero_()
        if self._count is not None:
            self._count.zero_()

    @torch.no_grad()
    def accumulate_grad_stats(
        self,
        *,
        step: int,
        v_means2d: torch.Tensor,
        v_means2d_abs: Optional[torch.Tensor],
        radii_2d: torch.Tensor,
        img_w: int,
        img_h: int,
        n_cameras: int,
    ) -> None:
        if not self.should_accumulate_grad_stats(step):
            return

        C, N, two = v_means2d.shape
        if two != 2:
            raise ValueError(f"v_means2d must end with dim 2, got {v_means2d.shape}")
        if radii_2d.shape[0] != C or radii_2d.shape[1] != N:
            raise ValueError(
                f"radii_2d shape {tuple(radii_2d.shape)} != (C,N)=({C},{N})"
            )

        if self.absgrad and v_means2d_abs is not None and v_means2d_abs.numel() > 0:
            g = v_means2d_abs.detach().clone()
        else:
            g = v_means2d.detach().clone()

        g[..., 0] *= (float(img_w) / 2.0) * float(n_cameras)
        g[..., 1] *= (float(img_h) / 2.0) * float(n_cameras)

        sel = radii_2d > 0
        if not sel.any():
            return

        gs_ids = torch.where(sel)[1]
        grads_sel = g[sel]
        contrib = grads_sel.norm(dim=-1).to(torch.float32)

        if self._grad2d is None or self._grad2d.numel() != N:
            dev = v_means2d.device
            self._grad2d = torch.zeros(N, device=dev, dtype=torch.float32)
            self._count = torch.zeros(N, device=dev, dtype=torch.float32)

        self._grad2d.index_add_(0, gs_ids, contrib)
        self._count.index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32))

    @torch.no_grad()
    def decide(
        self,
        gp: GaussianParams,
        *,
        scale_min: float,
        scale_max: float,
        grow_scale3d_abs: Optional[float] = None,
    ) -> StrategyOutput:
        N = gp.num_gaussians
        device = gp.means.device

        opacity = gp.opacity

        if self._grad2d is None or self._grad2d.numel() != N:
            accum = torch.zeros(N, device=device, dtype=torch.float32)
        else:
            accum = (self._grad2d / self._count.clamp_min(1.0)).to(
                device=device, dtype=gp.means.dtype
            )

        is_grad_high = accum > self.grow_grad2d

        scales = torch.sigmoid(gp.scale_ori) * (scale_max - scale_min) + scale_min
        scale3d = scales.max(dim=-1).values
        scale_threshold = self.grow_scale3d if grow_scale3d_abs is None else float(grow_scale3d_abs)
        is_large = scale3d > scale_threshold

        dupli = is_grad_high & ~is_large
        split = is_grad_high & is_large

        if self.max_gaussians > 0 and N >= self.max_gaussians:
            dupli = torch.zeros_like(dupli)
            split = torch.zeros_like(split)

        return StrategyOutput(dupli_mask=dupli, split_mask=split)

    @torch.no_grad()
    def apply(
        self,
        gp: GaussianParams,
        out: StrategyOutput,
        *,
        scale_min: float,
        scale_max: float,
        bbox: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
    ) -> Tuple[bool, StrategyTransition]:
        """Apply prune/duplicate/split and return transition metadata."""
        if not self.enabled:
            return False, StrategyTransition()

        changed = False
        transition = StrategyTransition()
        device = gp.means.device
        orig_n = gp.num_gaussians

        dupli_mask = out.dupli_mask
        split_mask = out.split_mask.clone()
        pruned_n = dupli_n = split_n = 0

        N = gp.num_gaussians
        if N == 0:
            return False, transition

        # --- grow: duplicate then split (gsplat order) ---
        dupli_n = int(dupli_mask.sum().item())
        if dupli_mask.any():
            sel = torch.where(dupli_mask[:N])[0]
            if sel.numel() > 0:
                transition.duplicate_sel = sel.detach().clone()
                means_new = gp.means[sel]
                quats_new = gp.quats[sel]
                scale_ori_new = gp.scale_ori[sel]
                opa_new = gp.opacity[sel] * 0.5
                gp.opacity[sel] = opa_new
                gp.means = torch.cat([gp.means, means_new], dim=0).contiguous()
                gp.quats = torch.cat([gp.quats, quats_new], dim=0).contiguous()
                gp.scale_ori = torch.cat([gp.scale_ori, scale_ori_new], dim=0).contiguous()
                gp.opacity = torch.cat([gp.opacity, opa_new], dim=0).contiguous()
                self._cat_running_stats_duplicate(sel)
                split_mask = torch.cat(
                    [
                        split_mask,
                        torch.zeros(sel.numel(), dtype=torch.bool, device=device),
                    ],
                    dim=0,
                )
                changed = True

        N = gp.num_gaussians
        split_n = int(split_mask.sum().item())
        if split_mask.any():
            sel = torch.where(split_mask[:N])[0]
            if sel.numel() > 0:
                transition.split_children = int(sel.numel() * 2)
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
                opa_split = (gp.opacity[sel] * 0.5).repeat(2)

                keep_parent = torch.ones(gp.num_gaussians, dtype=torch.bool, device=device)
                keep_parent[sel] = False
                transition.split_keep_parent = keep_parent.detach().clone()
                gp.means = torch.cat([gp.means[keep_parent], means_split], dim=0).contiguous()
                gp.quats = torch.cat([gp.quats[keep_parent], quats_split], dim=0).contiguous()
                gp.scale_ori = torch.cat(
                    [gp.scale_ori[keep_parent], scale_ori_split], dim=0
                ).contiguous()
                gp.opacity = torch.cat([gp.opacity[keep_parent], opa_split], dim=0).contiguous()
                self._running_stats_split(keep_parent, sel)
                changed = True

        # --- prune last: from current opacities (gsplat _prune_gs), after grow ---
        pruned_n = 0
        prune = gp.opacity.abs() < self.prune_opa_abs
        if bbox is not None:
            bbox_dev = bbox.to(device=device, dtype=gp.means.dtype)
            mins = bbox_dev[0]
            maxs = bbox_dev[1]
            prune_bbox = ((gp.means < mins) | (gp.means > maxs)).any(dim=-1)
            prune = prune | prune_bbox
        if self.min_remaining > 0:
            n = gp.num_gaussians
            if (n - int(prune.sum().item())) < self.min_remaining:
                keep_n = self.min_remaining
                topk = torch.topk(gp.opacity.abs(), k=min(keep_n, n), largest=True).indices
                new_prune = torch.ones(n, dtype=torch.bool, device=device)
                new_prune[topk] = False
                prune = new_prune
        pruned_n = int(prune.sum().item())
        if prune.any():
            keep = ~prune
            transition.prune_keep = keep.detach().clone()
            gp.means = gp.means[keep].contiguous()
            gp.quats = gp.quats[keep].contiguous()
            gp.scale_ori = gp.scale_ori[keep].contiguous()
            gp.opacity = gp.opacity[keep].contiguous()
            self._resize_running_stats(keep)
            changed = True

        if pruned_n > 0 or dupli_n > 0 or split_n > 0:
            prefix = f"[Strategy step {step}] " if step is not None else "[Strategy] "
            print(
                f"{prefix}split={split_n}, duplicate={dupli_n}, prune={pruned_n}, "
                f"N: {orig_n} -> {gp.num_gaussians}"
            )

        return changed, transition

    def _resize_running_stats(self, keep: torch.Tensor) -> None:
        if self._grad2d is None or self._grad2d.numel() != keep.numel():
            self._grad2d = None
            self._count = None
            return
        self._grad2d = self._grad2d[keep].contiguous()
        self._count = self._count[keep].contiguous()

    def _cat_running_stats_duplicate(self, sel: torch.Tensor) -> None:
        if self._grad2d is None or self._count is None:
            return
        n_new = int(sel.numel())
        if n_new <= 0:
            return
        z_grad = torch.zeros(n_new, device=self._grad2d.device, dtype=self._grad2d.dtype)
        z_count = torch.zeros(n_new, device=self._count.device, dtype=self._count.dtype)
        self._grad2d = torch.cat([self._grad2d, z_grad], dim=0).contiguous()
        self._count = torch.cat([self._count, z_count], dim=0).contiguous()

    def _running_stats_split(
        self, keep_parent: torch.Tensor, sel: torch.Tensor
    ) -> None:
        if self._grad2d is None or self._count is None:
            return
        kept_grad = self._grad2d[keep_parent].contiguous()
        kept_count = self._count[keep_parent].contiguous()
        n_children = int(sel.numel()) * 2
        if n_children <= 0:
            self._grad2d = kept_grad
            self._count = kept_count
            return
        z_grad = torch.zeros(n_children, device=kept_grad.device, dtype=kept_grad.dtype)
        z_count = torch.zeros(n_children, device=kept_count.device, dtype=kept_count.dtype)
        self._grad2d = torch.cat([kept_grad, z_grad], dim=0).contiguous()
        self._count = torch.cat([kept_count, z_count], dim=0).contiguous()
