from __future__ import annotations

from typing import Optional, Tuple

import torch

from gsplat2ct.model.gaussian_params import GaussianParams

from .default import StrategyOutput, StrategyTransition, _quat_to_rotmat


class KAMCMCStrategy:
    """Kernel-Agnostic MCMC primitive-count controller.

    The implementation follows the spirit of Beta Splatting's MCMC strategy:
    low-density primitives are relocated from high-density donors, and the
    population is slowly grown by sampling existing primitives as donors.
    It is intentionally independent of the kernel family, so it works for both
    Gaussian and Beta primitives.
    """

    def __init__(self, cfg):
        if isinstance(cfg, dict):
            cfg_obj = type("_Cfg", (), cfg)
        else:
            cfg_obj = cfg

        self.enabled = bool(getattr(cfg_obj, "enabled", True))
        self.absgrad = False
        self.refine_start_iter = int(getattr(cfg_obj, "refine_start_iter", 500))
        self.refine_stop_iter = int(getattr(cfg_obj, "refine_stop_iter", 25000))
        self.refine_every = int(getattr(cfg_obj, "refine_every", 100))
        self.cap_max = int(getattr(cfg_obj, "ka_mcmc_cap_max", getattr(cfg_obj, "max_gaussians", 1000000)))
        self.min_density = float(getattr(cfg_obj, "ka_mcmc_min_density", 0.005))
        self.grow_rate = float(getattr(cfg_obj, "ka_mcmc_grow_rate", 0.05))
        self.noise_scale = float(getattr(cfg_obj, "ka_mcmc_noise_scale", 0.25))

    def should_accumulate_grad_stats(self, step: int) -> bool:
        return False

    def accumulate_grad_stats(self, **kwargs) -> None:
        return None

    def reset_grad_accumulator(self) -> None:
        return None

    def want_refine(self, step: int) -> bool:
        if not self.enabled:
            return False
        if step >= self.refine_stop_iter or step <= self.refine_start_iter:
            return False
        return (step % max(self.refine_every, 1)) == 0

    @torch.no_grad()
    def decide(self, gp: GaussianParams, **kwargs) -> StrategyOutput:
        device = gp.means.device
        z = torch.zeros(gp.num_gaussians, dtype=torch.bool, device=device)
        return StrategyOutput(dupli_mask=z, split_mask=z)

    @torch.no_grad()
    def apply(
        self,
        gp: GaussianParams,
        out: Optional[StrategyOutput] = None,
        *,
        scale_min: float,
        scale_max: float,
        bbox: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
    ) -> Tuple[bool, StrategyTransition]:
        if not self.enabled or gp.num_gaussians == 0:
            return False, StrategyTransition()

        orig_n = gp.num_gaussians
        relocated = self._relocate_dead(gp)
        added = self._sample_add(gp, scale_min=scale_min, scale_max=scale_max)
        transition = StrategyTransition(append_new=added)

        if bbox is not None:
            bbox_dev = bbox.to(device=gp.means.device, dtype=gp.means.dtype)
            keep = ((gp.means >= bbox_dev[0]) & (gp.means <= bbox_dev[1])).all(dim=-1)
            if keep.numel() == gp.num_gaussians and not keep.all():
                gp.means = gp.means[keep].contiguous()
                gp.quats = gp.quats[keep].contiguous()
                gp.scale_ori = gp.scale_ori[keep].contiguous()
                gp.opacity = gp.opacity[keep].contiguous()
                if gp.beta is not None:
                    gp.beta = gp.beta[keep].contiguous()
                transition.prune_keep = keep.detach().clone()

        changed = relocated > 0 or added > 0 or transition.prune_keep is not None
        if changed:
            prefix = f"[KA-MCMC step {step}] " if step is not None else "[KA-MCMC] "
            pruned = 0 if transition.prune_keep is None else int((~transition.prune_keep).sum().item())
            print(
                f"{prefix}relocate={relocated}, add={added}, prune={pruned}, "
                f"N: {orig_n} -> {gp.num_gaussians}"
            )
        return changed, transition

    def _sampling_probs(self, gp: GaussianParams) -> torch.Tensor:
        weights = gp.opacity.detach().abs().to(torch.float32).clamp_min(1e-12)
        return weights / weights.sum().clamp_min(1e-12)

    @torch.no_grad()
    def _relocate_dead(self, gp: GaussianParams) -> int:
        dead = gp.opacity.detach().abs() <= self.min_density
        n_dead = int(dead.sum().item())
        if n_dead <= 0 or n_dead >= gp.num_gaussians:
            return 0
        dead_idx = torch.where(dead)[0]
        probs = self._sampling_probs(gp)
        probs = probs.masked_fill(dead, 0.0)
        probs = probs / probs.sum().clamp_min(1e-12)
        donor = torch.multinomial(probs, n_dead, replacement=True)

        gp.means[dead_idx] = gp.means[donor]
        gp.quats[dead_idx] = gp.quats[donor]
        gp.scale_ori[dead_idx] = gp.scale_ori[donor]
        donor_opa = gp.opacity[donor] * 0.5
        gp.opacity[donor] = donor_opa
        gp.opacity[dead_idx] = donor_opa.abs().clamp_min(self.min_density).to(gp.opacity.dtype)
        if gp.beta is not None:
            gp.beta[dead_idx] = gp.beta[donor]
        return n_dead

    @torch.no_grad()
    def _sample_add(self, gp: GaussianParams, *, scale_min: float, scale_max: float) -> int:
        if gp.num_gaussians >= self.cap_max:
            return 0
        target = min(self.cap_max, int(gp.num_gaussians * (1.0 + self.grow_rate)))
        n_add = max(0, target - gp.num_gaussians)
        if n_add <= 0:
            return 0

        probs = self._sampling_probs(gp)
        donor = torch.multinomial(probs, n_add, replacement=True)
        scales = torch.sigmoid(gp.scale_ori[donor]) * (scale_max - scale_min) + scale_min
        quats = gp.quats[donor] / gp.quats[donor].norm(dim=-1, keepdim=True).clamp_min(1e-12)
        rot = _quat_to_rotmat(quats)
        noise = torch.randn((n_add, 3), device=gp.means.device, dtype=gp.means.dtype)
        offsets = torch.einsum("nij,nj,nj->ni", rot, scales * self.noise_scale, noise)

        means_new = gp.means[donor] + offsets
        quats_new = gp.quats[donor]
        scale_ori_new = gp.scale_ori[donor]
        opa_new = gp.opacity[donor] * 0.5
        gp.opacity[donor] = opa_new

        gp.means = torch.cat([gp.means, means_new], dim=0).contiguous()
        gp.quats = torch.cat([gp.quats, quats_new], dim=0).contiguous()
        gp.scale_ori = torch.cat([gp.scale_ori, scale_ori_new], dim=0).contiguous()
        gp.opacity = torch.cat([gp.opacity, opa_new], dim=0).contiguous()
        if gp.beta is not None:
            gp.beta = torch.cat([gp.beta, gp.beta[donor]], dim=0).contiguous()
        return n_add


MCMCStrategy = KAMCMCStrategy

__all__ = ["KAMCMCStrategy", "MCMCStrategy"]
