"""
GS3DCT K-FAC (Empirical-Fisher) optimizer.

This optimizer approximates the per-Gaussian Hessian by the EMA of the
gradient outer product

    F_k(t) = alpha * F_k(t-1) + (1 - alpha) * g_k(t) g_k(t)^T

and performs a damped Gauss-Newton / natural-gradient step

    delta_k = -lr * ( F_k + lambda_k * I )^{-1}  g_k,

where ``lambda_k`` is either a fixed per-group Tikhonov term (``damping_mode:
absolute``) or ``lambda_rel * trace(F_k)/d + lambda_abs`` (``damping_mode:
relative``); see `KFACSecondOrderConfig`.

with optional eigenvalue-floor clipping on the damped matrix and a
per-element update magnitude cap.

Two decoupling modes are supported:

- ``mode = "per_gaussian"``: one (D x D) Fisher per Gaussian, with
  D = sum(param_dims) = 11 for GS3DCT. This keeps *within-Gaussian*
  cross-parameter curvature (e.g. means <-> quats).

- ``mode = "per_param_group"``: one (d x d) Fisher per parameter group
  per Gaussian (e.g. (N,3,3) for means). Cheaper, converges faster
  early on (each group fills its d-dim subspace with only d ~ 4 EMA
  steps), but discards cross-parameter curvature within a Gaussian.

Why Empirical Fisher instead of true Hessian?
- F is always PSD, so F + lambda*I is always PD. No eigenvalue sign flips,
  no need to take absolute value of diagonal.
- Free to compute: only requires the first-order gradient.
- Because each Gaussian's parameters mostly act independently across
  Gaussians in CT reconstruction, a block-diagonal (per-Gaussian)
  approximation captures most of the relevant curvature.

Known limitations (Kunstner et al. 2019):
- Empirical Fisher != true Fisher, especially far from the optimum.
  It reduces to a rescaling of the gradient by its own outer product,
  which is a conservative but reliable preconditioner.
- With mini-batch gradients we accumulate (E[g_batch])(E[g_batch])^T,
  which is sample-averaged-then-outer rather than sample-outer-averaged.
  This is the standard, cheap approximation used by Shampoo / K-FAC.

Author: Dy164
Date: 2026-04-23
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from gsplat2ct.ops.data import Params, GradResult, HessResult
from .base import (
    HessianMode,
    NewtonOptimizerBase,
    SecondOrderConfig,
    SignalStrategy,
)


# ============================================================================
# Constants
# ============================================================================

PARAM_NAMES: List[str] = ["means", "quats", "scale_ori", "opacity"]
PARAM_DIMS: List[int] = [3, 4, 3, 1]

# Unified "whole-Gaussian" bucket name for per_gaussian mode.
_KFAC_FULL_KEY = "kfac_full"


def _renormalize_quats(params: Params, eps: float = 1e-12) -> None:
    """Project quaternions back to the unit sphere in-place."""
    p = params.get("quats")
    if p is None:
        return
    q = p.data
    if q.dim() != 2 or q.shape[-1] != 4:
        return
    q_norm = q.norm(dim=-1, keepdim=True).clamp_min(eps)
    q.div_(q_norm)


# ============================================================================
# Config
# ============================================================================

@dataclass
class KFACSecondOrderConfig(SecondOrderConfig):
    """Configuration for the K-FAC (empirical-Fisher) optimizer.

    Per-parameter quantities are stored in small dicts keyed by parameter
    group name (``means`` / ``quats`` / ``scale_ori`` / ``opacity``):

    - ``lr_map``:         step size for each group.
    - ``damping_map``:    absolute (Tikhonov) part ``lambda_abs`` on the Fisher
                          diagonal, or the sole damping when ``damping_mode`` is
                          ``"absolute"``.
    - ``damping_mode``:   ``"absolute"`` -> ``F + lambda*I`` with
                          ``lambda = lambda_abs`` (after any topology / schedule
                          scaling). ``"relative"`` -> each block uses
                          ``lambda = lambda_rel * trace(F)/d + lambda_abs`` so
                          the regularizer tracks the Fisher scale.
    - ``damping_rel_map``: dimensionless ``lambda_rel`` per group (``relative`` mode
                          only; ignored in ``absolute`` mode).
    - ``ema_decay_map``:  EMA decay factor alpha in [0, 1) for the Fisher
                          update. Larger = more smoothing.
    - ``eig_clip_map``:   floor applied to eigenvalues of ``F + lambda*I``
                          before inverting. Set <= 0 to disable.
    - ``max_update_map``: hard cap on the absolute per-element update
                          magnitude. Set to ``inf`` to disable.

    Attributes:
        mode: ``"per_gaussian"`` or ``"per_param_group"``.
        bias_correction: whether to apply ``F / (1 - alpha^t)`` bias
            correction (Adam-style).
        min_step_for_fisher: number of initial steps before the Fisher
            estimate is trusted; during these warmup steps the update
            falls back to a damping-only step (``delta = -g / lambda``)
            to avoid ill-conditioned solves on a rank-deficient Fisher.
        warmup_damping_floor: when > 0 and ``warmup_match_fisher_scale`` is
            false, warmup uses ``delta = -g / max(lambda_abs, floor)``. When
            ``warmup_match_fisher_scale`` is true, per-row ``lambda`` is
            optionally clamped from below by this floor.
        warmup_match_fisher_scale: if true (and ``damping_mode`` is
            ``relative``), warmup applies the same Tikhonov law as the solve
            but with an instantaneous mean curvature ``||g||^2/d`` in place of
            ``trace(F)/d``, so ``lambda`` matches the scale of a one-sample
            empirical Fisher before the EMA has mass.
        verbose: emit per-step quantile summaries.
    """

    mode: str = "per_param_group"

    # ``absolute``: F + (lambda_abs * I). ``relative``: F + (lambda_rel*trace/d +
    # lambda_abs) * I. Warmup: see ``warmup_match_fisher_scale`` /
    # ``warmup_damping_floor``.
    damping_mode: str = "absolute"
    damping_rel_map: Dict[str, float] = field(default_factory=dict)

    warmup_damping_floor: float = 0.0
    warmup_match_fisher_scale: bool = False

    ema_decay_map: Dict[str, float] = field(default_factory=dict)
    grad_ema_decay_map: Dict[str, float] = field(default_factory=dict)
    eig_clip_map: Dict[str, float] = field(default_factory=dict)
    max_update_map: Dict[str, float] = field(default_factory=dict)

    bias_correction: bool = True
    min_step_for_fisher: int = 0

    def get_ema_decay(self, name: str, default: float = 0.95) -> float:
        return float(self.ema_decay_map.get(name, default))

    def set_ema_decay(self, name: str, value: float) -> None:
        self.ema_decay_map[name] = float(value)

    def get_grad_ema_decay(self, name: str, default: float = 0.9) -> float:
        return float(self.grad_ema_decay_map.get(name, default))

    def set_grad_ema_decay(self, name: str, value: float) -> None:
        self.grad_ema_decay_map[name] = float(value)

    def get_eig_clip(self, name: str, default: float = 0.0) -> float:
        return float(self.eig_clip_map.get(name, default))

    def set_eig_clip(self, name: str, value: float) -> None:
        self.eig_clip_map[name] = float(value)

    def get_max_update(self, name: str, default: float = float("inf")) -> float:
        return float(self.max_update_map.get(name, default))

    def set_max_update(self, name: str, value: float) -> None:
        self.max_update_map[name] = float(value)

    def get_damping_rel(self, name: str, default: float = 0.1) -> float:
        return float(self.damping_rel_map.get(name, default))

    def set_damping_rel(self, name: str, value: float) -> None:
        self.damping_rel_map[name] = float(value)


# ============================================================================
# Optimizer
# ============================================================================

class GS3DCTKFACOptimizer(NewtonOptimizerBase):
    """Block-diagonal empirical-Fisher (K-FAC-style) optimizer for GS3DCT.

    Reuses ``NewtonOptimizerBase`` so that topology transitions, damping
    reanneal, etc. all work out of the box. The Fisher state is stored
    inside ``self._moments`` under the keys:

    - per_param_group:   ``kfac_<name>`` -> dict("F": (N, d, d), "beta_prod": scalar)
    - per_gaussian:      ``kfac_full``   -> dict("F": (N, D, D), "beta_prod": scalar)

    Because ``self._moments`` is already migrated by
    ``apply_topology_transition``, newly-added Gaussians automatically
    get zero Fisher (equivalent to a fresh damping-only step) and pruned
    ones are correctly dropped.

    This class signals to the training loop that it does not need the
    true Hessian via the ``needs_hessian = False`` class attribute; the
    driver may therefore skip the expensive second-order backward.
    """

    # Advertise to the training loop that forward_backward1 is sufficient.
    needs_hessian: bool = False

    def __init__(self, params: Params, config: KFACSecondOrderConfig):
        if config.mode not in ("per_gaussian", "per_param_group"):
            raise ValueError(
                f"Unknown KFAC mode {config.mode!r}. "
                "Expected 'per_gaussian' or 'per_param_group'."
            )
        # hess_mode is unused but kept consistent with the parent class API.
        config.hess_mode = HessianMode.BLOCK
        super().__init__(
            params, config, PARAM_NAMES, PARAM_DIMS, cross_hess_names=[],
        )
        self.config: KFACSecondOrderConfig = config

        # Per-group offset into the concatenated 11-dim vector (per_gaussian mode).
        self._offsets: Dict[str, slice] = {}
        off = 0
        for name, d in zip(PARAM_NAMES, PARAM_DIMS):
            self._offsets[name] = slice(off, off + d)
            off += d
        self.total_dim = off  # 11

    # ---- Gradient intake ---------------------------------------------------

    def add_grad_hess(
        self, grad_result: GradResult, hess_result: HessResult,
    ) -> None:
        """Ignore the true Hessian; only gradients are used."""
        self.add_grad(grad_result)

    # ---- Helpers -----------------------------------------------------------

    @staticmethod
    def _expand_grad(g: torch.Tensor) -> torch.Tensor:
        """Turn (N,) opacity-like gradient into (N, 1); leave (N, d) alone."""
        if g.dim() == 1:
            return g.unsqueeze(-1)
        return g

    @staticmethod
    def _solve_with_eig_clip(
        F: torch.Tensor,       # (..., d, d), already symmetric (+ damping)
        g: torch.Tensor,       # (..., d)
        eig_floor: float,
    ) -> torch.Tensor:
        """Solve ``F x = g`` with an eigenvalue-floor on ``F``.

        ``F`` must already be symmetric PSD with damping applied. When
        ``eig_floor > 0``, eigenvalues smaller than ``eig_floor`` are
        clamped upward before inversion. This upper-bounds the spectral
        norm of the inverse to ``1 / eig_floor``, which directly caps the
        maximum per-Gaussian update amplification.

        The eigendecomposition branch is used uniformly (including d==1)
        so that behavior is consistent across parameter dimensions.
        """
        # Symmetrize to cancel floating-point asymmetry from accumulating gg^T.
        F_sym = 0.5 * (F + F.transpose(-1, -2))
        s, Q = torch.linalg.eigh(F_sym)
        # Guarantee strict positivity before reciprocal.
        floor = float(eig_floor) if eig_floor > 0.0 else 1e-30
        s = s.clamp_min(floor)
        # x = Q diag(1/s) Q^T g
        rhs = torch.matmul(Q.transpose(-1, -2), g.unsqueeze(-1))
        rhs = rhs / s.unsqueeze(-1)
        x = torch.matmul(Q, rhs).squeeze(-1)
        return x

    def _get_or_init_fisher(
        self,
        key: str,
        shape_NDD: torch.Size,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        """Lazily create the Fisher bucket in ``self._moments[key]``."""
        slot = self._moments.get(key)
        N, D, D2 = shape_NDD
        if (
            slot is None
            or "F" not in slot
            or slot["F"].shape != shape_NDD
            or slot["F"].device != device
            or slot["F"].dtype != dtype
        ):
            slot = {
                "F": torch.zeros(N, D, D2, device=device, dtype=dtype),
                "beta_prod": torch.ones((), device=device, dtype=dtype),
            }
            self._moments[key] = slot
        return slot

    def _get_or_init_grad_ema(
        self,
        key: str,
        shape: torch.Size,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        """Lazily create the gradient-EMA bucket in ``self._moments[key]``."""
        slot = self._moments.get(key)
        if (
            slot is None
            or "g_ema" not in slot
            or slot["g_ema"].shape != shape
            or slot["g_ema"].device != device
            or slot["g_ema"].dtype != dtype
        ):
            slot = {
                "g_ema": torch.zeros(shape, device=device, dtype=dtype),
                "beta_prod": torch.ones((), device=device, dtype=dtype),
            }
            self._moments[key] = slot
        return slot

    # ---- Step --------------------------------------------------------------

    def step(self) -> None:
        self._step_count += 1

        if bool(getattr(self.config, "warmup_beta_verbose", False)):
            # Kept for log parity with the other Newton optimizers, even
            # though KFAC uses a dedicated ``ema_decay_map``.
            beta_g, local_g, steps_g = self._resolve_beta1("grad", return_meta=True)
            print(
                f"[KFAC][WarmupBeta][step={self._step_count}] "
                f"beta_g={beta_g:.6f} (local={local_g}/{max(steps_g, 1)})"
            )
        if (
            bool(getattr(self.config, "topology_damping_reanneal_verbose", False))
            and self._damping_reanneal_active
        ):
            mult = self._resolve_damping_multiplier()
            local = int(self._step_count) - int(self._damping_reanneal_anchor_step)
            print(
                f"[KFAC][DampingReanneal][step={self._step_count}] "
                f"multiplier={mult:.4g} (local={local}/"
                f"{max(self._damping_reanneal_steps, 1)})"
            )

        if self.config.mode == "per_gaussian":
            self._step_per_gaussian()
        else:
            self._step_per_param_group()

        self._apply_constraints()

    # ---- Verbose logging ---------------------------------------------------

    def _verbose_dump(
        self,
        *,
        name: str,
        grad_raw: torch.Tensor,
        grad_used: torch.Tensor,
        hess_est: torch.Tensor,
        F_damped: torch.Tensor,
        eigvals_pre_clip: Optional[torch.Tensor],
        eigvals_post_clip: Optional[torch.Tensor],
        delta: torch.Tensor,
        update_raw: torch.Tensor,
        update: torch.Tensor,
        damping: float,
        eig_floor: float,
        max_update: float,
        eig_clip_hits: int,
        eig_clip_total: int,
        update_clip_hits: int,
        update_clip_total: int,
        param_before: torch.Tensor,
        param_after: torch.Tensor,
    ) -> None:
        if not bool(getattr(self.config, "verbose", False)):
            return
        print(f"[KFAC][step={self._step_count}][{name}]")
        print(f"  param_orig= {self._tensor_quantile_str(param_before)}")
        print(f"  grad_raw  = {self._tensor_quantile_str(grad_raw)}")
        print(f"  grad_used = {self._tensor_quantile_str(grad_used)}")
        print(f"  hess_est  = {self._tensor_quantile_str(hess_est)}")
        if F_damped.dim() >= 2:
            trace = torch.diagonal(F_damped, dim1=-2, dim2=-1).sum(dim=-1)
            print(f"  F_trace   = {self._tensor_quantile_str(trace)}")
        if eigvals_pre_clip is not None and eigvals_post_clip is not None:
            print(f"  eig_pre   = {self._tensor_quantile_str(eigvals_pre_clip)}")
            print(f"  eig_post  = {self._tensor_quantile_str(eigvals_post_clip)}")
        print(f"  delta     = {self._tensor_quantile_str(delta)}")
        print(f"  update_raw= {self._tensor_quantile_str(update_raw)}")
        print(f"  update    = {self._tensor_quantile_str(update)}")
        eig_pct = (
            100.0 * float(eig_clip_hits) / float(eig_clip_total)
            if eig_clip_total > 0
            else 0.0
        )
        upd_pct = (
            100.0 * float(update_clip_hits) / float(update_clip_total)
            if update_clip_total > 0
            else 0.0
        )
        print(
            "  eig_clip_hits"
            f"= {eig_clip_hits}/{eig_clip_total} ({eig_pct:.2f}%)"
            f" (eig_floor={eig_floor:.4e})"
        )
        print(
            "  update_clip_hits"
            f"= {update_clip_hits}/{update_clip_total} ({upd_pct:.2f}%)"
            f" (max_update={max_update:.4e}, damping={damping:.4e})"
        )
        print(f"  param_post= {self._tensor_quantile_str(param_after)}")

    # ---- per_param_group mode ---------------------------------------------

    def _step_per_param_group(self) -> None:
        for name, d in zip(PARAM_NAMES, PARAM_DIMS):
            g_raw = self._grad.get(name)
            if g_raw is None:
                continue

            g_in = self._expand_grad(g_raw)  # (N, d)
            N = g_in.shape[0]
            device, dtype = g_in.device, g_in.dtype

            lr = float(self.config.get_lr(name))
            damping_abs = float(self._effective_damping(name))
            wf = float(getattr(self.config, "warmup_damping_floor", 0.0))
            warm_denom = (
                max(damping_abs, wf) if wf > 0.0 else damping_abs
            )
            damp_mode = str(
                getattr(self.config, "damping_mode", "absolute")
            ).lower()
            alpha = float(self.config.get_ema_decay(name, default=0.95))
            grad_beta = float(self.config.get_grad_ema_decay(name, default=0.9))
            eig_floor = float(self.config.get_eig_clip(name, default=0.0))
            max_update = float(self.config.get_max_update(name, default=float("inf")))
            verbose = bool(getattr(self.config, "verbose", False))

            # Gradient EMA momentum: g <- beta * g_ema + (1-beta) * g_t
            g_slot = self._get_or_init_grad_ema(
                f"kfac_grad_{name}", g_in.shape, device, dtype
            )
            g_ema = g_slot["g_ema"]
            g_ema.mul_(grad_beta).add_(g_in, alpha=(1.0 - grad_beta))
            g_slot["beta_prod"] = g_slot["beta_prod"] * grad_beta
            if self.config.bias_correction:
                g_denom = (1.0 - g_slot["beta_prod"]).clamp_min(1e-12)
                g = g_ema / g_denom
            else:
                g = g_ema

            key = f"kfac_{name}"
            slot = self._get_or_init_fisher(key, (N, d, d), device, dtype)

            # Update Fisher EMA:  F <- alpha*F + (1-alpha) * g g^T
            gg = torch.einsum("ni,nj->nij", g, g)
            F = slot["F"]
            F.mul_(alpha).add_(gg, alpha=(1.0 - alpha))
            slot["beta_prod"] = slot["beta_prod"] * alpha

            # Bias correction (Adam-style).
            if self.config.bias_correction:
                denom = (1.0 - slot["beta_prod"]).clamp_min(1e-12)
                F_used = F / denom
            else:
                F_used = F

            # Early-warmup: either fixed divisor or ``lambda`` matched to a
            # one-sample Fisher scale ``||g||^2/d`` (same law as ``relative``).
            if self._step_count <= int(self.config.min_step_for_fisher):
                use_match = bool(
                    getattr(self.config, "warmup_match_fisher_scale", False)
                )
                if use_match and damp_mode == "relative":
                    rel_w = float(self.config.get_damping_rel(name, default=0.1))
                    tr_inst = (g * g).sum(dim=-1).clamp_min(1e-30)
                    mean_inst = tr_inst / float(d)
                    lam_row = rel_w * mean_inst + damping_abs
                    wf_m = float(getattr(self.config, "warmup_damping_floor", 0.0))
                    if wf_m > 0.0:
                        lam_row = lam_row.clamp(min=wf_m)
                    lam_effective = lam_row
                    delta = -g / lam_row.unsqueeze(-1)
                else:
                    lam_effective = None
                    delta = -g / warm_denom  # (N, d)
                eig_clip_hits = 0
                eig_clip_total = int(N * d)
                eigvals_pre = None
                eigvals_post = None
                F_damped = None
            else:
                F_sym = 0.5 * (F_used + F_used.transpose(-1, -2))
                eye = torch.eye(d, device=device, dtype=dtype)
                if damp_mode == "relative":
                    rel = float(self.config.get_damping_rel(name, default=0.1))
                    tr = torch.diagonal(F_sym, dim1=-2, dim2=-1).sum(dim=-1)
                    mean_eig = (tr / float(d)).clamp_min(1e-30)
                    lam = rel * mean_eig + damping_abs
                    F_damped = F_sym + lam.unsqueeze(-1).unsqueeze(-1) * eye
                    lam_effective = lam
                else:
                    F_damped = F_sym + damping_abs * eye
                    lam_effective = None
                eigvals_pre, Q = torch.linalg.eigh(F_damped)
                floor = float(eig_floor) if eig_floor > 0.0 else 1e-30
                eigvals_post = eigvals_pre.clamp_min(floor)
                eig_clip_hits = int((eigvals_pre < floor).sum().item())
                eig_clip_total = int(eigvals_pre.numel())
                rhs = torch.matmul(Q.transpose(-1, -2), g.unsqueeze(-1))
                rhs = rhs / eigvals_post.unsqueeze(-1)
                delta = -torch.matmul(Q, rhs).squeeze(-1)

            # Collapse trailing singleton for 1D params (opacity).
            if g_raw.dim() == 1:
                delta = delta.squeeze(-1)

            update_raw = lr * delta
            update_total = int(update_raw.numel())
            update_clip_hits = 0
            if math.isfinite(max_update) and max_update > 0.0:
                update_clip_hits = int((update_raw.abs() > max_update).sum().item())
                update = torch.clamp(update_raw, min=-max_update, max=max_update)
            else:
                update = update_raw

            p = self.params.get(name).data
            param_before = p.clone() if verbose else p
            p += update

            if verbose:
                if F_damped is not None:
                    F_for_log = F_damped
                else:
                    if (
                        self._step_count
                        <= int(self.config.min_step_for_fisher)
                        and lam_effective is not None
                    ):
                        log_denom = float(lam_effective.mean().item())
                    elif self._step_count <= int(
                        self.config.min_step_for_fisher
                    ):
                        log_denom = float(warm_denom)
                    else:
                        log_denom = float(damping_abs)
                    F_for_log = F_used + log_denom * torch.eye(
                        d, device=device, dtype=dtype
                    )
                log_damp = (
                    float(lam_effective.mean().item())
                    if lam_effective is not None
                    else (
                        float(warm_denom)
                        if self._step_count
                        <= int(self.config.min_step_for_fisher)
                        else damping_abs
                    )
                )
                self._verbose_dump(
                    name=name,
                    grad_raw=g_in if g_raw.dim() != 1 else g_raw,
                    grad_used=g,
                    hess_est=F_used.reshape(N, -1),
                    F_damped=F_for_log,
                    eigvals_pre_clip=eigvals_pre,
                    eigvals_post_clip=eigvals_post,
                    delta=delta,
                    update_raw=update_raw,
                    update=update,
                    damping=log_damp,
                    eig_floor=eig_floor,
                    max_update=max_update,
                    eig_clip_hits=eig_clip_hits,
                    eig_clip_total=eig_clip_total,
                    update_clip_hits=update_clip_hits,
                    update_clip_total=update_total,
                    param_before=param_before,
                    param_after=p,
                )

    # ---- per_gaussian mode -------------------------------------------------

    def _step_per_gaussian(self) -> None:
        # Gather a canonical reference tensor to infer N / device / dtype.
        ref = None
        for name in PARAM_NAMES:
            if name in self._grad:
                ref = self._grad[name]
                break
        if ref is None:
            return

        N = ref.shape[0]
        device, dtype = ref.device, ref.dtype
        D = self.total_dim  # 11

        # Concatenate gradients into a single (N, D) vector.
        g_cat_raw = torch.zeros(N, D, device=device, dtype=dtype)
        for name in PARAM_NAMES:
            g_raw = self._grad.get(name)
            if g_raw is None:
                continue
            gi = self._expand_grad(g_raw)  # (N, d)
            g_cat_raw[:, self._offsets[name]] = gi

        grad_beta = float(self.config.get_grad_ema_decay("means", default=0.9))
        g_slot = self._get_or_init_grad_ema(
            "kfac_grad_full", g_cat_raw.shape, device, dtype
        )
        g_ema = g_slot["g_ema"]
        g_ema.mul_(grad_beta).add_(g_cat_raw, alpha=(1.0 - grad_beta))
        g_slot["beta_prod"] = g_slot["beta_prod"] * grad_beta
        if self.config.bias_correction:
            g_denom = (1.0 - g_slot["beta_prod"]).clamp_min(1e-12)
            g_cat = g_ema / g_denom
        else:
            g_cat = g_ema

        # A single Fisher for the whole Gaussian vector.
        slot = self._get_or_init_fisher(_KFAC_FULL_KEY, (N, D, D), device, dtype)

        # Use the mean EMA decay across groups for the shared Fisher. If the
        # user set only a single per-name value they all agree; we just pick
        # means as the canonical one.
        alpha = float(self.config.get_ema_decay("means", default=0.95))
        gg = torch.einsum("ni,nj->nij", g_cat, g_cat)
        F = slot["F"]
        F.mul_(alpha).add_(gg, alpha=(1.0 - alpha))
        slot["beta_prod"] = slot["beta_prod"] * alpha

        if self.config.bias_correction:
            denom = (1.0 - slot["beta_prod"]).clamp_min(1e-12)
            F_used = F / denom
        else:
            F_used = F

        # Build per-element damping diagonal (respects per-group damping).
        damping_vec = torch.empty(D, device=device, dtype=dtype)
        for name, d in zip(PARAM_NAMES, PARAM_DIMS):
            damping_vec[self._offsets[name]] = float(self._effective_damping(name))
        damping_mat = torch.diag(damping_vec)  # (D, D)

        # Per-group eigenvalue floor: use the max over groups (the stricter
        # floor dominates the shared spectrum).
        eig_floor = max(
            float(self.config.get_eig_clip(name, default=0.0))
            for name in PARAM_NAMES
        )

        damp_mode = str(
            getattr(self.config, "damping_mode", "absolute")
        ).lower()
        damp_full: Optional[torch.Tensor] = None

        if self._step_count <= int(self.config.min_step_for_fisher):
            use_match_pg = bool(
                getattr(self.config, "warmup_match_fisher_scale", False)
            )
            wf_pg = float(getattr(self.config, "warmup_damping_floor", 0.0))
            if use_match_pg and damp_mode == "relative":
                wf_m = wf_pg
                parts_pg: List[torch.Tensor] = []
                for name, d in zip(PARAM_NAMES, PARAM_DIMS):
                    gi = g_cat[:, self._offsets[name]]
                    tr_i = (gi * gi).sum(dim=-1).clamp_min(1e-30)
                    mei = tr_i / float(d)
                    rel_k = float(self.config.get_damping_rel(name, default=0.1))
                    abs_k = float(self._effective_damping(name))
                    lam_k = rel_k * mei + abs_k
                    if wf_m > 0.0:
                        lam_k = lam_k.clamp(min=wf_m)
                    parts_pg.append(lam_k.unsqueeze(-1).expand(-1, d))
                damp_full = torch.cat(parts_pg, dim=-1)
                delta_cat = -g_cat / damp_full
            elif wf_pg > 0.0:
                damp_w = torch.clamp(damping_vec, min=wf_pg)
                delta_cat = -g_cat / damp_w  # (N, D)
            else:
                delta_cat = -g_cat / damping_vec  # (N, D)
            eig_clip_hits_all = 0
            eig_clip_total_all = int(N * D)
            eigvals_pre_all = None
            eigvals_post_all = None
            F_damped = None
        else:
            F_sym = 0.5 * (F_used + F_used.transpose(-1, -2))
            if damp_mode == "relative":
                tr_all = torch.diagonal(F_sym, dim1=-2, dim2=-1).sum(
                    dim=-1
                )  # (N,)
                mean_eig = (tr_all / float(D)).clamp_min(1e-30)
                damp_rows: List[torch.Tensor] = []
                for name, d in zip(PARAM_NAMES, PARAM_DIMS):
                    rel_k = float(
                        self.config.get_damping_rel(name, default=0.1)
                    )
                    abs_k = float(self._effective_damping(name))
                    lam_k = rel_k * mean_eig + abs_k  # (N,)
                    damp_rows.append(lam_k.unsqueeze(-1).expand(-1, d))
                damp_full = torch.cat(damp_rows, dim=-1)  # (N, D)
                F_damped = F_sym + torch.diag_embed(damp_full)
            else:
                damp_full = None
                F_damped = F_sym + damping_mat  # (N, D, D)
            eigvals_pre_all, Q_all = torch.linalg.eigh(F_damped)
            floor = float(eig_floor) if eig_floor > 0.0 else 1e-30
            eigvals_post_all = eigvals_pre_all.clamp_min(floor)
            eig_clip_hits_all = int((eigvals_pre_all < floor).sum().item())
            eig_clip_total_all = int(eigvals_pre_all.numel())
            rhs_all = torch.matmul(Q_all.transpose(-1, -2), g_cat.unsqueeze(-1))
            rhs_all = rhs_all / eigvals_post_all.unsqueeze(-1)
            delta_cat = -torch.matmul(Q_all, rhs_all).squeeze(-1)

        verbose = bool(getattr(self.config, "verbose", False))

        # Split back and apply per-group learning rate + update clip.
        for name, d in zip(PARAM_NAMES, PARAM_DIMS):
            lr = float(self.config.get_lr(name))
            max_update = float(self.config.get_max_update(name, default=float("inf")))
            delta = delta_cat[:, self._offsets[name]]  # (N, d)
            g_raw = self._grad.get(name)
            if g_raw is not None and g_raw.dim() == 1:
                delta = delta.squeeze(-1)

            update_raw = lr * delta
            update_clip_hits = 0
            update_total = int(update_raw.numel())
            if math.isfinite(max_update) and max_update > 0.0:
                update_clip_hits = int((update_raw.abs() > max_update).sum().item())
                update = torch.clamp(update_raw, min=-max_update, max=max_update)
            else:
                update = update_raw

            p = self.params.get(name).data
            param_before = p.clone() if verbose else p
            p += update

            if verbose:
                d_sel = d
                F_log_block = F_used[:, self._offsets[name], :][
                    :, :, self._offsets[name]
                ]
                if F_damped is not None:
                    F_damped_block = F_damped[:, self._offsets[name], :][
                        :, :, self._offsets[name]
                    ]
                else:
                    F_damped_block = F_log_block + float(
                        self._effective_damping(name)
                    ) * torch.eye(
                        d_sel, device=device, dtype=dtype
                    )
                if (
                    damp_mode == "relative"
                    and damp_full is not None
                ):
                    log_damp = float(
                        damp_full[:, self._offsets[name]]
                        .mean()
                        .item()
                    )
                else:
                    log_damp = float(self._effective_damping(name))
                self._verbose_dump(
                    name=name,
                    grad_raw=g_raw if g_raw is not None else delta,
                    grad_used=g_cat[:, self._offsets[name]] if d > 1 else g_cat[:, self._offsets[name]].squeeze(-1),
                    hess_est=F_log_block.reshape(N, -1),
                    F_damped=F_damped_block,
                    eigvals_pre_clip=eigvals_pre_all,
                    eigvals_post_clip=eigvals_post_all,
                    delta=delta,
                    update_raw=update_raw,
                    update=update,
                    damping=log_damp,
                    eig_floor=eig_floor,
                    max_update=max_update,
                    eig_clip_hits=eig_clip_hits_all,
                    eig_clip_total=eig_clip_total_all,
                    update_clip_hits=update_clip_hits,
                    update_clip_total=update_total,
                    param_before=param_before,
                    param_after=p,
                )

    # ---- Required abstract helpers (unused by KFAC) ------------------------

    def _extract_hess_diagonal(
        self, name: str, h: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError("KFAC does not consume an external Hessian.")

    def _reconstruct_block(
        self, name: str, h: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError("KFAC does not consume an external Hessian.")

    def _assemble_full_hessian(
        self,
        h: Dict[str, torch.Tensor],
        ch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError("KFAC does not consume an external Hessian.")

    # ---- Constraints -------------------------------------------------------

    def _apply_constraints(self) -> None:
        _renormalize_quats(self.params)
        p = self.params.get("opacity")
        allow_negative_opacity = bool(
            getattr(self.config, "allow_negative_opacity", False)
        )
        if p is not None and not allow_negative_opacity:
            p.data.clamp_(min=0.0)

    # ---- Serialization -----------------------------------------------------

    def state_dict(self) -> Dict:
        state = super().state_dict()
        state["kfac_mode"] = self.config.mode
        state["ema_decay_map"] = dict(self.config.ema_decay_map)
        state["grad_ema_decay_map"] = dict(self.config.grad_ema_decay_map)
        state["eig_clip_map"] = dict(self.config.eig_clip_map)
        state["max_update_map"] = dict(self.config.max_update_map)
        state["bias_correction"] = bool(self.config.bias_correction)
        state["min_step_for_fisher"] = int(self.config.min_step_for_fisher)
        state["damping_mode"] = str(getattr(self.config, "damping_mode", "absolute"))
        state["damping_rel_map"] = dict(
            getattr(self.config, "damping_rel_map", {}) or {}
        )
        state["warmup_damping_floor"] = float(
            getattr(self.config, "warmup_damping_floor", 0.0)
        )
        state["warmup_match_fisher_scale"] = bool(
            getattr(self.config, "warmup_match_fisher_scale", False)
        )
        return state

    def load_state_dict(self, state: Dict) -> None:
        super().load_state_dict(state)
        if "kfac_mode" in state:
            self.config.mode = str(state["kfac_mode"])
        if "ema_decay_map" in state:
            self.config.ema_decay_map = dict(state["ema_decay_map"])
        if "grad_ema_decay_map" in state:
            self.config.grad_ema_decay_map = dict(state["grad_ema_decay_map"])
        if "eig_clip_map" in state:
            self.config.eig_clip_map = dict(state["eig_clip_map"])
        if "max_update_map" in state:
            self.config.max_update_map = dict(state["max_update_map"])
        if "bias_correction" in state:
            self.config.bias_correction = bool(state["bias_correction"])
        if "min_step_for_fisher" in state:
            self.config.min_step_for_fisher = int(state["min_step_for_fisher"])
        if "damping_mode" in state:
            self.config.damping_mode = str(state["damping_mode"])
        if "damping_rel_map" in state and state["damping_rel_map"] is not None:
            self.config.damping_rel_map = dict(state["damping_rel_map"])
        if "warmup_damping_floor" in state:
            self.config.warmup_damping_floor = float(
                state["warmup_damping_floor"]
            )
        if "warmup_match_fisher_scale" in state:
            self.config.warmup_match_fisher_scale = bool(
                state["warmup_match_fisher_scale"]
            )


# ============================================================================
# Factory
# ============================================================================

def create_gs3d_ct_kfac(
    params: Params,
    *,
    mode: str = "per_param_group",
    lr: float = 1.0,
    damping: float = 1e-3,
    ema_decay: float = 0.95,
    grad_ema_decay: float = 0.9,
    eig_clip: float = 0.0,
    max_update: float = float("inf"),
    bias_correction: bool = True,
    min_step_for_fisher: int = 0,
    **kwargs,
) -> GS3DCTKFACOptimizer:
    """Convenience factory with uniform per-parameter defaults."""
    lr_map = {n: lr for n in PARAM_NAMES}
    damping_map = {n: damping for n in PARAM_NAMES}
    ema_decay_map = {n: ema_decay for n in PARAM_NAMES}
    grad_ema_decay_map = {n: grad_ema_decay for n in PARAM_NAMES}
    eig_clip_map = {n: eig_clip for n in PARAM_NAMES}
    max_update_map = {n: max_update for n in PARAM_NAMES}

    config = KFACSecondOrderConfig(
        lr_map=lr_map,
        damping_map=damping_map,
        ema_decay_map=ema_decay_map,
        grad_ema_decay_map=grad_ema_decay_map,
        eig_clip_map=eig_clip_map,
        max_update_map=max_update_map,
        mode=mode,
        bias_correction=bias_correction,
        min_step_for_fisher=min_step_for_fisher,
        hess_mode=HessianMode.BLOCK,
        grad_strategy=SignalStrategy.RAW,
        hess_strategy=SignalStrategy.RAW,
        **kwargs,
    )
    return GS3DCTKFACOptimizer(params, config)


__all__ = [
    "PARAM_NAMES",
    "PARAM_DIMS",
    "KFACSecondOrderConfig",
    "GS3DCTKFACOptimizer",
    "create_gs3d_ct_kfac",
]
