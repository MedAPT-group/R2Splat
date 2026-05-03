"""Reduced-space Newton optimizer for GS3D CT reconstruction.

This optimizer applies the 3DGS2-style reduced parameter updates:

- means: update only in the plane perpendicular to the main-view ray
- scale_ori: update in the raw-scale subspace that most changes projected
  2D covariance eigenvalues
- quats: update by a one-dimensional rotation around the main-view ray
- opacity: scalar damped Newton update

It intentionally does not inherit from the existing first/second-order
optimizer bases because the step is groupwise and re-runs forward/backward
between parameter groups.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from gsplat2ct.ops.data import Config, Data, ForwardResult, GradResult, HessResult, Params
from gsplat2ct.ops.parameter_transform import _quat_to_rotation_matrix


PARAM_NAMES = ["means", "quats", "scale_ori", "opacity"]


@dataclass
class ReducedNewtonConfig:
    """Configuration for :class:`GS3DCTReducedNewtonOptimizer`."""

    lr_map: Dict[str, float] = field(default_factory=dict)
    damping_map: Dict[str, float] = field(default_factory=dict)
    clamp_map: Dict[str, float] = field(default_factory=dict)
    max_update_map: Dict[str, float] = field(default_factory=dict)

    svd_threshold: float = 1e-8
    max_scale_rank: int = 2
    max_rotation_angle: float = 5.0e-2
    allow_negative_opacity: bool = False
    min_opacity: float = 1.0e-4
    means_bound: float = 1.0
    verbose: bool = False

    def get_lr(self, name: str, default: float = 1.0) -> float:
        return float(self.lr_map.get(name, default))

    def set_lr(self, name: str, lr: float) -> None:
        self.lr_map[name] = float(lr)

    def get_damping(self, name: str, default: float = 1e-3) -> float:
        return float(self.damping_map.get(name, default))

    def set_damping(self, name: str, damping: float) -> None:
        self.damping_map[name] = float(damping)

    def get_clamp(self, name: str, default: float = 1e-8) -> float:
        return float(self.clamp_map.get(name, default))

    def set_clamp(self, name: str, clamp_value: float) -> None:
        self.clamp_map[name] = float(clamp_value)

    def get_max_update(self, name: str, default: float = float("inf")) -> float:
        return float(self.max_update_map.get(name, default))

    def set_max_update(self, name: str, max_update: float) -> None:
        self.max_update_map[name] = float(max_update)


def _upper_tri3_to_sym(h: torch.Tensor) -> torch.Tensor:
    mat = torch.zeros(h.shape[0], 3, 3, device=h.device, dtype=h.dtype)
    mat[:, 0, 0] = h[:, 0]
    mat[:, 0, 1] = mat[:, 1, 0] = h[:, 1]
    mat[:, 0, 2] = mat[:, 2, 0] = h[:, 2]
    mat[:, 1, 1] = h[:, 3]
    mat[:, 1, 2] = mat[:, 2, 1] = h[:, 4]
    mat[:, 2, 2] = h[:, 5]
    return mat


def _upper_tri6_to_sym(h: torch.Tensor) -> torch.Tensor:
    n = h.shape[0]
    mat = torch.zeros(n, 6, 6, device=h.device, dtype=h.dtype)
    idx = 0
    for i in range(6):
        for j in range(i, 6):
            mat[:, i, j] = h[:, idx]
            mat[:, j, i] = h[:, idx]
            idx += 1
    return mat


def _pack_sym3_upper(m: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [m[:, 0, 0], m[:, 0, 1], m[:, 0, 2], m[:, 1, 1], m[:, 1, 2], m[:, 2, 2]],
        dim=-1,
    )


def _covars6_to_sym3(covars: torch.Tensor) -> torch.Tensor:
    return _upper_tri3_to_sym(covars)


def _quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def _normalize(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


def _camera_center_from_viewmat(viewmat: torch.Tensor) -> torch.Tensor:
    rot = viewmat[:3, :3]
    trans = viewmat[:3, 3]
    return -(rot.transpose(0, 1) @ trans)


def _view_rays(means: torch.Tensor, viewmats: torch.Tensor) -> torch.Tensor:
    center = _camera_center_from_viewmat(viewmats[0]).to(device=means.device, dtype=means.dtype)
    return _normalize(means - center.unsqueeze(0))


def _plane_basis(rays: torch.Tensor) -> torch.Tensor:
    ref_z = torch.zeros_like(rays)
    ref_z[:, 2] = 1.0
    ref_y = torch.zeros_like(rays)
    ref_y[:, 1] = 1.0
    ref = torch.where((rays[:, 2].abs() < 0.9).unsqueeze(-1), ref_z, ref_y)
    e1 = _normalize(torch.cross(rays, ref, dim=-1))
    e2 = _normalize(torch.cross(rays, e1, dim=-1))
    return torch.stack([e1, e2], dim=-1)


def _skew(rays: torch.Tensor) -> torch.Tensor:
    n = rays.shape[0]
    omega = torch.zeros(n, 3, 3, device=rays.device, dtype=rays.dtype)
    rx, ry, rz = rays[:, 0], rays[:, 1], rays[:, 2]
    omega[:, 0, 1] = -rz
    omega[:, 0, 2] = ry
    omega[:, 1, 0] = rz
    omega[:, 1, 2] = -rx
    omega[:, 2, 0] = -ry
    omega[:, 2, 1] = rx
    return omega


def _clip_by_abs(x: torch.Tensor, max_abs: float) -> torch.Tensor:
    if math.isfinite(max_abs) and max_abs > 0.0:
        return torch.clamp(x, min=-float(max_abs), max=float(max_abs))
    return x


def _solve_spd_like(H: torch.Tensor, g: torch.Tensor, damping: float, eig_floor: float) -> torch.Tensor:
    """Solve a damped reduced Newton system via eigenvalue flooring."""
    H = 0.5 * (H + H.transpose(-1, -2))
    evals, evecs = torch.linalg.eigh(H)
    evals = torch.clamp(evals, min=float(eig_floor))
    denom = evals + float(damping)
    denom = denom.clamp_min(max(float(eig_floor), 1e-12))
    g_eig = torch.einsum("...ji,...j->...i", evecs, g)
    delta_eig = -g_eig / denom
    return torch.einsum("...ij,...j->...i", evecs, delta_eig)


def _safe_item_stats(name: str, tensor: torch.Tensor) -> str:
    flat = tensor.detach().reshape(-1)
    if flat.numel() == 0:
        return f"{name}=empty"
    finite = flat[torch.isfinite(flat)]
    if finite.numel() == 0:
        return f"{name}=all_non_finite"
    q = _safe_quantiles(finite)
    std = finite.std(unbiased=False).item() if finite.numel() > 1 else 0.0
    return (
        f"{name}: min={finite.min().item():.4e}, mean={finite.mean().item():.4e}, "
        f"std={std:.4e}, max={finite.max().item():.4e}, {q}"
    )


def _safe_quantiles(tensor: torch.Tensor) -> str:
    flat = tensor.detach().reshape(-1)
    finite = flat[torch.isfinite(flat)]
    if finite.numel() == 0:
        return "quantiles=all_non_finite"
    probs = torch.tensor(
        [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99],
        device=finite.device,
        dtype=finite.dtype,
    )
    vals = torch.quantile(finite, probs)
    return (
        f"q01={vals[0].item():.4e}, q05={vals[1].item():.4e}, "
        f"q25={vals[2].item():.4e}, q50={vals[3].item():.4e}, "
        f"q75={vals[4].item():.4e}, q95={vals[5].item():.4e}, "
        f"q99={vals[6].item():.4e}"
    )


def _safe_item_sample(name: str, tensor: torch.Tensor, max_items: int = 8) -> str:
    flat = tensor.detach().reshape(-1)
    if flat.numel() == 0:
        return f"{name}_sample=[]"
    vals = flat[:max_items].cpu().tolist()
    body = ", ".join(f"{float(v):.4e}" for v in vals)
    suffix = ", ..." if flat.numel() > max_items else ""
    return f"{name}_sample=[{body}{suffix}]"


def _safe_eig_stats(name: str, eigvals: torch.Tensor) -> str:
    vals = eigvals.detach()
    if vals.numel() == 0:
        return f"{name}=empty"
    if vals.ndim == 1:
        vals = vals.unsqueeze(-1)
    flat = vals.reshape(-1)
    finite = flat[torch.isfinite(flat)]
    if finite.numel() == 0:
        return f"{name}=all_non_finite"
    pd = (vals > 0.0).all(dim=-1)
    nonpos = (vals <= 0.0).sum()
    negative = (vals < 0.0).sum()
    std = finite.std(unbiased=False).item() if finite.numel() > 1 else 0.0
    return (
        f"{name}: min={finite.min().item():.4e}, mean={finite.mean().item():.4e}, "
        f"std={std:.4e}, max={finite.max().item():.4e}, {_safe_quantiles(finite)}, "
        f"positive_definite={int(pd.sum().item())}/{vals.shape[0]}, "
        f"nonpos_eigs={int(nonpos.item())}, negative_eigs={int(negative.item())}"
    )


def _safe_clip_stats(raw_update: torch.Tensor, clipped_update: torch.Tensor) -> str:
    raw = raw_update.detach().reshape(-1)
    clipped = clipped_update.detach().reshape(-1)
    if raw.numel() == 0:
        return "clip_ratio: clipped=0/0 (0.00%)"
    changed = ~torch.isclose(raw, clipped, rtol=1e-6, atol=1e-12, equal_nan=True)
    n_clipped = int(changed.sum().item())
    n_total = int(raw.numel())
    ratio = 100.0 * float(n_clipped) / float(max(n_total, 1))
    raw_abs = raw.abs()
    clipped_abs = clipped.abs()
    finite_raw = raw_abs[torch.isfinite(raw_abs)]
    finite_clipped = clipped_abs[torch.isfinite(clipped_abs)]
    raw_max = float(finite_raw.max().item()) if finite_raw.numel() > 0 else float("nan")
    clipped_max = (
        float(finite_clipped.max().item()) if finite_clipped.numel() > 0 else float("nan")
    )
    return (
        f"clip_ratio: clipped={n_clipped}/{n_total} ({ratio:.2f}%), "
        f"max_abs_raw_update={raw_max:.4e}, max_abs_clipped_update={clipped_max:.4e}, "
        f"abs_raw_update_{_safe_quantiles(finite_raw) if finite_raw.numel() > 0 else 'all_non_finite'}, "
        f"abs_clipped_update_{_safe_quantiles(finite_clipped) if finite_clipped.numel() > 0 else 'all_non_finite'}"
    )


class GS3DCTReducedNewtonOptimizer:
    """Groupwise reduced-space optimizer for GS3D CT parameters."""

    needs_hessian = True
    groupwise_step = True

    def __init__(self, params: Params, config: ReducedNewtonConfig):
        self.params = params
        self.config = config
        self._step_count = 0
        self._last_stage_snapshots: Optional[List[Tuple[str, Dict[str, torch.Tensor]]]] = None
        self._last_operator_config: Optional[Config] = None

    def zero_grad(self) -> None:
        pass

    def add_grad_hess(self, grad_result: GradResult, hess_result: HessResult) -> None:
        self._last_grad = grad_result
        self._last_hess = hess_result

    def add_grad(self, grad_result: GradResult) -> None:
        self._last_grad = grad_result

    def reset_moments(self) -> None:
        self._step_count = 0

    def apply_topology_transition(self, **kwargs) -> None:
        pass

    def restart_beta_warmup(self, **kwargs) -> None:
        pass

    def restart_damping_reanneal(self, **kwargs) -> None:
        pass

    def pop_stage_snapshots(self) -> Optional[List[Tuple[str, Dict[str, torch.Tensor]]]]:
        snapshots = self._last_stage_snapshots
        self._last_stage_snapshots = None
        return snapshots

    def step(self) -> None:
        """No-op step used when topology changed after gradients were computed."""
        self._step_count += 1
        self._apply_constraints()
        self._last_stage_snapshots = None

    @torch.no_grad()
    def step_groupwise(
        self,
        operator,
        data: Data,
        config: Config,
        initial: Optional[Tuple[ForwardResult, GradResult, HessResult]] = None,
        capture_stages: bool = False,
    ) -> Tuple[ForwardResult, GradResult, HessResult]:
        """Run position, scale, rotation, and opacity updates sequentially."""
        self._step_count += 1
        self._last_operator_config = config
        snapshots: Optional[List[Tuple[str, Dict[str, torch.Tensor]]]] = [] if capture_stages else None
        self._last_stage_snapshots = None

        if snapshots is not None:
            snapshots.append(("before", self._clone_param_tensors()))

        if initial is None:
            fwd, grad, hess = operator.forward_backward2(self.params, data, config)
        else:
            fwd, grad, hess = initial
        self._step_means(grad, hess, data)
        if snapshots is not None:
            snapshots.append(("after_means", self._clone_param_tensors()))

        fwd, grad, hess = operator.forward_backward2(self.params, data, config)
        self._step_scale(grad, hess, data, config)
        if snapshots is not None:
            snapshots.append(("after_scales", self._clone_param_tensors()))

        fwd, grad, hess = operator.forward_backward2(self.params, data, config)
        self._step_rotation(fwd, grad, data)
        if snapshots is not None:
            snapshots.append(("after_quats", self._clone_param_tensors()))

        fwd, grad, hess = operator.forward_backward2(self.params, data, config)
        self._step_opacity(grad, hess)

        self._apply_constraints()
        if snapshots is not None:
            snapshots.append(("after_opacity", self._clone_param_tensors()))
            self._last_stage_snapshots = snapshots
        return fwd, grad, hess

    def _clone_param_tensors(self) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for name in PARAM_NAMES:
            param = self.params.get(name)
            if param is not None:
                out[name] = param.data.detach().clone()
        return out

    def _step_means(self, grad: GradResult, hess: HessResult, data: Data) -> None:
        g = grad.get("means")
        h = hess.get("means")
        if g is None or h is None:
            return
        means = self.params.get("means").data
        rays = _view_rays(means, data.get("viewmats"))
        Q = _plane_basis(rays)
        H = _upper_tri3_to_sym(h)
        g_red = torch.einsum("nki,nk->ni", Q, g)
        H_red = torch.einsum("nki,nkl,nlj->nij", Q, H, Q)
        delta_red = _solve_spd_like(
            H_red,
            g_red,
            self.config.get_damping("means"),
            self.config.get_clamp("means"),
        )
        delta = torch.einsum("nki,ni->nk", Q, delta_red)
        raw_update = self.config.get_lr("means") * delta
        update = _clip_by_abs(raw_update, self.config.get_max_update("means"))
        means_before = means.detach().clone()
        means.add_(update)
        bound_count = self._clamp_means_()
        effective_update = means - means_before
        if self.config.verbose:
            self._verbose_newton_debug(
                name="means",
                raw_grad=g,
                raw_hess=H,
                reduced_grad=g_red,
                reduced_hess=H_red,
                raw_hess_eigs=torch.linalg.eigvalsh(0.5 * (H + H.transpose(-1, -2))),
                reduced_hess_eigs=torch.linalg.eigvalsh(
                    0.5 * (H_red + H_red.transpose(-1, -2))
                ),
                raw_update=raw_update,
                clipped_update=effective_update,
                extra=(
                    f"means_bound: bound={self._means_bound():.4e}, "
                    f"clamped={bound_count}/{means.numel()}"
                ),
            )

    def _step_scale(self, grad: GradResult, hess: HessResult, data: Data, config: Config) -> None:
        g = grad.get("scale_ori")
        h = hess.get("scale_ori")
        if g is None or h is None:
            return
        scale_ori = self.params.get("scale_ori").data
        quats = self.params.get("quats").data
        means = self.params.get("means").data
        Q, rank = self._scale_basis(means, quats, scale_ori, data, config)
        if Q is None:
            return

        H = _upper_tri3_to_sym(h)
        update = torch.zeros_like(scale_ori)
        reduced_grad = torch.zeros(scale_ori.shape[0], 2, device=scale_ori.device, dtype=scale_ori.dtype)
        reduced_hess = torch.zeros(
            scale_ori.shape[0],
            2,
            2,
            device=scale_ori.device,
            dtype=scale_ori.dtype,
        )
        valid2 = rank >= 2
        valid1 = rank == 1

        if valid2.any():
            q2 = Q[valid2, :, :2]
            g2 = g[valid2]
            H2 = H[valid2]
            g_red = torch.einsum("nki,nk->ni", q2, g2)
            H_red = torch.einsum("nki,nkl,nlj->nij", q2, H2, q2)
            reduced_grad[valid2] = g_red
            reduced_hess[valid2] = H_red
            delta_red = _solve_spd_like(
                H_red,
                g_red,
                self.config.get_damping("scale_ori"),
                self.config.get_clamp("scale_ori"),
            )
            update[valid2] = torch.einsum("nki,ni->nk", q2, delta_red)

        if valid1.any():
            q1 = Q[valid1, :, 0]
            g1 = (q1 * g[valid1]).sum(dim=-1)
            h1 = torch.einsum("ni,nij,nj->n", q1, H[valid1], q1)
            reduced_grad[valid1, 0] = g1
            reduced_hess[valid1, 0, 0] = h1
            h1 = torch.clamp(h1, min=self.config.get_clamp("scale_ori"))
            delta1 = -g1 / (h1 + self.config.get_damping("scale_ori"))
            update[valid1] = q1 * delta1.unsqueeze(-1)

        raw_update = self.config.get_lr("scale_ori") * update
        update = _clip_by_abs(raw_update, self.config.get_max_update("scale_ori"))
        scale_ori.add_(update)
        if self.config.verbose:
            valid = rank > 0
            reduced_eigs = torch.linalg.eigvalsh(
                0.5 * (reduced_hess[valid] + reduced_hess[valid].transpose(-1, -2))
            ) if valid.any() else reduced_hess.new_empty(0, 2)
            self._verbose_newton_debug(
                name="scale_ori",
                raw_grad=g,
                raw_hess=H,
                reduced_grad=reduced_grad[valid],
                reduced_hess=reduced_hess[valid],
                raw_hess_eigs=torch.linalg.eigvalsh(0.5 * (H + H.transpose(-1, -2))),
                reduced_hess_eigs=reduced_eigs,
                raw_update=raw_update,
                clipped_update=update,
                extra=(
                    f"rank0={(rank == 0).sum().item()}, "
                    f"rank1={(rank == 1).sum().item()}, rank2={(rank >= 2).sum().item()}"
                ),
            )

    def _scale_basis(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        scale_ori: torch.Tensor,
        data: Data,
        config: Config,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        viewmats = data.get("viewmats")
        Ks = data.get("Ks")
        if viewmats is None or Ks is None:
            return None, None

        view = viewmats[0].to(device=means.device, dtype=means.dtype)
        K = Ks[0].to(device=means.device, dtype=means.dtype)
        R_view = view[:3, :3]
        t = view[:3, 3]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        dim1 = float(config.get("img_dim1"))
        dim0 = float(config.get("img_dim0"))

        pc = means @ R_view.transpose(0, 1) + t.unsqueeze(0)
        x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
        valid = z > 0.01
        z_safe = z.clamp_min(0.01)
        rz = 1.0 / z_safe
        rz2 = rz * rz

        W = torch.as_tensor(dim1, device=means.device, dtype=means.dtype)
        H = torch.as_tensor(dim0, device=means.device, dtype=means.dtype)
        tan_fovx = 0.5 * W / fx
        tan_fovy = 0.5 * H / fy
        lim_x_pos = (W - cx) / fx + 0.3 * tan_fovx
        lim_x_neg = cx / fx + 0.3 * tan_fovx
        lim_y_pos = (H - cy) / fy + 0.3 * tan_fovy
        lim_y_neg = cy / fy + 0.3 * tan_fovy

        tx = z_safe * torch.clamp(x * rz, min=-lim_x_neg, max=lim_x_pos)
        ty = z_safe * torch.clamp(y * rz, min=-lim_y_neg, max=lim_y_pos)

        zeros = torch.zeros_like(x)
        J2 = torch.stack(
            [
                torch.stack([fx * rz, zeros, -fx * tx * rz2], dim=-1),
                torch.stack([zeros, fy * rz, -fy * ty * rz2], dim=-1),
            ],
            dim=-2,
        )

        R_gauss = _quat_to_rotation_matrix(quats)
        B = torch.einsum("nij,jk,nkl->nil", J2, R_view, R_gauss)

        scale_min = float(config.get("scale_min", 0.01))
        scale_max = float(config.get("scale_max", 2.0))
        sig = torch.sigmoid(scale_ori)
        scales = scale_min + (scale_max - scale_min) * sig
        ds_draw = (scale_max - scale_min) * sig * (1.0 - sig)

        weighted_B = B * scales.unsqueeze(1)
        Sigma2 = weighted_B @ weighted_B.transpose(-1, -2)
        _, U_lam = torch.linalg.eigh(0.5 * (Sigma2 + Sigma2.transpose(-1, -2)))
        utb = torch.einsum("nma,nmi->nai", U_lam, B)
        J_lam_raw = 2.0 * utb.square() * scales.unsqueeze(1) * ds_draw.unsqueeze(1)
        J_lam_raw = torch.where(valid[:, None, None], J_lam_raw, torch.zeros_like(J_lam_raw))

        _, S, Vh = torch.linalg.svd(J_lam_raw, full_matrices=True)
        Q = Vh.transpose(-1, -2)[..., :2]
        threshold = float(self.config.svd_threshold)
        rank = (S > threshold).sum(dim=-1).clamp(max=max(int(self.config.max_scale_rank), 0))
        return Q, rank

    def _step_rotation(self, fwd: ForwardResult, grad: GradResult, data: Data) -> None:
        aux = grad.aux
        covars = fwd.intermediates.get("covars")
        g_covars = aux.get("v_covars_total")
        h_covars = aux.get("h_covars_total")
        if covars is None or g_covars is None or h_covars is None:
            return

        quats = self.params.get("quats").data
        means = self.params.get("means").data
        rays = _view_rays(means, data.get("viewmats"))
        Omega = _skew(rays)
        C = _covars6_to_sym3(covars)

        C_theta = Omega @ C - C @ Omega
        Omega2 = Omega @ Omega
        C_thetatheta = Omega2 @ C - 2.0 * (Omega @ C @ Omega) + C @ Omega2

        j = _pack_sym3_upper(C_theta)
        h = _pack_sym3_upper(C_thetatheta)
        H_cov = _upper_tri6_to_sym(h_covars)

        grad_theta = (g_covars * j).sum(dim=-1)
        hess_theta_raw = torch.einsum("ni,nij,nj->n", j, H_cov, j) + (g_covars * h).sum(dim=-1)
        hess_theta = torch.clamp(hess_theta_raw, min=self.config.get_clamp("quats"))

        delta_theta = -grad_theta / (hess_theta + self.config.get_damping("quats"))
        raw_angle = self.config.get_lr("quats") * delta_theta
        angle = raw_angle
        theta_max = float(getattr(self.config, "max_rotation_angle", 5.0e-2))
        if math.isfinite(theta_max) and theta_max > 0.0:
            angle = torch.clamp(angle, min=-theta_max, max=theta_max)
        max_update = self.config.get_max_update("quats")
        angle = _clip_by_abs(angle, max_update)

        half = 0.5 * angle
        dq = torch.cat(
            [torch.cos(half).unsqueeze(-1), rays * torch.sin(half).unsqueeze(-1)],
            dim=-1,
        )
        quats.copy_(_normalize(_quat_multiply(dq, quats)))
        if self.config.verbose:
            self._verbose_newton_debug(
                name="quats(theta)",
                raw_grad=g_covars,
                raw_hess=H_cov,
                reduced_grad=grad_theta.unsqueeze(-1),
                reduced_hess=hess_theta_raw.unsqueeze(-1).unsqueeze(-1),
                raw_hess_eigs=torch.linalg.eigvalsh(0.5 * (H_cov + H_cov.transpose(-1, -2))),
                reduced_hess_eigs=hess_theta_raw.unsqueeze(-1),
                raw_update=raw_angle.unsqueeze(-1),
                clipped_update=angle.unsqueeze(-1),
                extra="raw Hessian is d2L/d packed covars; reduced variable is view-axis angle",
            )

    def _step_opacity(self, grad: GradResult, hess: HessResult) -> None:
        g = grad.get("opacity")
        h = hess.get("opacity")
        if g is None or h is None:
            return
        opacity = self.params.get("opacity").data
        opacity_before = opacity.detach().clone()
        h_raw = h
        h_clamped = torch.clamp(h_raw, min=self.config.get_clamp("opacity"))
        delta = -g / (h_clamped + self.config.get_damping("opacity"))
        raw_update = self.config.get_lr("opacity") * delta
        update = _clip_by_abs(raw_update, self.config.get_max_update("opacity"))
        opacity.add_(update)
        min_opacity = float(getattr(self.config, "min_opacity", 1.0e-4))
        floor_count = 0
        if math.isfinite(min_opacity):
            floor_mask = opacity < min_opacity
            floor_count = int(floor_mask.sum().item())
            opacity.clamp_(min=min_opacity)
        effective_update = opacity - opacity_before
        if self.config.verbose:
            self._verbose_newton_debug(
                name="opacity",
                raw_grad=g,
                raw_hess=h_raw,
                reduced_grad=g.unsqueeze(-1),
                reduced_hess=h_raw.unsqueeze(-1).unsqueeze(-1),
                raw_hess_eigs=h_raw.unsqueeze(-1),
                reduced_hess_eigs=h_raw.unsqueeze(-1),
                raw_update=raw_update,
                clipped_update=effective_update,
                extra=(
                    f"opacity_floor: min_opacity={min_opacity:.4e}, "
                    f"floored={floor_count}/{opacity.numel()}"
                ),
            )

    def _apply_constraints(self) -> None:
        self._clamp_means_()
        q_param = self.params.get("quats")
        if q_param is not None:
            q_param.data.copy_(_normalize(q_param.data))
        opacity = self.params.get("opacity")
        if opacity is not None:
            min_opacity = float(getattr(self.config, "min_opacity", 1.0e-4))
            if math.isfinite(min_opacity):
                opacity.data.clamp_(min=min_opacity)
            elif not bool(self.config.allow_negative_opacity):
                opacity.data.clamp_(min=0.0)

    def _means_bound(self) -> float:
        return float(getattr(self.config, "means_bound", 1.0))

    def _clamp_means_(self) -> int:
        means_param = self.params.get("means")
        if means_param is None:
            return 0
        bound = self._means_bound()
        if not math.isfinite(bound) or bound <= 0.0:
            return 0
        means = means_param.data
        mask = (means < -bound) | (means > bound)
        count = int(mask.sum().item())
        means.clamp_(min=-bound, max=bound)
        return count

    def _verbose(self, name: str, update: torch.Tensor, extra: str = "") -> None:
        if not bool(self.config.verbose):
            return
        suffix = f", {extra}" if extra else ""
        print(f"[ReducedNewton][step={self._step_count}][{name}] {_safe_item_stats('update', update)}{suffix}")

    def _verbose_newton_debug(
        self,
        *,
        name: str,
        raw_grad: torch.Tensor,
        raw_hess: torch.Tensor,
        reduced_grad: torch.Tensor,
        reduced_hess: torch.Tensor,
        raw_hess_eigs: torch.Tensor,
        reduced_hess_eigs: torch.Tensor,
        raw_update: torch.Tensor,
        clipped_update: torch.Tensor,
        extra: str = "",
    ) -> None:
        if not bool(self.config.verbose):
            return
        lines = [
            f"[ReducedNewton][step={self._step_count}][{name}]",
            f"  {_safe_item_stats('raw_grad', raw_grad)}; {_safe_item_sample('raw_grad', raw_grad)}",
            f"  {_safe_item_stats('raw_hess', raw_hess)}; {_safe_item_sample('raw_hess', raw_hess)}",
            f"  {_safe_eig_stats('raw_hess_eigs', raw_hess_eigs)}",
            f"  {_safe_item_stats('reduced_grad', reduced_grad)}; {_safe_item_sample('reduced_grad', reduced_grad)}",
            f"  {_safe_item_stats('reduced_hess', reduced_hess)}; {_safe_item_sample('reduced_hess', reduced_hess)}",
            f"  {_safe_eig_stats('reduced_hess_eigs', reduced_hess_eigs)}",
            f"  {_safe_item_stats('raw_update', raw_update)}; {_safe_item_sample('raw_update', raw_update)}",
            f"  {_safe_item_stats('clipped_update', clipped_update)}; {_safe_item_sample('clipped_update', clipped_update)}",
            f"  {_safe_clip_stats(raw_update, clipped_update)}",
        ]
        lines.extend(self._verbose_param_distribution_lines())
        if extra:
            lines.append(f"  extra: {extra}")
        print("\n".join(lines))

    def _verbose_param_distribution_lines(self) -> List[str]:
        lines = ["  gaussian_params_after_update:"]

        means_param = self.params.get("means")
        if means_param is not None:
            means = means_param.data
            lines.append(f"    {_safe_item_stats('means_xyz', means)}")
            lines.append(f"    {_safe_item_stats('means_norm', means.norm(dim=-1))}")

        scale_param = self.params.get("scale_ori")
        if scale_param is not None:
            scale_raw = scale_param.data
            lines.append(f"    {_safe_item_stats('scale_raw', scale_raw)}")
            activated = self._activated_scale_for_verbose(scale_raw)
            if activated is not None:
                lines.append(f"    {_safe_item_stats('scale_activated', activated)}")

        quats_param = self.params.get("quats")
        if quats_param is not None:
            quats = quats_param.data
            lines.append(f"    {_safe_item_stats('quats_wxyz', quats)}")
            lines.append(f"    {_safe_item_stats('quat_norm', quats.norm(dim=-1))}")

        opacity_param = self.params.get("opacity")
        if opacity_param is not None:
            lines.append(f"    {_safe_item_stats('opacity', opacity_param.data)}")

        return lines

    def _activated_scale_for_verbose(self, scale_raw: torch.Tensor) -> Optional[torch.Tensor]:
        cfg = self._last_operator_config
        if cfg is None:
            return None
        try:
            scale_min = float(cfg.get("scale_min", 0.01))
            scale_max = float(cfg.get("scale_max", 2.0))
        except Exception:
            return None
        return scale_min + (scale_max - scale_min) * torch.sigmoid(scale_raw)

    def state_dict(self) -> Dict:
        return {
            "step_count": int(self._step_count),
            "lr_map": self.config.lr_map.copy(),
            "damping_map": self.config.damping_map.copy(),
            "clamp_map": self.config.clamp_map.copy(),
            "max_update_map": self.config.max_update_map.copy(),
            "min_opacity": float(getattr(self.config, "min_opacity", 1.0e-4)),
            "means_bound": float(getattr(self.config, "means_bound", 1.0)),
        }

    def load_state_dict(self, state: Dict) -> None:
        self._step_count = int(state.get("step_count", 0))
        if "lr_map" in state:
            self.config.lr_map = state["lr_map"].copy()
        if "damping_map" in state:
            self.config.damping_map = state["damping_map"].copy()
        if "clamp_map" in state:
            self.config.clamp_map = state["clamp_map"].copy()
        if "max_update_map" in state:
            self.config.max_update_map = state["max_update_map"].copy()
        if "min_opacity" in state:
            self.config.min_opacity = float(state["min_opacity"])
        if "means_bound" in state:
            self.config.means_bound = float(state["means_bound"])


def create_gs3d_ct_reduced_newton(
    params: Params,
    lr: float = 1.0,
    damping: float = 1e-3,
    clamp_min: float = 1e-8,
    max_update: float = float("inf"),
    **kwargs,
) -> GS3DCTReducedNewtonOptimizer:
    lr_map = {name: lr for name in PARAM_NAMES}
    damping_map = {name: damping for name in PARAM_NAMES}
    clamp_map = {name: clamp_min for name in PARAM_NAMES}
    max_update_map = {name: max_update for name in PARAM_NAMES}
    config = ReducedNewtonConfig(
        lr_map=lr_map,
        damping_map=damping_map,
        clamp_map=clamp_map,
        max_update_map=max_update_map,
        **kwargs,
    )
    return GS3DCTReducedNewtonOptimizer(params, config)


__all__ = [
    "ReducedNewtonConfig",
    "GS3DCTReducedNewtonOptimizer",
    "create_gs3d_ct_reduced_newton",
]
