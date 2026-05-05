"""Adam optimizer for GS3D CT reconstruction parameters."""

from __future__ import annotations

from gsplat2ct.ops.data import Params
from .base import FirstOrderConfig, FirstOrderOptimizerBase


PARAM_NAMES = ["means", "quats", "scale_ori", "opacity"]


def _renormalize_quats(params: Params, eps: float = 1e-12) -> None:
    p = params.get("quats")
    if p is None:
        return
    q = p.data
    if q.dim() != 2 or q.shape[-1] != 4:
        return
    q.div_(q.norm(dim=-1, keepdim=True).clamp_min(eps))


class GS3DCTAdamOptimizer(FirstOrderOptimizerBase):
    """Adam optimizer for raw GS3D CT parameters."""

    def __init__(self, params: Params, config: FirstOrderConfig):
        super().__init__(params, config, PARAM_NAMES)

    def _apply_constraints(self) -> None:
        _renormalize_quats(self.params)
        opacity = self.params.get("opacity")
        allow_negative_opacity = bool(getattr(self.config, "allow_negative_opacity", False))
        if opacity is not None and not allow_negative_opacity:
            opacity.data.clamp_(min=0.0)


def create_gs3d_ct_adam(
    params: Params,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    **per_param_lr,
) -> GS3DCTAdamOptimizer:
    lr_map = {name: float(lr) for name in PARAM_NAMES}
    lr_map.update({k: float(v) for k, v in per_param_lr.items()})
    config = FirstOrderConfig(lr_map=lr_map, beta1=beta1, beta2=beta2, eps=eps)
    return GS3DCTAdamOptimizer(params, config)


__all__ = ["PARAM_NAMES", "GS3DCTAdamOptimizer", "create_gs3d_ct_adam"]
