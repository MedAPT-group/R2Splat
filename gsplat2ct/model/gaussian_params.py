"""Gaussian parameter initialization and serialization.

``GaussianParams`` bridges the gap between on-disk data (``init_<scene>.npy``)
and the ``Params`` container consumed by ``GS3DCTLossCUDA``.

Unlike r2splat1 where ``GaussianModel`` is an ``nn.Module`` that applies
activation functions externally, here the raw (pre-activation) parameters
are stored directly because ``ParameterTransform`` inside the CUDA operator
handles all activations.

Scale activation in our kernel:
    activated_scale = scale_min + (scale_max - scale_min) * sigmoid(scale_ori)
    => scale_ori = logit((target_scale - scale_min) / (scale_max - scale_min))

The kernel expects ``scale_min`` / ``scale_max`` in world units.  Hydra
configs expose ``scale_min_relative`` / ``scale_max_relative`` as fractions
of ``max(scanner.sVoxel)``; training and validation scripts resolve them
before calling this class or any CUDA operator.
"""

import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from gsplat2ct.ops.data import Params
from .opacity import inverse_opacity_activation


class _ParamWrapper:
    """Minimal wrapper that satisfies the ``ParameterLike`` protocol."""

    def __init__(self, data: torch.Tensor):
        self.data = data


class GaussianParams:
    """Manage raw Gaussian parameters and convert to/from ``Params``.

    Parameters stored:
        means      (N, 3)  world-space centres
        quats      (N, 4)  unit quaternions (w, x, y, z)
        scale_ori  (N, 3)  raw scale values (pre-sigmoid)
        opacity    (N,)    raw opacity values (pre-softplus)

    Args:
        means: Centre positions.
        quats: Unit quaternions.
        scale_ori: Raw scale parameters.
        opacity: Raw opacity values.
        device: Torch device.
    """

    def __init__(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        scale_ori: torch.Tensor,
        opacity: torch.Tensor,
        beta: Optional[torch.Tensor] = None,
        primitive_type: str = "gaussian",
        beta_gamma_init: float = 3.5,
        beta_gamma_min: Optional[float] = None,
        device: torch.device = torch.device("cuda"),
    ):
        self.device = device
        self.primitive_type = str(primitive_type).lower()
        if self.primitive_type not in {"gaussian", "beta"}:
            raise ValueError("primitive_type must be 'gaussian' or 'beta'")
        # Legacy checkpoints/configs may still carry beta_gamma_min, but the
        # current activation is gamma = beta_gamma_init * exp(raw_beta).
        self.beta_gamma_min = 0.0
        self.beta_gamma_init = float(beta_gamma_init)
        self.means = means.to(device)
        self.quats = quats.to(device)
        self.scale_ori = scale_ori.to(device)
        self.opacity = opacity.to(device)
        self.beta = beta.to(device) if beta is not None else None
        if self.primitive_type == "beta" and self.beta is None:
            self.beta = torch.zeros(
                (self.means.shape[0],), dtype=self.means.dtype, device=device
            )

    @property
    def num_gaussians(self) -> int:
        return self.means.shape[0]

    # ------------------------------------------------------------------
    # Construction from data directory
    # ------------------------------------------------------------------

    @classmethod
    def from_data_dir(
        cls,
        data_dir: str,
        *,
        scale_min: float = 0.001,
        scale_max: float = 1.0,
        global_scale: float = 1.0,
        init_opa_scale: float = 1.0,
        init_scale_multiplier: float = 0.5,
        bbox: Optional[torch.Tensor] = None,
        primitive_type: str = "gaussian",
        beta_gamma_init: float = 3.5,
        beta_gamma_min: Optional[float] = None,
        device: str = "cuda",
    ) -> "GaussianParams":
        """Load ``init_<scene>.npy`` and create parameter tensors.

        The ``.npy`` file is expected to have shape ``(N, >=4)`` where the
        first three columns are xyz and the fourth is opacity.

        ``scale_min`` and ``scale_max`` are absolute world-space bounds.
        Initial per-Gaussian scale is derived from nearest-neighbour
        distances, then mapped to raw ``scale_ori`` via the inverse of the
        kernel's sigmoid activation so that the *activated* scale matches the
        NN distance.
        """
        scene = os.path.basename(os.path.normpath(data_dir))
        init_path = os.path.join(data_dir, f"init_{scene}.npy")
        if not os.path.exists(init_path):
            raise FileNotFoundError(
                f"Initialization file required but missing: {init_path}"
            )

        dev = torch.device(device)
        data = np.load(init_path)
        assert data.shape[1] >= 4, "init npy needs >= 4 columns (x, y, z, opacity)"

        xyz = torch.from_numpy(data[:, :3]).float() * global_scale
        opa_vals = torch.from_numpy(data[:, 3]).float()
        opa_vals = (opa_vals * init_opa_scale).clamp(min=1e-6, max=10.0)

        n_points = xyz.shape[0]
        xyz_dev = xyz.to(dev).contiguous()

        nn_dist = cls._compute_nn_distances(xyz_dev)
        nn_dist = nn_dist.clamp_min(2e-4)
        target_scale = nn_dist * init_scale_multiplier

        target_scale_3 = target_scale.unsqueeze(-1).expand(-1, 3)
        scale_ori = cls._inverse_scale_activation(
            target_scale_3, scale_min, scale_max
        )

        quats = torch.zeros((n_points, 4), device=dev)
        quats[:, 0] = 1.0

        opacity = inverse_opacity_activation(opa_vals.to(dev))
        beta = None
        if str(primitive_type).lower() == "beta":
            beta = torch.zeros((n_points,), dtype=xyz_dev.dtype, device=dev)

        gp = cls(
            means=xyz_dev,
            quats=quats,
            scale_ori=scale_ori,
            opacity=opacity,
            beta=beta,
            primitive_type=primitive_type,
            beta_gamma_init=beta_gamma_init,
            beta_gamma_min=beta_gamma_min,
            device=dev,
        )

        if bbox is not None:
            gp.crop_to_bbox(bbox)
        return gp

    @classmethod
    def from_points(
        cls,
        points: Union[np.ndarray, torch.Tensor],
        *,
        scale_min: float = 0.001,
        scale_max: float = 1.0,
        global_scale: float = 1.0,
        init_opa_scale: float = 1.0,
        init_scale_multiplier: float = 0.5,
        bbox: Optional[torch.Tensor] = None,
        primitive_type: str = "gaussian",
        beta_gamma_init: float = 3.5,
        beta_gamma_min: Optional[float] = None,
        device: str = "cuda",
    ) -> "GaussianParams":
        """Create Gaussian parameters from sampled points ``(N, >=4)``.

        Args:
            points: Array/tensor with columns ``(x, y, z, opacity, ...)``.
            scale_min: Absolute world-space lower scale bound.
            scale_max: Absolute world-space upper scale bound.
        """
        dev = torch.device(device)
        if isinstance(points, np.ndarray):
            data = points
        elif isinstance(points, torch.Tensor):
            data = points.detach().cpu().numpy()
        else:
            raise TypeError(f"Unsupported points type: {type(points)}")

        if data.ndim != 2 or data.shape[1] < 4:
            raise ValueError("points must have shape (N, >=4) with columns (x, y, z, opacity)")

        xyz = torch.from_numpy(data[:, :3]).float() * float(global_scale)
        opa_vals = torch.from_numpy(data[:, 3]).float()
        opa_vals = (opa_vals * float(init_opa_scale)).clamp(min=1e-6, max=10.0)

        n_points = xyz.shape[0]
        xyz_dev = xyz.to(dev).contiguous()

        nn_dist = cls._compute_nn_distances(xyz_dev)
        nn_dist = nn_dist.clamp_min(2e-4)
        target_scale = nn_dist * float(init_scale_multiplier)

        target_scale_3 = target_scale.unsqueeze(-1).expand(-1, 3)
        scale_ori = cls._inverse_scale_activation(
            target_scale_3, scale_min, scale_max
        )

        quats = torch.zeros((n_points, 4), device=dev)
        quats[:, 0] = 1.0
        opacity = inverse_opacity_activation(opa_vals.to(dev))
        beta = None
        if str(primitive_type).lower() == "beta":
            beta = torch.zeros((n_points,), dtype=xyz_dev.dtype, device=dev)

        gp = cls(
            means=xyz_dev,
            quats=quats,
            scale_ori=scale_ori,
            opacity=opacity,
            beta=beta,
            primitive_type=primitive_type,
            beta_gamma_init=beta_gamma_init,
            beta_gamma_min=beta_gamma_min,
            device=dev,
        )

        if bbox is not None:
            gp.crop_to_bbox(bbox)
        return gp

    # ------------------------------------------------------------------
    # Params conversion
    # ------------------------------------------------------------------

    def to_params(self) -> Params:
        """Wrap current tensors into a ``Params`` container."""
        params = Params()
        params.add_param("means", _ParamWrapper(self.means))
        params.add_param("quats", _ParamWrapper(self.quats))
        params.add_param("scale_ori", _ParamWrapper(self.scale_ori))
        params.add_param("opacity", _ParamWrapper(self.opacity))
        if self.primitive_type == "beta" and self.beta is not None:
            params.add_param("beta", _ParamWrapper(self.beta))
        return params

    def update_from_params(self, params: Params) -> None:
        """Sync internal tensors from a ``Params`` container (e.g. after
        optimizer step modified ``.data`` in-place)."""
        self.means = params.get("means").data
        self.quats = params.get("quats").data
        self.scale_ori = params.get("scale_ori").data
        self.opacity = params.get("opacity").data
        beta_param = params.get("beta")
        if beta_param is not None:
            self.beta = beta_param.data
            self.primitive_type = "beta"

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def crop_to_bbox(self, bbox: torch.Tensor) -> None:
        """Remove Gaussians outside the bounding box (2, 3)."""
        mins = bbox[0].to(self.device)
        maxs = bbox[1].to(self.device)
        mask = ((self.means >= mins) & (self.means <= maxs)).all(dim=-1)
        if mask.sum() < mask.numel():
            self.means = self.means[mask].contiguous()
            self.quats = self.quats[mask].contiguous()
            self.scale_ori = self.scale_ori[mask].contiguous()
            self.opacity = self.opacity[mask].contiguous()
            if self.beta is not None:
                self.beta = self.beta[mask].contiguous()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str, step: int = -1) -> None:
        """Save raw parameters to a ``.pt`` checkpoint."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "step": step,
                "means": self.means.detach().cpu(),
                "quats": self.quats.detach().cpu(),
                "scale_ori": self.scale_ori.detach().cpu(),
                "opacity": self.opacity.detach().cpu(),
                "opacity_activation": "softplus",
                "beta": None if self.beta is None else self.beta.detach().cpu(),
                "primitive_type": self.primitive_type,
                "beta_activation": "scaled_exp",
                "beta_gamma_min": self.beta_gamma_min,
                "beta_gamma_init": self.beta_gamma_init,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> Tuple["GaussianParams", int]:
        """Restore parameters from a ``.pt`` checkpoint."""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        opacity_activation = ckpt.get("opacity_activation", None)
        if str(opacity_activation).lower() != "softplus":
            raise ValueError(
                "Checkpoint must store raw opacity with "
                "opacity_activation='softplus'."
            )

        gp = cls(
            means=ckpt["means"],
            quats=ckpt["quats"],
            scale_ori=ckpt["scale_ori"],
            opacity=ckpt["opacity"],
            beta=ckpt.get("beta", None),
            primitive_type=str(ckpt.get("primitive_type", "beta" if ckpt.get("beta", None) is not None else "gaussian")),
            beta_gamma_init=float(ckpt.get("beta_gamma_init", 3.5)),
            beta_gamma_min=float(ckpt.get("beta_gamma_min", 0.0)),
            device=torch.device(device),
        )
        return gp, int(ckpt.get("step", -1))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_nn_distances(xyz: torch.Tensor) -> torch.Tensor:
        """Compute distance to nearest neighbour for each point."""
        n = xyz.shape[0]
        batch_size = 10000
        chunks = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            chunk = xyz[start:end]
            min_dists = []
            for t_start in range(0, n, batch_size):
                t_end = min(t_start + batch_size, n)
                targets = xyz[t_start:t_end]
                d = torch.cdist(chunk, targets)
                chunk_idx = torch.arange(
                    start, end, device=xyz.device
                ).unsqueeze(1)
                target_idx = torch.arange(
                    t_start, t_end, device=xyz.device
                ).unsqueeze(0)
                d = d.masked_fill(chunk_idx == target_idx, float("inf"))
                min_dists.append(d.min(dim=1).values)
            min_dists = torch.stack(min_dists, dim=1).min(dim=1).values
            min_dists[~torch.isfinite(min_dists)] = 1e-2
            chunks.append(min_dists)
        return torch.cat(chunks, dim=0)

    @staticmethod
    def _inverse_scale_activation(
        target_scale: torch.Tensor,
        scale_min: float,
        scale_max: float,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Compute raw ``scale_ori`` such that
        ``scale_min + (scale_max - scale_min) * sigmoid(scale_ori) == target_scale``."""
        target_clamped = target_scale.clamp(
            min=scale_min + eps, max=scale_max - eps
        )
        sigma = (target_clamped - scale_min) / (scale_max - scale_min)
        sigma = sigma.clamp(min=eps, max=1.0 - eps)
        return torch.logit(sigma)
