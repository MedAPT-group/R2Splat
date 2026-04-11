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
"""

import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from gsplat2ct.ops.data import Params


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
        opacity    (N,)    raw opacity values (non-negative, no activation)

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
        device: torch.device = torch.device("cuda"),
    ):
        self.device = device
        self.means = means.to(device)
        self.quats = quats.to(device)
        self.scale_ori = scale_ori.to(device)
        self.opacity = opacity.to(device)

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
        device: str = "cuda",
    ) -> "GaussianParams":
        """Load ``init_<scene>.npy`` and create parameter tensors.

        The ``.npy`` file is expected to have shape ``(N, >=4)`` where the
        first three columns are xyz and the fourth is opacity.

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

        opacity = opa_vals.to(dev)

        gp = cls(
            means=xyz_dev,
            quats=quats,
            scale_ori=scale_ori,
            opacity=opacity,
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
        device: str = "cuda",
    ) -> "GaussianParams":
        """Create Gaussian parameters from sampled points ``(N, >=4)``.

        Args:
            points: Array/tensor with columns ``(x, y, z, opacity, ...)``.
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
        opacity = opa_vals.to(dev)

        gp = cls(
            means=xyz_dev,
            quats=quats,
            scale_ori=scale_ori,
            opacity=opacity,
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
        return params

    def update_from_params(self, params: Params) -> None:
        """Sync internal tensors from a ``Params`` container (e.g. after
        optimizer step modified ``.data`` in-place)."""
        self.means = params.get("means").data
        self.quats = params.get("quats").data
        self.scale_ori = params.get("scale_ori").data
        self.opacity = params.get("opacity").data

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
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> Tuple["GaussianParams", int]:
        """Restore parameters from a ``.pt`` checkpoint."""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        gp = cls(
            means=ckpt["means"],
            quats=ckpt["quats"],
            scale_ori=ckpt["scale_ori"],
            opacity=ckpt["opacity"],
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
