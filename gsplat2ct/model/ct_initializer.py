"""CT Gaussian online initializer.

This module reconstructs a coarse volume from training projections and then
samples initial Gaussian centers/densities from that volume.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

try:
    import tigre
    import tigre.algorithms as tigre_algs
except Exception as e:  # pragma: no cover
    tigre = None
    tigre_algs = None
    _tigre_import_error = e
else:
    _tigre_import_error = None


@dataclass
class InitResult:
    """Result container for online initialization.

    Attributes:
        coarse_volume_wdh: Reconstructed coarse volume in (W, D, H) order.
            ``None`` when reconstruction is skipped (e.g. random mode).
        sampled_points_n4: Array of shape (N, 4) with columns (x, y, z, density).
    """

    coarse_volume_wdh: np.ndarray | None
    sampled_points_n4: np.ndarray


def _build_tigre_geometry(scanner_cfg: Dict[str, Any]):
    """Build TIGRE geometry from scanner metadata.

    Axis conversion follows the same convention as the legacy r2 pipeline.
    """
    if tigre is None:  # pragma: no cover
        raise ImportError(f"Failed to import TIGRE: {_tigre_import_error}")

    mode = str(scanner_cfg["mode"]).lower()
    if mode == "parallel":
        geo = tigre.geometry(mode="parallel", nVoxel=np.array(scanner_cfg["nVoxel"][::-1]))
    elif mode == "cone":
        geo = tigre.geometry(mode="cone")
    else:
        raise NotImplementedError(f"Unsupported scanner mode: {mode}")

    geo.DSD = scanner_cfg["DSD"]
    geo.DSO = scanner_cfg["DSO"]
    geo.nDetector = np.array(scanner_cfg["nDetector"])
    geo.sDetector = np.array(scanner_cfg["sDetector"])
    geo.dDetector = geo.sDetector / geo.nDetector

    geo.nVoxel = np.array(scanner_cfg["nVoxel"][::-1])
    geo.sVoxel = np.array(scanner_cfg["sVoxel"][::-1])
    geo.dVoxel = geo.sVoxel / geo.nVoxel

    geo.offOrigin = np.array(scanner_cfg["offOrigin"][::-1])
    geo.offDetector = np.array([scanner_cfg["offDetector"][1], scanner_cfg["offDetector"][0], 0])

    geo.accuracy = scanner_cfg["accuracy"]
    geo.filter = scanner_cfg.get("filter", None)
    return geo


def _load_train_projections_and_angles(data_dir: str, proj_train: List[Dict[str, Any]]):
    """Load train projections and corresponding angles from metadata."""
    projections = []
    angles = []
    for info in proj_train:
        proj_path = os.path.join(data_dir, info["file_path"])
        proj = np.load(proj_path)
        if proj.ndim == 3 and proj.shape[-1] == 1:
            proj = proj[..., 0]
        if proj.ndim != 2:
            raise ValueError(f"Unexpected projection shape {proj.shape} at {proj_path}")
        projections.append(proj.astype(np.float32, copy=False))
        angles.append(float(info["angle"]))

    if not projections:
        raise RuntimeError("No training projections found for initialization")

    projs = np.stack(projections, axis=0)
    angles_arr = np.asarray(angles, dtype=np.float32)
    return projs, angles_arr


def _reconstruct_volume(
    projs: np.ndarray,
    angles: np.ndarray,
    geo,
    recon_method: str,
    sart_n_iter: int,
) -> np.ndarray:
    """Run classic CT reconstruction and return (W, D, H) volume."""
    if tigre_algs is None:  # pragma: no cover
        raise ImportError(f"Failed to import TIGRE algorithms: {_tigre_import_error}")

    method = recon_method.lower()
    if method == "sart":
        # TIGRE uses detector-v axis opposite to dataset storage; flip v axis.
        vol = tigre_algs.sart(projs[:, ::-1, :], geo, angles, int(sart_n_iter), verbose=False)
    elif method == "fdk":
        vol = tigre_algs.fdk(projs[:, ::-1, :], geo, angles)
    else:
        raise ValueError(f"Unsupported recon_method={recon_method}, expected one of [sart, fdk, random]")

    # TIGRE outputs in (Z, Y, X); convert to project convention (W=X, D=Y, H=Z).
    return np.transpose(vol, (2, 1, 0)).astype(np.float32, copy=False)


def initialize_from_metadata(
    *,
    data_dir: str,
    scanner_cfg: Dict[str, Any],
    proj_train: List[Dict[str, Any]],
    recon_method: str = "sart",
    n_points: int = 50000,
    density_thresh: float = 0.05,
    density_rescale: float = 0.15,
    random_density_max: float = 1.0,
    sart_n_iter: int = 20,
) -> InitResult:
    """Initialize Gaussian points from projections + metadata.

    The sampling logic matches the legacy initialize_pcd script:
    threshold -> random voxel sampling -> world coordinate + density extraction.
    """
    method = recon_method.lower()
    if method not in {"random", "sart", "fdk"}:
        raise ValueError("recon_method must be one of [random, sart, fdk]")

    if method == "random":
        sampled_positions = np.array(scanner_cfg["offOrigin"], dtype=np.float32)[None, ...] + np.array(
            scanner_cfg["sVoxel"], dtype=np.float32
        )[None, ...] * (np.random.rand(int(n_points), 3).astype(np.float32) - 0.5)
        sampled_densities = np.random.rand(int(n_points)).astype(np.float32) * float(random_density_max)
        sampled_points = np.concatenate([sampled_positions, sampled_densities[:, None]], axis=-1)
        return InitResult(coarse_volume_wdh=None, sampled_points_n4=sampled_points)

    projs, angles = _load_train_projections_and_angles(data_dir, proj_train)
    geo = _build_tigre_geometry(scanner_cfg)
    coarse_vol = _reconstruct_volume(projs, angles, geo, method, sart_n_iter=sart_n_iter)

    density_mask = coarse_vol > float(density_thresh)
    valid_indices = np.argwhere(density_mask)
    if valid_indices.shape[0] < int(n_points):
        raise ValueError(
            "Valid voxels less than target number of sampling. "
            f"valid={valid_indices.shape[0]}, requested={int(n_points)}, "
            "consider lowering density_thresh or n_points."
        )

    sampled_indices = valid_indices[np.random.choice(len(valid_indices), int(n_points), replace=False)]

    off_origin = np.array(scanner_cfg["offOrigin"], dtype=np.float32)
    s_voxel = np.array(scanner_cfg["sVoxel"], dtype=np.float32)
    if "dVoxel" in scanner_cfg:
        d_voxel = np.array(scanner_cfg["dVoxel"], dtype=np.float32)
    else:
        n_voxel = np.array(scanner_cfg["nVoxel"], dtype=np.float32)
        d_voxel = s_voxel / n_voxel

    sampled_positions = sampled_indices.astype(np.float32) * d_voxel - s_voxel / 2.0 + off_origin
    sampled_densities = coarse_vol[
        sampled_indices[:, 0],
        sampled_indices[:, 1],
        sampled_indices[:, 2],
    ].astype(np.float32)
    sampled_densities = sampled_densities * float(density_rescale)

    sampled_points = np.concatenate([sampled_positions, sampled_densities[:, None]], axis=-1)
    return InitResult(coarse_volume_wdh=coarse_vol, sampled_points_n4=sampled_points)
