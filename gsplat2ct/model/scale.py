"""Scale-bound helpers shared by training and validation scripts."""

from __future__ import annotations

from typing import Any, Mapping, Tuple


def scanner_volume_to_world(scanner_cfg: Mapping[str, Any]) -> float:
    """Return r2_gaussian's volume-to-world scale, ``max(scanner.sVoxel)``."""
    s_voxel = scanner_cfg.get("sVoxel")
    if s_voxel is None:
        raise KeyError("scanner config is missing sVoxel")
    vals = [float(v) for v in s_voxel]
    if not vals:
        raise ValueError("scanner.sVoxel must not be empty")
    return max(vals)


def resolve_scale_bounds(
    scale_min_relative: float,
    scale_max_relative: float,
    scanner_cfg: Mapping[str, Any],
) -> Tuple[float, float, float]:
    """Convert relative r2_gaussian scale bounds to world-space bounds.

    In r2_gaussian, the Hydra-facing bounds are fractions of the reconstruction
    volume size.  The CUDA kernels, however, operate in the same world
    coordinate system as Gaussian means, so high-level scripts multiply by
    ``max(scanner.sVoxel)`` before initialization, rendering, voxelization, and
    densification.
    """
    volume_to_world = scanner_volume_to_world(scanner_cfg)
    return (
        float(scale_min_relative) * volume_to_world,
        float(scale_max_relative) * volume_to_world,
        volume_to_world,
    )


__all__ = ["resolve_scale_bounds", "scanner_volume_to_world"]
