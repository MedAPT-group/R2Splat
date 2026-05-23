#!/usr/bin/env python3
"""Run TIGRE classical CT baselines on the r2splat/R2-Gaussian dataset format.

The script reconstructs a volume from ``proj_train`` using FDK, SART, or
ASD-POCS, evaluates 3-D quality against ``vol_gt.npy``, then forward-projects
the reconstruction at ``proj_test`` angles for 2-D PSNR/SSIM.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import tigre
import tigre.algorithms as algs
from tigre.utilities.im3Dnorm import im3DNORM

from gsplat2ct.eval.metric import metric_proj, metric_vol


DEFAULT_SUBDIRS = [
    "cone_ntrain_25_angle_360",
    "cone_ntrain_50_angle_360",
    "cone_ntrain_75_angle_360",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TIGRE FDK/SART/ASD-POCS baselines."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Root directory containing cone_ntrain_* subsets.",
    )
    parser.add_argument(
        "--subdirs",
        nargs="+",
        default=DEFAULT_SUBDIRS,
        help="Subset directories under dataset root.",
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        default=[],
        help="Optional sample directory names to run in each selected subset.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["fdk", "cgls", "sart", "asd_pocs"],
        choices=["fdk", "cgls", "sart", "asd_pocs"],
        help="TIGRE algorithms to run.",
    )
    parser.add_argument(
        "--output-root",
        default="output/breast_exp/tigre",
        help="Parent output directory. Results are grouped by method/subdir/sample.",
    )
    parser.add_argument(
        "--sart-iters",
        type=int,
        default=20,
        help="Number of SART iterations.",
    )
    parser.add_argument(
        "--cgls-iters",
        type=int,
        default=20,
        help="Number of CGLS iterations.",
    )
    parser.add_argument(
        "--asd-pocs-iters",
        type=int,
        default=10,
        help="Number of ASD-POCS outer iterations.",
    )
    parser.add_argument(
        "--asd-tv-iters",
        type=int,
        default=20,
        help="Number of TV iterations per ASD-POCS outer iteration.",
    )
    parser.add_argument(
        "--asd-epsilon-scale",
        type=float,
        default=0.15,
        help="Scale applied to the FDK residual norm for ASD-POCS maxl2err.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for metric computation. TIGRE reconstruction still uses its CUDA backend.",
    )
    parser.add_argument(
        "--ssim-window",
        type=int,
        default=11,
        help="SSIM window size, matching r2splat evaluation.",
    )
    parser.add_argument(
        "--no-clamp",
        action="store_true",
        help="Do not clamp reconstructed volume/projections to [0, 1] before metrics.",
    )
    parser.add_argument(
        "--save-volume",
        action="store_true",
        help="Save reconstructed volume as pred_volume.npy.",
    )
    parser.add_argument(
        "--save-test-projections",
        action="store_true",
        help="Save TIGRE forward projections of the reconstruction.",
    )
    parser.add_argument(
        "--skip-finished",
        action="store_true",
        help="Skip a sample if a metrics_final.json already exists for it.",
    )
    parser.add_argument(
        "--verbose-tigre",
        action="store_true",
        help="Let TIGRE print iterative algorithm progress.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected jobs without running reconstruction.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fin:
        return json.load(fin)


def get_geometry_tigre(scanner: Dict[str, Any]) -> Any:
    """Build TIGRE geometry using the same convention as R2-Gaussian data generation."""
    mode = str(scanner["mode"]).lower()
    if mode == "parallel":
        geo = tigre.geometry(mode="parallel", nVoxel=np.array(scanner["nVoxel"][::-1]))
    elif mode == "cone":
        geo = tigre.geometry(mode="cone")
    else:
        raise NotImplementedError(f"Unsupported scanner mode: {scanner['mode']}")

    geo.DSD = float(scanner["DSD"])
    geo.DSO = float(scanner["DSO"])

    geo.nDetector = np.asarray(scanner["nDetector"], dtype=np.int32)
    geo.sDetector = np.asarray(scanner["sDetector"], dtype=np.float32)
    geo.dDetector = geo.sDetector / geo.nDetector

    geo.nVoxel = np.asarray(scanner["nVoxel"][::-1], dtype=np.int32)
    geo.sVoxel = np.asarray(scanner["sVoxel"][::-1], dtype=np.float32)
    geo.dVoxel = geo.sVoxel / geo.nVoxel

    geo.offOrigin = np.asarray(scanner.get("offOrigin", [0, 0, 0])[::-1], dtype=np.float32)
    off_detector = scanner.get("offDetector", [0, 0])
    geo.offDetector = np.asarray([off_detector[1], off_detector[0], 0], dtype=np.float32)

    geo.accuracy = float(scanner.get("accuracy", 0.5))
    geo.filter = scanner.get("filter", None)
    return geo


def load_projection_stack(sample_dir: Path, entries: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    projections: List[np.ndarray] = []
    angles: List[float] = []
    for entry in entries:
        proj = np.load(sample_dir / entry["file_path"]).astype(np.float32, copy=False)
        if proj.ndim != 2:
            raise ValueError(f"Expected 2-D projection, got {proj.shape}: {entry['file_path']}")
        projections.append(proj)
        angles.append(float(entry["angle"]))
    return np.ascontiguousarray(np.stack(projections, axis=0)), np.asarray(angles, dtype=np.float32)


def normalize_tigre_volume(vol_tigre: np.ndarray) -> np.ndarray:
    """Convert TIGRE output ``(Z, Y, X)`` to dataset volume order ``(X, Y, Z)``."""
    return np.ascontiguousarray(vol_tigre.transpose(2, 1, 0).astype(np.float32, copy=False))


def to_tigre_volume(vol_xyz: np.ndarray) -> np.ndarray:
    """Convert dataset volume order ``(X, Y, Z)`` to TIGRE order ``(Z, Y, X)``."""
    return np.ascontiguousarray(vol_xyz.transpose(2, 1, 0).astype(np.float32, copy=False))


def run_reconstruction(
    *,
    method: str,
    train_projs: np.ndarray,
    train_angles: np.ndarray,
    geo: Any,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    # The datasets were generated as tigre.Ax(... )[:, ::-1, :], so flip back
    # before handing projections to TIGRE reconstruction algorithms.
    projs_tigre = np.ascontiguousarray(train_projs[:, ::-1, :].astype(np.float32, copy=False))
    algorithm_info: Dict[str, Any] = {"method": method}

    if method == "fdk":
        vol_tigre = algs.fdk(projs_tigre, geo, train_angles, verbose=args.verbose_tigre)
    elif method == "cgls":
        algorithm_info.update({"niter": args.cgls_iters, "computel2": False})
        vol_tigre = algs.cgls(
            projs_tigre,
            geo,
            train_angles,
            args.cgls_iters,
            computel2=False,
            verbose=args.verbose_tigre,
        )
    elif method == "sart":
        algorithm_info.update({"niter": args.sart_iters, "computel2": False})
        vol_tigre = algs.sart(
            projs_tigre,
            geo,
            train_angles,
            args.sart_iters,
            computel2=False,
            verbose=args.verbose_tigre,
        )
    elif method == "asd_pocs":
        fdk_init = algs.fdk(projs_tigre, geo, train_angles, verbose=False)
        residual = tigre.Ax(fdk_init, geo, train_angles) - projs_tigre
        epsilon = float(im3DNORM(residual, 2) * args.asd_epsilon_scale)
        algorithm_info.update(
            {
                "niter": args.asd_pocs_iters,
                "tviter": args.asd_tv_iters,
                "maxl2err": epsilon,
                "epsilon_scale": args.asd_epsilon_scale,
                "alpha": 0.002,
                "lmbda": 1.0,
                "lmbda_red": 0.9999,
                "rmax": 0.94,
            }
        )
        vol_tigre = algs.asd_pocs(
            projs_tigre,
            geo,
            train_angles,
            args.asd_pocs_iters,
            tviter=args.asd_tv_iters,
            maxl2err=epsilon,
            alpha=0.002,
            lmbda=1.0,
            lmbda_red=0.9999,
            rmax=0.94,
            verbose=args.verbose_tigre,
        )
    else:
        raise ValueError(f"Unsupported method: {method}")

    return normalize_tigre_volume(vol_tigre), algorithm_info


def compute_metrics(
    *,
    pred_vol_xyz: np.ndarray,
    gt_vol_xyz: np.ndarray,
    test_projs: np.ndarray,
    test_angles: np.ndarray,
    geo: Any,
    args: argparse.Namespace,
) -> Tuple[Dict[str, float], np.ndarray]:
    clamp = not args.no_clamp
    pred_for_eval = pred_vol_xyz.astype(np.float32, copy=False)
    if clamp:
        pred_for_eval = np.clip(pred_for_eval, 0.0, 1.0)

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    pred_vol_t = torch.from_numpy(pred_for_eval).to(device=device, dtype=torch.float32)
    gt_vol_t = torch.from_numpy(gt_vol_xyz.astype(np.float32, copy=False)).to(device=device)
    vol_m = metric_vol(pred_vol_t, gt_vol_t, window_size=args.ssim_window)

    pred_test_tigre = tigre.Ax(to_tigre_volume(pred_for_eval), geo, test_angles)[:, ::-1, :].copy()
    pred_test = np.ascontiguousarray(pred_test_tigre.astype(np.float32, copy=False))
    if clamp:
        pred_test = np.clip(pred_test, 0.0, 1.0)

    pred_proj_t = torch.from_numpy(pred_test).to(device=device, dtype=torch.float32)
    gt_proj_t = torch.from_numpy(test_projs.astype(np.float32, copy=False)).to(device=device)
    proj_m = metric_proj(pred_proj_t, gt_proj_t, window_size=args.ssim_window)

    metrics = {
        "img_metrics/l1": float(proj_m["l1"].item()),
        "img_metrics/mse": float(proj_m["mse"].item()),
        "img_metrics/psnr": float(proj_m["psnr"].item()),
        "img_metrics/ssim": float(proj_m["ssim"].item()),
        "vol_metrics/l1": float(vol_m["l1"].item()),
        "vol_metrics/mse": float(vol_m["mse"].item()),
        "vol_metrics/psnr": float(vol_m["psnr"].item()),
        "vol_metrics/ssim": float(vol_m["ssim"].item()),
        "num_primitives": None,
        "primitive_count": None,
    }
    return metrics, pred_test


def discover_samples(dataset_root: Path, subdirs: Sequence[str], samples: Sequence[str]) -> List[Tuple[str, str, Path]]:
    jobs: List[Tuple[str, str, Path]] = []
    sample_filter = list(dict.fromkeys(samples))
    for subdir in subdirs:
        subset_dir = dataset_root / subdir
        if not subset_dir.is_dir():
            raise FileNotFoundError(f"Subset directory not found: {subset_dir}")
        if sample_filter:
            sample_dirs = [subset_dir / name for name in sample_filter]
        else:
            sample_dirs = sorted(path for path in subset_dir.iterdir() if path.is_dir())
        for sample_dir in sample_dirs:
            if not sample_dir.is_dir():
                raise FileNotFoundError(f"Sample directory not found: {sample_dir}")
            if not (sample_dir / "meta_data.json").is_file():
                raise FileNotFoundError(f"meta_data.json not found: {sample_dir}")
            jobs.append((subdir, sample_dir.name, sample_dir))
    return jobs


def latest_finished_run(method_root: Path, subdir: str, sample_name: str) -> Optional[Path]:
    subset_dir = method_root / subdir
    if not subset_dir.is_dir():
        return None
    candidates = [
        path
        for path in subset_dir.glob(f"*_{sample_name}")
        if (path / "metrics_final.json").is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2, sort_keys=False)


def evaluate_sample(
    *,
    method: str,
    subdir: str,
    sample_name: str,
    sample_dir: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    meta = load_json(sample_dir / "meta_data.json")
    scanner = meta["scanner"]
    train_geo = get_geometry_tigre(scanner)
    eval_geo = get_geometry_tigre(scanner)

    train_projs, train_angles = load_projection_stack(sample_dir, meta["proj_train"])
    test_projs, test_angles = load_projection_stack(sample_dir, meta["proj_test"])
    vol_relpath = meta.get("vol", meta.get("ct"))
    if vol_relpath is None:
        raise KeyError("meta_data.json must contain either 'vol' or 'ct'")
    gt_vol = np.load(sample_dir / vol_relpath).astype(np.float32, copy=False)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / method / subdir / f"{stamp}_{sample_name}"
    run_dir.mkdir(parents=True, exist_ok=False)

    start = time.time()
    pred_vol, algorithm_info = run_reconstruction(
        method=method,
        train_projs=train_projs,
        train_angles=train_angles,
        geo=train_geo,
        args=args,
    )
    reconstruction_sec = time.time() - start

    metrics_start = time.time()
    final_metrics, pred_test = compute_metrics(
        pred_vol_xyz=pred_vol,
        gt_vol_xyz=gt_vol,
        test_projs=test_projs,
        test_angles=test_angles,
        geo=eval_geo,
        args=args,
    )
    eval_sec = time.time() - metrics_start

    final_metrics["reconstruction_time_sec"] = float(reconstruction_sec)
    final_metrics["train_to_checkpoint_sec"] = float(reconstruction_sec)

    if args.save_volume:
        volume_to_save = np.clip(pred_vol, 0.0, 1.0) if not args.no_clamp else pred_vol
        np.save(run_dir / "pred_volume.npy", volume_to_save.astype(np.float32, copy=False))
    if args.save_test_projections:
        np.save(run_dir / "pred_test_projections.npy", pred_test.astype(np.float32, copy=False))

    payload: Dict[str, Any] = {
        "method": method,
        "algorithm": algorithm_info,
        "sample": {
            "subdir": subdir,
            "name": sample_name,
            "data_dir": str(sample_dir),
            "num_train_projections": int(train_projs.shape[0]),
            "num_test_projections": int(test_projs.shape[0]),
            "volume_shape": list(gt_vol.shape),
            "projection_shape": list(train_projs.shape[1:]),
        },
        "scanner": scanner,
        "postprocess": {
            "clamp_to_0_1": not args.no_clamp,
            "projection_flip_for_tigre": "projs[:, ::-1, :]",
            "volume_order": "dataset=(X,Y,Z), tigre=(Z,Y,X)",
        },
        "timing": {
            "reconstruction_sec": float(reconstruction_sec),
            "evaluation_sec": float(eval_sec),
            "train_to_checkpoint_sec": float(reconstruction_sec),
        },
        "final_metrics": final_metrics,
    }
    write_json(run_dir / "metrics_final.json", payload)
    return {"run_dir": str(run_dir), **payload}


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not output_root.is_absolute():
        output_root = (Path.cwd() / output_root).resolve()

    jobs = discover_samples(dataset_root, args.subdirs, args.samples)
    selected: List[Tuple[str, str, str, Path]] = []
    for method in args.methods:
        method_root = output_root / method
        for subdir, sample_name, sample_dir in jobs:
            if args.skip_finished and latest_finished_run(output_root / method, subdir, sample_name):
                print(f"[skip] {method} {subdir}/{sample_name}")
                continue
            selected.append((method, subdir, sample_name, sample_dir))

    print(f"Dataset root: {dataset_root}")
    print(f"Output root: {output_root}")
    print(f"Methods: {args.methods}")
    print(f"Subdirs: {args.subdirs}")
    print(f"Samples: {args.samples or '<all>'}")
    print(f"Total jobs: {len(selected)}")
    if args.dry_run:
        for idx, (method, subdir, sample_name, sample_dir) in enumerate(selected, start=1):
            print(f"  {idx:03d}. {method} {subdir}/{sample_name} -> {sample_dir}")
        return

    run_start = time.time()
    succeeded: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    for idx, (method, subdir, sample_name, sample_dir) in enumerate(selected, start=1):
        print(f"[{idx:03d}/{len(selected):03d}] start {method} {subdir}/{sample_name}", flush=True)
        try:
            result = evaluate_sample(
                method=method,
                subdir=subdir,
                sample_name=sample_name,
                sample_dir=sample_dir,
                output_root=output_root,
                args=args,
            )
            metrics = result["final_metrics"]
            print(
                "[done] "
                f"{method} {subdir}/{sample_name}: "
                f"3d_psnr={metrics['vol_metrics/psnr']:.3f}, "
                f"2d_psnr={metrics['img_metrics/psnr']:.3f}, "
                f"time={metrics['reconstruction_time_sec']:.1f}s",
                flush=True,
            )
            succeeded.append(result)
        except Exception as exc:
            print(f"[fail] {method} {subdir}/{sample_name}: {type(exc).__name__}: {exc}", flush=True)
            failed.append(
                {
                    "method": method,
                    "subdir": subdir,
                    "sample": sample_name,
                    "sample_dir": str(sample_dir),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    summary = {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "methods": list(args.methods),
        "subdirs": list(args.subdirs),
        "samples": list(args.samples),
        "sart_iters": args.sart_iters,
        "asd_pocs_iters": args.asd_pocs_iters,
        "asd_tv_iters": args.asd_tv_iters,
        "asd_epsilon_scale": args.asd_epsilon_scale,
        "clamp_to_0_1": not args.no_clamp,
        "total_wall_time_sec": time.time() - run_start,
        "succeeded": succeeded,
        "failed": failed,
    }
    summary_dir = output_root / "_summaries"
    summary_path = summary_dir / f"tigre_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    write_json(summary_path, summary)
    print("\n===== TIGRE summary =====")
    print(f"Succeeded: {len(succeeded)}")
    print(f"Failed: {len(failed)}")
    print(f"Summary JSON: {summary_path}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
