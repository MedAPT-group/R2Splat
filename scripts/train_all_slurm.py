#!/usr/bin/env python3
# Submit example:
# sbatch --time=12:00:00 --mem-per-cpu=8000M -w hardin01 --ntasks=8 --gres=gpu:4 scripts/train_all_slurm.py --gpus 4

import argparse
import os
import queue
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


DEFAULT_SUBDIRS = [
    "cone_ntrain_5_angle_360",
    "cone_ntrain_25_angle_360",
    "cone_ntrain_50_angle_360",
    "cone_ntrain_75_angle_360",
]


@dataclass(frozen=True)
class SampleJob:
    subset_name: str
    sample_name: str
    sample_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SLURM multi-GPU launcher: each GPU runs one sample training at a time, "
            "then pulls the next sample until all are finished."
        )
    )
    parser.add_argument(
        "--repo-root",
        default=os.environ.get("TRAIN_ALL_REPO_ROOT", ""),
        help=(
            "Repository root path (optional). If omitted, auto-detected from "
            "TRAIN_ALL_REPO_ROOT, SLURM_SUBMIT_DIR, then current working directory."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        default="~/terminus/users/xiaxiali/datasets/3DGS2CT/preprocessed/synthetic_dataset",
        help="Root directory containing cone_ntrain_* subsets.",
    )
    parser.add_argument(
        "--subdirs",
        nargs="+",
        default=DEFAULT_SUBDIRS,
        help=(
            "Subset directories under dataset root. "
            "Use space-separated values, e.g. --subdirs cone_ntrain_25_angle_360 cone_ntrain_50_angle_360"
        ),
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Total number of concurrent GPU workers (>=1).",
    )
    parser.add_argument(
        "--config-name",
        default="r2gaussian_train",
        help="Hydra config name passed to scripts.train.",
    )
    parser.add_argument(
        "--result-root",
        default="output/train_all",
        help="Parent directory for all per-sample result folders.",
    )
    parser.add_argument(
        "--logs-root",
        default="logs/train_all_slurm",
        help="Parent directory for per-sample launcher logs.",
    )
    parser.add_argument(
        "--activate-env-script",
        default=os.environ.get(
            "ACTIVATE_ENV_SCRIPT",
            f"{Path.home()}/myterminus/env/activate_env/activate_env_cu121.sh",
        ),
        help="Shell script to source before conda initialization.",
    )
    parser.add_argument(
        "--conda-env-name",
        default=os.environ.get("CONDA_ENV_NAME", "gsplat2"),
        help="Conda environment name to activate.",
    )
    parser.add_argument(
        "--conda-exe",
        default=os.environ.get(
            "CONDA_EXE",
            f"{os.environ.get('CONDA_ROOT', str(Path.home() / 'myterminus/miniconda3'))}/bin/conda",
        ),
        help="Path to conda executable used for shell hook initialization.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=True,
        help="Enable W&B tracking for each sample run (default: enabled).",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_false",
        dest="wandb",
        help="Disable W&B tracking.",
    )
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", ""))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--wandb-group", default="")
    parser.add_argument(
        "--skip-finished",
        action="store_true",
        help="Skip sample if output folder already exists and is non-empty.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run without actually launching training.",
    )
    parser.add_argument(
        "hydra_overrides",
        nargs=argparse.REMAINDER,
        help="Extra Hydra overrides passed through to scripts.train (prefix with --).",
    )
    return parser.parse_args()


def _strip_remainder_marker(overrides: Sequence[str]) -> List[str]:
    if overrides and overrides[0] == "--":
        return list(overrides[1:])
    return list(overrides)


def discover_sample_jobs(dataset_root: Path, subdirs: Sequence[str]) -> List[SampleJob]:
    jobs: List[SampleJob] = []
    for sub in subdirs:
        subset_dir = dataset_root / sub
        if not subset_dir.is_dir():
            raise FileNotFoundError(f"Subset directory not found: {subset_dir}")

        sample_dirs = sorted([p for p in subset_dir.iterdir() if p.is_dir()])
        if not sample_dirs:
            raise RuntimeError(f"No sample directories found under: {subset_dir}")

        for sample_dir in sample_dirs:
            jobs.append(SampleJob(subset_name=sub, sample_name=sample_dir.name, sample_dir=sample_dir))
    return jobs


def resolve_gpu_slots(requested: int) -> List[str]:
    if requested < 1:
        raise ValueError("--gpus must be >= 1")

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        visible = [tok.strip() for tok in cvd.split(",") if tok.strip()]
        if len(visible) < requested:
            raise RuntimeError(
                f"Requested {requested} GPUs, but CUDA_VISIBLE_DEVICES only exposes {len(visible)}: {visible}"
            )
        return visible[:requested]

    # Fallback when CUDA_VISIBLE_DEVICES is not set.
    return [str(i) for i in range(requested)]


def should_skip_output(subset_result_dir: Path, sample_name: str) -> bool:
    if not subset_result_dir.exists() or not subset_result_dir.is_dir():
        return False

    # Backward compatibility: if legacy non-timestamped sample dir contains outputs.
    legacy_dir = subset_result_dir / sample_name
    if legacy_dir.exists() and legacy_dir.is_dir() and any(legacy_dir.iterdir()):
        return True

    # Current behavior: outputs are written into sibling dirs like
    # "YYYYmmdd_HHMMSS_<sample_name>".
    pattern = f"*_{sample_name}"
    for candidate in subset_result_dir.glob(pattern):
        if candidate.is_dir() and any(candidate.iterdir()):
            return True

    return False


def build_worker_script(
    *,
    repo_root: Path,
    activate_env_script: Path,
    conda_exe: Path,
    conda_env_name: str,
    gpu_id: str,
    train_cmd: Sequence[str],
    wandb_enable: bool,
    wandb_project: str,
    wandb_entity: str,
    wandb_group: str,
    wandb_run_name: str,
    wandb_tags: str,
) -> str:
    train_cmd_str = shlex.join(list(train_cmd))

    lines = [
        "set -euo pipefail",
        f"source {shlex.quote(str(activate_env_script))}",
        f"if [[ ! -x {shlex.quote(str(conda_exe))} ]]; then echo 'ERROR: conda not executable: {str(conda_exe)}' >&2; exit 1; fi",
        f"eval \"$({shlex.quote(str(conda_exe))} shell.bash hook)\"",
        f"conda activate {shlex.quote(conda_env_name)}",
        f"cd {shlex.quote(str(repo_root))}",
        f"export CUDA_VISIBLE_DEVICES={shlex.quote(gpu_id)}",
        f"export WANDB_ENABLE={'1' if wandb_enable else '0'}",
    ]

    if wandb_enable:
        if wandb_project:
            lines.append(f"export WANDB_PROJECT={shlex.quote(wandb_project)}")
        if wandb_entity:
            lines.append(f"export WANDB_ENTITY={shlex.quote(wandb_entity)}")
        if wandb_group:
            lines.append(f"export WANDB_GROUP={shlex.quote(wandb_group)}")
        lines.append("export WANDB_JOB_TYPE=multi_dataset_train")
        lines.append(f"export WANDB_RUN_NAME={shlex.quote(wandb_run_name)}")
        lines.append(f"export WANDB_TAGS={shlex.quote(wandb_tags)}")

    lines.append(train_cmd_str)
    return "\n".join(lines)


def _looks_like_repo_root(path: Path) -> bool:
    return (path / "scripts" / "train.py").is_file()


def resolve_repo_root(repo_root_arg: str) -> Path:
    if repo_root_arg:
        candidate = Path(repo_root_arg).expanduser().resolve()
        if not _looks_like_repo_root(candidate):
            raise RuntimeError(f"--repo-root does not look like project root: {candidate}")
        return candidate

    for env_name in ("TRAIN_ALL_REPO_ROOT", "SLURM_SUBMIT_DIR"):
        raw = os.environ.get(env_name, "").strip()
        if not raw:
            continue
        candidate = Path(raw).expanduser().resolve()
        if _looks_like_repo_root(candidate):
            return candidate

    cwd_candidate = Path.cwd().resolve()
    if _looks_like_repo_root(cwd_candidate):
        return cwd_candidate

    script_candidate = Path(__file__).resolve().parents[1]
    if _looks_like_repo_root(script_candidate):
        return script_candidate

    raise RuntimeError(
        "Cannot resolve repository root. Please pass --repo-root /path/to/r2splat "
        "or set TRAIN_ALL_REPO_ROOT."
    )


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root(args.repo_root)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    result_root = (repo_root / args.result_root).resolve()
    logs_root = (repo_root / args.logs_root).resolve()
    activate_env_script = Path(args.activate_env_script).expanduser().resolve()
    conda_exe = Path(args.conda_exe).expanduser().resolve()
    hydra_overrides = _strip_remainder_marker(args.hydra_overrides)

    if not activate_env_script.is_file():
        raise FileNotFoundError(f"activate script not found: {activate_env_script}")
    if not conda_exe.is_file():
        raise FileNotFoundError(f"conda executable not found: {conda_exe}")

    gpu_slots = resolve_gpu_slots(args.gpus)
    jobs = discover_sample_jobs(dataset_root, args.subdirs)
    if not jobs:
        raise RuntimeError("No jobs found.")

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_run_dir = logs_root / run_stamp
    logs_run_dir.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)

    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    default_group = f"train_all_{run_stamp}" if not slurm_job_id else f"train_all_{slurm_job_id}"
    wandb_group = args.wandb_group or default_group

    # Apply skipping policy before enqueueing.
    filtered_jobs: List[SampleJob] = []
    for job in jobs:
        subset_out_dir = result_root / job.subset_name
        if args.skip_finished and should_skip_output(subset_out_dir, job.sample_name):
            print(
                f"[skip] {job.subset_name}/{job.sample_name} -> existing timestamped output under {subset_out_dir}"
            )
            continue
        filtered_jobs.append(job)

    if not filtered_jobs:
        print("All jobs were skipped. Nothing to do.")
        return

    if args.dry_run:
        print("Dry run job order:")
        for idx, job in enumerate(filtered_jobs, start=1):
            print(f"  {idx:03d}. {job.subset_name}/{job.sample_name} -> {job.sample_dir}")
        print(f"GPU slots: {gpu_slots}")
        print(f"Hydra overrides: {hydra_overrides}")
        return

    print(f"Dataset root: {dataset_root}")
    print(f"Total samples: {len(filtered_jobs)}")
    print(f"GPU workers: {len(gpu_slots)} -> {gpu_slots}")
    print(f"Result root: {result_root}")
    print(f"Logs root: {logs_run_dir}")
    print(f"W&B enabled: {args.wandb}")

    job_queue: "queue.Queue[SampleJob]" = queue.Queue()
    for job in filtered_jobs:
        job_queue.put(job)

    summary_lock = threading.Lock()
    succeeded: List[Tuple[SampleJob, Path, float]] = []
    failed: List[Tuple[SampleJob, Path, int, float]] = []

    def worker(gpu_slot_idx: int, gpu_id: str) -> None:
        while True:
            try:
                job = job_queue.get_nowait()
            except queue.Empty:
                return

            start = time.time()
            sample_key = f"{job.subset_name}/{job.sample_name}"
            out_dir = result_root / job.subset_name / job.sample_name

            log_file = logs_run_dir / f"gpu{gpu_slot_idx}_{job.subset_name}_{job.sample_name}.log"

            run_name = f"{job.subset_name}__{job.sample_name}"
            train_cmd = [
                "python",
                "-m",
                "scripts.train",
                "--config-name",
                args.config_name,
                f"exp.data_dir={str(job.sample_dir)}",
                f"exp.result_dir={str(out_dir)}",
                "trainer.device=cuda",
            ] + hydra_overrides

            script = build_worker_script(
                repo_root=repo_root,
                activate_env_script=activate_env_script,
                conda_exe=conda_exe,
                conda_env_name=args.conda_env_name,
                gpu_id=gpu_id,
                train_cmd=train_cmd,
                wandb_enable=args.wandb,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                wandb_group=wandb_group,
                wandb_run_name=run_name,
                wandb_tags=f"train_all,{job.subset_name},{job.sample_name}",
            )

            print(f"[start][gpu {gpu_id}] {sample_key}")
            with open(log_file, "w", encoding="utf-8") as fout:
                fout.write(f"# GPU: {gpu_id}\n")
                fout.write(f"# Sample: {sample_key}\n")
                fout.write(f"# Start: {datetime.now().isoformat()}\n\n")
                fout.flush()
                proc = subprocess.run(["bash", "-lc", script], stdout=fout, stderr=subprocess.STDOUT)

            elapsed = time.time() - start
            if proc.returncode == 0:
                print(f"[done ][gpu {gpu_id}] {sample_key} in {elapsed/60.0:.1f} min")
                with summary_lock:
                    succeeded.append((job, log_file, elapsed))
            else:
                print(
                    f"[fail ][gpu {gpu_id}] {sample_key} (code={proc.returncode}) in {elapsed/60.0:.1f} min, log={log_file}"
                )
                with summary_lock:
                    failed.append((job, log_file, proc.returncode, elapsed))

            job_queue.task_done()

    threads = [
        threading.Thread(target=worker, args=(slot_idx, gpu_id), daemon=False)
        for slot_idx, gpu_id in enumerate(gpu_slots)
    ]

    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total_elapsed = time.time() - t0

    print("\n===== train_all_slurm summary =====")
    print(f"Total queued jobs: {len(filtered_jobs)}")
    print(f"Succeeded: {len(succeeded)}")
    print(f"Failed: {len(failed)}")
    print(f"Total wall time: {total_elapsed/3600.0:.2f} h")
    print(f"Logs directory: {logs_run_dir}")

    if failed:
        print("Failed jobs:")
        for job, log_path, code, elapsed in failed:
            print(
                f"  - {job.subset_name}/{job.sample_name}: exit={code}, elapsed={elapsed/60.0:.1f} min, log={log_path}"
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
