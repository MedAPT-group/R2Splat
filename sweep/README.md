# Sweep and Multi-Sample Training Quickstart

This guide covers:
- Launching `scripts/train_all_slurm.py` with CLI arguments (`--gpus`, `--dataset-root`, `--subdirs`)
- Single-GPU W&B sweep
- Parallel multi-GPU (multi-agent) W&B sweep

## 1) Launch train_all_slurm

### 1.1 Dataset structure expected by launcher

`train_all_slurm.py` expects:
- One dataset root, for example:
  - `~/terminus/users/xiaxiali/datasets/3DGS2CT/preprocessed/synthetic_dataset`
- Multiple subset directories under the root, for example:
  - `cone_ntrain_5_angle_360`
  - `cone_ntrain_25_angle_360`
  - `cone_ntrain_50_angle_360`
  - `cone_ntrain_75_angle_360`
- Each subset contains sample dirs like `0_chest_cone`, `1_bonsai_cone`, etc.

### 1.2 Typical full run (4 GPUs)

From repo root:

```bash
sbatch \
  --time=12:00:00 \
  --mem-per-cpu=8000M \
  -w hardin01 \
  --ntasks=8 \
  --gres=gpu:4 \
  scripts/train_all_slurm.py \
  --gpus 4 \
  --dataset-root ~/terminus/users/xiaxiali/datasets/3DGS2CT/preprocessed/synthetic_dataset \
  --subdirs cone_ntrain_25_angle_360 cone_ntrain_5_angle_360 cone_ntrain_50_angle_360 cone_ntrain_75_angle_360
```

Meaning of key args:
- `--gpus`: number of concurrent workers (one sample per GPU at a time)
- `--dataset-root`: root containing all subsets
- `--subdirs`: which subsets to include

### 1.3 Run only one subset (example: 25 views)

```bash
sbatch \
  --time=12:00:00 \
  --mem-per-cpu=8000M \
  -w hardin01 \
  --ntasks=8 \
  --gres=gpu:4 \
  scripts/train_all_slurm.py \
  --gpus 4 \
  --dataset-root ~/terminus/users/xiaxiali/datasets/3DGS2CT/preprocessed/synthetic_dataset \
  --subdirs cone_ntrain_25_angle_360
```

### 1.4 Enable W&B monitoring in train_all_slurm

```bash
sbatch \
  --time=12:00:00 \
  --mem-per-cpu=8000M \
  -w hardin01 \
  --ntasks=8 \
  --gres=gpu:4 \
  scripts/train_all_slurm.py \
  --gpus 4 \
  --dataset-root ~/terminus/users/xiaxiali/datasets/3DGS2CT/preprocessed/synthetic_dataset \
  --subdirs cone_ntrain_5_angle_360 cone_ntrain_25_angle_360 cone_ntrain_50_angle_360 cone_ntrain_75_angle_360 \
  --wandb-project gsplat2ct \
  --wandb-entity 810364379denveryi-eth-zurich
```

Optional:
- add `--skip-finished` to skip non-empty output dirs
- add extra Hydra overrides after `--`, for example:

```bash
... scripts/train_all_slurm.py --gpus 4 --dataset-root ... --subdirs ... -- trainer.max_steps=2000 trainer.eval_interval=100
```

## 2) Single-GPU sweep

### 2.1 Create sweep

From repo root:

```bash
wandb login
wandb sweep sweep/sweep_r2gaussian.yaml
```

Copy the printed sweep id (`entity/project/sweep_id`), then set it when submitting agent:

```bash
SWEEP_ID=810364379denveryi-eth-zurich/gsplat2ct/<your_sweep_id> \
sbatch \
  --time=12:00:00 \
  --mem-per-cpu=8000M \
  -w hardin01 \
  --ntasks=8 \
  --gres=gpu:1 \
  sweep/sweep.sh
```

## 3) Parallel multi-GPU sweep (multiple agents)

W&B sweep parallelism is achieved by running multiple `wandb agent` jobs using the same `SWEEP_ID`.
Each `sweep/sweep.sh` job uses one GPU, so submit N jobs for N parallel agents.

### 3.1 Example: run 4 agents in parallel

```bash
export SWEEP_ID=810364379denveryi-eth-zurich/gsplat2ct/<your_sweep_id>

for i in 1 2 3 4; do
  sbatch \
    --time=12:00:00 \
    --mem-per-cpu=8000M \
    -w hardin01 \
    --ntasks=8 \
    --gres=gpu:1 \
    sweep/sweep.sh
done
```

If you need to pin specific nodes, keep `-w <node_name>` or adjust per submission.

## 4) Useful paths

- Sweep launcher: `sweep/sweep.sh`
- Sweep config: `sweep/sweep_r2gaussian.yaml`
- Multi-sample launcher: `scripts/train_all_slurm.py`
- Sweep logs: `logs/sweep_agent_<jobid>.log`
- train_all logs: `logs/train_all_slurm/<timestamp>/`
