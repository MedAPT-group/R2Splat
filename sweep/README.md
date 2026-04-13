# Multi-Sample Training Quickstart

## Sweep Experiments

Sweep assets are now organized into two subdirectories:

- Previous sweep experiment:
  - YAML: `sweep/previous_r2gaussian/sweep_r2gaussian.yaml`
  - Agent script: `sweep/previous_r2gaussian/sweep.sh`
- Current Newton sweep experiment:
  - YAML: `sweep/r2gaussian_newton/sweep_r2gaussian_newton.yaml`
  - Agent script: `sweep/r2gaussian_newton/sweep.sh`

Quick start (Newton sweep):

```bash
wandb sweep sweep/r2gaussian_newton/sweep_r2gaussian_newton.yaml
SWEEP_ID=<entity/project/sweep_id> sbatch sweep/r2gaussian_newton/sweep.sh
```

## 1 Dataset structure expected by launcher

`train_all_slurm.py` expects:
- One dataset root, for example:
  - `~/terminus/users/xiaxiali/datasets/3DGS2CT/preprocessed/synthetic_dataset`
- Multiple subset directories under the root, for example:
  - `cone_ntrain_5_angle_360`
  - `cone_ntrain_25_angle_360`
  - `cone_ntrain_50_angle_360`
  - `cone_ntrain_75_angle_360`
- Each subset contains sample dirs like `0_chest_cone`, `1_bonsai_cone`, etc.

## 2 Typical full run (4 GPUs)

```bash
sbatch \
  --time=12:00:00 \
  --mem-per-cpu=8000M \
  -w hardin01 \
  --ntasks=1 \
  --cpus-per-task=32 \
  --gres=gpu:4 \
  scripts/train_all_slurm.py \
  --gpus 4 \
  --config-name r2gaussian_train \
  --dataset-root ~/terminus/users/xiaxiali/datasets/3DGS2CT/preprocessed/synthetic_dataset \
  --subdirs cone_ntrain_25_angle_360 cone_ntrain_5_angle_360 cone_ntrain_50_angle_360 cone_ntrain_75_angle_360 \
  --wandb-project gsplat2ct_r2gaussianlike_final \
  --wandb-entity 810364379denveryi-eth-zurich
```

Meaning of key args:
- `--gpus`: number of concurrent workers (one sample per GPU at a time)
- `--dataset-root`: root containing all subsets
- `--subdirs`: which subsets to include
