#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/home/chengzyi/myterminus/code/r2splat-r2beta
DATASET_ROOT=/home/chengzyi/terminus/users/honlan/projects/binary_tree_3dgs_r2/datasets/breast_dataset
ACTIVATE_SCRIPT=/home/chengzyi/myterminus/env/activate_env/activate_env_cu121.sh
CONDA_EXE=/home/chengzyi/myterminus/env/miniconda3/bin/conda

mkdir -p "$REPO_ROOT/logs/slurm"

SBATCH_COMMON=(
  --time=08:00:00
  --ntasks=1
  --cpus-per-task=8
  --mem=40G
  -w hardin01
  --gres=gpu:1
)

for METHOD in fdk sart asd_pocs; do
  for VIEWS in 25 50 75; do
    SUBDIR="cone_ntrain_${VIEWS}_angle_360"

    sbatch "${SBATCH_COMMON[@]}" \
      --job-name="br_tigre_${METHOD}_${VIEWS}" \
      --output="$REPO_ROOT/logs/slurm/%x_%j.out" \
      --wrap="bash -lc 'set -euo pipefail; source $ACTIVATE_SCRIPT; eval \"\$($CONDA_EXE shell.bash hook)\"; conda activate r2splat-r2beta; cd $REPO_ROOT; export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-8}; export MKL_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-8}; python scripts/run_tigre_baselines.py --dataset-root $DATASET_ROOT --subdirs $SUBDIR --methods $METHOD --output-root output/breast_exp/tigre --sart-iters 20 --asd-pocs-iters 10 --asd-tv-iters 20 --skip-finished'"
  done
done
