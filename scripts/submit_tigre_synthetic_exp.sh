#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/home/chengzyi/myterminus/code/r2splat-r2beta
DATASET_ROOT=/home/chengzyi/myterminus/data/3DGS2CT/preprocessed/synthetic_dataset
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

VIEWS_LIST=(5 25 50 75)
SAMPLES=(
  0_chest_cone
  0_foot_cone
  0_head_cone
  0_jaw_cone
  0_pancreas_cone
  1_beetle_cone
  1_bonsai_cone
  1_broccoli_cone
  1_kingsnake_cone
  1_pepper_cone
  2_backpack_cone
  2_engine_cone
  2_mount_cone
  2_present_cone
  2_teapot_cone
)

# FDK and SART are stable when one task serially processes all 15 samples for
# a fixed view count.
for METHOD in fdk sart; do
  for VIEWS in "${VIEWS_LIST[@]}"; do
    SUBDIR="cone_ntrain_${VIEWS}_angle_360"

    sbatch "${SBATCH_COMMON[@]}" \
      --job-name="syn_tigre_${METHOD}_${VIEWS}" \
      --output="$REPO_ROOT/logs/slurm/%x_%j.out" \
      --wrap="bash -lc 'set -euo pipefail; source $ACTIVATE_SCRIPT; eval \"\$($CONDA_EXE shell.bash hook)\"; conda activate r2splat-r2beta; cd $REPO_ROOT; export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-8}; export MKL_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-8}; python scripts/run_tigre_baselines.py --dataset-root $DATASET_ROOT --subdirs $SUBDIR --methods $METHOD --output-root output/full_exp_tigre --sart-iters 20 --skip-finished'"
  done
done

# ASD-POCS can leave TIGRE/CUDA state dirty after one reconstruction in a
# long-lived Python process, so submit one sample per process.
for VIEWS in "${VIEWS_LIST[@]}"; do
  SUBDIR="cone_ntrain_${VIEWS}_angle_360"
  for SAMPLE in "${SAMPLES[@]}"; do
    sbatch "${SBATCH_COMMON[@]}" \
      --job-name="syn_asdpocs_${VIEWS}_${SAMPLE}" \
      --output="$REPO_ROOT/logs/slurm/%x_%j.out" \
      --wrap="bash -lc 'set -euo pipefail; source $ACTIVATE_SCRIPT; eval \"\$($CONDA_EXE shell.bash hook)\"; conda activate r2splat-r2beta; cd $REPO_ROOT; export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-8}; export MKL_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-8}; python scripts/run_tigre_baselines.py --dataset-root $DATASET_ROOT --subdirs $SUBDIR --samples $SAMPLE --methods asd_pocs --output-root output/full_exp_tigre --asd-pocs-iters 10 --asd-tv-iters 20 --skip-finished'"
  done
done
