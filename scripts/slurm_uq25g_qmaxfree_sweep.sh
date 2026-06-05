#!/usr/bin/env bash
#SBATCH --job-name=uq25g
#SBATCH --array=0-15%6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/uq25g_%A_%a.out

set -euo pipefail

cd /home/chengzyi/myterminus/code/r2splat-r2beta
export WANDB_MODE=disabled

source ~/my*/env/act*/*.sh

Q0S=(1e-5 1e-4 3e-4 1e-3)
LRS=(0.0001 0.0003 0.001 0.003)

idx=${SLURM_ARRAY_TASK_ID}
q0=${Q0S[$((idx % 4))]}
lr=${LRS[$((idx / 4))]}
run=uqfree_lam1_lr${lr}_q0${q0}

echo "[$(date)] task=${idx} run=${run} q0=${q0} lr=${lr} host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-unset}"

conda run -n r2splat-r2beta --no-capture-output python -m scripts.train \
  exp=0_chest_cone_25_views \
  exp.result_dir=output/unc_qmaxfree_lam1_30k_gaussian_sweep25/${run} \
  trainer.max_steps=30000 \
  trainer.eval_at_start=false \
  trainer.eval_interval=0 \
  trainer.eval_at_end=true \
  trainer.loss_log_interval=200 \
  trainer.uncertainty_enabled=true \
  trainer.lambda_l1=0.0 \
  trainer.lambda_dssim=0.25 \
  trainer.lambda_tv=0.05 \
  trainer.lambda_nll=1.0 \
  trainer.lambda_uncertainty_prior=0.0 \
  trainer.uncertainty_var_max=0.0 \
  model.uncertainty_q_max=null \
  model.init_uncertainty=${q0} \
  optim.lr_uncertainty=${lr}

echo "[$(date)] done ${run}"
