#!/usr/bin/env bash
#SBATCH --job-name=uqnodet
#SBATCH --array=0-5%6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/uqnodet_%A_%a.out

set -euo pipefail

cd /home/chengzyi/myterminus/code/r2splat-r2beta
export WANDB_MODE=disabled

source ~/my*/env/act*/*.sh

SPECS=(
  "gaussian:5"
  "gaussian:25"
  "gaussian:75"
  "beta:5"
  "beta:25"
  "beta:75"
)

spec=${SPECS[${SLURM_ARRAY_TASK_ID}]}
prim=${spec%%:*}
views=${spec##*:}
q0=1e-4
lr=0.001

data=/home/chengzyi/myterminus/data/3DGS2CT/preprocessed/synthetic_dataset/cone_ntrain_${views}_angle_360/0_chest_cone
run=${prim}_unc_no_detach_${views}views_q0${q0}_lr${lr}

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} run=${run} data=${data} host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-unset}"

conda run -n r2splat-r2beta --no-capture-output python -m scripts.train --config-name ${prim} \
  exp.data_dir=${data} \
  exp.result_dir=output/unc_qmaxfree_lam1_30k_no_detach_fixed/${run} \
  trainer.max_steps=30000 \
  trainer.eval_at_start=false \
  trainer.eval_interval=0 \
  trainer.eval_at_end=true \
  trainer.loss_log_interval=200 \
  trainer.uncertainty_enabled=true \
  trainer.uncertainty_detach_contribution=false \
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
