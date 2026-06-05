#!/usr/bin/env bash
#SBATCH --job-name=uqadcsv
#SBATCH --array=0-7%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/uqadcsv_%A_%a.out

set -euo pipefail

cd /home/chengzyi/myterminus/code/r2splat-r2beta
export WANDB_MODE=disabled

source ~/my*/env/act*/*.sh

SPECS=(
  "gaussian:5:0:long_axis"
  "gaussian:75:0:long_axis"
  "gaussian:5:1:long_axis"
  "gaussian:75:1:long_axis"
  "beta:5:0:long_axis"
  "beta:75:0:long_axis"
  "beta:5:3:long_axis"
  "beta:75:3:long_axis"
)

spec=${SPECS[${SLURM_ARRAY_TASK_ID}]}
prim=$(echo "${spec}" | cut -d: -f1)
views=$(echo "${spec}" | cut -d: -f2)
weight=$(echo "${spec}" | cut -d: -f3)
split=$(echo "${spec}" | cut -d: -f4)
q0=1e-4
lr=0.001

data=/home/chengzyi/myterminus/data/3DGS2CT/preprocessed/synthetic_dataset/cone_ntrain_${views}_angle_360/0_chest_cone
run=${prim}_nodet_adcuw${weight}_split${split}_${views}views_q0${q0}_lr${lr}

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} run=${run} data=${data} weight=${weight} split=${split} host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-unset}"

conda run -n r2splat-r2beta --no-capture-output python -m scripts.train --config-name ${prim} \
  exp.data_dir=${data} \
  exp.result_dir=output/unc_no_detach_adc_selected_views/${run} \
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
  trainer.detector_uncertainty_var=0.0 \
  model.uncertainty_q_max=null \
  model.init_uncertainty=${q0} \
  optim.lr_uncertainty=${lr} \
  strategy.uncertainty_grad_weight=${weight} \
  strategy.uncertainty_grad_mode=std \
  strategy.split_sampling=${split}

echo "[$(date)] done ${run}"
