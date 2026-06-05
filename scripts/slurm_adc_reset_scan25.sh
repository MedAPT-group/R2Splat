#!/usr/bin/env bash
#SBATCH --job-name=uqadcinit25
#SBATCH --array=0-11%6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/uqadcinit25_%A_%a.out

set -euo pipefail

cd /home/chengzyi/myterminus/code/r2splat-r2beta
export WANDB_MODE=disabled

source ~/my*/env/act*/*.sh

SPECS=(
  "gaussian:0:1:5e-4"
  "gaussian:1:1:5e-4"
  "gaussian:3:3:5e-4"
  "gaussian:3:3:7.5e-4"
  "gaussian:5:5:7.5e-4"
  "gaussian:3:1:1e-3"
  "beta:0:1:5e-4"
  "beta:1:1:5e-4"
  "beta:3:3:5e-4"
  "beta:3:3:7.5e-4"
  "beta:5:5:7.5e-4"
  "beta:3:1:1e-3"
)

spec=${SPECS[${SLURM_ARRAY_TASK_ID}]}
prim=$(echo "${spec}" | cut -d: -f1)
weight=$(echo "${spec}" | cut -d: -f2)
uscale=$(echo "${spec}" | cut -d: -f3)
grow=$(echo "${spec}" | cut -d: -f4)
views=25
q0=1e-4
lr=0.001

data=/home/chengzyi/myterminus/data/3DGS2CT/preprocessed/synthetic_dataset/cone_ntrain_${views}_angle_360/0_chest_cone
run=${prim}_childinit_adcuw${weight}_uscale${uscale}_grow${grow}_long_${views}views_q0${q0}_lr${lr}

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} run=${run} host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-unset}"

conda run -n r2splat-r2beta --no-capture-output python -m scripts.train --config-name ${prim} \
  exp.data_dir=${data} \
  exp.result_dir=output/unc_adc_childinit_scan25/${run} \
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
  trainer.calibration_mode=none \
  model.uncertainty_q_max=null \
  model.init_uncertainty=${q0} \
  optim.lr_uncertainty=${lr} \
  strategy.new_uncertainty_mode=init \
  strategy.uncertainty_grad_weight=${weight} \
  strategy.uncertainty_grad_mode=std \
  strategy.uncertainty_grad_scale=${uscale} \
  strategy.uncertainty_grad_max_multiplier=4 \
  strategy.split_sampling=long_axis \
  strategy.grow_grad2d=${grow}

echo "[$(date)] done ${run}"
