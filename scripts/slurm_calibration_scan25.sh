#!/usr/bin/env bash
#SBATCH --job-name=uqcal25
#SBATCH --array=0-9%5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/uqcal25_%A_%a.out

set -euo pipefail

cd /home/chengzyi/myterminus/code/r2splat-r2beta
export WANDB_MODE=disabled

source ~/my*/env/act*/*.sh

SPECS=(
  "gaussian:uncertainty:0:0"
  "gaussian:uncertainty:1e-4:0"
  "gaussian:uncertainty:3e-4:0"
  "gaussian:global_scale:0:0.01"
  "gaussian:global_scale:0:0.03"
  "beta:uncertainty:0:0"
  "beta:uncertainty:1e-4:0"
  "beta:uncertainty:3e-4:0"
  "beta:global_scale:0:0.01"
  "beta:global_scale:0:0.03"
)

spec=${SPECS[${SLURM_ARRAY_TASK_ID}]}
prim=$(echo "${spec}" | cut -d: -f1)
mode=$(echo "${spec}" | cut -d: -f2)
cal_lr_unc=$(echo "${spec}" | cut -d: -f3)
cal_lr_scale=$(echo "${spec}" | cut -d: -f4)
views=25
q0=1e-4
lr=0.001
cal_views=3
cal_interval=500

data=/home/chengzyi/myterminus/data/3DGS2CT/preprocessed/synthetic_dataset/cone_ntrain_${views}_angle_360/0_chest_cone
if [[ "${mode}" == "uncertainty" ]]; then
  tag=${prim}_calq_lr${cal_lr_unc}_${cal_views}holdout_${views}views_q0${q0}_lr${lr}
else
  tag=${prim}_calscale_lr${cal_lr_scale}_${cal_views}holdout_${views}views_q0${q0}_lr${lr}
fi

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} run=${tag} host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-unset}"

conda run -n r2splat-r2beta --no-capture-output python -m scripts.train --config-name ${prim} \
  exp.data_dir=${data} \
  exp.result_dir=output/unc_calibration_scan25/${tag} \
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
  trainer.calibration_mode=${mode} \
  trainer.calibration_num_views=${cal_views} \
  trainer.calibration_interval=${cal_interval} \
  trainer.calibration_start_iter=500 \
  trainer.calibration_batch_size=0 \
  trainer.calibration_lr_uncertainty=${cal_lr_unc} \
  trainer.calibration_lr_scale=${cal_lr_scale} \
  trainer.calibration_scale_init=1.0 \
  model.uncertainty_q_max=null \
  model.init_uncertainty=${q0} \
  optim.lr_uncertainty=${lr} \
  strategy.uncertainty_grad_weight=0 \
  strategy.split_sampling=random \
  strategy.new_uncertainty_mode=inherit

echo "[$(date)] done ${tag}"
