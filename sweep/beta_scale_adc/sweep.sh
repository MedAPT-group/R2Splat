#!/usr/bin/env bash
#SBATCH --job-name=r2beta-scale-adc
#SBATCH --output=logs/sweep_beta_scale_adc_agent_%j.log
#SBATCH --error=logs/sweep_beta_scale_adc_agent_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/chengzyi/myterminus/code/r2splat-r2beta}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-r2splat-r2beta}"
ACTIVATE_SCRIPT="${ACTIVATE_SCRIPT:-${HOME}/myterminus/env/activate_env/activate_env_cu121.sh}"

if [[ -z "${SWEEP_ID:-}" ]]; then
    echo "Please set SWEEP_ID, e.g. SWEEP_ID=entity/project/sweep_id sbatch $0"
    exit 1
fi

cd "${PROJECT_DIR}"
mkdir -p logs

if [[ -f "${ACTIVATE_SCRIPT}" ]]; then
    # shellcheck disable=SC1090
    source "${ACTIVATE_SCRIPT}"
fi

if [[ -n "${CONDA_EXE:-}" ]]; then
    exec "${CONDA_EXE}" run -n "${CONDA_ENV_NAME}" --no-capture-output wandb agent "${SWEEP_ID}"
fi

exec conda run -n "${CONDA_ENV_NAME}" --no-capture-output wandb agent "${SWEEP_ID}"
