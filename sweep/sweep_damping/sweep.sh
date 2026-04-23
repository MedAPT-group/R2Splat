#!/usr/bin/env bash
# Single GPU, single wandb agent for the damping sweep.
# Submit multiple copies (same SWEEP_ID) for parallel agents.
#
# Usage:
#   wandb sweep sweep/sweep_damping/sweep_damping.yaml
#   SWEEP_ID=<entity/project/sweep_id> sbatch sweep/sweep_damping/sweep.sh

#SBATCH --job-name=wb_damping_1g
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=8000M
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1
#SBATCH -o /home/chengzyi/myterminus/code/gsplat-3dgs2ct/logs/sweep_damping_agent_%j.log

set -euo pipefail

ACTIVATE_ENV_SCRIPT="${ACTIVATE_ENV_SCRIPT:-${HOME}/myterminus/env/activate_env/activate_env_cu121.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-gsplat2}"
CONDA_EXE="${CONDA_EXE:-${CONDA_ROOT:-${HOME}/myterminus/miniconda3}/bin/conda}"

if [[ ! -f "${ACTIVATE_ENV_SCRIPT}" ]]; then
  echo "ERROR: activate script not found: ${ACTIVATE_ENV_SCRIPT}" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${ACTIVATE_ENV_SCRIPT}"

if [[ ! -x "${CONDA_EXE}" ]]; then
  echo "ERROR: conda not executable: ${CONDA_EXE}" >&2
  exit 1
fi

cd /home/chengzyi/myterminus/code/gsplat-3dgs2ct
mkdir -p logs

SWEEP_ID="${SWEEP_ID:-}"
if [[ "${SWEEP_ID}" == *"YOUR_"* ]] || [[ -z "${SWEEP_ID}" ]]; then
  echo "ERROR: Set SWEEP_ID from: wandb sweep sweep/sweep_damping/sweep_damping.yaml" >&2
  exit 1
fi

exec "${CONDA_EXE}" run -n "${CONDA_ENV_NAME}" --no-capture-output wandb agent "${SWEEP_ID}"
