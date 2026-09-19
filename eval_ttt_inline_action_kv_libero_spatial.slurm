#!/usr/bin/env bash
# LIBERO-Plus evaluation: inline TTT + action-token K/V memory.
# This evaluates the already-trained checkpoint; it does not update weights.
#SBATCH --job-name=eval-ttt-inline-action
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.out
#SBATCH --error=/home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.err

set -euo pipefail

REPO_ROOT="/home/ha865618/project/LeVJEPA_WAM_TTT"
CHECKPOINT="${REPO_ROOT}/checkpoints/jepa-wam-ttt-inline-action-kv-vjepa21-context8-gb1-pb1/checkpoints/latest-checkpoint.pt"
RUN_TAG="jepa-wam-ttt-inline-action-kv-vjepa21-context8-gb1-pb1"

export SAVE_ROLLOUTS=False
export NUM_OPEN_LOOP_STEPS=8
export EVAL_TIMESTAMP="${SLURM_JOB_ID}"
export RUN_TAG
export RESULT_ROOT="${REPO_ROOT}/rollout_ttt_eval/${RUN_TAG}"
export LOCAL_LOG_DIR="${REPO_ROOT}/experiments/logs/ttt-eval-${RUN_TAG}-${SLURM_JOB_ID}"
export LIBERO_CONFIG_PATH="${REPO_ROOT}/.libero_plus_config/ttt-eval-${RUN_TAG}-${SLURM_JOB_ID}"

cd "${REPO_ROOT}"
echo "===== LIBERO-Plus EVAL: ${RUN_TAG} ====="
echo "checkpoint=${CHECKPOINT}"
echo "result_root=${RESULT_ROOT}"
echo "save_rollouts=${SAVE_ROLLOUTS}"
echo "num_open_loop_steps=${NUM_OPEN_LOOP_STEPS}"
echo "host=$(hostname)"
echo "date=$(date)"
nvidia-smi

bash vla-scripts/libero_plus_ttt.sh "${CHECKPOINT}" libero_object all 1
