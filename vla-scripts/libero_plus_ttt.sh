#!/usr/bin/env bash

# Convenient LIBERO-Plus evaluation entry point for the four JEPA-WAM + TTT
# training routes. Positional arguments match the released launcher:
#
#   bash vla-scripts/libero_plus_ttt.sh CHECKPOINT TASK_SUITE CATEGORIES TRIALS
#
# This script supplies local asset paths, disables rollout videos by default,
# and creates an isolated result directory for every invocation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ASSET_ROOT="${ASSET_ROOT:-${REPO_ROOT}/jepa_wam_assets}"

CHECKPOINT="${1:-${CHECKPOINT:-}}"
TASK_SUITE="${2:-${TASK_SUITE:-libero_spatial}}"
CATEGORIES="${3:-${CATEGORIES:-all}}"
TRIALS="${4:-${TRIALS:-1}}"

if [[ -z "${CHECKPOINT}" ]]; then
    echo "Usage: bash vla-scripts/libero_plus_ttt.sh CHECKPOINT [TASK_SUITE] [CATEGORIES] [TRIALS]" >&2
    echo "Example: bash vla-scripts/libero_plus_ttt.sh checkpoints/<run>/checkpoints/latest-checkpoint.pt libero_spatial all 1" >&2
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-/home/ha865618/.conda/envs/jepa_wam/bin/python}"
BASE_VLM_RUN="${BASE_VLM_RUN:-${ASSET_ROOT}/JEPA_WAM/checkpoints/pretrained_vlm/prism-qwen25-vjepa21-vitl-384px+0_5b+stage-finetune+x7}"
QWEN_PATH="${QWEN_PATH:-${ASSET_ROOT}/Qwen2.5-0.5B}"
VJEPA_CKPT="${VJEPA_CKPT:-${ASSET_ROOT}/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt}"
LIBERO_PATH="${LIBERO_PATH:-/home/ha865618/project/LIBERO-plus}"
SAVE_ROLLOUTS="${SAVE_ROLLOUTS:-False}"
NUM_OPEN_LOOP_STEPS="${NUM_OPEN_LOOP_STEPS:-8}"
STAMP="${EVAL_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"

# Checkpoint layout: .../<run_id>/checkpoints/<checkpoint>.pt
CHECKPOINT_RUN_DIR="$(dirname "$(dirname "${CHECKPOINT}")")"
DEFAULT_RUN_TAG="$(basename "${CHECKPOINT_RUN_DIR}")"
RUN_TAG="${RUN_TAG:-${DEFAULT_RUN_TAG}}"
RUN_TAG="$(printf '%s' "${RUN_TAG}" | tr '/ ' '__' | tr -cd '[:alnum:]_.-')"
if [[ -z "${RUN_TAG}" ]]; then
    RUN_TAG="checkpoint"
fi

LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-${REPO_ROOT}/experiments/logs/ttt-${RUN_TAG}-${STAMP}}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/rollout_ttt/${RUN_TAG}-${STAMP}}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${REPO_ROOT}/.libero_plus_config/ttt-${RUN_TAG}-${STAMP}}"

for path in "${CHECKPOINT}" "${BASE_VLM_RUN}" "${QWEN_PATH}" "${VJEPA_CKPT}" "${LIBERO_PATH}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Required path does not exist: ${path}" >&2
        exit 1
    fi
done

export PYTHON_BIN BASE_VLM_RUN QWEN_PATH VJEPA_CKPT LIBERO_PATH
export SAVE_ROLLOUTS NUM_OPEN_LOOP_STEPS LOCAL_LOG_DIR RESULT_ROOT LIBERO_CONFIG_PATH

printf '%s\n' '===== JEPA-WAM + TTT LIBERO-Plus EVALUATION ====='
echo "checkpoint=${CHECKPOINT}"
echo "task_suite=${TASK_SUITE}"
echo "categories=${CATEGORIES}"
echo "trials=${TRIALS}"
echo "save_rollouts=${SAVE_ROLLOUTS}"
echo "num_open_loop_steps=${NUM_OPEN_LOOP_STEPS}"
echo "log_dir=${LOCAL_LOG_DIR}"
echo "result_root=${RESULT_ROOT}"

cd "${REPO_ROOT}"
bash "${SCRIPT_DIR}/libero_plus.sh" "${CHECKPOINT}" "${TASK_SUITE}" "${CATEGORIES}" "${TRIALS}"
