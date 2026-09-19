#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || true)}"

RUNS_DIR="${RUNS_DIR:-${REPO_ROOT}/runs}"

CHECKPOINT="${1:-${CHECKPOINT:-}}"
TASK_SUITE="${2:-${TASK_SUITE:-libero_spatial}}"
CATEGORIES="${3:-${CATEGORIES:-all}}"
TRIALS="${4:-${TRIALS:-1}}"

if [[ -z "${CHECKPOINT}" && -d "${RUNS_DIR}" ]]; then
    CHECKPOINT="$(
        find "${RUNS_DIR}" -path '*/checkpoints/latest-checkpoint.pt' -type f -printf '%T@ %p\n' \
            | sort -n \
            | tail -n 1 \
            | cut -d' ' -f2-
    )"
fi

if [[ -z "${CHECKPOINT}" ]]; then
    echo "Usage: bash vla-scripts/libero_plus.sh CHECKPOINT [TASK_SUITE] [CATEGORIES] [TRIALS]" >&2
    echo "Example: bash vla-scripts/libero_plus.sh ./runs/xxx/checkpoints/latest-checkpoint.pt libero_spatial all 1" >&2
    echo "Download released checkpoints from https://huggingface.co/CokeAnd1ce/JEPA_WAM" >&2
    exit 1
fi

: "${BASE_VLM_RUN:?Set BASE_VLM_RUN to the downloaded pretrained VLM run directory}"
: "${QWEN_PATH:?Set QWEN_PATH to the downloaded Qwen2.5-0.5B directory}"
VJEPA_CKPT="${VJEPA_CKPT:-}"
LEVJEPA_CKPT="${LEVJEPA_CKPT:-}"
if [[ -z "${VJEPA_CKPT}" && -z "${LEVJEPA_CKPT}" ]]; then
    echo "Set VJEPA_CKPT or LEVJEPA_CKPT for evaluation." >&2
    exit 1
fi
: "${LIBERO_PATH:?Set LIBERO_PATH to the LIBERO-Plus checkout}"
MAX_TASKS="${MAX_TASKS:-0}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-0}"
NUM_OPEN_LOOP_STEPS="${NUM_OPEN_LOOP_STEPS:-0}"
SAVE_ROLLOUTS="${SAVE_ROLLOUTS:-True}"
RESULT_ROOT="${RESULT_ROOT:-}"
RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}. Activate the project environment or set PYTHON_BIN." >&2
    exit 1
fi

for path in "${CHECKPOINT}" "${BASE_VLM_RUN}" "${QWEN_PATH}" "${LIBERO_PATH}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Required path does not exist: ${path}" >&2
        exit 1
    fi
done

for vision_path in "${VJEPA_CKPT}" "${LEVJEPA_CKPT}"; do
    if [[ -n "${vision_path}" && ! -e "${vision_path}" ]]; then
        echo "Required vision checkpoint does not exist: ${vision_path}" >&2
        exit 1
    fi
done

case "${TASK_SUITE}" in
    all)
        SUITES=(libero_spatial libero_object libero_goal libero_10)
        ;;
    libero_spatial|libero_object|libero_goal|libero_10|libero_90)
        SUITES=("${TASK_SUITE}")
        ;;
    *)
        echo "Invalid task suite: ${TASK_SUITE}" >&2
        exit 1
        ;;
esac

CLASSIFICATION_FILE="${LIBERO_PATH}/libero/libero/benchmark/task_classification.json"
if [[ ! -f "${CLASSIFICATION_FILE}" ]]; then
    echo "LIBERO-Plus classification file not found: ${CLASSIFICATION_FILE}" >&2
    echo "Set LIBERO_PATH to a LIBERO-Plus checkout rather than the standard LIBERO repository." >&2
    exit 1
fi

LIBERO_CONFIG_DIR="${LIBERO_CONFIG_PATH:-${REPO_ROOT}/.libero_plus_config}"
mkdir -p "${LIBERO_CONFIG_DIR}"
cat > "${LIBERO_CONFIG_DIR}/config.yaml" <<EOF
benchmark_root: ${LIBERO_PATH}/libero/libero
bddl_files: ${LIBERO_PATH}/libero/libero/bddl_files
init_states: ${LIBERO_PATH}/libero/libero/init_files
datasets: ${LIBERO_PATH}/libero/datasets
assets: ${LIBERO_PATH}/libero/libero/assets
EOF

export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_DIR}"
export LIBERO_PATH
export PYTHONPATH="${REPO_ROOT}:${LIBERO_PATH}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    DEFAULT_EGL_DEVICE="${CUDA_VISIBLE_DEVICES%%,*}"
else
    DEFAULT_EGL_DEVICE=0
fi
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${DEFAULT_EGL_DEVICE}}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/jepa_wam_matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

cd "${REPO_ROOT}"

for suite in "${SUITES[@]}"; do
    CMD=(
        "${PYTHON_BIN}" experiments/robot/libero/run_libero_eval.py
        --pretrained_checkpoint "${CHECKPOINT}"
        --base_vlm "${BASE_VLM_RUN}"
        --llm_checkpoint_path "${QWEN_PATH}"
        --task_suite_name "${suite}"
        --libero_plus_categories "${CATEGORIES}"
        --num_trials_per_task "${TRIALS}"
        --save_rollouts "${SAVE_ROLLOUTS}"
    )

    if [[ -n "${RESULT_ROOT}" ]]; then
        CMD+=(--result_root "${RESULT_ROOT}")
    fi

    if [[ -n "${VJEPA_CKPT}" ]]; then
        CMD+=(--vjepa_checkpoint_path "${VJEPA_CKPT}")
    fi
    if [[ -n "${LEVJEPA_CKPT}" ]]; then
        CMD+=(--levjepa_checkpoint_path "${LEVJEPA_CKPT}")
    fi

    if [[ "${MAX_TASKS}" -gt 0 ]]; then
        CMD+=(--max_tasks "${MAX_TASKS}")
    fi
    if [[ "${MAX_EPISODE_STEPS}" -gt 0 ]]; then
        CMD+=(--max_episode_steps "${MAX_EPISODE_STEPS}")
    fi
    if [[ "${NUM_OPEN_LOOP_STEPS}" -gt 0 ]]; then
        CMD+=(--num_open_loop_steps "${NUM_OPEN_LOOP_STEPS}")
    fi
    if [[ "${RESUME}" == "1" ]]; then
        RESUME_PATH=""
        RESUME_SEARCH_ROOT="${RESULT_ROOT:-${REPO_ROOT}/rollout/${suite}}"
        if [[ -d "${RESUME_SEARCH_ROOT}" ]]; then
            RESUME_PATH="$(find "${RESUME_SEARCH_ROOT}" -type f -name 'summary-*.json' -printf '%T@ %p\n' \
                | sort -nr \
                | head -n 1 \
                | cut -d' ' -f2-)"
        fi
        if [[ -n "${RESUME_PATH}" ]]; then
            CMD+=(--resume_from "${RESUME_PATH}")
        else
            echo "No previous summary found for ${suite}; starting it from the beginning."
        fi
    fi

    if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'Command:'
        printf ' %q' "${CMD[@]}"
        printf '\n'
    else
        echo "Starting LIBERO-Plus suite: ${suite}"
        "${CMD[@]}"
    fi
done
