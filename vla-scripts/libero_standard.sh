#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || true)}"

CHECKPOINT="${1:-${CHECKPOINT:-}}"
TASK_SUITE="${2:-${TASK_SUITE:-libero_spatial}}"
TRIALS="${3:-${TRIALS:-50}}"

if [[ -z "${CHECKPOINT}" ]]; then
    echo "Usage: bash vla-scripts/libero_standard.sh CHECKPOINT [TASK_SUITE|all] [TRIALS]" >&2
    exit 1
fi

: "${BASE_VLM_RUN:?Set BASE_VLM_RUN to the pretrained VLM run directory}"
: "${QWEN_PATH:?Set QWEN_PATH to the Qwen2.5-0.5B directory}"
VJEPA_CKPT="${VJEPA_CKPT:-}"
LEVJEPA_CKPT="${LEVJEPA_CKPT:-}"
if [[ -z "${VJEPA_CKPT}" && -z "${LEVJEPA_CKPT}" ]]; then
    echo "Set VJEPA_CKPT or LEVJEPA_CKPT for evaluation." >&2
    exit 1
fi
: "${LIBERO_PATH:?Set LIBERO_PATH to a standard LIBERO checkout}"

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

if [[ ! -f "${LIBERO_PATH}/libero/libero/benchmark/__init__.py" ]]; then
    echo "Standard LIBERO benchmark package not found under: ${LIBERO_PATH}" >&2
    exit 1
fi

case "${TASK_SUITE}" in
    libero_spatial|libero_object|libero_goal|libero_10|all)
        ;;
    *)
        echo "Invalid task suite: ${TASK_SUITE}" >&2
        exit 1
        ;;
esac

LIBERO_CONFIG_DIR="${LIBERO_CONFIG_PATH:-${REPO_ROOT}/.libero_config}"
mkdir -p "${LIBERO_CONFIG_DIR}"
cat > "${LIBERO_CONFIG_DIR}/config.yaml" <<EOF
benchmark_root: ${LIBERO_PATH}/libero/libero
bddl_files: ${LIBERO_PATH}/libero/libero/bddl_files
init_states: ${LIBERO_PATH}/libero/libero/init_files
datasets: ${LIBERO_PATH}/libero/datasets
assets: ${LIBERO_PATH}/libero/libero/assets
EOF

export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_DIR}"
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

SUITES=("${TASK_SUITE}")
if [[ "${TASK_SUITE}" == "all" ]]; then
    SUITES=(libero_spatial libero_object libero_goal libero_10)
fi

for suite in "${SUITES[@]}"; do
    CMD=(
        "${PYTHON_BIN}" experiments/robot/libero/run_libero_standard_eval.py
        --pretrained_checkpoint "${CHECKPOINT}"
        --base_vlm "${BASE_VLM_RUN}"
        --llm_checkpoint_path "${QWEN_PATH}"
        --task_suite_name "${suite}"
        --num_trials_per_task "${TRIALS}"
        --save_rollouts "${SAVE_ROLLOUTS:-False}"
    )

    if [[ -n "${VJEPA_CKPT}" ]]; then
        CMD+=(--vjepa_checkpoint_path "${VJEPA_CKPT}")
    fi
    if [[ -n "${LEVJEPA_CKPT}" ]]; then
        CMD+=(--levjepa_checkpoint_path "${LEVJEPA_CKPT}")
    fi

    if [[ "${MAX_TASKS:-0}" -gt 0 ]]; then
        CMD+=(--max_tasks "${MAX_TASKS}")
    fi
    if [[ "${MAX_EPISODE_STEPS:-0}" -gt 0 ]]; then
        CMD+=(--max_episode_steps "${MAX_EPISODE_STEPS}")
    fi
    if [[ "${NUM_OPEN_LOOP_STEPS:-0}" -gt 0 ]]; then
        CMD+=(--num_open_loop_steps "${NUM_OPEN_LOOP_STEPS}")
    fi
    "${CMD[@]}"
done
