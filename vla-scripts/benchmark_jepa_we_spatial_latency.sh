#!/usr/bin/env bash
# One visible H100, four sequential JEPA-WAM/WE latency groups.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
: "${LIBERO_PATH:?Set LIBERO_PATH to the LIBERO-Plus checkout}"
: "${SREWAM_WE_CKPT:?Set SREWAM_WE_CKPT to the WE best-validation-checkpoint.pt}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new benchmark result directory}"

ASSET_ROOT="${ASSET_ROOT:-${PROJECT_ROOT}/jepa_wam_assets}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || true)}"
BASE_POLICY_RUN="${BASE_POLICY_RUN:-${ASSET_ROOT}/JEPA_WAM/checkpoints/libero/jepavla-qvv25-vjepa-224px+0_5b+mx-libero-90+n1+b32+x7--visual-cosine-projector-allviews--20260723_232305}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${BASE_POLICY_RUN}/checkpoints/step-040000-epoch-37-loss=0.0262.pt}"
BASE_VLM_RUN="${BASE_VLM_RUN:-${ASSET_ROOT}/JEPA_WAM/checkpoints/pretrained_vlm/prism-qvv25-vjepa21-vitl-384px+0_5b+stage-finetune+x7}"
QVV_PATH="${QVV_PATH:-${ASSET_ROOT}/Qvv2.5-0.5B}"
VJEPA_CKPT="${VJEPA_CKPT:-${ASSET_ROOT}/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt}"

for path in "${BASE_CHECKPOINT}" "${SREWAM_WE_CKPT}" "${BASE_VLM_RUN}" "${QVV_PATH}" \
    "${VJEPA_CKPT}" "${LIBERO_PATH}/libero/libero/benchmark/task_classification.json"; do
    [[ -e "${path}" ]] || { echo "Missing benchmark input: ${path}" >&2; exit 1; }
done
[[ -n "${PYTHON_BIN}" && -x "${PYTHON_BIN}" ]] || { echo "Python executable not found: ${PYTHON_BIN}" >&2; exit 1; }
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "Refusing to overwrite: ${OUTPUT_DIR}" >&2; exit 1; }

export PYTHONPATH="${PROJECT_ROOT}:${LIBERO_PATH}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NUMBA_DISABLE_JIT=1
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID%%,*}"
export TTT_INFERENCE_MEMORY_UPDATE=first_denoise
unset FM_NUM_INFERENCE_TIMESTEPS TTT_INFERENCE_SLOW_ONLY TTT_INFERENCE_FIXED_OBSERVATION_POSITION

RUNTIME_DIR="${TMPDIR:-/tmp}/srewam-latency-$$"
export LIBERO_CONFIG_PATH="${RUNTIME_DIR}/libero_config"
export MPLCONFIGDIR="${RUNTIME_DIR}/matplotlib"
mkdir -p "${LIBERO_CONFIG_PATH}" "${MPLCONFIGDIR}"
printf 'benchmark_root: %s/libero/libero\nbddl_files: %s/libero/libero/bddl_files\ninit_states: %s/libero/libero/init_files\ndatasets: %s/libero/datasets\nassets: %s/libero/libero/assets\n' \
    "${LIBERO_PATH}" "${LIBERO_PATH}" "${LIBERO_PATH}" "${LIBERO_PATH}" "${LIBERO_PATH}" \
    > "${LIBERO_CONFIG_PATH}/config.yaml"

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" experiments/robot/libero/benchmark_predict_action_latency.py \
    --base-checkpoint "${BASE_CHECKPOINT}" \
    --we-checkpoint "${SREWAM_WE_CKPT}" \
    --base-vlm "${BASE_VLM_RUN}" \
    --llm-checkpoint-path "${QVV_PATH}" \
    --vjepa-checkpoint-path "${VJEPA_CKPT}" \
    --output-dir "${OUTPUT_DIR}"
