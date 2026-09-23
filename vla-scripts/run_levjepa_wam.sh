#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

ENV_ROOT="${ENV_ROOT:-/home/ha865618/.conda/envs/jepa_wam}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_ROOT}/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${ENV_ROOT}/bin/torchrun}"
DATA_ROOT="${DATA_ROOT:-/home/ha865618/data/modified_libero_rlds}"
ASSET_ROOT="${ASSET_ROOT:-${REPO_ROOT}/jepa_wam_assets}"
BASE_VLM_RUN="${BASE_VLM_RUN:-${ASSET_ROOT}/JEPA_WAM/checkpoints/pretrained_vlm/prism-qvv25-vjepa21-vitl-384px+0_5b+stage-finetune+x7}"
QVV_PATH="${QVV_PATH:-${ASSET_ROOT}/Qvv2.5-0.5B}"
LEVJEPA_CHECKPOINT="${LEVJEPA_CHECKPOINT:-${ASSET_ROOT}/LeVJEPA-VideoMix-Large}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/checkpoints}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
EXPECTED_WORLD_SIZE="${EXPECTED_WORLD_SIZE:-${NPROC_PER_NODE}}"

TTT_ENABLED="${TTT_ENABLED:-0}"
RESUME="${RESUME:-0}"
RUN_ID="${RUN_ID:-}"
SMOKE_TEST="${SMOKE_TEST:-0}"
DRY_RUN="${DRY_RUN:-0}"
MAX_STEPS="${MAX_STEPS:-40000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2500}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-20000}"
CPU_MEMORY_LOG_INTERVAL="${CPU_MEMORY_LOG_INTERVAL:-10}"
DEBUG_BATCH_SHAPES="${DEBUG_BATCH_SHAPES:-False}"
if [[ "${SMOKE_TEST}" == "1" ]]; then
    MAX_STEPS=1
    SAVE_INTERVAL=1
    GLOBAL_BATCH_SIZE=1
    PER_DEVICE_BATCH_SIZE=1
    SHUFFLE_BUFFER_SIZE=128
    CPU_MEMORY_LOG_INTERVAL=0
    DEBUG_BATCH_SHAPES=True
fi

if [[ "${TTT_ENABLED}" != "0" && "${TTT_ENABLED}" != "1" ]]; then
    echo "TTT_ENABLED must be 0 or 1, got ${TTT_ENABLED}." >&2
    exit 1
fi

for path in "${DATA_ROOT}" "${BASE_VLM_RUN}" "${QVV_PATH}" "${LEVJEPA_CHECKPOINT}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Required path does not exist: ${path}" >&2
        exit 1
    fi
done

for dataset_name in \
    libero_spatial_no_noops \
    libero_object_no_noops \
    libero_goal_no_noops \
    libero_10_no_noops; do
    dataset_path="${DATA_ROOT}/${dataset_name}"
    if [[ ! -e "${dataset_path}" ]]; then
        echo "Required LIBERO dataset does not exist: ${dataset_path}" >&2
        exit 1
    fi
done
if [[ ! -f "${LEVJEPA_CHECKPOINT}/config.json" ]]; then
    echo "LeVJEPA checkpoint must be a local Hugging Face directory containing config.json: ${LEVJEPA_CHECKPOINT}" >&2
    exit 1
fi
if [[ ! -x "${PYTHON_BIN}" || ! -x "${TORCHRUN_BIN}" ]]; then
    echo "Missing Python or torchrun under ${ENV_ROOT}." >&2
    exit 1
fi

RUN_ID_NOTE="levjepa-wam"
TTT_ARGS=(--vla.ttt_enabled False --vla.ttt_context_length 1)
if [[ "${TTT_ENABLED}" == "1" ]]; then
    RUN_ID_NOTE="levjepa-wam-ttt-context8"
    TTT_ARGS=(
        --vla.ttt_enabled True
        --vla.ttt_context_length "${TTT_CONTEXT_LENGTH:-8}"
        --vla.ttt_num_register_tokens 16
        --vla.ttt_tbptt_step_size 8
    )
fi

EXTRA_ARGS=(
    --vla.expected_world_size "${EXPECTED_WORLD_SIZE}"
    --vla.global_batch_size "${GLOBAL_BATCH_SIZE}"
    --vla.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}"
    --vla.max_steps "${MAX_STEPS}"
    --vla.shuffle_buffer_size "${SHUFFLE_BUFFER_SIZE}"
    --vla.data_mix libero_4_task_suites_no_noops
    --vla.train_qvv_lora True
    --vla.train_projector True
    --vla.train_action_head True
    --vla.train_visual_token_cosine_head True
    --vla.vision_backbone_id levjepa-vit-l-224px
    --vla.levjepa_checkpoint_path "${LEVJEPA_CHECKPOINT}"
        --vla.vla_id "jepavla-qvv25-levjepa-224px+0_5b+mx-libero-90"
    --save_interval "${SAVE_INTERVAL}"
    --cpu_memory_log_interval "${CPU_MEMORY_LOG_INTERVAL}"
    --debug_memory_stats False
    --debug_batch_shapes "${DEBUG_BATCH_SHAPES}"
    --use_swanlab False
)
EXTRA_ARGS+=("${TTT_ARGS[@]}" )

RESUME_ARGS=()
if [[ "${RESUME}" == "1" ]]; then
    if [[ -z "${RUN_ID}" ]]; then
        RUN_ID="${RUN_ID_NOTE}"
    fi
    RESUME_CHECKPOINT="${RUN_ROOT}/${RUN_ID}/checkpoints/latest-checkpoint.pt"
    if [[ -f "${RESUME_CHECKPOINT}" ]]; then
        RESUME_ARGS=(--resume True --run_id "${RUN_ID}" --initial_checkpoint "${RESUME_CHECKPOINT}")
        echo "resume_checkpoint=${RESUME_CHECKPOINT}"
    else
        echo "No checkpoint found for run_id=${RUN_ID}; starting a new run."
        RESUME_ARGS=(--run_id "${RUN_ID}")
    fi
fi

mkdir -p "${RUN_ROOT}"
echo "===== LeVJEPA + JEPA-WAM training ====="
echo "ttt_enabled=${TTT_ENABLED}"
echo "levjepa_checkpoint=${LEVJEPA_CHECKPOINT}"
echo "base_vlm=${BASE_VLM_RUN}"
echo "data=${DATA_ROOT}"
echo "run_root=${RUN_ROOT}"

if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'Command:'
    printf ' %q' "${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc-per-node="${NPROC_PER_NODE}" --module prismatic.training.train \
        --vla.base_vlm "${BASE_VLM_RUN}" \
        --llm_checkpoint_path "${QVV_PATH}" \
        --data_root_dir "${DATA_ROOT}" \
        --run_root_dir "${RUN_ROOT}" \
        --run_id_note "${RUN_ID_NOTE}" \
        "${RESUME_ARGS[@]}" \
        "${EXTRA_ARGS[@]}"
    printf '\n'
    exit 0
fi

"${TORCHRUN_BIN}" \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="${NPROC_PER_NODE}" \
    --module prismatic.training.train \
    --vla.base_vlm "${BASE_VLM_RUN}" \
    --llm_checkpoint_path "${QVV_PATH}" \
    --data_root_dir "${DATA_ROOT}" \
    --run_root_dir "${RUN_ROOT}" \
    --run_id_note "${RUN_ID_NOTE}" \
    "${RESUME_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
