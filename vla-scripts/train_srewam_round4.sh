#!/usr/bin/env bash
# Portable single-node, four-GPU Round-4 TTT-only training entry point.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ARCH="${R4_ARCH:-wrapper}"
MEMORY_SOURCE="${R4_MEMORY_SOURCE:-jepa_current}"
CONTEXT_FRAMES="${R4_CONTEXT_FRAMES:-128}"
OBSERVATION_STRIDE=8
DRY_RUN="${DRY_RUN:-0}"

case "${ARCH}" in wrapper|inline) ;; *) echo "R4_ARCH must be wrapper or inline" >&2; exit 2 ;; esac
case "${MEMORY_SOURCE}" in
    jepa_current) MEMORY_TAG=explicit-current-vjepa-kv ;;
    action_tokens) MEMORY_TAG=implicit-full-dit-kv ;;
    *) echo "R4_MEMORY_SOURCE must be jepa_current or action_tokens" >&2; exit 2 ;;
esac
if [[ ! "${CONTEXT_FRAMES}" =~ ^[0-9]+$ ]] || (( CONTEXT_FRAMES < 16 || CONTEXT_FRAMES % OBSERVATION_STRIDE != 0 )); then
    echo "R4_CONTEXT_FRAMES must be a multiple of 8 and at least 16" >&2
    exit 2
fi
CONTEXT_UPDATES=$((CONTEXT_FRAMES / OBSERVATION_STRIDE))
SEGMENT_SIZE=$((CONTEXT_UPDATES < 8 ? CONTEXT_UPDATES : 8))
if (( CONTEXT_UPDATES % SEGMENT_SIZE != 0 )); then
    echo "The context update count must be divisible by the TBPTT segment size" >&2
    exit 2
fi

: "${LIBERO_DATA:?Set LIBERO_DATA to the modified LIBERO RLDS root}"
: "${BASE_VLM_RUN:?Set BASE_VLM_RUN to the pretrained VLM run directory}"
: "${QVV_PATH:?Set QVV_PATH to the Qvv2.5 model/tokenizer directory}"
: "${VJEPA_CKPT:?Set VJEPA_CKPT to the V-JEPA checkpoint file}"

TORCHRUN_BIN="${TORCHRUN_BIN:-$(command -v torchrun || true)}"
RUN_ROOT="${R4_RUN_ROOT:-${PROJECT_ROOT}/checkpoints/4th_round}"
RUN_ID="${R4_RUN_ID:-srewam-r4-${ARCH}-${MEMORY_TAG}-frames${CONTEXT_FRAMES}-stride8-updates${CONTEXT_UPDATES}-seg${SEGMENT_SIZE}-gb64-pb4}"
if [[ ! "${RUN_ID}" =~ ^[a-zA-Z0-9._-]+$ ]]; then
    echo "R4_RUN_ID must be a single safe directory name" >&2
    exit 2
fi
RUN_DIR="${RUN_ROOT}/${RUN_ID}"
LATEST="${RUN_DIR}/checkpoints/latest-checkpoint.pt"
if [[ -s "${LATEST}" ]]; then
    CHECKPOINT_ARGS=(--initial_checkpoint "${LATEST}" --resume True)
elif [[ "${R4_REQUIRE_RESUME:-0}" == 1 ]]; then
    echo "Required resume checkpoint missing: ${LATEST}" >&2
    exit 1
elif [[ -e "${RUN_DIR}" ]]; then
    echo "Run directory exists without a usable latest checkpoint: ${RUN_DIR}" >&2
    exit 1
else
    : "${R4_INITIAL_CHECKPOINT:?Set R4_INITIAL_CHECKPOINT to the pretrained JEPA-WAM policy checkpoint}"
    CHECKPOINT_ARGS=(--initial_checkpoint "${R4_INITIAL_CHECKPOINT}")
fi

WANDB_PROJECT="${WANDB_PROJECT:-srewam-r4-ttt-only}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_MODE="${WANDB_MODE:-online}"

TRAIN_ARGS=(
    --module prismatic.training.train
    --vla.base_vlm "${BASE_VLM_RUN}"
    "${CHECKPOINT_ARGS[@]}"
    --run_id "${RUN_ID}"
    --llm_checkpoint_path "${QVV_PATH}"
    --vla.vjepa_checkpoint_path "${VJEPA_CKPT}"
    --data_root_dir "${LIBERO_DATA}"
    --run_root_dir "${RUN_ROOT}"
    --run_id_note "r4-${ARCH}-${MEMORY_TAG}-ttt-only-frames${CONTEXT_FRAMES}-stride8"
    --vla.expected_world_size 4
    --vla.global_batch_size 64
    --vla.per_device_batch_size 4
    --vla.max_steps 30000
    --vla.learning_rate 2e-5
    --vla.min_learning_rate 1e-6
    --vla.weight_decay 1e-5
    --vla.shuffle_buffer_size 2500
    --vla.data_mix libero_4_task_suites_no_noops
    --vla.train_qvv_lora False
    --vla.train_projector False
    --vla.train_action_head False
    --vla.train_ttt_only True
    --vla.train_visual_token_cosine_head False
    --vla.ttt_enabled True
    --vla.ttt_memory_source "${MEMORY_SOURCE}"
    --vla.ttt_architecture "${ARCH}"
    --vla.ttt_token_scope state_register_action
    --vla.ttt_action_kv_scope full_dit_tokens
    --vla.ttt_wrapper_register_tokens True
    --vla.ttt_context_length "${CONTEXT_UPDATES}"
    --vla.ttt_observation_stride "${OBSERVATION_STRIDE}"
    --vla.ttt_carry_between_segments True
    --vla.ttt_require_full_context True
    --vla.ttt_num_register_tokens 16
    --vla.ttt_tbptt_step_size "${SEGMENT_SIZE}"
    --vla.visual_token_pair_offset 0
    --vla.enable_gradient_checkpointing True
    --save_interval 500
    --cpu_memory_log_interval 10
    --debug_batch_shapes True
    --debug_memory_stats False
    --validation_percent 5
    --validation_sequences_per_suite 16
    --validation_interval 1000
    --validation_seed 7
    --check_finite True
    --parameter_check_interval 100
    --use_swanlab False
    --use_wandb True
    --wandb_project "${WANDB_PROJECT}"
    --wandb_entity "${WANDB_ENTITY}"
)

if [[ -z "${TORCHRUN_BIN}" || ! -x "${TORCHRUN_BIN}" ]]; then
    echo "torchrun is unavailable; activate the project environment or set TORCHRUN_BIN" >&2
    exit 1
fi
if [[ "${DRY_RUN}" == 1 ]]; then
    printf 'Command:'
    printf ' %q' "${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc-per-node=4 "${TRAIN_ARGS[@]}"
    printf '\n'
    exit 0
fi

for path in "${LIBERO_DATA}/libero_spatial_no_noops" "${LIBERO_DATA}/libero_object_no_noops" \
    "${LIBERO_DATA}/libero_goal_no_noops" "${LIBERO_DATA}/libero_10_no_noops" \
    "${BASE_VLM_RUN}" "${QVV_PATH}" "${VJEPA_CKPT}" "${CHECKPOINT_ARGS[1]}"; do
    [[ -e "${path}" ]] || { echo "Missing training input: ${path}" >&2; exit 1; }
done

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2
export VLA_RLDS_FRAME_TRANSFORM_THREADS=1
source "${SCRIPT_DIR}/wandb_auth.sh"
configure_wandb_auth "${PROJECT_ROOT}"
mkdir -p "${RUN_ROOT}"
cd "${PROJECT_ROOT}"
"${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc-per-node=4 "${TRAIN_ARGS[@]}"
