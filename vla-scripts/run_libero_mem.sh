#!/usr/bin/env bash

# Train JEPA-WAM on LIBERO-Mem in two explicit stages:
#   STAGE=policy  - adapt the ordinary JEPA-WAM policy to LIBERO-Mem.
#   STAGE=ttt     - load that policy, freeze it, and train only TTT memory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

STAGE="${STAGE:-policy}"
case "${STAGE}" in
    policy|ttt) ;;
    *) echo "STAGE must be policy or ttt, got ${STAGE}." >&2; exit 1 ;;
esac

ENV_ROOT="${ENV_ROOT:-/home/ha865618/.conda/envs/jepa_wam}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_ROOT}/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${ENV_ROOT}/bin/torchrun}"
DATA_ROOT="${DATA_ROOT:-/home/ha865618/data/modified_libero_rlds}"
DATASET_VERSION_DIR="${DATA_ROOT}/libero_mem/1.0.0"
ASSET_ROOT="${ASSET_ROOT:-${REPO_ROOT}/jepa_wam_assets}"
BASE_VLM_RUN="${BASE_VLM_RUN:-${ASSET_ROOT}/JEPA_WAM/checkpoints/pretrained_vlm/prism-qvv25-vjepa21-vitl-384px+0_5b+stage-finetune+x7}"
QVV_PATH="${QVV_PATH:-${ASSET_ROOT}/Qvv2.5-0.5B}"
VJEPA_CHECKPOINT="${VJEPA_CHECKPOINT:-${ASSET_ROOT}/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt}"
PUBLIC_POLICY_CHECKPOINT="${PUBLIC_POLICY_CHECKPOINT:-${ASSET_ROOT}/JEPA_WAM/checkpoints/libero/jepavla-qvv25-vjepa-224px+0_5b+mx-libero-90+n1+b32+x7--visual-cosine-projector-allviews--20260723_232305/checkpoints/step-040000-epoch-37-loss=0.0262.pt}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/checkpoints/libero_mem}"

NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NODE_RANK="${NODE_RANK:-0}"
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-2500}"
SEED="${SEED:-7}"
USE_WANDB="${USE_WANDB:-False}"
WANDB_PROJECT="${WANDB_PROJECT:-jepa-wam-libero-mem}"

for integer_name in NNODES NPROC_PER_NODE NODE_RANK SAVE_INTERVAL SHUFFLE_BUFFER_SIZE SEED; do
    integer_value="${!integer_name}"
    if [[ ! "${integer_value}" =~ ^[0-9]+$ ]] || [[ "${integer_name}" != "NODE_RANK" && "${integer_value}" -lt 1 ]]; then
        echo "${integer_name} must be a valid non-negative/positive integer, got ${integer_value}." >&2
        exit 1
    fi
done
case "${USE_WANDB}" in True|False) ;; *) echo "USE_WANDB must be True or False." >&2; exit 1 ;; esac
case "${RESUME}" in 0|1) ;; *) echo "RESUME must be 0 or 1." >&2; exit 1 ;; esac
case "${DRY_RUN}" in 0|1) ;; *) echo "DRY_RUN must be 0 or 1." >&2; exit 1 ;; esac

for required in "${PYTHON_BIN}" "${TORCHRUN_BIN}" "${BASE_VLM_RUN}" "${QVV_PATH}" "${VJEPA_CHECKPOINT}"; do
    [[ -e "${required}" ]] || { echo "Missing required path: ${required}" >&2; exit 1; }
done

# TFDS publishes dataset_info.json only after all shards are complete. This
# prevents training from opening a partially converted LIBERO-Mem dataset.
if [[ ! -s "${DATASET_VERSION_DIR}/dataset_info.json" && "${DRY_RUN}" != "1" ]]; then
    echo "LIBERO-Mem RLDS conversion is incomplete: ${DATASET_VERSION_DIR}/dataset_info.json is missing." >&2
    echo "Wait for the converter to print 'LIBERO-Mem RLDS conversion complete'." >&2
    exit 1
fi
if [[ "${DRY_RUN}" != "1" ]] && ! find "${DATASET_VERSION_DIR}" -maxdepth 1 -type f -name '*.tfrecord*' -print -quit | grep -q .; then
    echo "No completed TFRecord shards found under ${DATASET_VERSION_DIR}." >&2
    exit 1
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-1}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-1}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-1}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export VLA_RLDS_FRAME_TRANSFORM_THREADS="${VLA_RLDS_FRAME_TRANSFORM_THREADS:-1}"

check_checkpoint_contract() {
    local checkpoint="$1"
    local checkpoint_run_dir
    checkpoint_run_dir="$(cd "$(dirname "${checkpoint}")/.." && pwd)"
    for metadata in config.json dataset_statistics.json; do
        [[ -s "${checkpoint_run_dir}/${metadata}" ]] || {
            echo "Checkpoint metadata is missing: ${checkpoint_run_dir}/${metadata}" >&2
            return 1
        }
    done
    "${PYTHON_BIN}" -c '
import json, sys
config = json.load(open(sys.argv[1]))
horizon = int(config["vla"].get("action_horizon", 20))
if horizon != 20:
    raise SystemExit(f"Checkpoint action_horizon={horizon}; this launcher requires the public H=20 contract.")
' "${checkpoint_run_dir}/config.json"
}

COMMON_ARGS=(
    --vla.base_vlm "${BASE_VLM_RUN}"
    --llm_checkpoint_path "${QVV_PATH}"
    --vla.vjepa_checkpoint_path "${VJEPA_CHECKPOINT}"
    --data_root_dir "${DATA_ROOT}"
    --run_root_dir "${RUN_ROOT}"
    --vla.data_mix libero_mem
    --vla.expected_world_size "${WORLD_SIZE}"
    --vla.shuffle_buffer_size "${SHUFFLE_BUFFER_SIZE}"
    --vla.action_horizon 20
    --vla.d_action 7
    --vla.d_proprio 8
    --vla.enable_gradient_checkpointing True
    --save_interval "${SAVE_INTERVAL}"
    --seed "${SEED}"
    --cpu_memory_log_interval 10
    --debug_batch_shapes True
    --debug_memory_stats False
    --check_finite True
    --parameter_check_interval 100
    --use_swanlab False
    --use_wandb "${USE_WANDB}"
    --wandb_project "${WANDB_PROJECT}"
)

if [[ "${STAGE}" == "policy" ]]; then
    # The paper's LIBERO optimizer schedule is retained. The public checkpoint
    # predicts 20 actions, so this launcher deliberately preserves horizon 20
    # instead of silently changing checkpoint semantics to the paper's H=8.
    MAX_STEPS="${MAX_STEPS:-60000}"
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
    PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-16}"
    POLICY_INIT="${POLICY_INIT:-public}"
    case "${POLICY_INIT}" in public|base) ;; *) echo "POLICY_INIT must be public or base." >&2; exit 1 ;; esac
    RUN_ID="${RUN_ID:-jepa-wam-libero-mem-policy-vjepa21-h20-gb${GLOBAL_BATCH_SIZE}-steps${MAX_STEPS}-s${SEED}}"

    INITIAL_ARGS=()
    if [[ "${POLICY_INIT}" == "public" ]]; then
        POLICY_INIT_CHECKPOINT="${POLICY_INIT_CHECKPOINT:-${PUBLIC_POLICY_CHECKPOINT}}"
        [[ -f "${POLICY_INIT_CHECKPOINT}" ]] || { echo "Missing public policy checkpoint: ${POLICY_INIT_CHECKPOINT}" >&2; exit 1; }
        check_checkpoint_contract "${POLICY_INIT_CHECKPOINT}"
        INITIAL_ARGS=(--initial_checkpoint "${POLICY_INIT_CHECKPOINT}")
    fi

    STAGE_ARGS=(
        --vla.global_batch_size "${GLOBAL_BATCH_SIZE}"
        --vla.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}"
        --vla.max_steps "${MAX_STEPS}"
        --vla.learning_rate 2e-4
        --vla.min_learning_rate 1e-5
        --vla.weight_decay 0.0
        --vla.max_grad_norm 1.0
        --vla.warmup_ratio 0.03
        --vla.visual_token_pair_offset 31
        --vla.lambda_visual_token_cosine 0.5
        --vla.train_qvv_lora True
        --vla.train_projector False
        --vla.train_action_head True
        --vla.train_ttt_only False
        --vla.train_visual_token_cosine_head True
        --vla.ttt_enabled False
        --vla.ttt_context_length 1
        --validation_percent 0
        --run_id_note libero-mem-policy
    )
else
    POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-}"
    if [[ -z "${POLICY_CHECKPOINT}" || ! -f "${POLICY_CHECKPOINT}" ]]; then
        echo "STAGE=ttt requires POLICY_CHECKPOINT pointing to the completed LIBERO-Mem policy checkpoint." >&2
        exit 1
    fi
    check_checkpoint_contract "${POLICY_CHECKPOINT}"

    TTT_ARCHITECTURE="${TTT_ARCHITECTURE:-wrapper}"
    TTT_MEMORY_SOURCE="${TTT_MEMORY_SOURCE:-jepa}"
    TTT_CONTEXT_FRAMES="${TTT_CONTEXT_FRAMES:-128}"
    TTT_OBSERVATION_STRIDE="${TTT_OBSERVATION_STRIDE:-8}"
    TTT_CONTEXT_LENGTH=$((TTT_CONTEXT_FRAMES / TTT_OBSERVATION_STRIDE))
    TTT_TBPTT_STEP_SIZE="${TTT_TBPTT_STEP_SIZE:-8}"
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1}"
    PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
    TTT_OBSERVATION_BUDGET="${TTT_OBSERVATION_BUDGET:-5120000}"
    case "${TTT_ARCHITECTURE}" in wrapper|inline) ;; *) echo "TTT_ARCHITECTURE must be wrapper or inline." >&2; exit 1 ;; esac
    case "${TTT_MEMORY_SOURCE}" in
        jepa|jepa_current|action_tokens) ;;
        *) echo "TTT_MEMORY_SOURCE must be jepa, jepa_current, or action_tokens." >&2; exit 1 ;;
    esac
    for integer_name in TTT_CONTEXT_FRAMES TTT_OBSERVATION_STRIDE TTT_CONTEXT_LENGTH TTT_TBPTT_STEP_SIZE GLOBAL_BATCH_SIZE PER_DEVICE_BATCH_SIZE TTT_OBSERVATION_BUDGET; do
        integer_value="${!integer_name}"
        [[ "${integer_value}" =~ ^[1-9][0-9]*$ ]] || { echo "${integer_name} must be positive." >&2; exit 1; }
    done
    (( TTT_CONTEXT_FRAMES % TTT_OBSERVATION_STRIDE == 0 )) || {
        echo "TTT_CONTEXT_FRAMES must be divisible by TTT_OBSERVATION_STRIDE." >&2; exit 1;
    }
    (( TTT_TBPTT_STEP_SIZE >= 2 && TTT_CONTEXT_LENGTH % TTT_TBPTT_STEP_SIZE == 0 )) || {
        echo "TTT_CONTEXT_LENGTH must be divisible by TTT_TBPTT_STEP_SIZE>=2." >&2; exit 1;
    }
    OBSERVATIONS_PER_UPDATE=$((GLOBAL_BATCH_SIZE * TTT_CONTEXT_LENGTH))
    if [[ -z "${MAX_STEPS:-}" ]]; then
        (( TTT_OBSERVATION_BUDGET % OBSERVATIONS_PER_UPDATE == 0 )) || {
            echo "TTT_OBSERVATION_BUDGET must divide by GLOBAL_BATCH_SIZE * TTT_CONTEXT_LENGTH." >&2; exit 1;
        }
        MAX_STEPS=$((TTT_OBSERVATION_BUDGET / OBSERVATIONS_PER_UPDATE))
    fi
    MEMORY_TAG="explicit-jepa-kv"
    [[ "${TTT_MEMORY_SOURCE}" == "jepa_current" ]] && MEMORY_TAG="explicit-current-vjepa-kv"
    [[ "${TTT_MEMORY_SOURCE}" == "action_tokens" ]] && MEMORY_TAG="implicit-full-dit-kv"
    RUN_ID="${RUN_ID:-jepa-wam-libero-mem-ttt-${TTT_ARCHITECTURE}-${MEMORY_TAG}-frames${TTT_CONTEXT_FRAMES}-stride${TTT_OBSERVATION_STRIDE}-updates${TTT_CONTEXT_LENGTH}-seg${TTT_TBPTT_STEP_SIZE}-gb${GLOBAL_BATCH_SIZE}-steps${MAX_STEPS}-s${SEED}}"
    INITIAL_ARGS=(--initial_checkpoint "${POLICY_CHECKPOINT}")
    STAGE_ARGS=(
        --vla.global_batch_size "${GLOBAL_BATCH_SIZE}"
        --vla.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}"
        --vla.max_steps "${MAX_STEPS}"
        --vla.train_qvv_lora False
        --vla.train_projector False
        --vla.train_action_head False
        --vla.train_ttt_only True
        --vla.train_visual_token_cosine_head False
        --vla.ttt_enabled True
        --vla.ttt_memory_source "${TTT_MEMORY_SOURCE}"
        --vla.ttt_architecture "${TTT_ARCHITECTURE}"
        --vla.ttt_token_scope state_register_action
        --vla.ttt_action_kv_scope full_dit_tokens
        --vla.ttt_wrapper_register_tokens True
        --vla.ttt_context_length "${TTT_CONTEXT_LENGTH}"
        --vla.ttt_observation_stride "${TTT_OBSERVATION_STRIDE}"
        --vla.ttt_carry_between_segments True
        --vla.ttt_require_full_context True
        --vla.ttt_num_register_tokens 16
        --vla.ttt_tbptt_step_size "${TTT_TBPTT_STEP_SIZE}"
        --vla.visual_token_pair_offset 0
        --validation_percent 5
        --validation_sequences_per_suite 16
        --validation_interval 1000
        --validation_seed "${SEED}"
        --run_id_note libero-mem-ttt
    )
fi

for integer_name in MAX_STEPS GLOBAL_BATCH_SIZE PER_DEVICE_BATCH_SIZE; do
    integer_value="${!integer_name}"
    [[ "${integer_value}" =~ ^[1-9][0-9]*$ ]] || { echo "${integer_name} must be positive." >&2; exit 1; }
done
if (( GLOBAL_BATCH_SIZE % (PER_DEVICE_BATCH_SIZE * WORLD_SIZE) != 0 )); then
    echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be divisible by PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE} * WORLD_SIZE=${WORLD_SIZE}." >&2
    exit 1
fi

mkdir -p "${RUN_ROOT}"
LATEST_CHECKPOINT="${RUN_ROOT}/${RUN_ID}/checkpoints/latest-checkpoint.pt"
if [[ "${RESUME}" == "1" ]]; then
    [[ -f "${LATEST_CHECKPOINT}" ]] || { echo "Resume checkpoint not found: ${LATEST_CHECKPOINT}" >&2; exit 1; }
    check_checkpoint_contract "${LATEST_CHECKPOINT}"
    INITIAL_ARGS=(--initial_checkpoint "${LATEST_CHECKPOINT}" --resume True)
fi

TRAIN_ARGS=(
    --module prismatic.training.train
    "${COMMON_ARGS[@]}"
    "${INITIAL_ARGS[@]}"
    "${STAGE_ARGS[@]}"
    --run_id "${RUN_ID}"
)

echo "===== LIBERO-Mem JEPA-WAM training ====="
echo "stage=${STAGE} run_id=${RUN_ID} world_size=${WORLD_SIZE}"
echo "data=${DATASET_VERSION_DIR} global_batch=${GLOBAL_BATCH_SIZE} per_device_batch=${PER_DEVICE_BATCH_SIZE} max_steps=${MAX_STEPS}"
if [[ "${STAGE}" == "policy" ]]; then
    echo "policy_init=${POLICY_INIT} action_horizon=20 pair_offset=31"
else
    echo "policy_checkpoint=${POLICY_CHECKPOINT} ttt=${TTT_ARCHITECTURE}/${TTT_MEMORY_SOURCE} frames=${TTT_CONTEXT_FRAMES} stride=${TTT_OBSERVATION_STRIDE} updates=${TTT_CONTEXT_LENGTH} segment=${TTT_TBPTT_STEP_SIZE}"
fi

if [[ "${NNODES}" == "1" ]]; then
    TORCHRUN_ARGS=(--standalone --nnodes=1 --nproc-per-node="${NPROC_PER_NODE}")
else
    : "${MASTER_ADDR:?Set MASTER_ADDR for multi-node training}"
    : "${MASTER_PORT:?Set MASTER_PORT for multi-node training}"
    TORCHRUN_ARGS=(
        --nnodes="${NNODES}"
        --nproc-per-node="${NPROC_PER_NODE}"
        --node-rank="${NODE_RANK}"
        --master-addr="${MASTER_ADDR}"
        --master-port="${MASTER_PORT}"
    )
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'Command:'
    printf ' %q' "${TORCHRUN_BIN}" "${TORCHRUN_ARGS[@]}" "${TRAIN_ARGS[@]}"
    printf '\n'
    exit 0
fi

exec "${TORCHRUN_BIN}" "${TORCHRUN_ARGS[@]}" "${TRAIN_ARGS[@]}"
