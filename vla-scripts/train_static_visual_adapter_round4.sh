#!/usr/bin/env bash
# Stateless, single-observation control with the Round-4 WE adapter modules.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
: "${LIBERO_DATA:?Set LIBERO_DATA to the LIBERO RLDS root}"
: "${BASE_VLM_RUN:?Set BASE_VLM_RUN to the pretrained VLM run directory}"
: "${QVV_PATH:?Set QVV_PATH to the Qvv2.5 model/tokenizer directory}"
: "${VJEPA_CKPT:?Set VJEPA_CKPT to the V-JEPA checkpoint}"
: "${R4_INITIAL_CHECKPOINT:?Set R4_INITIAL_CHECKPOINT to the pretrained JEPA-WAM checkpoint}"

TORCHRUN_BIN="${TORCHRUN_BIN:-$(command -v torchrun || true)}"
[[ -n "${TORCHRUN_BIN}" && -x "${TORCHRUN_BIN}" ]] || { echo "torchrun is unavailable" >&2; exit 1; }
R4_NNODES="${R4_NNODES:-1}"
R4_GPUS_PER_NODE="${R4_GPUS_PER_NODE:-4}"
[[ "${R4_NNODES}" =~ ^[1-9][0-9]*$ && "${R4_GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || {
    echo "R4_NNODES and R4_GPUS_PER_NODE must be positive integers" >&2; exit 2;
}
(( R4_NNODES * R4_GPUS_PER_NODE == 4 )) || {
    echo "This recipe requires exactly four GPUs in total" >&2; exit 2;
}
RUN_ROOT="${R4_RUN_ROOT:-${PROJECT_ROOT}/checkpoints/4th_round}"
RUN_ID="${R4_RUN_ID:-srewam-r4-we-static-visual-adapter-single-observation-gb32-pb8}"
[[ "${RUN_ID}" =~ ^[a-zA-Z0-9._-]+$ ]] || { echo "Invalid R4_RUN_ID" >&2; exit 2; }
RUN_DIR="${RUN_ROOT}/${RUN_ID}"
LATEST="${RUN_DIR}/checkpoints/latest-checkpoint.pt"
if [[ -s "${LATEST}" ]]; then
    CHECKPOINT_ARGS=(--initial_checkpoint "${LATEST}" --resume True)
elif [[ -e "${RUN_DIR}" ]]; then
    echo "Run directory exists without a usable latest checkpoint: ${RUN_DIR}" >&2
    exit 1
else
    CHECKPOINT_ARGS=(--initial_checkpoint "${R4_INITIAL_CHECKPOINT}")
fi

TRAIN_ARGS=(
    --module prismatic.training.train
    --vla.base_vlm "${BASE_VLM_RUN}"
    "${CHECKPOINT_ARGS[@]}"
    --run_id "${RUN_ID}"
    --llm_checkpoint_path "${QVV_PATH}"
    --vla.vjepa_checkpoint_path "${VJEPA_CKPT}"
    --data_root_dir "${LIBERO_DATA}"
    --run_root_dir "${RUN_ROOT}"
    --run_id_note static-visual-adapter-single-observation
    --vla.expected_world_size 4
    --vla.global_batch_size 32
    --vla.per_device_batch_size 8
    --vla.max_steps 30000
    --vla.learning_rate 2e-5
    --vla.min_learning_rate 1e-6
    --vla.weight_decay 1e-5
    --vla.shuffle_buffer_size 2500
    --vla.data_mix libero_4_task_suites_no_noops
    --vla.train_qvv_lora False
    --vla.train_projector False
    --vla.train_action_head False
    --vla.train_visual_token_cosine_head False
    --vla.static_visual_adapter True
    --vla.visual_token_pair_offset 0
    --vla.enable_gradient_checkpointing True
    --save_interval 500
    --cpu_memory_log_interval 10
    --debug_batch_shapes True
    --validation_percent 5
    --validation_sequences_per_suite 16
    --validation_interval 1000
    --validation_seed 7
    --check_finite True
    --parameter_check_interval 100
    --use_swanlab False
    --use_wandb True
    --wandb_project "${WANDB_PROJECT:-srewam-static-adapter-ablation}"
    --wandb_entity "${WANDB_ENTITY:-}"
)

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf 'Training arguments:'
    printf ' %q' "${TRAIN_ARGS[@]}"
    printf '\n'
    printf 'Launcher: %s node(s), %s GPU(s) per node\n' "${R4_NNODES}" "${R4_GPUS_PER_NODE}"
    exit 0
fi

for required in "${LIBERO_DATA}/libero_spatial_no_noops" "${LIBERO_DATA}/libero_object_no_noops" \
    "${LIBERO_DATA}/libero_goal_no_noops" "${LIBERO_DATA}/libero_10_no_noops" \
    "${BASE_VLM_RUN}/config.json" "${QVV_PATH}" "${VJEPA_CKPT}" "${CHECKPOINT_ARGS[1]}"; do
    [[ -e "${required}" ]] || { echo "Missing training input: ${required}" >&2; exit 1; }
done

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2
export VLA_RLDS_FRAME_TRANSFORM_THREADS=1
export WANDB_MODE="${WANDB_MODE:-offline}"
source "${SCRIPT_DIR}/wandb_auth.sh"
configure_wandb_auth "${PROJECT_ROOT}"
mkdir -p "${RUN_ROOT}"
cd "${PROJECT_ROOT}"
if (( R4_NNODES == 1 )); then
    "${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc-per-node="${R4_GPUS_PER_NODE}" "${TRAIN_ARGS[@]}"
else
    [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_JOB_NODELIST:-}" ]] || {
        echo "Multi-node training requires a Slurm allocation" >&2; exit 2;
    }
    R4_MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
    R4_MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
    export R4_MASTER_ADDR R4_MASTER_PORT R4_NNODES R4_GPUS_PER_NODE TORCHRUN_BIN
    echo "Launching four ranks on ${R4_NNODES} nodes; rendezvous=${R4_MASTER_ADDR}:${R4_MASTER_PORT}"
    srun --nodes="${R4_NNODES}" --ntasks="${R4_NNODES}" --ntasks-per-node=1 \
        --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" --gpus-per-task="${R4_GPUS_PER_NODE}" \
        --gpu-bind=none --kill-on-bad-exit=1 \
        bash -c 'exec "${TORCHRUN_BIN}" --nnodes="${R4_NNODES}" --nproc-per-node="${R4_GPUS_PER_NODE}" --node-rank="${SLURM_PROCID}" --master-addr="${R4_MASTER_ADDR}" --master-port="${R4_MASTER_PORT}" "$@"' \
        bash "${TRAIN_ARGS[@]}"
fi
