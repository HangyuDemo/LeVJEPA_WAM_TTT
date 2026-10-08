#!/usr/bin/env bash
# Stampede3 TACC H100: one Round-4 run on four GPUs.
#
# Configuration:
#   one H100 node / four GPUs, 48 hours
#   per-device batch = 4
#   gradient accumulation = 4
#   global batch = 4 GPUs * 4 samples * 4 accumulation = 64
#   physical context = 128 frames, stride = 8 => 16 TTT context updates
#
# Submit from the TACC project directory after activating the JEPA-WAM
# environment (or set JEPA_PYTHON to its absolute Python executable):
#   sbatch v4_WE.slurm
#SBATCH -J jepa-wam-r4-w-exp-gb64
#SBATCH -p h100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 72
#SBATCH -t 48:00:00
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit this script with sbatch, not bash." >&2
    exit 2
fi

ARCH="${R4_ARCH:-wrapper}"
MEMORY_SOURCE="${R4_MEMORY_SOURCE:-jepa_current}"
case "${ARCH}" in
    wrapper|inline) ;;
    *) echo "R4_ARCH must be wrapper or inline" >&2; exit 2 ;;
esac
case "${MEMORY_SOURCE}" in
    jepa_current) MEMORY_TAG=explicit-current-vjepa-kv ;;
    action_tokens) MEMORY_TAG=implicit-full-dit-kv ;;
    *) echo "R4_MEMORY_SOURCE must be jepa_current or action_tokens" >&2; exit 2 ;;
esac

PROJECT_ROOT="/scratch/11441/demo23331/project/LeVJEPA_WAM_TTT"
ASSET_ROOT="${PROJECT_ROOT}/jepa_wam_assets"
DATA_ROOT="/scratch/11441/demo23331/data/modified_libero_rlds"
RUN_ROOT="${PROJECT_ROOT}/checkpoints/4th_round"
LOG_ROOT="${PROJECT_ROOT}/slurm_logs"

BASE_VLM_RUN="${ASSET_ROOT}/JEPA_WAM/checkpoints/pretrained_vlm/prism-qvv25-vjepa21-vitl-384px+0_5b+stage-finetune+x7"
INITIAL_CHECKPOINT="${ASSET_ROOT}/JEPA_WAM/checkpoints/libero/jepavla-qvv25-vjepa-224px+0_5b+mx-libero-90+n1+b32+x7--visual-cosine-projector-allviews--20260723_232305/checkpoints/step-040000-epoch-37-loss=0.0262.pt"
QVV_PATH="${ASSET_ROOT}/Qvv2.5-0.5B"
VJEPA_CHECKPOINT="${ASSET_ROOT}/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt"

JEPA_PYTHON="${JEPA_PYTHON:-$(command -v python || true)}"
TORCHRUN_BIN="$(dirname "${JEPA_PYTHON:-/nonexistent}")/torchrun"

for path in \
    "${PROJECT_ROOT}/prismatic/training/train.py" \
    "${PROJECT_ROOT}/prismatic/conf/vla.py" \
    "${DATA_ROOT}/libero_spatial_no_noops" \
    "${DATA_ROOT}/libero_object_no_noops" \
    "${DATA_ROOT}/libero_goal_no_noops" \
    "${DATA_ROOT}/libero_10_no_noops" \
    "${BASE_VLM_RUN}" "${INITIAL_CHECKPOINT}" "${QVV_PATH}" "${VJEPA_CHECKPOINT}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing required code, data, or asset: ${path}" >&2
        exit 1
    fi
done
if [[ ! -x "${JEPA_PYTHON}" || ! -x "${TORCHRUN_BIN}" ]]; then
    echo "JEPA_PYTHON/torchrun missing. Activate the TACC environment or set JEPA_PYTHON." >&2
    exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2
export VLA_RLDS_FRAME_TRANSFORM_THREADS=1
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-hang-yu-team}"
export WANDB_PROJECT="${WANDB_PROJECT:-jepa-wam-ttt-r4-action-expert-frames128-stride8-gb64}"

"${JEPA_PYTHON}" -c 'import torch, wandb; print("torch", torch.__version__, "cuda", torch.cuda.device_count())' || {
    echo "The TACC Python environment cannot import torch/wandb." >&2
    exit 1
}

cd "${PROJECT_ROOT}"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

# New run ID: deliberately separate from the earlier gb16/pb1 run.
RUN_ID="jepa-wam-ttt-r4-${ARCH}-sra16-fullkv-v5-${MEMORY_TAG}-action-expert-vjepa21-frames128-stride8-updates16-seg8-gb64-pb4-tacc-4gpu"
RUN_DIR="${RUN_ROOT}/${RUN_ID}"
CHECKPOINT="${RUN_DIR}/checkpoints/latest-checkpoint.pt"
if [[ -f "${CHECKPOINT}" ]]; then
    CHECKPOINT_ARGS=(--initial_checkpoint "${CHECKPOINT}" --resume True --run_id "${RUN_ID}")
    echo "Resuming ${RUN_ID} from ${CHECKPOINT}"
else
    CHECKPOINT_ARGS=(--initial_checkpoint "${INITIAL_CHECKPOINT}" --run_id "${RUN_ID}")
    echo "Starting ${RUN_ID} from the public JEPA-WAM checkpoint"
fi

echo "job=${SLURM_JOB_ID} host=$(hostname) started=$(date)"
echo "python=${JEPA_PYTHON} data=${DATA_ROOT} run_root=${RUN_ROOT}"
echo "arch=${ARCH} memory_source=${MEMORY_SOURCE} world_size=4 global_batch=64 per_device_batch=4 gradient_accumulation=4 context_frames=128 stride=8"

# Verify all four GPUs before starting the distributed job.
srun --nodes=1 --ntasks=1 \
    "${JEPA_PYTHON}" -c 'import torch; expected=4; print("cuda_devices", torch.cuda.device_count()); assert torch.cuda.is_available() and torch.cuda.device_count() == expected; [torch.cuda.synchronize(i) for i in range(expected)]'

MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT="$((20000 + SLURM_JOB_ID % 10000))"
export MASTER_ADDR MASTER_PORT

TRAIN_ARGS=(--module prismatic.training.train \
    --vla.base_vlm "${BASE_VLM_RUN}" \
    "${CHECKPOINT_ARGS[@]}" \
    --llm_checkpoint_path "${QVV_PATH}" \
    --vla.vjepa_checkpoint_path "${VJEPA_CHECKPOINT}" \
    --data_root_dir "${DATA_ROOT}" \
    --run_root_dir "${RUN_ROOT}" \
    --run_id_note "r4-${ARCH}-${MEMORY_TAG}-tacc-4gpu-gb64-frames128-stride8" \
    --vla.expected_world_size 4 \
    --vla.global_batch_size 64 \
    --vla.per_device_batch_size 4 \
    --vla.max_steps 40000 \
    --vla.shuffle_buffer_size 2500 \
    --vla.data_mix libero_4_task_suites_no_noops \
    --vla.train_qvv_lora False \
    --vla.train_projector False \
    --vla.train_action_head True \
    --vla.train_ttt_only False \
    --vla.train_visual_token_cosine_head False \
    --vla.ttt_enabled True \
    --vla.ttt_memory_source "${MEMORY_SOURCE}" \
    --vla.ttt_architecture "${ARCH}" \
    --vla.ttt_token_scope state_register_action \
    --vla.ttt_action_kv_scope full_dit_tokens \
    --vla.ttt_wrapper_register_tokens True \
    --vla.ttt_context_length 16 \
    --vla.ttt_observation_stride 8 \
    --vla.ttt_carry_between_segments True \
    --vla.ttt_require_full_context True \
    --vla.ttt_num_register_tokens 16 \
    --vla.ttt_tbptt_step_size 8 \
    --vla.visual_token_pair_offset 0 \
    --vla.enable_gradient_checkpointing True \
    --save_interval 500 \
    --cpu_memory_log_interval 10 \
    --debug_batch_shapes True \
    --debug_memory_stats False \
    --validation_percent 5 \
    --validation_sequences_per_suite 16 \
    --validation_interval 1000 \
    --validation_seed 7 \
    --check_finite True \
    --parameter_check_interval 100 \
    --use_swanlab False \
    --use_wandb True \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_entity "${WANDB_ENTITY}")

echo "Launching ${ARCH}-${MEMORY_TAG} on four GPUs; rendezvous=${MASTER_ADDR}:${MASTER_PORT}"
srun --nodes=1 --ntasks=1 --kill-on-bad-exit=1 \
    "${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc-per-node=4 \
    "${TRAIN_ARGS[@]}"

echo "Completed at $(date)"
