#!/usr/bin/env bash
# PSC Bridges-2: one Round-4 implicit action-expert run on 4 H100 GPUs.
# Submit once for each architecture:
#   sbatch run_ttt_r4_psc_4gpu.slurm wrapper
#   sbatch run_ttt_r4_psc_4gpu.slurm inline
# GPU-shared allows at most four GPUs, so this requests one node and launches
# four distributed workers. Global batch 64 = 4 GPUs x PB4 x accumulation 4.
# Physical context is 128 frames; stride 8 gives 16 memory updates.
#SBATCH -J jepa-wam-r4-psc
#SBATCH -A cis260071p
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus=h100-80:4
#SBATCH --mem=512G
#SBATCH -t 48:00:00
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

ARCH="${1:-}"
case "${ARCH}" in
    wrapper|inline) ;;
    *) echo "Usage: sbatch $0 {wrapper|inline}" >&2; exit 2 ;;
esac
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Submit this script with sbatch." >&2; exit 2; }

PROJECT_ROOT="/ocean/projects/cis260071p/ha865618/project/LeVJEPA_WAM_TTT"
ASSET_ROOT="${PROJECT_ROOT}/jepa_wam_assets"
DATA_ROOT="/ocean/projects/cis260071p/ha865618/data/modified_libero_rlds"
RUN_ROOT="${PROJECT_ROOT}/checkpoints/4th_round"
LOG_ROOT="${PROJECT_ROOT}/slurm_logs/4th_round"
BASE_VLM_RUN="${ASSET_ROOT}/JEPA_WAM/checkpoints/pretrained_vlm/prism-qvv25-vjepa21-vitl-384px+0_5b+stage-finetune+x7"
INITIAL_CHECKPOINT="${ASSET_ROOT}/JEPA_WAM/checkpoints/libero/jepavla-qvv25-vjepa-224px+0_5b+mx-libero-90+n1+b32+x7--visual-cosine-projector-allviews--20260723_232305/checkpoints/step-040000-epoch-37-loss=0.0262.pt"
QVV_PATH="${ASSET_ROOT}/Qvv2.5-0.5B"
VJEPA_CHECKPOINT="${ASSET_ROOT}/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt"
JEPA_PYTHON="${JEPA_PYTHON:-/ocean/projects/cis260071p/ha865618/conda-envs/jepa-wam/bin/python}"
TORCHRUN_BIN="$(dirname "${JEPA_PYTHON}")/torchrun"

for path in "${PROJECT_ROOT}/prismatic/training/train.py" "${DATA_ROOT}/libero_spatial_no_noops" \
    "${BASE_VLM_RUN}" "${INITIAL_CHECKPOINT}" "${QVV_PATH}" "${VJEPA_CHECKPOINT}" \
    "${JEPA_PYTHON}" "${TORCHRUN_BIN}"; do
    [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 1; }
done

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2
export VLA_RLDS_FRAME_TRANSFORM_THREADS=2
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-hang-yu-team}"
export WANDB_PROJECT="${WANDB_PROJECT:-jepa-wam-ttt}"

RUN_ID="jepa-wam-ttt-r4-${ARCH}-sra16-fullkv-v5-implicit-full-dit-kv-action-expert-vjepa21-frames128-stride8-updates16-seg8-gb64-pb4-steps40000-val5-n16-s7"
CHECKPOINT="${RUN_ROOT}/${RUN_ID}/checkpoints/latest-checkpoint.pt"
if [[ -f "${CHECKPOINT}" ]]; then
    CHECKPOINT_ARGS=(--initial_checkpoint "${CHECKPOINT}" --resume True --run_id "${RUN_ID}")
else
    CHECKPOINT_ARGS=(--initial_checkpoint "${INITIAL_CHECKPOINT}" --run_id "${RUN_ID}")
fi

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
cd "${PROJECT_ROOT}"
echo "job=${SLURM_JOB_ID} arch=${ARCH} host=$(hostname) started=$(date)"
nvidia-smi -L

srun --nodes=1 --ntasks=1 --gpus=4 --kill-on-bad-exit=1 \
    "${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc-per-node=4 \
    --module prismatic.training.train \
    --vla.base_vlm "${BASE_VLM_RUN}" \
    "${CHECKPOINT_ARGS[@]}" \
    --llm_checkpoint_path "${QVV_PATH}" \
    --vla.vjepa_checkpoint_path "${VJEPA_CHECKPOINT}" \
    --data_root_dir "${DATA_ROOT}" \
    --run_root_dir "${RUN_ROOT}" \
    --run_id_note "r4-${ARCH}-implicit-action-expert-frames128-stride8-updates16-seg8-gb64-pb4-psc4" \
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
    --vla.ttt_memory_source action_tokens \
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
    --wandb_entity "${WANDB_ENTITY}"
