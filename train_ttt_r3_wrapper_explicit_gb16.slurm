#!/usr/bin/env bash
#SBATCH -J ttt-r3-wrapper-explicit
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc24,evc25,evc27,evc34,evc38,evc44
#SBATCH --mem=480G
#SBATCH -t 7-00:00:00
#SBATCH -o /home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.out
#SBATCH -e /home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.err

set -euo pipefail

echo "Slurm job ${SLURM_JOB_ID:-unknown} running on node ${SLURMD_NODENAME:-unknown}"

PROJECT_ROOT="/home/ha865618/project/LeVJEPA_WAM_TTT"
DATA_ROOT="/home/ha865618/data/modified_libero_rlds"
RUN_ROOT="${PROJECT_ROOT}/checkpoints/3rd_round"
ASSET_ROOT="${PROJECT_ROOT}/jepa_wam_assets"
PYTHON_BIN="/home/ha865618/.conda/envs/jepa_wam/bin/python"
TORCHRUN_BIN="/home/ha865618/.conda/envs/jepa_wam/bin/torchrun"
BASE_VLM_RUN="${ASSET_ROOT}/JEPA_WAM/checkpoints/pretrained_vlm/prism-qvv25-vjepa21-vitl-384px+0_5b+stage-finetune+x7"
INITIAL_CHECKPOINT="${ASSET_ROOT}/JEPA_WAM/checkpoints/libero/jepavla-qvv25-vjepa-224px+0_5b+mx-libero-90+n1+b32+x7--visual-cosine-projector-allviews--20260723_232305/checkpoints/step-040000-epoch-37-loss=0.0262.pt"
QVV_PATH="${ASSET_ROOT}/Qvv2.5-0.5B"
VJEPA_CHECKPOINT="${ASSET_ROOT}/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt"
RUN_ID="jepa-wam-ttt-r3-wrapper-sra16-fullkv-v4-explicit-jepa-kv-vjepa21-context128-seg8-gb1-pb1-steps40000-val5-n16-s7"
export WANDB_API_KEY='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'
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2
export VLA_RLDS_FRAME_TRANSFORM_THREADS=1
export WANDB_MODE="${WANDB_MODE:-online}"

mkdir -p "${RUN_ROOT}" "${PROJECT_ROOT}/slurm_logs"
cd "${PROJECT_ROOT}"

for required in "${PYTHON_BIN}" "${TORCHRUN_BIN}" "${DATA_ROOT}/libero_spatial_no_noops" "${DATA_ROOT}/libero_object_no_noops" "${DATA_ROOT}/libero_goal_no_noops" "${DATA_ROOT}/libero_10_no_noops" "${BASE_VLM_RUN}" "${INITIAL_CHECKPOINT}" "${QVV_PATH}" "${VJEPA_CHECKPOINT}"; do
    [[ -e "${required}" ]] || { echo "Missing required path: ${required}" >&2; exit 1; }
done

cuda_ready=0
for attempt in 1 2 3; do
    if timeout --kill-after=10s 120s "${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1; x=torch.ones(1, device="cuda"); torch.cuda.synchronize(); print(torch.cuda.get_device_name(0))'; then
        cuda_ready=1
        break
    fi
    echo "WARNING: PyTorch CUDA preflight attempt ${attempt}/3 failed" >&2
    sleep 15
done
[[ "${cuda_ready}" -eq 1 ]] || { echo "CUDA remained unavailable after three attempts" >&2; exit 1; }

CHECKPOINT="${RUN_ROOT}/${RUN_ID}/checkpoints/latest-checkpoint.pt"
if [[ -f "${CHECKPOINT}" ]]; then
    CHECKPOINT_ARGS=(--initial_checkpoint "${CHECKPOINT}" --resume True --run_id "${RUN_ID}")
    echo "Resuming ${CHECKPOINT}"
else
    CHECKPOINT_ARGS=(--initial_checkpoint "${INITIAL_CHECKPOINT}" --run_id "${RUN_ID}")
    echo "Starting ${RUN_ID} from the public JEPA-WAM checkpoint"
fi

"${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc-per-node=1 --module prismatic.training.train \
    --vla.base_vlm "${BASE_VLM_RUN}" \
    "${CHECKPOINT_ARGS[@]}" \
    --llm_checkpoint_path "${QVV_PATH}" \
    --vla.vjepa_checkpoint_path "${VJEPA_CHECKPOINT}" \
    --data_root_dir "${DATA_ROOT}" \
    --run_root_dir "${RUN_ROOT}" \
    --run_id_note r3-wrapper-explicit-context128-seg8-gb1 \
    --vla.expected_world_size 1 \
    --vla.global_batch_size 1 \
    --vla.per_device_batch_size 1 \
    --vla.max_steps 40000 \
    --vla.shuffle_buffer_size 2500 \
    --vla.data_mix libero_4_task_suites_no_noops \
    --vla.train_qvv_lora False \
    --vla.train_projector False \
    --vla.train_action_head False \
    --vla.train_ttt_only True \
    --vla.train_visual_token_cosine_head False \
    --vla.ttt_enabled True \
    --vla.ttt_memory_source jepa \
    --vla.ttt_architecture wrapper \
    --vla.ttt_token_scope state_register_action \
    --vla.ttt_action_kv_scope full_dit_tokens \
    --vla.ttt_wrapper_register_tokens True \
    --vla.ttt_context_length 128 \
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
    --wandb_project jepa-wam-ttt-context128-gb1 \
    --wandb_entity hang-yu-team
