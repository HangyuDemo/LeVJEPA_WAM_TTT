#!/usr/bin/env bash
# JEPA-WAM + TTT implicit-memory route. State/register/action tokens provide
# queries; the complete post-attention DiT sequence provides K/V.
# This is a separate route from run_ttt_action_expert_1gpu.slurm, which uses
# the predicted JEPA representation as explicit TTT memory.
#
# Default/fixed entry point: RoboTTT wrapper + implicit full-DiT K/V memory. The
# separate inline entry point is run_ttt_action_tokens_inline_1gpu.slurm.
# Other useful overrides: GLOBAL_BATCH_SIZE=1 MAX_STEPS=40000
#
#SBATCH --job-name=jepa-wam-r2-wrapper-implicit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h100_pcie:1
# Avoid H100 nodes with confirmed CUDA initialization/FSDP failures.
#SBATCH --exclude=evc22,evc30,evc38,evc43,evc45
# Extreme low-CPU trial: keep one GPU while limiting host-side preprocessing.
# The RLDS pipeline still keeps one reader/trajectory thread per LIBERO suite.
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.out
#SBATCH --error=/home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.err

set -euo pipefail
export WANDB_API_KEY="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"
REPO_ROOT="/home/ha865618/project/LeVJEPA_WAM_TTT"
ENV_ROOT="/home/ha865618/.conda/envs/jepa_wam"
PYTHON_BIN="${ENV_ROOT}/bin/python"
TORCHRUN_BIN="${ENV_ROOT}/bin/torchrun"

DATA_ROOT="/home/ha865618/data/modified_libero_rlds"
ASSET_ROOT="${REPO_ROOT}/jepa_wam_assets"
BASE_VLM_RUN="${ASSET_ROOT}/JEPA_WAM/checkpoints/pretrained_vlm/prism-qvv25-vjepa21-vitl-384px+0_5b+stage-finetune+x7"
INITIAL_CHECKPOINT="${ASSET_ROOT}/JEPA_WAM/checkpoints/libero/jepavla-qvv25-vjepa-224px+0_5b+mx-libero-90+n1+b32+x7--visual-cosine-projector-allviews--20260723_232305/checkpoints/step-040000-epoch-37-loss=0.0262.pt"
QVV_PATH="${ASSET_ROOT}/Qvv2.5-0.5B"
VJEPA_CHECKPOINT="${ASSET_ROOT}/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt"
TTT_MEMORY_TAG="implicit-full-dit-kv"
# Explicitly run the full 40,000 optimizer steps for this launcher.  The
# sourced round-2 config derives OBSERVATION_BUDGET from MAX_STEPS.
MAX_STEPS="${MAX_STEPS:-40000}"
source "${REPO_ROOT}/vla-scripts/ttt_round2_config.sh"

mkdir -p "${RUN_ROOT}" "${REPO_ROOT}/slurm_logs"

for path in \
    "${REPO_ROOT}" \
    "${DATA_ROOT}/libero_spatial_no_noops" \
    "${DATA_ROOT}/libero_object_no_noops" \
    "${DATA_ROOT}/libero_goal_no_noops" \
    "${DATA_ROOT}/libero_10_no_noops" \
    "${BASE_VLM_RUN}" \
    "${INITIAL_CHECKPOINT}" \
    "${QVV_PATH}" \
    "${VJEPA_CHECKPOINT}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Required path does not exist: ${path}" >&2
        exit 1
    fi
done

if [[ ! -x "${PYTHON_BIN}" || ! -x "${TORCHRUN_BIN}" ]]; then
    echo "jepa_wam Python or torchrun executable is missing." >&2
    exit 1
fi

RESUME_CHECKPOINT="${RUN_ROOT}/${RUN_ID}/checkpoints/latest-checkpoint.pt"
if [[ -f "${RESUME_CHECKPOINT}" ]]; then
    RESUME_ARGS=(
        --initial_checkpoint "${RESUME_CHECKPOINT}"
        --resume True
        --run_id "${RUN_ID}"
    )
    echo "resume_checkpoint=${RESUME_CHECKPOINT}"
else
    RESUME_ARGS=(
        --initial_checkpoint "${INITIAL_CHECKPOINT}"
        --run_id "${RUN_ID}"
    )
    echo "starting_from_initial_checkpoint=${INITIAL_CHECKPOINT}"
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VLA_RLDS_FRAME_TRANSFORM_THREADS="${VLA_RLDS_FRAME_TRANSFORM_THREADS:-2}"
export TF_CPP_MIN_LOG_LEVEL=2
export WANDB_PROJECT="${WANDB_PROJECT:-jepa-wam-ttt-2nd-round}"
export WANDB_DIR="${WANDB_DIR:-${REPO_ROOT}/wandb}"

cd "${REPO_ROOT}"

echo "===== JEPA-WAM TTT IMPLICIT FULL-DIT K/V TRAINING ====="
echo "host=$(hostname)"
echo "date=$(date)"
echo "repo=${REPO_ROOT}"
echo "python=${PYTHON_BIN}"
echo "python_version=$(${PYTHON_BIN} --version)"
echo "data=${DATA_ROOT}"
echo "initial_checkpoint=${INITIAL_CHECKPOINT}"
echo "ttt_memory_source=action_tokens"
echo "ttt_action_kv_scope=${TTT_ACTION_KV_SCOPE}"
echo "ttt_architecture=${TTT_ARCHITECTURE}"
if [[ "${TTT_ARCHITECTURE}" == "wrapper" ]]; then
    echo "ttt_layer_indices=3,7,11,15"
else
    echo "ttt_layer_indices=all_dit_blocks"
fi
echo "ttt_context_length=${TTT_CONTEXT_LENGTH}"
echo "global_batch_size=${GLOBAL_BATCH_SIZE}"
echo "per_device_batch_size=${PER_DEVICE_BATCH_SIZE}"
echo "max_steps=${MAX_STEPS}"
echo "save_interval=${SAVE_INTERVAL}"
nvidia-smi

"${TORCHRUN_BIN}" \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=1 \
    --module prismatic.training.train \
    --vla.base_vlm "${BASE_VLM_RUN}" \
    "${RESUME_ARGS[@]}" \
    --llm_checkpoint_path "${QVV_PATH}" \
    --vla.vjepa_checkpoint_path "${VJEPA_CHECKPOINT}" \
    --data_root_dir "${DATA_ROOT}" \
    --run_root_dir "${RUN_ROOT}" \
    --run_id_note "r2-${TTT_ARCHITECTURE_TAG}-implicit-full-dit-kv-context${TTT_CONTEXT_LENGTH}-seg${TTT_TBPTT_STEP_SIZE}" \
    --vla.expected_world_size 1 \
    --vla.global_batch_size "${GLOBAL_BATCH_SIZE}" \
    --vla.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --vla.max_steps "${MAX_STEPS}" \
    --vla.shuffle_buffer_size "${SHUFFLE_BUFFER_SIZE:-5000}" \
    --vla.data_mix libero_4_task_suites_no_noops \
    --vla.train_qvv_lora False \
    --vla.train_projector False \
    --vla.train_action_head False \
    --vla.train_ttt_only True \
    --vla.train_visual_token_cosine_head False \
    --vla.ttt_enabled True \
    --vla.ttt_memory_source action_tokens \
    --vla.ttt_architecture "${TTT_ARCHITECTURE}" \
    --vla.ttt_token_scope "${TTT_TOKEN_SCOPE}" \
    --vla.ttt_action_kv_scope "${TTT_ACTION_KV_SCOPE}" \
    --vla.ttt_wrapper_register_tokens "${TTT_WRAPPER_REGISTER_TOKENS}" \
    --vla.ttt_context_length "${TTT_CONTEXT_LENGTH}" \
    --vla.ttt_carry_between_segments True \
    --vla.ttt_require_full_context True \
    --vla.ttt_num_register_tokens 16 \
    --vla.ttt_tbptt_step_size "${TTT_TBPTT_STEP_SIZE}" \
    --vla.visual_token_pair_offset 0 \
    --vla.enable_gradient_checkpointing True \
    --save_interval "${SAVE_INTERVAL}" \
    --cpu_memory_log_interval 10 \
    --debug_batch_shapes True \
    --debug_memory_stats False \
    --validation_percent "${VALIDATION_PERCENT}" \
    --validation_sequences_per_suite "${VALIDATION_SEQUENCES_PER_SUITE}" \
    --validation_interval "${VALIDATION_INTERVAL}" \
    --validation_seed "${VALIDATION_SEED}" \
    --check_finite True \
    --parameter_check_interval "${PARAMETER_CHECK_INTERVAL}" \
    --use_swanlab False \
    --use_wandb True \
    --wandb_project "${WANDB_PROJECT}"

echo "Training completed at $(date)"
