#!/usr/bin/env bash
#SBATCH --job-name=jepa-wam-ttt-action
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.out
#SBATCH --error=/home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.err

set -euo pipefail

REPO_ROOT="/home/ha865618/project/LeVJEPA_WAM_TTT"
ENV_ROOT="/home/ha865618/.conda/envs/jepa_wam"
PYTHON_BIN="${ENV_ROOT}/bin/python"
TORCHRUN_BIN="${ENV_ROOT}/bin/torchrun"

DATA_ROOT="/home/ha865618/data/modified_libero_rlds"
ASSET_ROOT="${REPO_ROOT}/jepa_wam_assets"
BASE_VLM_RUN="${ASSET_ROOT}/JEPA_WAM/checkpoints/pretrained_vlm/prism-qwen25-vjepa21-vitl-384px+0_5b+stage-finetune+x7"
INITIAL_CHECKPOINT="${ASSET_ROOT}/JEPA_WAM/checkpoints/libero/jepavla-qwen25-vjepa-224px+0_5b+mx-libero-90+n1+b32+x7--visual-cosine-projector-allviews--20260723_232305/checkpoints/step-040000-epoch-37-loss=0.0262.pt"
QWEN_PATH="${ASSET_ROOT}/Qwen2.5-0.5B"
VJEPA_CHECKPOINT="${ASSET_ROOT}/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt"
RUN_ROOT="${REPO_ROOT}/checkpoints"

# These can be overridden at submission time, for example:
# TTT_CONTEXT_LENGTH=8 MAX_STEPS=100 sbatch run_ttt_action_expert_1gpu.slurm
TTT_CONTEXT_LENGTH="${TTT_CONTEXT_LENGTH:-2}"
MAX_STEPS="${MAX_STEPS:-40000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2500}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"

mkdir -p "${RUN_ROOT}" "${REPO_ROOT}/slurm_logs"

for path in "${REPO_ROOT}" "${DATA_ROOT}" "${BASE_VLM_RUN}" "${INITIAL_CHECKPOINT}" "${QWEN_PATH}" "${VJEPA_CHECKPOINT}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Required path does not exist: ${path}" >&2
        exit 1
    fi
done

if [[ ! -x "${PYTHON_BIN}" || ! -x "${TORCHRUN_BIN}" ]]; then
    echo "jepa_wam Python or torchrun executable is missing." >&2
    exit 1
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=2

cd "${REPO_ROOT}"

echo "===== JEPA-WAM TTT ACTION-EXPERT TRAINING ====="
echo "host=$(hostname)"
echo "date=$(date)"
echo "repo=${REPO_ROOT}"
echo "python=${PYTHON_BIN}"
echo "python_version=$(${PYTHON_BIN} --version)"
echo "data=${DATA_ROOT}"
echo "initial_checkpoint=${INITIAL_CHECKPOINT}"
echo "ttt_context_length=${TTT_CONTEXT_LENGTH}"
echo "max_steps=${MAX_STEPS}"
echo "save_interval=${SAVE_INTERVAL}"
nvidia-smi

"${TORCHRUN_BIN}" \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=1 \
    --module prismatic.training.train \
    --vla.base_vlm "${BASE_VLM_RUN}" \
    --initial_checkpoint "${INITIAL_CHECKPOINT}" \
    --llm_checkpoint_path "${QWEN_PATH}" \
    --vla.vjepa_checkpoint_path "${VJEPA_CHECKPOINT}" \
    --data_root_dir "${DATA_ROOT}" \
    --run_root_dir "${RUN_ROOT}" \
    --run_id_note "ttt-action-expert-only" \
    --vla.expected_world_size 1 \
    --vla.global_batch_size "${GLOBAL_BATCH_SIZE}" \
    --vla.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --vla.max_steps "${MAX_STEPS}" \
    --vla.shuffle_buffer_size 20000 \
    --vla.train_qwen_lora False \
    --vla.train_action_head True \
    --vla.train_visual_token_cosine_head False \
    --vla.ttt_enabled True \
    --vla.ttt_context_length "${TTT_CONTEXT_LENGTH}" \
    --vla.ttt_num_register_tokens 16 \
    --vla.ttt_tbptt_step_size 8 \
    --vla.visual_token_pair_offset 0 \
    --save_interval "${SAVE_INTERVAL}" \
    --cpu_memory_log_interval 10 \
    --debug_batch_shapes True \
    --debug_memory_stats False \
    --use_swanlab False

echo "Training completed at $(date)"
