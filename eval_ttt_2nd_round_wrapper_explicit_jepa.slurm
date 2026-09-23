#!/usr/bin/env bash
# Second-round LIBERO-Plus evaluation: wrapper + full-DiT K/V,
# with explicit JEPA representation memory.
# Array entries evaluate: spatial, object, goal, and LIBERO-10.
# No rollout videos are saved.
#SBATCH --job-name=eval-r2-explicit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-3
#SBATCH --gres=gpu:nvidia_h100_pcie:1
# Keep the evaluation CPU footprint small; MuJoCo runs on the allocated core(s).
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
# Four suites can take several days at one trial per LIBERO-Plus variant.
#SBATCH --time=1-00:00:00
# The previous resume attempt stalled on this node during PyTorch CUDA init.
#SBATCH --exclude=evc25
#SBATCH --output=/home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%A_%a.out
#SBATCH --error=/home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%A_%a.err

set -euo pipefail

REPO_ROOT="/home/ha865618/project/LeVJEPA_WAM_TTT"
PYTHON_BIN="${PYTHON_BIN:-/home/ha865618/.conda/envs/jepa_wam/bin/python}"
CHECKPOINT="${REPO_ROOT}/checkpoints/2nd_round/jepa-wam-ttt-r2-wrapper-sra16-fullkv-v4-explicit-jepa-kv-vjepa21-context32-seg8-gb1-pb1-steps40000-val5-n16-s7/checkpoints/step-029000-epoch-00-loss=0.0029.pt"
RUN_TAG="jepa-wam-ttt-r2-wrapper-sra16-fullkv-v4-explicit-jepa-kv-vjepa21-context32-seg8-gb1-pb1-steps40000-val5-n16-s7"
SUITES=(libero_spatial libero_object libero_goal libero_10)
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
TASK_SUITE="${SUITES[ARRAY_INDEX]}"
JOB_TAG="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}_${ARRAY_INDEX}"

export SAVE_ROLLOUTS=False
export NUM_OPEN_LOOP_STEPS=8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export TOKENIZERS_PARALLELISM=false
export EVAL_TIMESTAMP="${JOB_TAG}"
export RUN_TAG
export RESULT_ROOT="${REPO_ROOT}/rollout_ttt_eval/2nd_round/${RUN_TAG}"
export LOCAL_LOG_DIR="${REPO_ROOT}/experiments/logs/ttt-eval-${RUN_TAG}-${JOB_TAG}"
export LIBERO_CONFIG_PATH="${REPO_ROOT}/.libero_plus_config/ttt-eval-${RUN_TAG}-${JOB_TAG}"

BASE_VLM_RUN="${REPO_ROOT}/jepa_wam_assets/JEPA_WAM/checkpoints/pretrained_vlm/prism-qvv25-vjepa21-vitl-384px+0_5b+stage-finetune+x7"
QVV_PATH="${REPO_ROOT}/jepa_wam_assets/Qvv2.5-0.5B"
VJEPA_CKPT="${REPO_ROOT}/jepa_wam_assets/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt"
LIBERO_PATH="/home/ha865618/project/LIBERO-plus"
export BASE_VLM_RUN QVV_PATH VJEPA_CKPT LIBERO_PATH

for required_path in "${PYTHON_BIN}" "${CHECKPOINT}" "${BASE_VLM_RUN}" "${QVV_PATH}" "${VJEPA_CKPT}" "${LIBERO_PATH}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path does not exist: ${required_path}" >&2
        exit 1
    fi
done

mkdir -p "${LOCAL_LOG_DIR}" "${LIBERO_CONFIG_PATH}" "${RESULT_ROOT}"
cd "${REPO_ROOT}"

echo "===== SECOND-ROUND LIBERO-Plus EVALUATION ====="
echo "model=${RUN_TAG}"
echo "checkpoint=${CHECKPOINT}"
echo "task_suite=${TASK_SUITE}"
echo "result_root=${RESULT_ROOT}"
echo "save_rollouts=${SAVE_ROLLOUTS}"
echo "num_open_loop_steps=${NUM_OPEN_LOOP_STEPS}"
echo "cpus=${SLURM_CPUS_PER_TASK:-unset}"
echo "host=$(hostname)"
echo "date=$(date)"
nvidia-smi -L
if ! timeout 120 "${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available(), "torch.cuda.is_available() is False"; assert torch.cuda.device_count() > 0, "torch.cuda.device_count() == 0"; print(f"torch_cuda_device={torch.cuda.get_device_name(0)}", flush=True)'; then
    echo "GPU preflight failed or timed out; refusing to run evaluation on CPU." >&2
    exit 1
fi

bash vla-scripts/libero_plus_ttt.sh "${CHECKPOINT}" "${TASK_SUITE}" all 1
