#!/usr/bin/env bash
# Resume the exact 2026-09-18 LIBERO-10 summary with one CPU core.
#SBATCH --job-name=eval-i-jepa-r18
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:nvidia_h100_pcie:1
# evc25 stalled during the PyTorch CUDA initialization check in the previous run.
# Keep this resume job on a different H100 node.
#SBATCH --exclude=evc25
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.out
#SBATCH --error=/home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.err

set -euo pipefail

REPO_ROOT="/home/ha865618/project/LeVJEPA_WAM_TTT"
PYTHON_BIN="/home/ha865618/.conda/envs/jepa_wam/bin/python"
CHECKPOINT="${REPO_ROOT}/checkpoints/1st_round/jepa-wam-ttt-inline-jepa-memory-vjepa21-context8-gb1-pb1/checkpoints/latest-checkpoint.pt"
RESUME_SUMMARY="${REPO_ROOT}/rollout_ttt_eval/1st_round/jepa-wam-ttt-inline-jepa-memory-vjepa21-context8-gb1-pb1/libero_10/summary-2026_09_18-03_18_58.json"
RESULT_ROOT="${REPO_ROOT}/rollout_ttt_eval/1st_round/jepa-wam-ttt-inline-jepa-memory-vjepa21-context8-gb1-pb1"
BASE_VLM_RUN="${REPO_ROOT}/jepa_wam_assets/JEPA_WAM/checkpoints/pretrained_vlm/prism-qvv25-vjepa21-vitl-384px+0_5b+stage-finetune+x7"
QVV_PATH="${REPO_ROOT}/jepa_wam_assets/Qvv2.5-0.5B"
VJEPA_CKPT="${REPO_ROOT}/jepa_wam_assets/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt"
LIBERO_PATH="/home/ha865618/project/LIBERO-plus"
LOCAL_LOG_DIR="${REPO_ROOT}/experiments/logs/ttt-eval-i-jepa-r18-${SLURM_JOB_ID}"
LIBERO_CONFIG_PATH="${REPO_ROOT}/.libero_plus_config/ttt-eval-i-jepa-r18-${SLURM_JOB_ID}"

# Bound every common worker pool to the single CPU allocated by Slurm.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export TOKENIZERS_PARALLELISM=false

export LIBERO_PATH LIBERO_CONFIG_PATH
export PYTHONPATH="${REPO_ROOT}:${LIBERO_PATH}:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export PYOPENGL_PLATFORM=egl
export NUMBA_DISABLE_JIT=1
export MPLCONFIGDIR="/tmp/jepa_wam_matplotlib_${SLURM_JOB_ID}"

mkdir -p "${LOCAL_LOG_DIR}" "${LIBERO_CONFIG_PATH}" "${MPLCONFIGDIR}" "${REPO_ROOT}/slurm_logs"
printf '%s\n' \
    "benchmark_root: ${LIBERO_PATH}/libero/libero" \
    "bddl_files: ${LIBERO_PATH}/libero/libero/bddl_files" \
    "init_states: ${LIBERO_PATH}/libero/libero/init_files" \
    "datasets: ${LIBERO_PATH}/libero/datasets" \
    "assets: ${LIBERO_PATH}/libero/libero/assets" \
    > "${LIBERO_CONFIG_PATH}/config.yaml"

for required_path in \
    "${PYTHON_BIN}" "${CHECKPOINT}" "${RESUME_SUMMARY}" \
    "${BASE_VLM_RUN}" "${QVV_PATH}" "${VJEPA_CKPT}" "${LIBERO_PATH}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path does not exist: ${required_path}" >&2
        exit 1
    fi
done

cd "${REPO_ROOT}"
echo "resume_summary=${RESUME_SUMMARY}"
echo "checkpoint=${CHECKPOINT}"
echo "cpu_threads=${OMP_NUM_THREADS}"
echo "host=$(hostname)"
echo "date=$(date)"
nvidia-smi -L
timeout 120 "${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'

"${PYTHON_BIN}" experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint "${CHECKPOINT}" \
    --base_vlm "${BASE_VLM_RUN}" \
    --llm_checkpoint_path "${QVV_PATH}" \
    --vjepa_checkpoint_path "${VJEPA_CKPT}" \
    --task_suite_name libero_10 \
    --libero_plus_categories all \
    --num_trials_per_task 1 \
    --num_open_loop_steps 8 \
    --save_rollouts False \
    --local_log_dir "${LOCAL_LOG_DIR}" \
    --result_root "${RESULT_ROOT}" \
    --resume_from "${RESUME_SUMMARY}"
