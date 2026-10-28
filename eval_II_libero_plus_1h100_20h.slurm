#!/usr/bin/env bash
# Fourth-round II: four LIBERO-Plus suites concurrently on one H100.
# Flow Matching takes four denoising steps; TTT memory updates at every step.
#SBATCH --job-name=r4-ii-plus4
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --no-requeue
#SBATCH --output=/home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.out
#SBATCH --error=/home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT=/home/ha865618/project/LeVJEPA_WAM_TTT
RUN_ID=jepa-wam-ttt-r4-inline-sra16-fullkv-v5-implicit-full-dit-kv-ttt-only-vjepa21-frames128-stride8-updates16-seg8-gb64-pb4-steps30000-tacc-4gpu
CHECKPOINT="${PROJECT_ROOT}/checkpoints/4th_round/${RUN_ID}/checkpoints/latest-checkpoint.pt"
RESULT_ROOT="${PROJECT_ROOT}/rollout_ttt_eval/newton_r4_ii_full_denoise_${SLURM_JOB_ID}"
LOG_ROOT="${PROJECT_ROOT}/slurm_logs/r4-ii-plus4-${SLURM_JOB_ID}"
TMP_ROOT="/tmp/r4-ii-plus4-${SLURM_JOB_ID}"

[[ -s "${CHECKPOINT}" ]] || { echo "Missing II checkpoint: ${CHECKPOINT}" >&2; exit 1; }
[[ ! -e "${RESULT_ROOT}" ]] || { echo "Refusing to overwrite ${RESULT_ROOT}" >&2; exit 1; }
mkdir -p "${RESULT_ROOT}" "${LOG_ROOT}" "${TMP_ROOT}"
cd "${PROJECT_ROOT}"

export PYTHON_BIN=/home/ha865618/.conda/envs/jepa_wam/bin/python
export LIBERO_PATH=/home/ha865618/project/LIBERO-plus
export NUM_OPEN_LOOP_STEPS=8 SAVE_ROLLOUTS=False EVAL_SEED=7
export TTT_INFERENCE_MEMORY_UPDATE=all_denoise TTT_INFERENCE_UPDATES=4
export MAX_TASKS=0 MAX_EPISODE_STEPS=0 RESUME=0 DRY_RUN=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
unset FM_NUM_INFERENCE_TIMESTEPS EPISODE_FILTER_FILE

echo "checkpoint=${CHECKPOINT}"
echo "result_root=${RESULT_ROOT}"
echo "flow_denoise_steps=4 memory_updates=4 suites=libero_10,libero_goal,libero_spatial,libero_object"
echo "gpu=${CUDA_VISIBLE_DEVICES:-unset} cpu_allocation=${SLURM_CPUS_PER_TASK:-unset}"

pids=()
suites=(libero_10 libero_goal libero_spatial libero_object)
stop_children() {
    for pid in "${pids[@]}"; do kill "${pid}" 2>/dev/null || true; done
}
trap stop_children TERM INT

for suite in "${suites[@]}"; do
    (
        export RESULT_ROOT
        export LOCAL_LOG_DIR="${LOG_ROOT}/${suite}"
        export LIBERO_CONFIG_PATH="${TMP_ROOT}/${suite}/config"
        export MPLCONFIGDIR="${TMP_ROOT}/${suite}/matplotlib"
        mkdir -p "${LOCAL_LOG_DIR}" "${LIBERO_CONFIG_PATH}" "${MPLCONFIGDIR}"
        exec bash vla-scripts/libero_plus_ttt.sh "${CHECKPOINT}" "${suite}" all 1
    ) >"${LOG_ROOT}/${suite}.console.log" 2>&1 &
    pids+=("$!")
    echo "started ${suite} pid=$!"
done

failed=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "completed ${suites[$i]}"
    else
        echo "failed ${suites[$i]}; see ${LOG_ROOT}/${suites[$i]}.console.log" >&2
        failed=1
    fi
done
echo "result_root=${RESULT_ROOT}"
exit "${failed}"
