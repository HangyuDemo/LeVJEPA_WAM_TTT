#!/usr/bin/env bash
# Four LIBERO-Plus suites run concurrently on one allocated H100.
# Submit twice with WE_TTT_INFERENCE_UPDATES=1 and =4, respectively.
# The four Flow Matching Euler steps stay fixed; only the number of TTT writes changes.
#SBATCH -J we-plus
#SBATCH -p highgpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --mem=256G
#SBATCH -t 2-00:00:00
#SBATCH -o /home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.out
#SBATCH -e /home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.err

set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Submit with sbatch, not bash." >&2; exit 2; }

PROJECT_ROOT="/home/ha865618/project/LeVJEPA_WAM_TTT"
RUN_ID="jepa-wam-ttt-r4-wrapper-sra16-fullkv-v5-explicit-current-vjepa-kv-ttt-only-vjepa21-frames128-stride8-updates16-seg8-gb64-pb4-steps30000-highgpu-4gpu"
RUN_DIR="${PROJECT_ROOT}/checkpoints/4th_round/${RUN_ID}"
CHECKPOINT_KIND="${WE_EVAL_CHECKPOINT_KIND:-best}"
case "${CHECKPOINT_KIND}" in
    best) CHECKPOINT_NAME=best-validation-checkpoint.pt ;;
    latest) CHECKPOINT_NAME=latest-checkpoint.pt ;;
    *) echo "WE_EVAL_CHECKPOINT_KIND must be best or latest." >&2; exit 2 ;;
esac
CHECKPOINT="${RUN_DIR}/checkpoints/${CHECKPOINT_NAME}"
UPDATES="${WE_TTT_INFERENCE_UPDATES:?Submit with WE_TTT_INFERENCE_UPDATES=1 or 4}"
[[ "${UPDATES}" == 1 || "${UPDATES}" == 4 ]] || { echo "WE_TTT_INFERENCE_UPDATES must be 1 or 4." >&2; exit 2; }

PYTHON_BIN="/home/ha865618/.conda/envs/jepa_wam/bin/python"
LIBERO_PATH="/home/ha865618/project/LIBERO-plus"
for required in \
    "${CHECKPOINT}" \
    "${RUN_DIR}/config.json" \
    "${RUN_DIR}/dataset_statistics.json" \
    "${PROJECT_ROOT}/vla-scripts/libero_plus_ttt.sh" \
    "${LIBERO_PATH}/libero/libero/benchmark/task_classification.json"; do
    [[ -e "${required}" ]] || { echo "Missing WE evaluation input: ${required}" >&2; exit 1; }
done
[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python environment: ${PYTHON_BIN}" >&2; exit 1; }

export PYTHON_BIN LIBERO_PATH
export PYTHONPATH="${PROJECT_ROOT}:${LIBERO_PATH}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export NUM_OPEN_LOOP_STEPS=8
export TTT_INFERENCE_UPDATES="${UPDATES}"
export SAVE_ROLLOUTS=False
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false

OUTPUT_ROOT="${PROJECT_ROOT}/rollout_ttt_eval/WE-${CHECKPOINT_KIND}-fm4-memory${UPDATES}"
LOG_ROOT="${PROJECT_ROOT}/slurm_logs/we-plus-${CHECKPOINT_KIND}-memory${UPDATES}-${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
cd "${PROJECT_ROOT}"
echo "WE checkpoint=${CHECKPOINT} memory_updates_per_observation=${UPDATES} flow_matching_steps=4 rollout_stride=8"
echo "job=${SLURM_JOB_ID} visible_gpu=${CUDA_VISIBLE_DEVICES:-unset} started=$(date)"

"${PYTHON_BIN}" -c 'import torch; n=torch.cuda.device_count(); print("visible GPUs:", n); assert n == 1, f"Expected exactly one allocated H100, got {n}"'

pids=()
suites=(libero_spatial libero_object libero_goal libero_10)
stop_children() {
    for pid in "${pids[@]}"; do kill "${pid}" 2>/dev/null || true; done
}
trap stop_children INT TERM

for suite in "${suites[@]}"; do
    (
        export RESULT_ROOT="${OUTPUT_ROOT}/${suite}"
        export LOCAL_LOG_DIR="${LOG_ROOT}/${suite}"
        export LIBERO_CONFIG_PATH="${PROJECT_ROOT}/.libero_plus_config/we-${CHECKPOINT_KIND}-memory${UPDATES}-${SLURM_JOB_ID}-${suite}"
        export MPLCONFIGDIR="${LOG_ROOT}/${suite}/matplotlib"
        export RESUME=1
        mkdir -p "${LOCAL_LOG_DIR}" "${MPLCONFIGDIR}"
        exec bash "${PROJECT_ROOT}/vla-scripts/libero_plus_ttt.sh" "${CHECKPOINT}" "${suite}" all 1
    ) >"${LOG_ROOT}/${suite}.out" 2>"${LOG_ROOT}/${suite}.err" &
    pid=$!
    pids+=("${pid}")
    echo "Started ${suite} pid=${pid} (same allocated H100)"
done

failed=0
for index in "${!pids[@]}"; do
    if wait "${pids[index]}"; then
        echo "Completed ${suites[index]}"
    else
        echo "FAILED ${suites[index]}; see ${LOG_ROOT}/${suites[index]}.err" >&2
        failed=1
    fi
done
exit "${failed}"
