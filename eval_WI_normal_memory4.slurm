#!/usr/bin/env bash
# WI (wrapper implicit) LIBERO-Plus evaluation: four suites concurrently on one H100.
# Flow Matching runs four denoising steps; TTT memory updates on all four steps.
# No node feature constraint or node exclusion list is used.
#SBATCH -J wi-plus-m4
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --mem=256G
#SBATCH -t 2-00:00:00
#SBATCH -o /home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.out
#SBATCH -e /home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.err

set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Submit with sbatch, not bash." >&2; exit 2; }

PROJECT_ROOT="/home/ha865618/project/LeVJEPA_WAM_TTT"
RUN_ID="jepa-wam-ttt-r4-wrapper-sra16-fullkv-v5-implicit-full-dit-kv-ttt-only-vjepa21-frames128-stride8-updates16-seg8-gb64-pb4-steps30000-tacc-4gpu"
RUN_DIR="${PROJECT_ROOT}/checkpoints/4th_round/${RUN_ID}"
CHECKPOINT_KIND="${WI_EVAL_CHECKPOINT_KIND:-best}"
case "${CHECKPOINT_KIND}" in
    best) CHECKPOINT_NAME=best-validation-checkpoint.pt ;;
    latest) CHECKPOINT_NAME=latest-checkpoint.pt ;;
    *) echo "WI_EVAL_CHECKPOINT_KIND must be best or latest." >&2; exit 2 ;;
esac
CHECKPOINT="${RUN_DIR}/checkpoints/${CHECKPOINT_NAME}"
PYTHON_BIN="/home/ha865618/.conda/envs/jepa_wam/bin/python"
LIBERO_PATH="/home/ha865618/project/LIBERO-plus"
ASSET_ROOT="${PROJECT_ROOT}/jepa_wam_assets"
BASE_VLM_RUN="${ASSET_ROOT}/JEPA_WAM/checkpoints/pretrained_vlm/prism-qvv25-vjepa21-vitl-384px+0_5b+stage-finetune+x7"
QVV_PATH="${ASSET_ROOT}/Qvv2.5-0.5B"
VJEPA_CKPT="${ASSET_ROOT}/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt"

for required in \
    "${RUN_DIR}/config.json" "${RUN_DIR}/dataset_statistics.json" \
    "${BASE_VLM_RUN}/config.json" "${QVV_PATH}" "${VJEPA_CKPT}" \
    "${PROJECT_ROOT}/vla-scripts/libero_plus_ttt.sh" \
    "${PROJECT_ROOT}/vla-scripts/libero_plus.sh" \
    "${LIBERO_PATH}/libero/libero/benchmark/task_classification.json"; do
    [[ -e "${required}" ]] || { echo "Missing WI evaluation input: ${required}" >&2; exit 1; }
done
[[ -s "${CHECKPOINT}" && -x "${PYTHON_BIN}" ]] || {
    echo "Checkpoint is empty or Python environment is missing: ${CHECKPOINT}" >&2
    exit 1
}

"${PYTHON_BIN}" - "${RUN_DIR}/config.json" "${CHECKPOINT}" <<'PY'
import json
import sys
import zipfile

with open(sys.argv[1], encoding="utf-8") as handle:
    vla = json.load(handle)["vla"]
expected = {
    "ttt_enabled": True,
    "ttt_architecture": "wrapper",
    "ttt_memory_source": "action_tokens",
    "fm_num_inference_timesteps": 4,
    "ttt_observation_stride": 8,
}
mismatches = {key: (vla.get(key), value) for key, value in expected.items() if vla.get(key) != value}
if mismatches or not zipfile.is_zipfile(sys.argv[2]):
    raise SystemExit(f"Wrong WI model recipe or incomplete checkpoint: {mismatches}")
PY

export PYTHON_BIN LIBERO_PATH ASSET_ROOT BASE_VLM_RUN QVV_PATH VJEPA_CKPT
export PYTHONPATH="${PROJECT_ROOT}:${LIBERO_PATH}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export NUM_OPEN_LOOP_STEPS=8 TTT_INFERENCE_UPDATES=4 SAVE_ROLLOUTS=False
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

OUTPUT_ROOT="${PROJECT_ROOT}/rollout_ttt_eval/WI-${CHECKPOINT_KIND}-fm4-memory4"
LOG_ROOT="${PROJECT_ROOT}/slurm_logs/wi-plus-${CHECKPOINT_KIND}-memory4-${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
cd "${PROJECT_ROOT}"
echo "job=${SLURM_JOB_ID} checkpoint=${CHECKPOINT} flow_steps=4 ttt_updates=4 rollout_stride=8"
"${PYTHON_BIN}" -c 'import torch; n=torch.cuda.device_count(); assert n == 1, f"Expected one H100, got {n}"; print(torch.cuda.get_device_name(0), flush=True)'

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
        export LIBERO_CONFIG_PATH="${PROJECT_ROOT}/.libero_plus_config/wi-${CHECKPOINT_KIND}-memory4-${SLURM_JOB_ID}-${suite}"
        export MPLCONFIGDIR="${LOG_ROOT}/${suite}/matplotlib"
        export RESUME=1
        mkdir -p "${LOCAL_LOG_DIR}" "${MPLCONFIGDIR}"
        exec bash "${PROJECT_ROOT}/vla-scripts/libero_plus_ttt.sh" "${CHECKPOINT}" "${suite}" all 1
    ) >"${LOG_ROOT}/${suite}.out" 2>"${LOG_ROOT}/${suite}.err" &
    pids+=("$!")
    echo "Started ${suite} pid=$! (same H100, concurrent)"
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
