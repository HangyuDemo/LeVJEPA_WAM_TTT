#!/usr/bin/env bash
# Stampede3 h100: one full node, three WE context models on GPUs 0/1/2.
# Each model launches all four LIBERO-Plus suites concurrently on its GPU.
# Submit from the TACC repository root with:
#   sbatch eval_WE_tacc_context_ablation_3models_1node_24h.slurm
#SBATCH -J we-ctx-plus
#SBATCH -p h100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 72
#SBATCH -t 24:00:00
#SBATCH --no-requeue
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Submit with sbatch." >&2; exit 2; }

PROJECT_ROOT="/scratch/11441/demo23331/project/LeVJEPA_WAM_TTT"
ASSET_ROOT="${PROJECT_ROOT}/jepa_wam_assets"
BASE_VLM_RUN="${ASSET_ROOT}/JEPA_WAM/checkpoints/pretrained_vlm/prism-qvv25-vjepa21-vitl-384px+0_5b+stage-finetune+x7"
QVV_PATH="${ASSET_ROOT}/Qvv2.5-0.5B"
VJEPA_CKPT="${ASSET_ROOT}/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt"
PYTHON_BIN="${JEPA_PYTHON:-/home1/11441/demo23331/miniforge3/envs/jepa-wam/bin/python}"
LIBERO_PATH="${LIBERO_PATH:-${PROJECT_ROOT}/../LIBERO-plus}"
LAUNCHER="${PROJECT_ROOT}/vla-scripts/libero_plus_ttt.sh"

OUTPUT_BASE="${PROJECT_ROOT}/rollout_ttt_eval/tacc_r4_we_context_ablation_best_fm4_memory1"
LOG_ROOT="${PROJECT_ROOT}/slurm_logs/tacc-we-context-plus-${SLURM_JOB_ID}"
JOB_TMP_BASE="${TMPDIR:-/tmp}/tacc-we-context-plus-${SLURM_JOB_ID}"

# Avoid a user-exported setting silently changing this comparison.
unset FM_NUM_INFERENCE_TIMESTEPS EPISODE_FILTER_FILE LEVJEPA_CKPT
unset CUDA_VISIBLE_DEVICES MUJOCO_EGL_DEVICE_ID
export TTT_INFERENCE_MEMORY_UPDATE=first_denoise
export TTT_INFERENCE_UPDATES=1
export TTT_INFERENCE_FIXED_OBSERVATION_POSITION=0
export NUM_OPEN_LOOP_STEPS=8 SAVE_ROLLOUTS=False EVAL_SEED=7
export MAX_TASKS=0 MAX_EPISODE_STEPS=0 RESUME=0 DRY_RUN=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHON_BIN LIBERO_PATH BASE_VLM_RUN QVV_PATH VJEPA_CKPT ASSET_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${LIBERO_PATH}:${PYTHONPATH:-}"

for required in \
    "${LAUNCHER}" \
    "${PROJECT_ROOT}/vla-scripts/libero_plus.sh" \
    "${BASE_VLM_RUN}/config.json" \
    "${QVV_PATH}" \
    "${VJEPA_CKPT}" \
    "${LIBERO_PATH}/libero/libero/benchmark/task_classification.json"; do
    [[ -e "${required}" ]] || { echo "Missing evaluation input: ${required}" >&2; exit 1; }
done
[[ -x "${PYTHON_BIN}" ]] || { echo "Python executable missing: ${PYTHON_BIN}" >&2; exit 1; }
command -v setsid >/dev/null || { echo "setsid is required for independent process groups." >&2; exit 1; }

"${PYTHON_BIN}" -c 'import torch; n = torch.cuda.device_count(); print("Visible H100 GPUs:", n); assert n >= 3, "Need GPUs 0, 1, and 2"'

frames_list=(16 32 64)
gpu_list=(0 1 2)
suites=(libero_10 libero_goal libero_spatial libero_object)
checkpoints=()
output_roots=()

# Preflight all three models and output locations before starting any evaluation.
for index in "${!frames_list[@]}"; do
    frames="${frames_list[index]}"
    updates=$((frames / 8))
    segment=$((updates < 8 ? updates : 8))
    run_id="jepa-wam-ttt-r4-wrapper-sra16-fullkv-v5-explicit-current-vjepa-kv-ttt-only-vjepa21-frames${frames}-stride8-updates${updates}-seg${segment}-gb64-pb4-steps30000-tacc-4gpu-context-length-ablation"
    run_dir="${PROJECT_ROOT}/checkpoints/4th_round/${run_id}"
    checkpoint="${run_dir}/checkpoints/best-validation-checkpoint.pt"
    output_root="${OUTPUT_BASE}/${frames}f/job-${SLURM_JOB_ID}"
    for required in "${checkpoint}" "${run_dir}/config.json" "${run_dir}/dataset_statistics.json"; do
        [[ -s "${required}" ]] || { echo "Missing or empty model input: ${required}" >&2; exit 1; }
    done
    "${PYTHON_BIN}" - "${run_dir}/config.json" "${updates}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)["vla"]
expected = {
    "ttt_enabled": True,
    "ttt_architecture": "wrapper",
    "ttt_memory_source": "jepa_current",
    "ttt_context_length": int(sys.argv[2]),
    "ttt_observation_stride": 8,
    "fm_num_inference_timesteps": 4,
}
mismatches = {
    key: (config.get(key), value)
    for key, value in expected.items()
    if config.get(key) != value
}
if mismatches:
    raise SystemExit(f"Checkpoint configuration mismatch in {sys.argv[1]}: {mismatches}")
PY
    [[ ! -e "${output_root}" ]] || { echo "Refusing to overwrite result directory: ${output_root}" >&2; exit 1; }
    checkpoints+=("${checkpoint}")
    output_roots+=("${output_root}")
done

mkdir -p "${LOG_ROOT}" "${JOB_TMP_BASE}"
cd "${PROJECT_ROOT}"
for output_root in "${output_roots[@]}"; do mkdir -p "${output_root}"; done

pids=()
labels=()
summary_dirs=()
stop_children() {
    trap - INT TERM
    for pid in "${pids[@]}"; do
        kill -- "-${pid}" 2>/dev/null || true
    done
    exit 143
}
trap stop_children INT TERM

echo "Job ${SLURM_JOB_ID}: three best checkpoints, one H100 per model, four suites per GPU"
echo "Logs: ${LOG_ROOT}"

for index in "${!frames_list[@]}"; do
    frames="${frames_list[index]}"
    gpu="${gpu_list[index]}"
    checkpoint="${checkpoints[index]}"
    output_root="${output_roots[index]}"
    echo "Model ${frames}f -> physical GPU ${gpu}: ${checkpoint}"
    for suite in "${suites[@]}"; do
        label="${frames}f/${suite}"
        task_log_dir="${LOG_ROOT}/${frames}f/${suite}"
        task_tmp_dir="${JOB_TMP_BASE}/${frames}f/${suite}"
        mkdir -p "${task_log_dir}" "${task_tmp_dir}/matplotlib" "${task_tmp_dir}/config"
        CUDA_VISIBLE_DEVICES="${gpu}" \
        MUJOCO_EGL_DEVICE_ID="${gpu}" \
        RESULT_ROOT="${output_root}" \
        LOCAL_LOG_DIR="${task_log_dir}" \
        LIBERO_CONFIG_PATH="${task_tmp_dir}/config" \
        MPLCONFIGDIR="${task_tmp_dir}/matplotlib" \
        TMPDIR="${task_tmp_dir}" \
            setsid bash "${LAUNCHER}" "${checkpoint}" "${suite}" all 1 \
            >"${task_log_dir}/console.log" 2>&1 &
        pids+=("$!")
        labels+=("${label}")
        summary_dirs+=("${output_root}/${suite}")
        echo "Started ${label} on GPU ${gpu}, pid=$!"
    done
done

failed=0
for index in "${!pids[@]}"; do
    if wait "${pids[index]}"; then
        summary="$(find "${summary_dirs[index]}" -maxdepth 1 -type f -name 'summary-*.json' -print -quit 2>/dev/null || true)"
        if [[ -n "${summary}" ]]; then
            echo "Completed ${labels[index]}: ${summary}"
        else
            echo "FAILED ${labels[index]}: no summary JSON under ${summary_dirs[index]}" >&2
            failed=1
        fi
    else
        echo "FAILED ${labels[index]}: see ${LOG_ROOT}/${labels[index]}/console.log" >&2
        failed=1
    fi
done
exit "${failed}"
