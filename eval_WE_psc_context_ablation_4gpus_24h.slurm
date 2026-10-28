#!/usr/bin/env bash
# PSC Bridges-2: four allocated H100s, one WE model per GPU.
# Each GPU runs all four LIBERO-Plus suites as independent concurrent processes.
# GPU 3 evaluates the 128-frame WE checkpoint using slow TTT parameters only.
# Submit from the PSC project root: sbatch eval_WE_psc_context_ablation_4gpus_24h.slurm
#SBATCH -J we-ctx-plus4
#SBATCH -A cis260071p
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus=h100-80:4
#SBATCH --mem=512G
#SBATCH -t 24:00:00
#SBATCH --no-requeue
#SBATCH --export=ALL
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Submit with sbatch." >&2; exit 2; }

PROJECT_ROOT="/ocean/projects/cis260071p/ha865618/project/LeVJEPA_WAM_TTT"
RUN_ROOT="${PROJECT_ROOT}/checkpoints/4th_round"
ASSET_ROOT="${PROJECT_ROOT}/jepa_wam_assets"
BASE_VLM_RUN="${ASSET_ROOT}/JEPA_WAM/checkpoints/pretrained_vlm/prism-qvv25-vjepa21-vitl-384px+0_5b+stage-finetune+x7"
QVV_PATH="${ASSET_ROOT}/Qvv2.5-0.5B"
VJEPA_CKPT="${ASSET_ROOT}/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt"
PYTHON_BIN="${JEPA_PYTHON:-/ocean/projects/cis260071p/ha865618/conda-envs/jepa-wam/bin/python}"
LIBERO_PATH="${LIBERO_PATH:-/ocean/projects/cis260071p/ha865618/project/LIBERO-plus}"
LAUNCHER="${PROJECT_ROOT}/vla-scripts/libero_plus_ttt.sh"

RESULT_BASE="${PROJECT_ROOT}/rollout_ttt_eval/psc_we_context_ablation_best_fm4_memory1_plus_128_slow_only/job-${SLURM_JOB_ID}"
LOG_ROOT="${PROJECT_ROOT}/slurm_logs/psc-we-ctx-plus4-${SLURM_JOB_ID}"
JOB_TMP_BASE="${SLURM_TMPDIR:-/tmp}/psc-we-ctx-plus4-${SLURM_JOB_ID}"

if [[ "${1:-}" != "--worker" ]]; then
    # Run all 16 children inside one Slurm step that owns all four GPUs.
    # The worker maps the four scheduler-assigned CUDA devices, not physical 0..3.
    exec srun --nodes=1 --ntasks=1 --gpus=4 --cpus-per-task=32 --kill-on-bad-exit=1 \
        bash "${BASH_SOURCE[0]}" --worker
fi

[[ -x "${PYTHON_BIN}" ]] || { echo "Missing PSC Python: ${PYTHON_BIN}" >&2; exit 1; }
for required in \
    "${LAUNCHER}" "${PROJECT_ROOT}/vla-scripts/libero_plus.sh" \
    "${BASE_VLM_RUN}/config.json" "${QVV_PATH}" "${VJEPA_CKPT}" \
    "${LIBERO_PATH}/libero/libero/benchmark/task_classification.json" \
    "${PROJECT_ROOT}/experiments/robot/openvla_utils.py" \
    "${PROJECT_ROOT}/prismatic/models/flow_gr00t_action_head.py"; do
    [[ -e "${required}" ]] || { echo "Missing evaluation input: ${required}" >&2; exit 1; }
done
for source in \
    "${PROJECT_ROOT}/experiments/robot/openvla_utils.py" \
    "${PROJECT_ROOT}/prismatic/models/flow_gr00t_action_head.py"; do
    grep -Fq 'ttt_inference_slow_only' "${source}" || {
        echo "PSC code lacks slow-only inference support: ${source}; copy the updated source before submitting." >&2
        exit 1
    }
done
command -v setsid >/dev/null || { echo "setsid is required." >&2; exit 1; }

# In GPU-shared, the assigned CUDA devices may not be physical IDs 0..3.
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "Slurm did not set CUDA_VISIBLE_DEVICES." >&2; exit 1; }
IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
[[ "${#gpu_ids[@]}" -eq 4 ]] || {
    echo "Expected four assigned CUDA devices, got: ${CUDA_VISIBLE_DEVICES}" >&2
    exit 1
}
# EGL normally uses the same device index; Slurm's physical list takes priority
# when it differs from the CUDA-visible list on a shared node.
egl_ids=("${gpu_ids[@]}")
if [[ -n "${SLURM_STEP_GPUS:-}" ]]; then
    IFS=',' read -r -a step_gpu_ids <<< "${SLURM_STEP_GPUS}"
    if [[ "${#step_gpu_ids[@]}" -eq 4 ]]; then egl_ids=("${step_gpu_ids[@]}"); fi
fi
"${PYTHON_BIN}" -c 'import torch; n=torch.cuda.device_count(); print("Allocated CUDA devices:", n); assert n == 4'

tags=(ctx16 ctx32 ctx64 ctx128_slow_only)
frames_list=(16 32 64 128)
suites=(libero_10 libero_goal libero_spatial libero_object)
checkpoints=()
result_names=()

# Validate all inputs before creating results or starting any suite.
[[ ! -e "${RESULT_BASE}" ]] || { echo "Refusing to overwrite ${RESULT_BASE}" >&2; exit 1; }
for index in "${!tags[@]}"; do
    frames="${frames_list[index]}"
    updates=$((frames / 8))
    segment=$((updates < 8 ? updates : 8))
    if [[ "${frames}" -eq 128 ]]; then
        run_id="jepa-wam-ttt-r4-wrapper-sra16-fullkv-v5-explicit-current-vjepa-kv-ttt-only-vjepa21-frames128-stride8-updates16-seg8-gb64-pb4-steps30000-highgpu-4gpu"
    else
        run_id="jepa-wam-ttt-r4-wrapper-sra16-fullkv-v5-explicit-current-vjepa-kv-ttt-only-vjepa21-frames${frames}-stride8-updates${updates}-seg${segment}-gb64-pb4-steps30000-psc-4gpu-context-length-ablation"
    fi
    run_dir="${RUN_ROOT}/${run_id}"
    checkpoint="${run_dir}/checkpoints/best-validation-checkpoint.pt"
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
mismatches = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
if mismatches:
    raise SystemExit(f"Checkpoint configuration mismatch in {sys.argv[1]}: {mismatches}")
PY
    checkpoints+=("${checkpoint}")
    if [[ "${frames}" -eq 128 ]]; then
        result_names+=("${run_id}--best-fm4-slow-parameters-only-no-fast-weights")
    else
        result_names+=("${run_id}--best-fm4-first-denoise-memory1")
    fi
done

unset FM_NUM_INFERENCE_TIMESTEPS EPISODE_FILTER_FILE LEVJEPA_CKPT
export PYTHON_BIN LIBERO_PATH BASE_VLM_RUN QVV_PATH VJEPA_CKPT ASSET_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${LIBERO_PATH}:${PYTHONPATH:-}"
export TTT_INFERENCE_MEMORY_UPDATE=first_denoise TTT_INFERENCE_UPDATES=1
export TTT_INFERENCE_FIXED_OBSERVATION_POSITION=0
export NUM_OPEN_LOOP_STEPS=8 SAVE_ROLLOUTS=False EVAL_SEED=7
export MAX_TASKS=0 MAX_EPISODE_STEPS=0 RESUME=0 DRY_RUN=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=1
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

mkdir -p "${RESULT_BASE}" "${LOG_ROOT}" "${JOB_TMP_BASE}"
cd "${PROJECT_ROOT}"
pids=()
labels=()
summary_dirs=()
stop_children() {
    trap - INT TERM
    for pid in "${pids[@]}"; do kill -- "-${pid}" 2>/dev/null || true; done
    exit 143
}
trap stop_children INT TERM

for index in "${!tags[@]}"; do
    tag="${tags[index]}"
    gpu="${gpu_ids[index]}"
    egl="${egl_ids[index]}"
    checkpoint="${checkpoints[index]}"
    slow_only=0
    [[ "${frames_list[index]}" -eq 128 ]] && slow_only=1
    result_root="${RESULT_BASE}/${result_names[index]}"
    echo "${tag}: CUDA device ${gpu}, best=${checkpoint}, slow_only=${slow_only}, result_root=${result_root}"
    for suite in "${suites[@]}"; do
        label="${tag}/${suite}"
        log_dir="${LOG_ROOT}/${tag}/${suite}"
        task_tmp="${JOB_TMP_BASE}/${tag}/${suite}"
        mkdir -p "${log_dir}" "${task_tmp}/config" "${task_tmp}/matplotlib"
        CUDA_VISIBLE_DEVICES="${gpu}" \
        MUJOCO_EGL_DEVICE_ID="${egl}" \
        TTT_INFERENCE_SLOW_ONLY="${slow_only}" \
        RESULT_ROOT="${result_root}" \
        LOCAL_LOG_DIR="${log_dir}" \
        LIBERO_CONFIG_PATH="${task_tmp}/config" \
        MPLCONFIGDIR="${task_tmp}/matplotlib" \
        TMPDIR="${task_tmp}" \
            setsid bash "${LAUNCHER}" "${checkpoint}" "${suite}" all 1 \
            >"${log_dir}/console.log" 2>&1 &
        pids+=("$!")
        labels+=("${label}")
        summary_dirs+=("${result_root}/${suite}")
        echo "Started ${label}, pid=$!"
    done
done

failed=0
for index in "${!pids[@]}"; do
    if wait "${pids[index]}"; then
        summary="$(find "${summary_dirs[index]}" -maxdepth 1 -type f -name 'summary-*.json' -print -quit 2>/dev/null || true)"
        if [[ -n "${summary}" ]]; then
            echo "Completed ${labels[index]}: ${summary}"
        else
            echo "FAILED ${labels[index]}: no summary JSON in ${summary_dirs[index]}" >&2
            failed=1
        fi
    else
        echo "FAILED ${labels[index]}: ${LOG_ROOT}/${labels[index]}/console.log" >&2
        failed=1
    fi
done
exit "${failed}"
