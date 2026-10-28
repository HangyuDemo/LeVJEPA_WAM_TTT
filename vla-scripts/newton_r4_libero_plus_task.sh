#!/usr/bin/env bash
# Run exactly one LIBERO-Plus mode/suite inside an allocated GPU tmux pane.
set -euo pipefail

mode="${1:?mode}"
suite="${2:?suite}"
gpu="${3:?gpu}"
job_id="${4:?job id}"
attempt="${5:?attempt}"
[[ "${SLURM_JOB_ID:-}" == "${job_id}" ]] || { echo "Not inside the requested Slurm allocation" >&2; exit 2; }
[[ "${gpu}" =~ ^[0-3]$ ]] || { echo "GPU must be 0..3" >&2; exit 2; }
case "${suite}" in libero_10|libero_goal|libero_spatial|libero_object) ;; *) exit 2 ;; esac
case "${mode}" in one|full) ;; *) exit 2 ;; esac
case "${attempt}" in first|serial) ;; *) exit 2 ;; esac

project_root="/home/ha865618/project/LeVJEPA_WAM_TTT"
run_id="jepa-wam-ttt-r4-wrapper-sra16-fullkv-v5-explicit-current-vjepa-kv-ttt-only-vjepa21-frames128-stride8-updates16-seg8-gb64-pb4-steps30000-highgpu-4gpu"
run_dir="${project_root}/checkpoints/4th_round/${run_id}"
checkpoint="${run_dir}/checkpoints/latest-checkpoint.pt"
tmp_root="/tmp/newton-r4-libero-plus-${job_id}"
status_file="${tmp_root}/status/${mode}-${suite}-${attempt}.exit"
result_root="${project_root}/rollout_ttt_eval/newton_r4_${mode}_denoise"
suite_dir="${result_root}/${suite}"
mkdir -p "${tmp_root}/status"
task_exit=1
trap 'printf "%s\n" "${task_exit}" > "${status_file}"' EXIT

if [[ "${attempt}" == first && ( -e "${suite_dir}" || -L "${suite_dir}" ) ]]; then
    echo "Refusing to touch pre-existing result directory: ${suite_dir}" >&2
    exit 73
fi

export PYTHON_BIN="/home/ha865618/.conda/envs/jepa_wam/bin/python"
"${PYTHON_BIN}" - "${run_dir}/config.json" "${run_id}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)
vla = config["vla"]
assert config.get("run_id") == sys.argv[2]
assert vla.get("fm_num_inference_timesteps") == 4
assert vla.get("ttt_enabled") is True
assert vla.get("ttt_architecture") == "wrapper"
assert vla.get("ttt_memory_source") == "jepa_current"
PY
export LIBERO_PATH="/home/ha865618/project/LIBERO-plus"
export CUDA_VISIBLE_DEVICES="${gpu}"
export MUJOCO_EGL_DEVICE_ID="${gpu}"
export NUM_OPEN_LOOP_STEPS=8 SAVE_ROLLOUTS=False EVAL_SEED=7
export MAX_TASKS=0 MAX_EPISODE_STEPS=0 RESUME=0 DRY_RUN=0
export RESULT_ROOT="${result_root}"
export LOCAL_LOG_DIR="${tmp_root}/logs/${mode}-${suite}-${attempt}"
export LIBERO_CONFIG_PATH="${tmp_root}/config/${mode}-${suite}-${attempt}"
export MPLCONFIGDIR="${tmp_root}/matplotlib/${mode}-${suite}-${attempt}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset EPISODE_FILTER_FILE
if [[ "${mode}" == one ]]; then
    export TTT_INFERENCE_MEMORY_UPDATE=first_denoise
    export FM_NUM_INFERENCE_TIMESTEPS=1
else
    export TTT_INFERENCE_MEMORY_UPDATE=all_denoise
    unset FM_NUM_INFERENCE_TIMESTEPS
fi

mkdir -p "${LOCAL_LOG_DIR}" "${LIBERO_CONFIG_PATH}" "${MPLCONFIGDIR}"
cd "${project_root}"
echo "mode=${mode} suite=${suite} gpu=${gpu} attempt=${attempt} checkpoint=${checkpoint}"
bash vla-scripts/libero_plus_ttt.sh "${checkpoint}" "${suite}" all 1
"${PYTHON_BIN}" - "${suite_dir}" "${suite}" "${mode}" <<'PY'
import json
import pathlib
import sys

directory, suite, mode = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
summaries = sorted(directory.glob("summary-*.json"))
assert summaries, f"No summary JSON in {directory}"
summary = json.loads(summaries[-1].read_text())
assert summary.get("status") == "completed", summaries[-1]
assert summary.get("task_suite") == suite and summary.get("categories") == "all"
assert summary.get("trials_per_task") == 1
assert summary.get("ttt_inference_updates") == (1 if mode == "one" else 4)
assert summary.get("total_episodes", 0) > 0
extras = [path for path in directory.iterdir() if not (path.is_file() and path.name.startswith("summary-") and path.suffix == ".json")]
assert not extras, f"Unexpected non-summary results: {extras}"
print(f"Verified completed summary: {summaries[-1]}")
PY
task_exit=0
