#!/usr/bin/env bash
# Eight LIBERO-Plus evaluations, two modes x four suites, on four allocated H100s.
# Every evaluation has its own tmux session. Console/config/evaluator logs stay in /tmp.
#SBATCH -J r4plus-eight
#SBATCH -p highgpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH --mem=800G
#SBATCH -t 3-00:00:00
#SBATCH -o /tmp/r4plus-%j.out
#SBATCH -e /tmp/r4plus-%j.err

set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Submit with sbatch" >&2; exit 2; }

project_root="/home/ha865618/project/LeVJEPA_WAM_TTT"
task_script="${project_root}/vla-scripts/newton_r4_libero_plus_task.sh"
tmp_root="/tmp/newton-r4-libero-plus-${SLURM_JOB_ID}"
socket="r4plus-${SLURM_JOB_ID}"
mkdir -p "${tmp_root}/logs" "${tmp_root}/status"
cd "${project_root}"

run_id="jepa-wam-ttt-r4-wrapper-sra16-fullkv-v5-explicit-current-vjepa-kv-ttt-only-vjepa21-frames128-stride8-updates16-seg8-gb64-pb4-steps30000-highgpu-4gpu"
run_dir="${project_root}/checkpoints/4th_round/${run_id}"
for required in "${task_script}" "${run_dir}/checkpoints/latest-checkpoint.pt" "${run_dir}/config.json" "${run_dir}/dataset_statistics.json"; do
    [[ -s "${required}" ]] || { echo "Missing required file: ${required}" >&2; exit 1; }
done

# Do not start if any of these exact, user-selected result directories already exist.
for mode in one full; do
    for suite in libero_10 libero_goal libero_spatial libero_object; do
        target="${project_root}/rollout_ttt_eval/newton_r4_${mode}_denoise/${suite}"
        [[ ! -e "${target}" && ! -L "${target}" ]] || { echo "Existing result directory; refusing overwrite: ${target}" >&2; exit 1; }
    done
done

gpu_count="$(nvidia-smi -L | wc -l)"
[[ "${gpu_count}" -eq 4 ]] || { echo "Expected four allocated H100s, nvidia-smi sees ${gpu_count}" >&2; exit 1; }
"/home/ha865618/.conda/envs/jepa_wam/bin/python" -c 'import torch; assert torch.cuda.device_count() == 4, torch.cuda.device_count()'

gpu_state() {
    local gpu="$1" line free temp
    line="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free,temperature.gpu --format=csv,noheader,nounits | head -n 1)" || return 1
    IFS=, read -r free temp <<< "${line}"
    free="${free//[[:space:]]/}"
    temp="${temp//[[:space:]]/}"
    [[ "${free}" =~ ^[0-9]+$ && "${temp}" =~ ^[0-9]+$ ]] || return 1
    printf '%s %s\n' "${free}" "${temp}"
}

wait_for_headroom() {
    local gpu="$1" required_mb="$2" waited=0 free temp
    while (( waited < 3600 )); do
        if read -r free temp < <(gpu_state "${gpu}") && [[ -n "${free:-}" && -n "${temp:-}" ]]; then
            if (( temp <= 80 && free >= required_mb )); then return 0; fi
        fi
        sleep 15
        (( waited += 15 ))
    done
    echo "GPU${gpu} never had ${required_mb}MB free and temperature <=80C" >&2
    return 1
}

session_name() { printf 'r4p%s_%s_%s_%s' "${SLURM_JOB_ID}" "$1" "$2" "$3"; }
status_path() { printf '%s/status/%s-%s-%s.exit' "${tmp_root}" "$1" "$2" "$3"; }
session_alive() { tmux -L "${socket}" has-session -t "$1" 2>/dev/null; }

start_task() {
    local mode="$1" suite="$2" gpu="$3" attempt="$4" name
    name="$(session_name "${mode}" "${suite}" "${attempt}")"
    echo "Starting ${mode}/${suite} on GPU${gpu}, attempt=${attempt}, tmux=${name}"
    tmux -L "${socket}" new-session -d -s "${name}" \
        "bash '${task_script}' '${mode}' '${suite}' '${gpu}' '${SLURM_JOB_ID}' '${attempt}' > '${tmp_root}/logs/${name}.console.log' 2>&1"
}

task_code() {
    local file
    file="$(status_path "$1" "$2" "$3")"
    [[ -s "${file}" ]] || return 1
    read -r REPLY < "${file}"
    [[ "${REPLY}" == 0 ]]
}

summary_started() {
    local mode="$1" suite="$2"
    compgen -G "${project_root}/rollout_ttt_eval/newton_r4_${mode}_denoise/${suite}/summary-*.json" >/dev/null
}

stop_second() {
    local name="$1"
    if session_alive "${name}"; then
        tmux -L "${socket}" send-keys -t "${name}" C-c || true
        sleep 3
        tmux -L "${socket}" kill-session -t "${name}" 2>/dev/null || true
    fi
}

worker() {
    local gpu="$1" mode="$2" first_suite="$3" second_suite="$4"
    local first_name second_name first_baseline free temp used need second_started=0 second_deferred=0
    local second_success=0 first_success=0 first_status second_status
    wait_for_headroom "${gpu}" 30720 || return 1
    read -r first_baseline temp < <(gpu_state "${gpu}")
    start_task "${mode}" "${first_suite}" "${gpu}" first
    first_name="$(session_name "${mode}" "${first_suite}" first)"
    second_name="$(session_name "${mode}" "${second_suite}" first)"

    # The first summary appears only after its model has loaded, giving a
    # useful steady-state memory measurement before starting the second copy.
    while session_alive "${first_name}" && ! summary_started "${mode}" "${first_suite}"; do sleep 10; done
    if session_alive "${first_name}"; then
        read -r free temp < <(gpu_state "${gpu}")
        used=$(( first_baseline > free ? first_baseline - free : 0 ))
        need=$(( used + 14336 ))
        (( need < 30720 )) && need=30720
        if (( temp <= 80 && free >= need )); then
            start_task "${mode}" "${second_suite}" "${gpu}" first
            second_started=1
        else
            echo "GPU${gpu}: second task deferred (free=${free}MB, temp=${temp}C, need=${need}MB)"
            second_deferred=1
        fi
    else
        second_deferred=1
    fi

    if (( second_started )); then
        while session_alive "${first_name}" || session_alive "${second_name}"; do
            if session_alive "${second_name}"; then
                read -r free temp < <(gpu_state "${gpu}")
                if (( free < 10240 )); then
                    echo "GPU${gpu} below 10GB free (${free}MB); stopping second task ${second_name}"
                    stop_second "${second_name}"
                    second_deferred=1
                fi
            fi
            sleep 2
        done
    else
        while session_alive "${first_name}"; do sleep 10; done
    fi

    first_status="$(status_path "${mode}" "${first_suite}" first)"
    if task_code "${mode}" "${first_suite}" first; then first_success=1; fi

    if (( second_started && ! second_deferred )); then
        second_status="$(status_path "${mode}" "${second_suite}" first)"
        if task_code "${mode}" "${second_suite}" first; then
            second_success=1
        elif rg -qi 'out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED' "${tmp_root}/logs/${second_name}.console.log"; then
            echo "GPU${gpu}: second task OOM; retrying once serially after first completes"
            second_deferred=1
        else
            echo "FAILED ${mode}/${second_suite}; see ${tmp_root}/logs/${second_name}.console.log" >&2
        fi
    fi

    if (( second_deferred )) && task_code "${mode}" "${second_suite}" first; then
        second_success=1
        second_deferred=0
    fi
    if (( second_deferred )); then
        # Keep partial summaries from this job untouched; a later serial run
        # gets a new timestamped summary rather than overwriting one.
        sleep 2
        wait_for_headroom "${gpu}" 30720 || return 1
        start_task "${mode}" "${second_suite}" "${gpu}" serial
        second_name="$(session_name "${mode}" "${second_suite}" serial)"
        while session_alive "${second_name}"; do sleep 10; done
        if task_code "${mode}" "${second_suite}" serial; then second_success=1; fi
    fi

    if (( ! first_success )); then echo "FAILED ${mode}/${first_suite}; status=${first_status}" >&2; fi
    if (( ! second_success )); then echo "FAILED ${mode}/${second_suite}" >&2; fi
    (( first_success && second_success ))
}

cleanup() { tmux -L "${socket}" kill-server 2>/dev/null || true; }
trap cleanup EXIT
tmux -L "${socket}" new-session -d -s "keeper-${SLURM_JOB_ID}" 'sleep 1000000'

worker 0 one libero_10 libero_goal & p0=$!
worker 1 one libero_spatial libero_object & p1=$!
worker 2 full libero_10 libero_goal & p2=$!
worker 3 full libero_spatial libero_object & p3=$!
failed=0
for pid in "${p0}" "${p1}" "${p2}" "${p3}"; do
    if ! wait "${pid}"; then failed=1; fi
done

"/home/ha865618/.conda/envs/jepa_wam/bin/python" - "${project_root}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]) / "rollout_ttt_eval"
print("mode suite successes total_episodes success_rate summary_json")
for mode in ("one", "full"):
    for suite in ("libero_10", "libero_goal", "libero_spatial", "libero_object"):
        result_dir = root / f"newton_r4_{mode}_denoise" / suite
        summaries = sorted(result_dir.glob("summary-*.json"))
        if not summaries:
            print(mode, suite, "MISSING", "MISSING", "MISSING", "MISSING")
            continue
        summary_path = summaries[-1]
        summary = json.loads(summary_path.read_text())
        if summary.get("status") != "completed":
            print(mode, suite, "INCOMPLETE", summary.get("total_episodes"), "INCOMPLETE", summary_path)
        else:
            print(mode, suite, summary.get("successes"), summary.get("total_episodes"),
                  summary.get("success_rate"), summary_path)
        extras = [p for p in result_dir.iterdir() if not (p.is_file() and p.name.startswith("summary-") and p.suffix == ".json")]
        if extras:
            print(f"Unexpected non-summary files in {result_dir}: {extras}", file=sys.stderr)
PY
exit "${failed}"
