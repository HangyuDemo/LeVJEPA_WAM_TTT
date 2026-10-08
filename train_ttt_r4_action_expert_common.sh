#!/usr/bin/env bash

set -euo pipefail

: "${R4_ARCH:?R4_ARCH must be wrapper or inline}"
: "${R4_MEMORY_SOURCE:?R4_MEMORY_SOURCE must be jepa, jepa_current, or action_tokens}"
: "${R4_MEMORY_TAG:?R4_MEMORY_TAG is required}"
: "${R4_RUN_NOTE:?R4_RUN_NOTE is required}"
R4_WORLD_SIZE="${R4_WORLD_SIZE:-4}"
R4_GLOBAL_BATCH_SIZE="${R4_GLOBAL_BATCH_SIZE:-4}"
R4_PER_DEVICE_BATCH_SIZE="${R4_PER_DEVICE_BATCH_SIZE:-1}"
R4_SAVE_INTERVAL="${R4_SAVE_INTERVAL:-1000}"
R4_MAX_STEPS="${R4_MAX_STEPS:-40000}"
R4_NNODES="${R4_NNODES:-1}"
R4_GPUS_PER_NODE="${R4_GPUS_PER_NODE:-${R4_WORLD_SIZE}}"
R4_SHUFFLE_BUFFER_SIZE="${R4_SHUFFLE_BUFFER_SIZE:-5000}"
R4_CONTEXT_FRAMES="${R4_CONTEXT_FRAMES:-128}"
R4_OBSERVATION_STRIDE="${R4_OBSERVATION_STRIDE:-8}"
R4_CONTEXT_UPDATES=$((R4_CONTEXT_FRAMES / R4_OBSERVATION_STRIDE))
R4_EFFECTIVE_MEMORY_SOURCE="${R4_MEMORY_SOURCE}"
R4_EFFECTIVE_MEMORY_TAG="${R4_MEMORY_TAG}"
R4_EFFECTIVE_RUN_NOTE="${R4_RUN_NOTE}"
R4_LAUNCH_PARTITION="${SLURM_JOB_PARTITION:-normal}"

if [[ "${R4_LAUNCH_PARTITION}" == "normal" && "${R4_MEMORY_SOURCE}" == "jepa" ]]; then
    # Pending normal explicit jobs source this file at allocation time and
    # therefore adopt current-observation V-JEPA K/V without resubmission.
    R4_EFFECTIVE_MEMORY_SOURCE=jepa_current
    R4_EFFECTIVE_MEMORY_TAG=explicit-current-vjepa-kv
    R4_EFFECTIVE_RUN_NOTE="${R4_RUN_NOTE/explicit/current-explicit}"
elif [[ "${R4_LAUNCH_PARTITION}" == "highgpu" && "${R4_MEMORY_SOURCE}" == "jepa" ]]; then
    # Highgpu explicit jobs retain the paper-style predicted future JEPA K/V.
    R4_EFFECTIVE_MEMORY_SOURCE=jepa
    R4_EFFECTIVE_MEMORY_TAG=explicit-future-wam-kv
fi

[[ "${R4_NNODES}" =~ ^[1-9][0-9]*$ ]] || { echo "R4_NNODES must be a positive integer" >&2; exit 1; }
[[ "${R4_GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || { echo "R4_GPUS_PER_NODE must be a positive integer" >&2; exit 1; }
[[ "${R4_PER_DEVICE_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || { echo "R4_PER_DEVICE_BATCH_SIZE must be a positive integer" >&2; exit 1; }
[[ "${R4_GLOBAL_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || { echo "R4_GLOBAL_BATCH_SIZE must be a positive integer" >&2; exit 1; }
[[ "${R4_SHUFFLE_BUFFER_SIZE}" =~ ^[1-9][0-9]*$ ]] || { echo "R4_SHUFFLE_BUFFER_SIZE must be a positive integer" >&2; exit 1; }
[[ "${R4_MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || { echo "R4_MAX_STEPS must be a positive integer" >&2; exit 1; }
[[ "${R4_CONTEXT_FRAMES}" =~ ^[1-9][0-9]*$ ]] || { echo "R4_CONTEXT_FRAMES must be a positive integer" >&2; exit 1; }
[[ "${R4_OBSERVATION_STRIDE}" =~ ^[1-9][0-9]*$ ]] || { echo "R4_OBSERVATION_STRIDE must be a positive integer" >&2; exit 1; }
(( R4_CONTEXT_FRAMES % R4_OBSERVATION_STRIDE == 0 )) || {
    echo "R4_CONTEXT_FRAMES must be divisible by R4_OBSERVATION_STRIDE" >&2
    exit 1
}
(( R4_CONTEXT_UPDATES % 8 == 0 )) || {
    echo "The number of TTT updates must be divisible by the TBPTT segment length (8)" >&2
    exit 1
}
[[ $((R4_NNODES * R4_GPUS_PER_NODE)) -eq "${R4_WORLD_SIZE}" ]] || {
    echo "R4_WORLD_SIZE must equal R4_NNODES * R4_GPUS_PER_NODE" >&2
    exit 1
}
case "${R4_EFFECTIVE_MEMORY_SOURCE}" in
    jepa|jepa_current|action_tokens) ;;
    *) echo "Unsupported effective Round-4 memory source: ${R4_EFFECTIVE_MEMORY_SOURCE}" >&2; exit 1 ;;
esac

if [[ "${R4_RESOLVE_ONLY:-0}" == "1" ]]; then
    printf 'partition=%s source=%s tag=%s frames=%s stride=%s updates=%s\n' \
        "${R4_LAUNCH_PARTITION}" "${R4_EFFECTIVE_MEMORY_SOURCE}" "${R4_EFFECTIVE_MEMORY_TAG}" \
        "${R4_CONTEXT_FRAMES}" "${R4_OBSERVATION_STRIDE}" "${R4_CONTEXT_UPDATES}"
    exit 0
fi

PROJECT_ROOT="/home/ha865618/project/LeVJEPA_WAM_TTT"
DATA_ROOT="/home/ha865618/data/modified_libero_rlds"
RUN_ROOT="${PROJECT_ROOT}/checkpoints/4th_round"
ASSET_ROOT="${PROJECT_ROOT}/jepa_wam_assets"
PYTHON_BIN="/home/ha865618/.conda/envs/jepa_wam/bin/python"
TORCHRUN_BIN="/home/ha865618/.conda/envs/jepa_wam/bin/torchrun"
BASE_VLM_RUN="${ASSET_ROOT}/JEPA_WAM/checkpoints/pretrained_vlm/prism-qvv25-vjepa21-vitl-384px+0_5b+stage-finetune+x7"
INITIAL_CHECKPOINT="${ASSET_ROOT}/JEPA_WAM/checkpoints/libero/jepavla-qvv25-vjepa-224px+0_5b+mx-libero-90+n1+b32+x7--visual-cosine-projector-allviews--20260723_232305/checkpoints/step-040000-epoch-37-loss=0.0262.pt"
QVV_PATH="${ASSET_ROOT}/Qvv2.5-0.5B"
VJEPA_CHECKPOINT="${ASSET_ROOT}/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt"
RUN_ID="${R4_RUN_ID_OVERRIDE:-jepa-wam-ttt-r4-${R4_ARCH}-sra16-fullkv-v5-${R4_EFFECTIVE_MEMORY_TAG}-action-expert-vjepa21-frames${R4_CONTEXT_FRAMES}-stride${R4_OBSERVATION_STRIDE}-updates${R4_CONTEXT_UPDATES}-seg8-gb${R4_GLOBAL_BATCH_SIZE}-pb${R4_PER_DEVICE_BATCH_SIZE}-steps40000-val5-n16-s7}"
export WANDB_API_KEY='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'
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2
export VLA_RLDS_FRAME_TRANSFORM_THREADS=1
export WANDB_MODE="${WANDB_MODE:-online}"

mkdir -p "${RUN_ROOT}" "${PROJECT_ROOT}/slurm_logs"
cd "${PROJECT_ROOT}"

for required in "${PYTHON_BIN}" "${TORCHRUN_BIN}" "${DATA_ROOT}/libero_spatial_no_noops" "${DATA_ROOT}/libero_object_no_noops" "${DATA_ROOT}/libero_goal_no_noops" "${DATA_ROOT}/libero_10_no_noops" "${BASE_VLM_RUN}" "${INITIAL_CHECKPOINT}" "${QVV_PATH}" "${VJEPA_CHECKPOINT}"; do
    [[ -e "${required}" ]] || { echo "Missing required path: ${required}" >&2; exit 1; }
done

# Any implicit jobs that resolve to the same run hold one advisory lock for the
# lifetime of the winning batch script, preventing duplicate checkpoint writers.
if [[ "${R4_EFFECTIVE_MEMORY_SOURCE}" == "action_tokens" ]]; then
    mkdir -p "${RUN_ROOT}/${RUN_ID}"
    R4_WRITER_LOCK="${RUN_ROOT}/${RUN_ID}/.writer.lock"
    exec {R4_WRITER_LOCK_FD}>"${R4_WRITER_LOCK}"
    if ! flock -n "${R4_WRITER_LOCK_FD}"; then
        echo "Another Slurm job already owns ${RUN_ID}; exiting without creating a second checkpoint writer."
        exit 0
    fi
    printf 'job_id=%s host=%s acquired=%s\n' \
        "${SLURM_JOB_ID:-interactive}" "$(hostname)" "$(date --iso-8601=seconds)" >&"${R4_WRITER_LOCK_FD}"

    # Normal and highgpu implicit jobs deliberately race for the same run.
    # The lock winner cancels only the matching architecture in the other
    # partition, so no second allocation can train or write the same run.
    counterpart_name=""
    if [[ "${R4_LAUNCH_PARTITION}" == "normal" ]]; then
        counterpart_name="ttt-r4-${R4_ARCH}-implicit-ae-h8"
    elif [[ "${R4_LAUNCH_PARTITION}" == "highgpu" ]]; then
        counterpart_name="ttt-r4-${R4_ARCH}-implicit-ae-n1"
    fi
    if [[ -n "${counterpart_name}" && -n "${SLURM_JOB_ID:-}" ]]; then
        while IFS='|' read -r counterpart_id counterpart_state; do
            [[ -n "${counterpart_id}" && "${counterpart_id}" != "${SLURM_JOB_ID}" ]] || continue
            scancel "${counterpart_id}"
            echo "Canceled counterpart ${counterpart_id} (${counterpart_name}, ${counterpart_state}); this job owns ${RUN_ID}."
        done < <(
            squeue -h -u "${SLURM_JOB_USER:-ha865618}" --name="${counterpart_name}" \
                --states=PENDING,CONFIGURING,RUNNING,COMPLETING --format='%A|%T' 2>/dev/null || true
        )
    fi
fi

cuda_ready=0
for attempt in 1 2 3; do
    if [[ "${R4_NNODES}" -eq 1 ]]; then
        if CUDA_EXPECTED_LOCAL_SIZE="${R4_GPUS_PER_NODE}" timeout --kill-after=10s 120s "${PYTHON_BIN}" -c 'import os, torch; expected=int(os.environ["CUDA_EXPECTED_LOCAL_SIZE"]); assert torch.cuda.is_available() and torch.cuda.device_count() == expected; [(torch.cuda.set_device(i), torch.ones(1, device=f"cuda:{i}"), torch.cuda.synchronize(i), print(i, torch.cuda.get_device_name(i))) for i in range(expected)]'; then
            cuda_ready=1
            break
        fi
    else
        if srun --nodes="${R4_NNODES}" --ntasks="${R4_NNODES}" --ntasks-per-node=1 \
            --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" --gpus-per-task="${R4_GPUS_PER_NODE}" --gpu-bind=none --kill-on-bad-exit=1 \
            env CUDA_EXPECTED_LOCAL_SIZE="${R4_GPUS_PER_NODE}" timeout --kill-after=10s 120s "${PYTHON_BIN}" \
            -c 'import os, socket, torch; expected=int(os.environ["CUDA_EXPECTED_LOCAL_SIZE"]); assert torch.cuda.is_available() and torch.cuda.device_count() == expected; [(torch.cuda.set_device(i), torch.ones(1, device=f"cuda:{i}"), torch.cuda.synchronize(i), print(socket.gethostname(), i, torch.cuda.get_device_name(i))) for i in range(expected)]'; then
            cuda_ready=1
            break
        fi
    fi
    echo "WARNING: PyTorch CUDA preflight attempt ${attempt}/3 failed" >&2
    sleep 15
done
[[ "${cuda_ready}" -eq 1 ]] || { echo "CUDA remained unavailable after three attempts" >&2; exit 1; }

CHECKPOINT="${RUN_ROOT}/${RUN_ID}/checkpoints/latest-checkpoint.pt"
if [[ -n "${R4_HANDOFF_NORMAL_JOB_ID:-}" ]]; then
    [[ "${R4_WORLD_SIZE}" -eq 4 ]] || { echo "Handoff requires the four-GPU job" >&2; exit 1; }
    if [[ "${R4_EFFECTIVE_MEMORY_SOURCE}" == "jepa" || "${R4_EFFECTIVE_MEMORY_SOURCE}" == "jepa_current" ]]; then
        normal_variant="explicit"
    else
        normal_variant="implicit"
    fi
    normal_job_name="ttt-r4-${R4_ARCH}-${normal_variant}-ae-n1"
    normal_job_id="${R4_HANDOFF_NORMAL_JOB_ID}"
    normal_state="$(squeue -h -j "${normal_job_id}" -o '%T' 2>/dev/null || true)"

    # The normal job may have been resubmitted after this highgpu job entered
    # the queue. Resolve its replacement by exact job name when the recorded
    # job ID is no longer active, preferring a running job over a pending one.
    if [[ -z "${normal_state}" ]]; then
        normal_match="$(squeue -h -u "${SLURM_JOB_USER:-ha865618}" --name="${normal_job_name}" \
            --states=RUNNING,COMPLETING --format='%A|%T' 2>/dev/null \
            | awk -F'|' -v current="${SLURM_JOB_ID:-}" '$1 != current {print; exit}' || true)"
        if [[ -z "${normal_match}" ]]; then
            normal_match="$(squeue -h -u "${SLURM_JOB_USER:-ha865618}" --name="${normal_job_name}" \
                --states=CONFIGURING,PENDING --format='%A|%T' 2>/dev/null \
                | awk -F'|' -v current="${SLURM_JOB_ID:-}" '$1 != current {print; exit}' || true)"
        fi
        if [[ -n "${normal_match}" ]]; then
            normal_job_id="${normal_match%%|*}"
            normal_state="${normal_match#*|}"
            echo "Resolved replacement normal job ${normal_job_id} (${normal_job_name}, ${normal_state})"
        fi
    fi

    if [[ "${normal_state}" == RUNNING || "${normal_state}" == COMPLETING ]]; then
        if [[ -f "${CHECKPOINT}" ]]; then
            # An atomically published latest already exists; do not idle four GPUs.
            scancel "${normal_job_id}"
            while [[ -n "$(squeue -h -j "${normal_job_id}" -o '%T' 2>/dev/null || true)" ]]; do sleep 5; done
            echo "Stopped normal job ${normal_job_id}; resuming its latest checkpoint"
        else
            handoff_request="${RUN_ROOT}/${RUN_ID}/.handoff-request"
            mkdir -p "${RUN_ROOT}/${RUN_ID}"
            touch "${handoff_request}"
            echo "Requested an optimizer-step checkpoint from normal job ${normal_job_id}"
            handoff_deadline=$(( $(date +%s) + ${R4_HANDOFF_TIMEOUT_SECONDS:-7200} ))
            while [[ -n "$(squeue -h -j "${normal_job_id}" -o '%T' 2>/dev/null || true)" ]]; do
                if (( $(date +%s) >= handoff_deadline )); then
                    rm -f "${handoff_request}"
                    echo "Normal job did not finish its checkpoint handoff before the deadline" >&2
                    exit 1
                fi
                sleep 30
            done
            rm -f "${handoff_request}"
            [[ -f "${CHECKPOINT}" ]] || { echo "Normal job exited without a handoff checkpoint" >&2; exit 1; }
        fi
    elif [[ "${normal_state}" == PENDING || "${normal_state}" == CONFIGURING ]]; then
        # No training step has run yet. Avoid two writers to the same run ID.
        scancel "${normal_job_id}"
        while [[ -n "$(squeue -h -j "${normal_job_id}" -o '%T' 2>/dev/null || true)" ]]; do sleep 5; done
        echo "Canceled pending normal job ${normal_job_id} before starting highgpu"
    elif [[ -n "${normal_state}" ]]; then
        echo "Unexpected normal-job state ${normal_state}; refusing to start a second writer" >&2
        exit 1
    fi
fi
if [[ -f "${CHECKPOINT}" ]]; then
    CHECKPOINT_ARGS=(--initial_checkpoint "${CHECKPOINT}" --resume True --run_id "${RUN_ID}")
    echo "Resuming ${CHECKPOINT}"
else
    CHECKPOINT_ARGS=(--initial_checkpoint "${INITIAL_CHECKPOINT}" --run_id "${RUN_ID}")
    echo "Starting ${RUN_ID} from the public JEPA-WAM checkpoint"
fi

TRAIN_ARGS=(--module prismatic.training.train \
    --vla.base_vlm "${BASE_VLM_RUN}" \
    "${CHECKPOINT_ARGS[@]}" \
    --llm_checkpoint_path "${QVV_PATH}" \
    --vla.vjepa_checkpoint_path "${VJEPA_CHECKPOINT}" \
    --data_root_dir "${DATA_ROOT}" \
    --run_root_dir "${RUN_ROOT}" \
    --run_id_note "${R4_EFFECTIVE_RUN_NOTE}" \
    --vla.expected_world_size "${R4_WORLD_SIZE}" \
    --vla.global_batch_size "${R4_GLOBAL_BATCH_SIZE}" \
    --vla.per_device_batch_size "${R4_PER_DEVICE_BATCH_SIZE}" \
    --vla.max_steps "${R4_MAX_STEPS}" \
    --vla.shuffle_buffer_size "${R4_SHUFFLE_BUFFER_SIZE}" \
    --vla.data_mix libero_4_task_suites_no_noops \
    --vla.train_qvv_lora False \
    --vla.train_projector False \
    --vla.train_action_head True \
    --vla.train_ttt_only False \
    --vla.train_visual_token_cosine_head False \
    --vla.ttt_enabled True \
    --vla.ttt_memory_source "${R4_EFFECTIVE_MEMORY_SOURCE}" \
    --vla.ttt_architecture "${R4_ARCH}" \
    --vla.ttt_token_scope state_register_action \
    --vla.ttt_action_kv_scope full_dit_tokens \
    --vla.ttt_wrapper_register_tokens True \
    --vla.ttt_context_length "${R4_CONTEXT_UPDATES}" \
    --vla.ttt_observation_stride "${R4_OBSERVATION_STRIDE}" \
    --vla.ttt_carry_between_segments True \
    --vla.ttt_require_full_context True \
    --vla.ttt_num_register_tokens 16 \
    --vla.ttt_tbptt_step_size 8 \
    --vla.visual_token_pair_offset 0 \
    --vla.enable_gradient_checkpointing True \
    --save_interval "${R4_SAVE_INTERVAL}" \
    --cpu_memory_log_interval 10 \
    --debug_batch_shapes True \
    --debug_memory_stats False \
    --validation_percent 5 \
    --validation_sequences_per_suite 16 \
    --validation_interval 1000 \
    --validation_seed 7 \
    --check_finite True \
    --parameter_check_interval 100 \
    --use_swanlab False \
    --use_wandb True \
    --wandb_project "jepa-wam-ttt-r4-action-expert-frames128-stride8-gb${R4_GLOBAL_BATCH_SIZE}" \
    --wandb_entity hang-yu-team)

if [[ "${R4_NNODES}" -eq 1 ]]; then
    "${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc-per-node="${R4_GPUS_PER_NODE}" "${TRAIN_ARGS[@]}"
else
    R4_MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
    R4_MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
    export R4_MASTER_ADDR R4_MASTER_PORT R4_NNODES R4_GPUS_PER_NODE TORCHRUN_BIN
    echo "Launching ${R4_WORLD_SIZE} ranks on ${R4_NNODES} nodes (${R4_GPUS_PER_NODE} GPUs/node); rendezvous=${R4_MASTER_ADDR}:${R4_MASTER_PORT}"
    srun --nodes="${R4_NNODES}" --ntasks="${R4_NNODES}" --ntasks-per-node=1 \
        --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" --gpus-per-task="${R4_GPUS_PER_NODE}" --gpu-bind=none --kill-on-bad-exit=1 \
        bash -c 'exec "${TORCHRUN_BIN}" --nnodes="${R4_NNODES}" --nproc-per-node="${R4_GPUS_PER_NODE}" --node-rank="${SLURM_PROCID}" --master-addr="${R4_MASTER_ADDR}" --master-port="${R4_MASTER_PORT}" "$@"' \
        bash "${TRAIN_ARGS[@]}"
fi
