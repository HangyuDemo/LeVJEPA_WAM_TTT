#!/usr/bin/env bash
# TACC Stampede3 h100: one full node, four independent single-GPU round-3 runs.
# Submit twice, with arguments 1 and 2 and an afterany dependency; h100 jobs
# cannot request four days in a single allocation (48-hour per-job limit).
#SBATCH -J jepa-wam-r3-4gpu
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 72
#SBATCH -t 48:00:00
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

STAGE="${1:-}"
if [[ "${STAGE}" != 1 && "${STAGE}" != 2 ]]; then
    echo "Usage: sbatch $0 {1|2}" >&2
    exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit this script with sbatch, not bash." >&2
    exit 2
fi

PROJECT_ROOT="/scratch/11441/demo23331/project/LeVJEPA_WAM_TTT"
ASSET_ROOT="${PROJECT_ROOT}/jepa_wam_assets"
DATA_ROOT="/scratch/11441/demo23331/data/modified_libero_rlds"
# LIBERO-Plus is for evaluation, not this training data loader.
LIBERO_PATH="/scratch/11441/demo23331/data/LIBERO-plus"
RUN_ROOT="${PROJECT_ROOT}/checkpoints/3rd_round"
LOG_ROOT="${PROJECT_ROOT}/slurm_logs"

BASE_VLM_RUN="${ASSET_ROOT}/JEPA_WAM/checkpoints/pretrained_vlm/prism-qvv25-vjepa21-vitl-384px+0_5b+stage-finetune+x7"
INITIAL_CHECKPOINT="${ASSET_ROOT}/JEPA_WAM/checkpoints/libero/jepavla-qvv25-vjepa-224px+0_5b+mx-libero-90+n1+b32+x7--visual-cosine-projector-allviews--20260723_232305/checkpoints/step-040000-epoch-37-loss=0.0262.pt"
QVV_PATH="${ASSET_ROOT}/Qvv2.5-0.5B"
VJEPA_CHECKPOINT="${ASSET_ROOT}/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt"

# Activate the TACC Python environment before sbatch, or set JEPA_PYTHON to
# its absolute executable.
JEPA_PYTHON="${JEPA_PYTHON:-$(command -v python || true)}"
TORCHRUN_BIN="$(dirname "${JEPA_PYTHON:-/nonexistent}")/torchrun"

for path in \
    "${PROJECT_ROOT}/prismatic/training/train.py" \
    "${PROJECT_ROOT}/prismatic/conf/vla.py" \
    "${DATA_ROOT}/libero_spatial_no_noops" \
    "${DATA_ROOT}/libero_object_no_noops" \
    "${DATA_ROOT}/libero_goal_no_noops" \
    "${DATA_ROOT}/libero_10_no_noops" \
    "${BASE_VLM_RUN}" "${INITIAL_CHECKPOINT}" "${QVV_PATH}" "${VJEPA_CHECKPOINT}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing required code, data, or asset: ${path}" >&2
        exit 1
    fi
done
if [[ ! -x "${JEPA_PYTHON}" || ! -x "${TORCHRUN_BIN}" ]]; then
    echo "JEPA_PYTHON/torchrun missing. Activate the TACC environment or set JEPA_PYTHON." >&2
    exit 1
fi
# Detect an old source tree before reserving all four GPUs for training.
if ! grep -q 'ttt_carry_between_segments' "${PROJECT_ROOT}/prismatic/conf/vla.py"; then
    echo "TACC project does not contain the round-2/3 TTT implementation." >&2
    exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export VLA_RLDS_FRAME_TRANSFORM_THREADS=2
export TF_CPP_MIN_LOG_LEVEL=2
export WANDB_API_KEY='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'
export WANDB_MODE=online
export WANDB_ENTITY=hang-yu-team
export WANDB_PROJECT=jepa-wam-ttt
"${JEPA_PYTHON}" -c 'import wandb' || {
    echo "The TACC Python environment cannot import wandb." >&2
    exit 1
}
export LIBERO_PATH
cd "${PROJECT_ROOT}"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

echo "job=${SLURM_JOB_ID} stage=${STAGE} host=$(hostname) started=$(date)"
echo "python=${JEPA_PYTHON} data=${DATA_ROOT} run_root=${RUN_ROOT}"
timeout --kill-after=10s 30s nvidia-smi -L
GPU_COUNT="$(timeout --kill-after=10s 30s nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [[ "${GPU_COUNT}" -ne 4 ]]; then
    echo "Expected four H100 GPUs on this node, found ${GPU_COUNT}. No training started." >&2
    exit 1
fi
timeout --kill-after=10s 60s "${JEPA_PYTHON}" -c 'import torch, tensorflow; print("torch", torch.__version__, "cuda_devices", torch.cuda.device_count()); assert torch.cuda.device_count() == 4'
for gpu in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="${gpu}" timeout --kill-after=10s 120s "${JEPA_PYTHON}" -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1; x=torch.ones(1,device="cuda"); assert x.item()==1; torch.cuda.synchronize()' || {
        echo "GPU ${gpu} failed CUDA compute preflight. No training started." >&2
        exit 1
    }
done

# Stage 2 must have all four checkpoints. This prevents a fresh restart when
# the first allocation ended before a checkpoint was written.
routes=(wrapper-explicit wrapper-implicit inline-explicit inline-implicit)
run_ids=()
for route in "${routes[@]}"; do
    architecture="${route%%-*}"
    memory_tag="${route#*-}"
    if [[ "${memory_tag}" == explicit ]]; then
        memory_tag=explicit-jepa-kv
    else
        memory_tag=implicit-full-dit-kv
    fi
    run_id="jepa-wam-ttt-r3-${architecture}-sra16-fullkv-v4-${memory_tag}-vjepa21-context64-seg8-gb4-pb1-steps40000-val5-n16-s7"
    run_ids+=("${run_id}")
    checkpoint="${RUN_ROOT}/${run_id}/checkpoints/latest-checkpoint.pt"
    done_marker="${RUN_ROOT}/${run_id}/tacc-r3-40000.done"
    if [[ "${STAGE}" == 2 && ! -f "${checkpoint}" && ! -f "${done_marker}" ]]; then
        echo "Stage 2 cannot resume ${route}: ${checkpoint} is missing." >&2
        exit 1
    fi
done

run_route() {
    local gpu="$1" route="$2" run_id="$3"
    local architecture="${route%%-*}" variant="${route#*-}"
    local source checkpoint done_marker log_file
    local -a checkpoint_args
    if [[ "${variant}" == explicit ]]; then source=jepa; else source=action_tokens; fi
    checkpoint="${RUN_ROOT}/${run_id}/checkpoints/latest-checkpoint.pt"
    done_marker="${RUN_ROOT}/${run_id}/tacc-r3-40000.done"
    log_file="${LOG_ROOT}/tacc-r3-${route}-${SLURM_JOB_ID}.log"
    if [[ -f "${done_marker}" ]]; then
        echo "${route}: already completed; skip."
        return 0
    fi
    if [[ -f "${checkpoint}" ]]; then
        checkpoint_args=(--initial_checkpoint "${checkpoint}" --resume True --run_id "${run_id}")
        echo "${route}: resume ${checkpoint} on GPU ${gpu}; log ${log_file}"
    else
        checkpoint_args=(--initial_checkpoint "${INITIAL_CHECKPOINT}" --run_id "${run_id}")
        echo "${route}: initialize from public JEPA-WAM checkpoint on GPU ${gpu}; log ${log_file}"
    fi
    mkdir -p "${RUN_ROOT}/${run_id}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${TORCHRUN_BIN}" \
        --standalone --nnodes=1 --nproc-per-node=1 --module prismatic.training.train \
        --vla.base_vlm "${BASE_VLM_RUN}" \
        "${checkpoint_args[@]}" \
        --llm_checkpoint_path "${QVV_PATH}" \
        --vla.vjepa_checkpoint_path "${VJEPA_CHECKPOINT}" \
        --data_root_dir "${DATA_ROOT}" \
        --run_root_dir "${RUN_ROOT}" \
        --run_id_note "r3-${architecture}-${variant}-context64-seg8" \
        --vla.expected_world_size 1 \
        --vla.global_batch_size 4 \
        --vla.per_device_batch_size 1 \
        --vla.max_steps 40000 \
        --vla.shuffle_buffer_size 5000 \
        --vla.data_mix libero_4_task_suites_no_noops \
        --vla.train_qvv_lora False \
        --vla.train_projector False \
        --vla.train_action_head False \
        --vla.train_ttt_only True \
        --vla.train_visual_token_cosine_head False \
        --vla.ttt_enabled True \
        --vla.ttt_memory_source "${source}" \
        --vla.ttt_architecture "${architecture}" \
        --vla.ttt_token_scope state_register_action \
        --vla.ttt_action_kv_scope full_dit_tokens \
        --vla.ttt_wrapper_register_tokens True \
        --vla.ttt_context_length 64 \
        --vla.ttt_carry_between_segments True \
        --vla.ttt_require_full_context True \
        --vla.ttt_num_register_tokens 16 \
        --vla.ttt_tbptt_step_size 8 \
        --vla.visual_token_pair_offset 0 \
        --vla.enable_gradient_checkpointing True \
        --save_interval 1000 \
        --cpu_memory_log_interval 10 \
        --debug_batch_shapes True \
        --debug_memory_stats False \
        --validation_percent 5 \
        --validation_sequences_per_suite 16 \
        --validation_interval 125 \
        --validation_seed 7 \
        --check_finite True \
        --parameter_check_interval 100 \
        --use_swanlab False \
        --use_wandb True \
        --wandb_project "${WANDB_PROJECT}" \
        --wandb_entity "${WANDB_ENTITY}" >"${log_file}" 2>&1
    touch "${done_marker}"
    echo "${route}: completed 40000 steps at $(date)"
}

pids=()
for idx in 0 1 2 3; do
    run_route "${idx}" "${routes[$idx]}" "${run_ids[$idx]}" &
    pids+=("$!")
done

cleanup() {
    for pid in "${pids[@]}"; do kill "${pid}" 2>/dev/null || true; done
}
trap cleanup TERM INT
failed=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then failed=1; fi
done
trap - TERM INT
if [[ "${failed}" -ne 0 ]]; then
    echo "At least one route failed; inspect ${LOG_ROOT}/tacc-r3-*-${SLURM_JOB_ID}.log" >&2
    exit 1
fi
echo "All routes finished at $(date)"
