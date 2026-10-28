#!/usr/bin/env bash
# PSC Bridges-2 WE context-length ablation: 16 physical frames, stride 8, 2 observations.
# Four H100 GPUs; global batch 64 = 4 GPUs x per-device batch 4 x accumulation 4.
# Submit from the PSC repository root: sbatch train_ttt_r4_WE_psc_context16f_4gpu_gb64_2d.slurm
#SBATCH -J r4-we-c16-psc
#SBATCH -A cis260071p
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus=h100-80:4
#SBATCH --mem=512G
#SBATCH -t 48:00:00
#SBATCH --export=ALL
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Submit with sbatch, not bash." >&2; exit 2; }

PROJECT_ROOT="/ocean/projects/cis260071p/ha865618/project/LeVJEPA_WAM_TTT"
DATA_ROOT="/ocean/projects/cis260071p/ha865618/data/modified_libero_rlds"
ASSET_ROOT="${PROJECT_ROOT}/jepa_wam_assets"
RUN_ROOT="${PROJECT_ROOT}/checkpoints/4th_round"
RUN_ID="jepa-wam-ttt-r4-wrapper-sra16-fullkv-v5-explicit-current-vjepa-kv-ttt-only-vjepa21-frames16-stride8-updates2-seg2-gb64-pb4-steps30000-psc-4gpu-context-length-ablation"
RUN_DIR="${RUN_ROOT}/${RUN_ID}"
CHECKPOINT="${RUN_DIR}/checkpoints/latest-checkpoint.pt"

BASE_VLM_RUN="${ASSET_ROOT}/JEPA_WAM/checkpoints/pretrained_vlm/prism-qvv25-vjepa21-vitl-384px+0_5b+stage-finetune+x7"
INITIAL_CHECKPOINT="${ASSET_ROOT}/JEPA_WAM/checkpoints/libero/jepavla-qvv25-vjepa-224px+0_5b+mx-libero-90+n1+b32+x7--visual-cosine-projector-allviews--20260723_232305/checkpoints/step-040000-epoch-37-loss=0.0262.pt"
QVV_PATH="${ASSET_ROOT}/Qvv2.5-0.5B"
VJEPA_CHECKPOINT="${ASSET_ROOT}/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt"
JEPA_PYTHON="${JEPA_PYTHON:-/ocean/projects/cis260071p/ha865618/conda-envs/jepa-wam/bin/python}"
TORCHRUN_BIN="$(dirname "${JEPA_PYTHON}")/torchrun"

for path in \
    "${PROJECT_ROOT}/prismatic/training/train.py" \
    "${PROJECT_ROOT}/prismatic/conf/vla.py" \
    "${PROJECT_ROOT}/prismatic/util/checkpoint_compat.py" \
    "${DATA_ROOT}/libero_spatial_no_noops" \
    "${DATA_ROOT}/libero_object_no_noops" \
    "${DATA_ROOT}/libero_goal_no_noops" \
    "${DATA_ROOT}/libero_10_no_noops" \
    "${BASE_VLM_RUN}/config.json" \
    "${BASE_VLM_RUN}/checkpoints/latest-checkpoint.pt" \
    "${INITIAL_CHECKPOINT}" "${QVV_PATH}" "${VJEPA_CHECKPOINT}"; do
    [[ -e "${path}" ]] || { echo "Missing required code, data, or model: ${path}" >&2; exit 1; }
done
[[ -x "${JEPA_PYTHON}" && -x "${TORCHRUN_BIN}" ]] || { echo "Missing JEPA_PYTHON or torchrun." >&2; exit 1; }

mkdir -p "${RUN_ROOT}"
exec 9>"${RUN_ROOT}/.${RUN_ID}.psc-ablation.lock"
flock -n 9 || { echo "Another PSC job is already writing ${RUN_ID}." >&2; exit 1; }
if [[ -s "${CHECKPOINT}" ]]; then
    for path in "${RUN_DIR}/config.json" "${RUN_DIR}/validation-split.json" "${RUN_DIR}/dataset_statistics.json"; do
        [[ -s "${path}" ]] || { echo "Incomplete resume directory: ${path}" >&2; exit 1; }
    done
    "${JEPA_PYTHON}" - "${CHECKPOINT}" <<'PY'
import sys, zipfile
if not zipfile.is_zipfile(sys.argv[1]):
    raise SystemExit("The resume checkpoint is not a complete PyTorch archive.")
PY
    CHECKPOINT_ARGS=(--initial_checkpoint "${CHECKPOINT}" --resume True --run_id "${RUN_ID}")
    echo "Resuming ${RUN_ID} from ${CHECKPOINT}"
elif [[ -e "${RUN_DIR}" ]]; then
    echo "Run directory already exists without a valid latest checkpoint: ${RUN_DIR}; refusing to overwrite." >&2
    exit 1
else
    CHECKPOINT_ARGS=(--initial_checkpoint "${INITIAL_CHECKPOINT}" --run_id "${RUN_ID}")
    echo "Starting ${RUN_ID} from the public JEPA-WAM checkpoint"
fi

mkdir -p "${RUN_DIR}/tmp" "${RUN_DIR}/cache"
export TMPDIR="${RUN_DIR}/tmp" MPLCONFIGDIR="${RUN_DIR}/cache"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export SLURM_EXPORT_ENV=ALL
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2
export VLA_RLDS_FRAME_TRANSFORM_THREADS=2
export WANDB_API_KEY='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'
export WANDB_MODE=online
export WANDB_ENTITY=hang-yu-team
export WANDB_PROJECT=prewam-we-context-length-ablation
export WANDB_DIR="${RUN_DIR}"
cd "${PROJECT_ROOT}"
echo "job=${SLURM_JOB_ID} route=WE context_frames=16 stride=8 updates=2 tbptt_segment=2 world_size=4 global_batch=64 started=$(date)"

srun --nodes=1 --ntasks=1 --gpus=4 --kill-on-bad-exit=1 \
    "${JEPA_PYTHON}" -c 'import torch, tensorflow, wandb; assert torch.cuda.is_available() and torch.cuda.device_count() == 4, torch.cuda.device_count(); print("GPUs:", [torch.cuda.get_device_name(i) for i in range(4)])'

srun --nodes=1 --ntasks=1 --gpus=4 --kill-on-bad-exit=1 \
    "${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc-per-node=4 \
    --module prismatic.training.train \
    --vla.base_vlm "${BASE_VLM_RUN}" \
    "${CHECKPOINT_ARGS[@]}" \
    --llm_checkpoint_path "${QVV_PATH}" \
    --vla.vjepa_checkpoint_path "${VJEPA_CHECKPOINT}" \
    --data_root_dir "${DATA_ROOT}" \
    --run_root_dir "${RUN_ROOT}" \
    --run_id_note "r4-wrapper-explicit-current-vjepa-kv-ttt-only-frames16-stride8-updates2-seg2-gb64-pb4-psc-context-ablation" \
    --vla.expected_world_size 4 \
    --vla.global_batch_size 64 \
    --vla.per_device_batch_size 4 \
    --vla.max_steps 30000 \
    --vla.learning_rate 2e-5 \
    --vla.min_learning_rate 1e-6 \
    --vla.weight_decay 1e-5 \
    --vla.shuffle_buffer_size 2500 \
    --vla.data_mix libero_4_task_suites_no_noops \
    --vla.train_qvv_lora False \
    --vla.train_projector False \
    --vla.train_action_head False \
    --vla.train_ttt_only True \
    --vla.train_visual_token_cosine_head False \
    --vla.ttt_enabled True \
    --vla.ttt_memory_source jepa_current \
    --vla.ttt_architecture wrapper \
    --vla.ttt_token_scope state_register_action \
    --vla.ttt_action_kv_scope full_dit_tokens \
    --vla.ttt_wrapper_register_tokens True \
    --vla.ttt_context_length 2 \
    --vla.ttt_observation_stride 8 \
    --vla.ttt_carry_between_segments True \
    --vla.ttt_require_full_context True \
    --vla.ttt_num_register_tokens 16 \
    --vla.ttt_tbptt_step_size 2 \
    --vla.visual_token_pair_offset 0 \
    --vla.enable_gradient_checkpointing True \
    --save_interval 500 \
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
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_entity "${WANDB_ENTITY}"

echo "Completed at $(date)"
