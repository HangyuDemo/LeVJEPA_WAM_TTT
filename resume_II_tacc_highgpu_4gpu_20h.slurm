#!/usr/bin/env bash
# Resume the transferred TACC inline-implicit run on four highgpu H100s.
# 4 CPUs for 20 hours = at most 4,800 requested CPU-minutes.
#SBATCH -J r4-ii-resume20h
#SBATCH -p highgpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH --mem=768G
#SBATCH -t 20:00:00
#SBATCH -o /home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.out
#SBATCH -e /home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.err

set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Submit with sbatch, not bash." >&2; exit 2; }

PROJECT_ROOT="/home/ha865618/project/LeVJEPA_WAM_TTT"
R4_RUN_ID_OVERRIDE="jepa-wam-ttt-r4-inline-sra16-fullkv-v5-implicit-full-dit-kv-ttt-only-vjepa21-frames128-stride8-updates16-seg8-gb64-pb4-steps30000-tacc-4gpu"
RUN_DIR="${PROJECT_ROOT}/checkpoints/4th_round/${R4_RUN_ID_OVERRIDE}"
CHECKPOINT="${RUN_DIR}/checkpoints/latest-checkpoint.pt"
[[ -s "${CHECKPOINT}" ]] || { echo "Missing or empty II resume checkpoint: ${CHECKPOINT}" >&2; exit 1; }
[[ -s "${RUN_DIR}/config.json" && -s "${RUN_DIR}/validation-split.json" ]] || {
    echo "Missing original II run configuration or validation split: ${RUN_DIR}" >&2
    exit 1
}

"/home/ha865618/.conda/envs/jepa_wam/bin/python" - "${RUN_DIR}/config.json" "${CHECKPOINT}" "${R4_RUN_ID_OVERRIDE}" <<'PY'
import json
import sys
import zipfile

with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)
vla = config["vla"]
expected = {
    "ttt_architecture": "inline",
    "ttt_memory_source": "action_tokens",
    "train_action_head": False,
    "train_ttt_only": True,
    "expected_world_size": 4,
    "global_batch_size": 64,
    "per_device_batch_size": 4,
    "max_steps": 30000,
    "learning_rate": 2e-5,
}
mismatches = {key: (vla.get(key), value) for key, value in expected.items() if vla.get(key) != value}
if config.get("run_id") != sys.argv[3] or mismatches or not zipfile.is_zipfile(sys.argv[2]):
    raise SystemExit(f"Wrong II run or incomplete checkpoint: {mismatches}")
PY

R4_ARCH=inline
R4_MEMORY_SOURCE=action_tokens
R4_MEMORY_TAG=implicit-full-dit-kv
R4_RUN_NOTE=r4-inline-implicit-tacc-checkpoint-continuation-frames128-stride8-updates16-seg8-gb64-pb4-highgpu-4gpu
R4_NNODES=1
R4_GPUS_PER_NODE=4
R4_WORLD_SIZE=4
R4_GLOBAL_BATCH_SIZE=64
R4_PER_DEVICE_BATCH_SIZE=4
R4_SHUFFLE_BUFFER_SIZE=2500
R4_SAVE_INTERVAL=500
R4_MAX_STEPS=30000
R4_CONTEXT_FRAMES=128
R4_OBSERVATION_STRIDE=8
R4_TRAIN_ACTION_HEAD=False
R4_TRAIN_TTT_ONLY=True
R4_LEARNING_RATE=2e-5
R4_MIN_LEARNING_RATE=1e-6
R4_WEIGHT_DECAY=1e-5
R4_TRAINING_TAG=ttt-only
R4_WANDB_PROJECT=jepa-wam-ttt-r4-ttt-only-frames128-stride8-gb64
R4_REQUIRE_RESUME=1
R4_DISABLE_COUNTERPART_CANCEL=1
R4_RESOLVE_ONLY=0
WANDB_MODE=online
unset R4_HANDOFF_NORMAL_JOB_ID

source "${PROJECT_ROOT}/train_ttt_r4_action_expert_common.sh"
