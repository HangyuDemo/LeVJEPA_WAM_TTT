#!/usr/bin/env bash
# Continue the PSC-transferred inline-explicit run on Newton normal H100 nodes.
# normal has two H100 GPUs per node, so four GPUs require two nodes.
# 4 CPUs/node * 2 nodes * 48 hours = 23,040 requested CPU-minutes.
#SBATCH -J r4-ie-resume2d
#SBATCH -p normal
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH --gres=gpu:nvidia_h100_pcie:2
#SBATCH --mem=400G
#SBATCH -t 2-00:00:00
#SBATCH -o /home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.out
#SBATCH -e /home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.err

set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Submit with sbatch, not bash." >&2; exit 2; }

PROJECT_ROOT="/home/ha865618/project/LeVJEPA_WAM_TTT"
R4_RUN_ID_OVERRIDE="jepa-wam-ttt-r4-inline-sra16-fullkv-v5-explicit-current-vjepa-kv-ttt-only-vjepa21-frames128-stride8-updates16-seg8-gb64-pb4-steps30000-highgpu-4gpu"
RUN_DIR="${PROJECT_ROOT}/checkpoints/4th_round/${R4_RUN_ID_OVERRIDE}"
CHECKPOINT="${RUN_DIR}/checkpoints/latest-checkpoint.pt"
[[ -s "${CHECKPOINT}" && -s "${RUN_DIR}/config.json" && -s "${RUN_DIR}/validation-split.json" ]] || {
    echo "Missing IE checkpoint, config.json, or validation-split.json in ${RUN_DIR}" >&2
    exit 1
}

exec {ie_lock_fd}>"${RUN_DIR}/.writer-normal-ie.lock"
flock -n "${ie_lock_fd}" || { echo "Another IE continuation owns ${RUN_DIR}" >&2; exit 1; }

"/home/ha865618/.conda/envs/jepa_wam/bin/python" - "${RUN_DIR}/config.json" "${CHECKPOINT}" "${R4_RUN_ID_OVERRIDE}" <<'PY'
import json
import pickletools
import sys
import zipfile

with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)
expected = {
    "ttt_architecture": "inline",
    "ttt_memory_source": "jepa_current",
    "train_action_head": False,
    "train_ttt_only": True,
    "expected_world_size": 4,
    "global_batch_size": 64,
    "per_device_batch_size": 4,
    "max_steps": 30000,
    "learning_rate": 2e-5,
}
mismatches = {key: (config["vla"].get(key), value) for key, value in expected.items() if config["vla"].get(key) != value}
if config.get("run_id") != sys.argv[3] or mismatches or not zipfile.is_zipfile(sys.argv[2]):
    raise SystemExit(f"Wrong IE run or invalid checkpoint: {mismatches}")
with zipfile.ZipFile(sys.argv[2]) as checkpoint:
    ops = list(pickletools.genops(checkpoint.read("archive/data.pkl")))
step = next(ops[index + 2][1] for index, (_, value, _) in enumerate(ops[:-2]) if value == "global_step")
print(f"IE resume checkpoint global_step={step}")
if not 19000 <= step < 30000:
    raise SystemExit("IE checkpoint step must be in [19000, 30000) to resume this job")
PY

R4_ARCH=inline
R4_MEMORY_SOURCE=jepa_current
R4_MEMORY_TAG=explicit-current-vjepa-kv
R4_RUN_NOTE=r4-inline-explicit-current-vjepa-kv-ttt-only-frames128-stride8-updates16-seg8-gb64-pb4-highgpu-4gpu
R4_NNODES=2
R4_GPUS_PER_NODE=2
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
WANDB_MODE=online
unset R4_HANDOFF_NORMAL_JOB_ID

source "${PROJECT_ROOT}/train_ttt_r4_action_expert_common.sh"
