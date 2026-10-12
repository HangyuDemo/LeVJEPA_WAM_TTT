#!/usr/bin/env bash
# One-day continuation of highgpu job 882699.
# This intentionally reuses the exact existing Round-4 wrapper-explicit run ID.
# It starts only after 882699 finishes, so both jobs never write the checkpoint
# directory at the same time.
#SBATCH -J ttt-r4-wrapper-exp-gb64-cont1d
#SBATCH -p highgpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH --mem=768G
#SBATCH -t 1-00:00:00
#SBATCH -o /home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.out
#SBATCH -e /home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.err

set -euo pipefail

R4_ARCH=wrapper
R4_MEMORY_SOURCE=jepa_current
R4_MEMORY_TAG=explicit-current-vjepa-kv
R4_RUN_NOTE=r4-wrapper-explicit-current-vjepa-kv-ttt-only-continuation-frames128-stride8-updates16-seg8-gb64-pb4-highgpu-4gpu
R4_RUN_ID_OVERRIDE=jepa-wam-ttt-r4-wrapper-sra16-fullkv-v5-explicit-current-vjepa-kv-ttt-only-vjepa21-frames128-stride8-updates16-seg8-gb64-pb4-steps30000-highgpu-4gpu
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

# This job is queued immediately, but must not read or write the same run as
# 882699 until the original writer has left the queue.
if [[ "${R4_RESOLVE_ONLY:-0}" != 1 ]]; then
    run_dir="/home/ha865618/project/LeVJEPA_WAM_TTT/checkpoints/4th_round/${R4_RUN_ID_OVERRIDE}"
    checkpoint="${run_dir}/checkpoints/latest-checkpoint.pt"
    mkdir -p "${run_dir}"
    exec {continuation_lock_fd}>"${run_dir}/.continuation-wrapper-explicit-882699.lock"
    if ! flock -n "${continuation_lock_fd}"; then
        echo "Another wrapper-explicit continuation owns ${run_dir}; stopping." >&2
        exit 1
    fi

    while true; do
        if ! original_state="$(squeue -h -j 882699 -o '%T')"; then
            echo "Could not query job 882699; retrying before checkpoint handoff." >&2
            sleep 60
            continue
        fi
        [[ -n "${original_state}" ]] || break
        echo "Waiting for original job 882699 (${original_state}) before resuming ${checkpoint}"
        sleep 60
    done

    if [[ ! -s "${checkpoint}" ]]; then
        echo "Original job 882699 ended without a usable resume checkpoint: ${checkpoint}" >&2
        exit 1
    fi
fi

source /home/ha865618/project/LeVJEPA_WAM_TTT/train_ttt_r4_action_expert_common.sh
