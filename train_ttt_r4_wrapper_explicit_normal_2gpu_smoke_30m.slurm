#!/usr/bin/env bash
#SBATCH -J ttt-r4-wrapper-explicit-smoke-2g
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:nvidia_h100_pcie:2
#SBATCH --exclude=evc22,evc24,evc25,evc27,evc34,evc38,evc44
#SBATCH --mem=384G
#SBATCH -t 00:30:00
#SBATCH -o /home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.out
#SBATCH -e /home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.err

set -euo pipefail

# Match the formal wrapper-explicit recipe while keeping this smoke run small
# and isolated from the formal checkpoint directory.
R4_ARCH=wrapper
R4_MEMORY_SOURCE=jepa
R4_MEMORY_TAG=explicit-jepa-kv
R4_RUN_NOTE=r4-wrapper-explicit-current-vjepa-smoke-2gpu-30m
R4_RUN_ID_OVERRIDE="jepa-wam-ttt-r4-wrapper-explicit-current-vjepa-smoke-2gpu-30m-${SLURM_JOB_ID}"
R4_NNODES=1
R4_GPUS_PER_NODE=2
R4_WORLD_SIZE=2
R4_GLOBAL_BATCH_SIZE=16
R4_SHUFFLE_BUFFER_SIZE=2500
R4_MAX_STEPS=2
R4_SAVE_INTERVAL=1
source /home/ha865618/project/LeVJEPA_WAM_TTT/train_ttt_r4_action_expert_common.sh
