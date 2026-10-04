#!/usr/bin/env bash
#SBATCH -J ttt-r4-wrapper-explicit-smoke-n4
#SBATCH -p normal
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --gres=gpu:nvidia_h100_pcie:2
#SBATCH --exclude=evc22,evc24,evc25,evc27,evc34,evc38,evc44
#SBATCH --mem=320G
# The first checkpoint is at step 100 and this route takes about one minute per
# step, so a 30-minute allocation can only prove launch/forward progress.  Give
# the smoke run enough time to reach and verify its first checkpoint.
#SBATCH -t 02:30:00
#SBATCH --time-min=02:00:00
#SBATCH -o /home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.out
#SBATCH -e /home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.err

set -euo pipefail
R4_ARCH=wrapper
R4_MEMORY_SOURCE=jepa
R4_MEMORY_TAG=explicit-jepa-kv
R4_RUN_NOTE=r4-wrapper-explicit-smoke-2h-normal-2node-4gpu-gb16
R4_RUN_ID_OVERRIDE="jepa-wam-ttt-r4-wrapper-explicit-smoke-2h-${SLURM_JOB_ID}"
R4_NNODES=2
R4_GPUS_PER_NODE=2
R4_WORLD_SIZE=4
R4_GLOBAL_BATCH_SIZE=16
R4_SHUFFLE_BUFFER_SIZE=2500
R4_SAVE_INTERVAL=100
source /home/ha865618/project/LeVJEPA_WAM_TTT/train_ttt_r4_action_expert_common.sh
