#!/usr/bin/env bash
#SBATCH -J ttt-r4-wrapper-implicit-ae-n1
#SBATCH -p normal
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --gres=gpu:nvidia_h100_pcie:2
#SBATCH --exclude=evc22,evc24,evc25,evc27,evc34,evc38,evc44
#SBATCH --mem=480G
#SBATCH -t 7-00:00:00
#SBATCH -o /home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.out
#SBATCH -e /home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.err

set -euo pipefail

R4_ARCH=wrapper
R4_MEMORY_SOURCE=action_tokens
R4_MEMORY_TAG=implicit-full-dit-kv
R4_RUN_NOTE=r4-wrapper-implicit-action-expert-frames128-stride8-updates16-seg8-gb16-normal-4node-8gpu
R4_NNODES=4
R4_GPUS_PER_NODE=2
R4_WORLD_SIZE=8
R4_GLOBAL_BATCH_SIZE=16
R4_SHUFFLE_BUFFER_SIZE=2500
R4_SAVE_INTERVAL=500
source /home/ha865618/project/LeVJEPA_WAM_TTT/train_ttt_r4_action_expert_common.sh
