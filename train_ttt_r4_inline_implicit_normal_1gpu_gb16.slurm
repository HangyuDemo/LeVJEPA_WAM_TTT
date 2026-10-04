#!/usr/bin/env bash
#SBATCH -J ttt-r4-inline-implicit-ae-n1
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc24,evc25,evc27,evc34
#SBATCH --mem=384G
#SBATCH -t 7-00:00:00
#SBATCH -o /home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.out
#SBATCH -e /home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.err

set -euo pipefail
R4_ARCH=inline
R4_MEMORY_SOURCE=action_tokens
R4_MEMORY_TAG=implicit-full-dit-kv
R4_RUN_NOTE=r4-inline-implicit-action-expert-frames128-stride8-updates16-seg8-gb16-normal-1gpu
R4_WORLD_SIZE=1
R4_GLOBAL_BATCH_SIZE=16
R4_SAVE_INTERVAL=100
source /home/ha865618/project/LeVJEPA_WAM_TTT/train_ttt_r4_action_expert_common.sh
