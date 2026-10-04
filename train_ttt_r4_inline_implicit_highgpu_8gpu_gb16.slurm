#!/usr/bin/env bash
#SBATCH -J ttt-r4-inline-implicit-ae-h8
#SBATCH -p highgpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:8
#SBATCH --mem=768G
#SBATCH -t 4-00:00:00
#SBATCH -o /home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.out
#SBATCH -e /home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.err

set -euo pipefail
R4_ARCH=inline
R4_MEMORY_SOURCE=action_tokens
R4_MEMORY_TAG=implicit-full-dit-kv
R4_RUN_NOTE=r4-inline-implicit-action-expert-frames128-stride8-updates16-seg8-gb16-highgpu-8gpu
R4_NNODES=1
R4_GPUS_PER_NODE=8
R4_WORLD_SIZE=8
R4_GLOBAL_BATCH_SIZE=16
R4_SHUFFLE_BUFFER_SIZE=2500
R4_SAVE_INTERVAL=500
source /home/ha865618/project/LeVJEPA_WAM_TTT/train_ttt_r4_action_expert_common.sh
