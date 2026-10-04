#!/usr/bin/env bash
#SBATCH -J ttt-r4-inline-implicit-ae
#SBATCH -p highgpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH --mem=768G
#SBATCH -t 5-00:00:00
#SBATCH -o /home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.out
#SBATCH -e /home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.err

R4_ARCH=inline
R4_MEMORY_SOURCE=action_tokens
R4_MEMORY_TAG=implicit-full-dit-kv
R4_RUN_NOTE=r4-inline-implicit-action-expert-frames128-stride8-updates16-seg8-gb4
source /home/ha865618/project/LeVJEPA_WAM_TTT/train_ttt_r4_action_expert_common.sh
