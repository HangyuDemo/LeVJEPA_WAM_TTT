#!/usr/bin/env bash
#SBATCH -J ttt-r4-inline-implicit-ae-n1
#SBATCH -p normal
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --gres=gpu:nvidia_h100_pcie:2
#SBATCH --exclude=evc22,evc24,evc25,evc27,evc34,evc38,evc44
#SBATCH --mem=480G
#SBATCH -t 7-00:00:00
#SBATCH -o /home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.out
#SBATCH -e /home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.err

set -euo pipefail

# Whichever four-GPU allocation starts first owns this run. If the matching
# highgpu job is already active, exit instead of creating a second writer.
highgpu_state="$(squeue -h -j 868950 -o '%T' 2>/dev/null || true)"
if [[ "${highgpu_state}" == RUNNING || "${highgpu_state}" == COMPLETING || "${highgpu_state}" == CONFIGURING ]]; then
    echo "Matching highgpu job 868950 is already ${highgpu_state}; this normal job will exit."
    exit 0
fi

R4_ARCH=inline
R4_MEMORY_SOURCE=action_tokens
R4_MEMORY_TAG=implicit-full-dit-kv
R4_RUN_NOTE=r4-inline-implicit-action-expert-frames128-stride8-updates16-seg8-gb16-normal-2node-4gpu
R4_NNODES=2
R4_GPUS_PER_NODE=2
R4_WORLD_SIZE=4
R4_GLOBAL_BATCH_SIZE=16
R4_SHUFFLE_BUFFER_SIZE=2500
R4_SAVE_INTERVAL=500
source /home/ha865618/project/LeVJEPA_WAM_TTT/train_ttt_r4_action_expert_common.sh
