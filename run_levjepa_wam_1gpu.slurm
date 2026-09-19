#!/usr/bin/env bash
#SBATCH --job-name=levjepa-wam-4gpu
#SBATCH --partition=highgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.out
#SBATCH --error=/home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.err

set -euo pipefail
cd /home/ha865618/project/LeVJEPA_WAM_TTT

# Match the original JEPA-WAM effective batch size: 8 GPUs x 32 = 256.
# One H100 uses micro-batch 16 and accumulates 16 micro-batches.
NPROC_PER_NODE=4 \
EXPECTED_WORLD_SIZE=4 \
PER_DEVICE_BATCH_SIZE=16 \
GLOBAL_BATCH_SIZE=256 \
MAX_STEPS=40000 \
SAVE_INTERVAL=250 \
RESUME=1 \
RUN_ID=levjepa-wam-4gpu \
TTT_ENABLED=0 \
bash vla-scripts/run_levjepa_wam.sh
