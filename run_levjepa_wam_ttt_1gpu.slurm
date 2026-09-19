#!/usr/bin/env bash
#SBATCH --job-name=levjepa-wam-ttt
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.out
#SBATCH --error=/home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.err

set -euo pipefail
cd /home/ha865618/project/LeVJEPA_WAM_TTT

# Match the original JEPA-WAM effective batch size: 8 GPUs x 32 = 256.
# One H100 uses micro-batch 16 and accumulates 16 micro-batches.
NPROC_PER_NODE=1 \
EXPECTED_WORLD_SIZE=1 \
PER_DEVICE_BATCH_SIZE=16 \
GLOBAL_BATCH_SIZE=256 \
MAX_STEPS=40000 \
SAVE_INTERVAL=250 \
RESUME=1 \
RUN_ID=levjepa-wam-ttt-single-gpu \
TTT_ENABLED=1 \
bash vla-scripts/run_levjepa_wam.sh
