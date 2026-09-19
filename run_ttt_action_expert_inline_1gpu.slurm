#!/usr/bin/env bash
# Legacy inline TTT + explicit JEPA representation memory.
# Every DiT block contains its own TTT layer.
#
# This is an explicit inline entry point.  The common training body remains
# in run_ttt_action_expert_1gpu.slurm so the two JEPA-memory routes cannot
# silently drift apart.
#
#SBATCH --job-name=jepa-wam-r2-inline-jepa
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

REPO_ROOT="/home/ha865618/project/LeVJEPA_WAM_TTT"
export TTT_ARCHITECTURE="inline"

exec bash "${REPO_ROOT}/run_ttt_action_expert_1gpu.slurm"
