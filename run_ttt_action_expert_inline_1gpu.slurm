#!/usr/bin/env bash
# Inline TTT + explicit JEPA representation K/V memory.
# Every DiT block contains its own TTT layer.
#
# This is an explicit inline entry point.  The common training body remains
# in run_ttt_action_expert_1gpu.slurm so the two JEPA-memory routes cannot
# silently drift apart.
#
#SBATCH --job-name=jepa-wam-r2-inline-explicit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h100_pcie:1
# Avoid H100 nodes with confirmed CUDA initialization/FSDP failures.
#SBATCH --exclude=evc22,evc30,evc38,evc43,evc44,evc45
# Extreme low-CPU trial; the delegated main launcher sets the preprocessing cap.
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.out
#SBATCH --error=/home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.err

set -euo pipefail
export WANDB_API_KEY="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"
REPO_ROOT="/home/ha865618/project/LeVJEPA_WAM_TTT"
export TTT_ARCHITECTURE="inline"
# The shared launcher defaults to the unified full-K/V v4 recipe.

exec bash "${REPO_ROOT}/run_ttt_action_expert_1gpu.slurm"
