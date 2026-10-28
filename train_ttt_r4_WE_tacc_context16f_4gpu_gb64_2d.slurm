#!/usr/bin/env bash
# Stampede3 TACC H100: WE context-length ablation, 16 physical frames.
# Stride 8 => 2 TTT observations; four GPUs; global batch 64; 48 hours.
# Submit from the TACC repository root: sbatch train_ttt_r4_WE_tacc_context16f_4gpu_gb64_2d.slurm
#SBATCH -J r4-we-ctx16f-gb64
#SBATCH -p h100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 72
#SBATCH -t 48:00:00
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail
PROJECT_ROOT="/scratch/11441/demo23331/project/LeVJEPA_WAM_TTT"
export WANDB_API_KEY='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'
export WANDB_MODE=online
export WANDB_ENTITY=hang-yu-team
export WANDB_PROJECT=prewam-we-context-length-ablation
export R4_CONTEXT_FRAMES=16
export R4_ABLATION_TAG=context-length-ablation
export R4_ARCH=wrapper
export R4_MEMORY_SOURCE=jepa_current
exec bash "${PROJECT_ROOT}/v4_WE.slurm"
