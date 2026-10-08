#!/usr/bin/env bash
# Stampede3 TACC H100: Round-4 wrapper-explicit, four GPUs, 48 hours.
# Physical context is 128 frames (16 TTT updates at observation stride 8).
# Global batch is 4 GPUs * per-device batch 4 * accumulation 4 = 64.
#SBATCH -J jepa-wam-r4-w-exp-gb64
#SBATCH -p h100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 72
#SBATCH -t 48:00:00
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail
PROJECT_ROOT="/scratch/11441/demo23331/project/LeVJEPA_WAM_TTT"
R4_ARCH=wrapper R4_MEMORY_SOURCE=jepa_current \
    exec bash "${PROJECT_ROOT}/v4_WE.slurm"
