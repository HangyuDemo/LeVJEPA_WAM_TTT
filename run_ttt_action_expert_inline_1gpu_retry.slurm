#!/usr/bin/env bash
# Retry the inline explicit-JEPA TTT training after a node-level CUDA failure.
# CPU-constrained: one allocated CPU core and one thread for host-side workers.
#SBATCH --job-name=jepa-wam-r2-inline-explicit-r
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h100_pcie:1
# evc27 produced "No devices were found"; keep the known bad nodes excluded.
#SBATCH --exclude=evc22,evc27,evc30,evc38,evc43,evc44,evc45
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.out
#SBATCH --error=/home/ha865618/project/LeVJEPA_WAM_TTT/slurm_logs/%x-%j.err

set -euo pipefail

REPO_ROOT="/home/ha865618/project/LeVJEPA_WAM_TTT"

export TTT_ARCHITECTURE="inline"
export VLA_RLDS_FRAME_TRANSFORM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export TOKENIZERS_PARALLELISM=false

cd "${REPO_ROOT}"
echo "===== RETRY JEPA-WAM TTT ACTION-EXPERT TRAINING ====="
echo "host=$(hostname)"
echo "date=$(date)"
echo "cpus=${SLURM_CPUS_PER_TASK:-unset}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L

PYTHON_BIN="/home/ha865618/.conda/envs/jepa_wam/bin/python"
"${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available(), "torch.cuda.is_available() is False"; assert torch.cuda.device_count() > 0, "torch.cuda.device_count() == 0"; print(f"torch_cuda_device={torch.cuda.get_device_name(0)}", flush=True)'

exec bash "${REPO_ROOT}/run_ttt_action_expert_1gpu.slurm"
