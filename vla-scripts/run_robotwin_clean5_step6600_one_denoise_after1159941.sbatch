#!/usr/bin/env bash
# Run the same Clean5 evaluation as job 1159941 with one FM denoising step.
# Start only after the existing four-step evaluation finishes successfully.
#SBATCH -J rw-clean5-6600-100ep-1denoise
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=96G
#SBATCH -t 1-00:00:00
#SBATCH --dependency=afterok:1159941
#SBATCH --chdir=/lustre/fs1/home/ha865618/project/openpi_jepawam
#SBATCH -o slurm_logs/%x-%j.out
#SBATCH -e slurm_logs/%x-%j.err

set -euo pipefail

SOURCE_SCRIPT="/lustre/fs1/home/ha865618/project/openpi_jepawam/scripts/run_robotwin_clean5_step6600_fourdenoise.sbatch"
TEMP_SCRIPT="${SLURM_TMPDIR:-/tmp}/run_robotwin_clean5_step6600_1denoise_${SLURM_JOB_ID}.sh"

[[ -f "${SOURCE_SCRIPT}" ]] || {
    echo "Missing source evaluation script: ${SOURCE_SCRIPT}" >&2
    exit 2
}

# Reuse the existing verified evaluation pipeline, changing only the inference
# step count and result identifier so this run cannot overwrite four-step data.
sed \
    -e 's/--num-steps 4/--num-steps 1/' \
    -e 's/num_inference_steps=4/num_inference_steps=1/' \
    -e 's/CHECKPOINT_ID="step6600_${SLURM_JOB_ID}"/CHECKPOINT_ID="step6600_1denoise_${SLURM_JOB_ID}"/' \
    "${SOURCE_SCRIPT}" > "${TEMP_SCRIPT}"

grep -q -- '--num-steps 1' "${TEMP_SCRIPT}" || {
    echo "Failed to set the policy server to one denoising step" >&2
    exit 2
}
grep -q 'CHECKPOINT_ID="step6600_1denoise_' "${TEMP_SCRIPT}" || {
    echo "Failed to create a distinct one-denoise result identifier" >&2
    exit 2
}

echo "Starting one-denoise Clean5 evaluation after successful completion of job 1159941"
exec bash "${TEMP_SCRIPT}"
