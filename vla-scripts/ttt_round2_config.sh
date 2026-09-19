#!/usr/bin/env bash
# Sourced by all four round-2 launchers. Batch size counts long sequences.
# Equal OBSERVATION_BUDGET means equal non-padding observation exposures
# across GB=1/4 (windows may overlap; these are not unique dataset frames).
RUN_ROOT="${REPO_ROOT}/checkpoints/2nd_round"
TTT_CONTEXT_LENGTH="${TTT_CONTEXT_LENGTH:-32}"
TTT_TBPTT_STEP_SIZE="${TTT_TBPTT_STEP_SIZE:-8}"
TTT_ARCHITECTURE="${TTT_ARCHITECTURE:-wrapper}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
OBSERVATION_BUDGET="${OBSERVATION_BUDGET:-320000}"

for round2_key in TTT_CONTEXT_LENGTH TTT_TBPTT_STEP_SIZE GLOBAL_BATCH_SIZE PER_DEVICE_BATCH_SIZE OBSERVATION_BUDGET; do
    if [[ ! "${!round2_key}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${round2_key} must be a positive integer." >&2
        exit 1
    fi
done
if (( TTT_TBPTT_STEP_SIZE < 2 || TTT_CONTEXT_LENGTH % TTT_TBPTT_STEP_SIZE != 0 )); then
    echo "Context must be divisible by segment size, and segment size must be >=2." >&2
    exit 1
fi
if (( GLOBAL_BATCH_SIZE % PER_DEVICE_BATCH_SIZE != 0 )); then
    echo "Global batch must be divisible by per-device batch (one GPU)." >&2
    exit 1
fi
OBSERVATIONS_PER_UPDATE=$((GLOBAL_BATCH_SIZE * TTT_CONTEXT_LENGTH))
if [[ -n "${MAX_STEPS:-}" ]]; then
    if [[ ! "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "MAX_STEPS must be a positive integer." >&2
        exit 1
    fi
    OBSERVATION_BUDGET=$((MAX_STEPS * OBSERVATIONS_PER_UPDATE))
else
    if (( OBSERVATION_BUDGET % OBSERVATIONS_PER_UPDATE != 0 )); then
        echo "OBSERVATION_BUDGET must be divisible by global_batch * context_length." >&2
        exit 1
    fi
    MAX_STEPS=$((OBSERVATION_BUDGET / OBSERVATIONS_PER_UPDATE))
fi
GRAD_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / PER_DEVICE_BATCH_SIZE))
SAVE_INTERVAL="${SAVE_INTERVAL:-$(((8000 + OBSERVATIONS_PER_UPDATE - 1) / OBSERVATIONS_PER_UPDATE))}"
RUN_ID="${RUN_ID:-jepa-wam-ttt-r2-${TTT_ARCHITECTURE}-${TTT_MEMORY_TAG}-vjepa21-context${TTT_CONTEXT_LENGTH}-seg${TTT_TBPTT_STEP_SIZE}-gb${GLOBAL_BATCH_SIZE}-pb${PER_DEVICE_BATCH_SIZE}-steps${MAX_STEPS}}"

echo "round=2 architecture=${TTT_ARCHITECTURE} memory=${TTT_MEMORY_TAG}"
echo "context=${TTT_CONTEXT_LENGTH} segment=${TTT_TBPTT_STEP_SIZE} carry_between_segments=True full_context=True"
echo "global_batch=${GLOBAL_BATCH_SIZE} per_device_batch=${PER_DEVICE_BATCH_SIZE} grad_accumulation=${GRAD_ACCUMULATION_STEPS}"
echo "valid_observation_budget=${OBSERVATION_BUDGET} observations_per_update=${OBSERVATIONS_PER_UPDATE} max_steps=${MAX_STEPS}"
echo "run_root=${RUN_ROOT} run_id=${RUN_ID}"
