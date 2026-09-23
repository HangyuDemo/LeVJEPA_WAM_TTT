#!/usr/bin/env bash
# Sourced by all four round-2 launchers. Batch size counts long sequences.
# Equal OBSERVATION_BUDGET means equal non-padding observation exposures
# across GB=1/4 (windows may overlap; these are not unique dataset frames).
RUN_ROOT="${REPO_ROOT}/checkpoints/2nd_round"
TTT_CONTEXT_LENGTH="${TTT_CONTEXT_LENGTH:-32}"
TTT_TBPTT_STEP_SIZE="${TTT_TBPTT_STEP_SIZE:-8}"
TTT_ARCHITECTURE="${TTT_ARCHITECTURE:-wrapper}"
TTT_TOKEN_SCOPE="${TTT_TOKEN_SCOPE:-state_register_action}"
case "${TTT_TOKEN_SCOPE}" in
    legacy|state_register_action) ;;
    *) echo "TTT_TOKEN_SCOPE must be legacy or state_register_action." >&2; exit 1 ;;
esac
TTT_ACTION_KV_SCOPE="${TTT_ACTION_KV_SCOPE:-full_dit_tokens}"
case "${TTT_ACTION_KV_SCOPE}" in
    query_tokens|full_dit_tokens) ;;
    *) echo "TTT_ACTION_KV_SCOPE must be query_tokens or full_dit_tokens." >&2; exit 1 ;;
esac
TTT_WRAPPER_REGISTER_TOKENS="${TTT_WRAPPER_REGISTER_TOKENS:-True}"
case "${TTT_WRAPPER_REGISTER_TOKENS}" in
    True|False) ;;
    *) echo "TTT_WRAPPER_REGISTER_TOKENS must be True or False." >&2; exit 1 ;;
esac
TTT_ARCHITECTURE_TAG="${TTT_ARCHITECTURE}"
if [[ "${TTT_TOKEN_SCOPE}" == "state_register_action" ]]; then
    TTT_ARCHITECTURE_TAG="${TTT_ARCHITECTURE}-sra16-fullkv-v4"
elif [[ "${TTT_ARCHITECTURE}" == "wrapper" && "${TTT_WRAPPER_REGISTER_TOKENS}" == "True" ]]; then
    TTT_ARCHITECTURE_TAG="wrapper-reg16-state-action-v2"
fi
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
OBSERVATION_BUDGET="${OBSERVATION_BUDGET:-320000}"
VALIDATION_PERCENT="${VALIDATION_PERCENT:-5}"
VALIDATION_SEQUENCES_PER_SUITE="${VALIDATION_SEQUENCES_PER_SUITE:-16}"
VALIDATION_OBSERVATION_INTERVAL="${VALIDATION_OBSERVATION_INTERVAL:-32000}"
VALIDATION_SEED="${VALIDATION_SEED:-7}"
PARAMETER_CHECK_INTERVAL="${PARAMETER_CHECK_INTERVAL:-100}"

for round2_key in TTT_CONTEXT_LENGTH TTT_TBPTT_STEP_SIZE GLOBAL_BATCH_SIZE PER_DEVICE_BATCH_SIZE OBSERVATION_BUDGET; do
    if [[ ! "${!round2_key}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${round2_key} must be a positive integer." >&2
        exit 1
    fi
done
if [[ ! "${VALIDATION_PERCENT}" =~ ^[0-9]+$ ]] || (( VALIDATION_PERCENT >= 50 )); then
    echo "VALIDATION_PERCENT must be an integer in [0, 50)." >&2
    exit 1
fi
for round2_key in VALIDATION_SEQUENCES_PER_SUITE VALIDATION_OBSERVATION_INTERVAL PARAMETER_CHECK_INTERVAL; do
    if [[ ! "${!round2_key}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${round2_key} must be a positive integer." >&2
        exit 1
    fi
done
if [[ ! "${VALIDATION_SEED}" =~ ^[0-9]+$ ]]; then
    echo "VALIDATION_SEED must be a non-negative integer." >&2
    exit 1
fi
if (( TTT_TBPTT_STEP_SIZE < 2 || TTT_CONTEXT_LENGTH % TTT_TBPTT_STEP_SIZE != 0 )); then
    echo "Context must be divisible by segment size, and segment size must be >=2." >&2
    exit 1
fi
if (( GLOBAL_BATCH_SIZE % PER_DEVICE_BATCH_SIZE != 0 )); then
    echo "Global batch must be divisible by per-device batch (one GPU)." >&2
    exit 1
fi
OBSERVATIONS_PER_UPDATE=$((GLOBAL_BATCH_SIZE * TTT_CONTEXT_LENGTH))
VALIDATION_INTERVAL=$(((VALIDATION_OBSERVATION_INTERVAL + OBSERVATIONS_PER_UPDATE - 1) / OBSERVATIONS_PER_UPDATE))
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
# Keep periodic checkpoints at a manageable cadence; callers can override with
# SAVE_INTERVAL when a different checkpoint frequency is needed.
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
RUN_ID="${RUN_ID:-jepa-wam-ttt-r2-${TTT_ARCHITECTURE_TAG}-${TTT_MEMORY_TAG}-vjepa21-context${TTT_CONTEXT_LENGTH}-seg${TTT_TBPTT_STEP_SIZE}-gb${GLOBAL_BATCH_SIZE}-pb${PER_DEVICE_BATCH_SIZE}-steps${MAX_STEPS}-val${VALIDATION_PERCENT}-n${VALIDATION_SEQUENCES_PER_SUITE}-s${VALIDATION_SEED}}"

echo "round=2 architecture=${TTT_ARCHITECTURE} memory=${TTT_MEMORY_TAG}"
echo "ttt_token_scope=${TTT_TOKEN_SCOPE} (state_register_action selects state + 16 registers + actions for TTT input/residual)"
echo "ttt_action_kv_scope=${TTT_ACTION_KV_SCOPE} (implicit route K/V uses the complete post-attention DiT sequence)"
echo "wrapper_register_tokens=${TTT_WRAPPER_REGISTER_TOKENS} (legacy scope only; unified scope always includes registers)"
echo "context=${TTT_CONTEXT_LENGTH} segment=${TTT_TBPTT_STEP_SIZE} carry_between_segments=True full_context=True"
echo "global_batch=${GLOBAL_BATCH_SIZE} per_device_batch=${PER_DEVICE_BATCH_SIZE} grad_accumulation=${GRAD_ACCUMULATION_STEPS}"
echo "valid_observation_budget=${OBSERVATION_BUDGET} observations_per_update=${OBSERVATIONS_PER_UPDATE} max_steps=${MAX_STEPS}"
echo "run_root=${RUN_ROOT} run_id=${RUN_ID}"
echo "validation_percent=${VALIDATION_PERCENT} sequences_per_suite=${VALIDATION_SEQUENCES_PER_SUITE} validation_interval=${VALIDATION_INTERVAL} validation_seed=${VALIDATION_SEED}"
