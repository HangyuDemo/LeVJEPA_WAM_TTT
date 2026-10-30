#!/usr/bin/env bash
# Load W&B credentials without storing them in Git-tracked launchers.

configure_wandb_auth() {
    local project_root="${1:?project root is required}"
    local key_file="${WANDB_API_KEY_FILE:-${project_root}/.wandb_api_key}"

    if [[ "${WANDB_MODE:-online}" != "online" ]]; then
        return 0
    fi
    if [[ -z "${WANDB_API_KEY:-}" && -f "${key_file}" ]]; then
        IFS= read -r WANDB_API_KEY < "${key_file}" || true
        export WANDB_API_KEY
    fi
    if [[ -z "${WANDB_API_KEY:-}" ]]; then
        echo "W&B online mode requires WANDB_API_KEY or a local key file (${key_file})." >&2
        return 1
    fi
}
