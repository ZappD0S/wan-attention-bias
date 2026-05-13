#!/bin/bash

set -e

[ -f .env ] && set -a && source .env && set +a

if command -v apptainer >/dev/null 2>&1; then
  RUNTIME_CMD="apptainer"
  ENV_PREFIX="APPTAINERENV"
elif command -v singularity >/dev/null 2>&1; then
  RUNTIME_CMD="singularity"
  ENV_PREFIX="SINGULARITYENV"
else
  echo "Error: Neither apptainer nor singularity is installed." >&2
  exit 1
fi

CONTAINER_ROOT="/workspace"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${CONTAINER_IMAGE:-$PROJECT_ROOT/containers/cuda_ubuntu.sif}"

HOST_HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
HOST_TMP="${CONTAINER_TMPDIR:-$PROJECT_ROOT/.tmp}"
mkdir -p "$HOST_HF_HOME" "$HOST_TMP"

# Export external variables for the container
export ${ENV_PREFIX}_HF_HOME="/.cache/huggingface"
export ${ENV_PREFIX}_HF_TOKEN="$HF_TOKEN"

# apptainer execution flags
OPTS=(
  --nv --cleanenv --contain
  --workdir "$HOST_TMP"
  --bind "$PROJECT_ROOT:$CONTAINER_ROOT"
  --bind "$HOST_HF_HOME:/.cache/huggingface"
  --pwd "$CONTAINER_ROOT"
)

if [ -d "$STORAGE_DIR" ]; then
  OPTS+=(--bind "$STORAGE_DIR:${STORAGE_BIND_PATH:-/storage}")
fi

# initialize command array
CMD=()

# handle debug logic safely
if [ "${DEBUG:-0}" == "1" ]; then
  # safeguard: debugpy requires a target Python script to attach to.
  if [ $# -eq 0 ]; then
    echo "Error: DEBUG=1 requires a Python script."
    exit 1
  fi

  # allow port assignment (defaults to 5678)
  PORT="${DEBUG_PORT:-5678}"
  echo ">>> Debugger enabled! Waiting for VS Code to attach on port $PORT..."

  # prefix the execution with debugpy
  CMD+=(python -m debugpy --listen "0.0.0.0:$PORT" --wait-for-client)
fi

# execute based on arguments provided
if [ $# -eq 0 ]; then
  # no arguments: Drop into an interactive bash shell
  exec "$RUNTIME_CMD" exec "${OPTS[@]}" "$IMAGE" bash
else
  # arguments provided: Run the script
  # because /opt/venv/bin is natively in the PATH, calling 'python script.py'
  # works automatically without needing 'uv run'.
  exec "$RUNTIME_CMD" exec "${OPTS[@]}" "$IMAGE" "${CMD[@]}" "$@"
fi
