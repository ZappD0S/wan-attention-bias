#!/bin/bash

set -e

# get the absolute path of the directory this script lives in
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE="$PROJECT_ROOT/containers/arch_ml.sif"
HOST_HF_PATH="${HF_HOME:-$HOME/.cache/huggingface}"

# ensure host directories exist (using absolute paths)
mkdir -p "$PROJECT_ROOT/.uv_cache" "$PROJECT_ROOT/.tmp"

# export environment variables for the container
export APPTAINERENV_HF_HOME="/huggingface_cache"
export APPTAINERENV_UV_CACHE_DIR="/workspace/.uv_cache"
export APPTAINERENV_TMPDIR="/workspace/.tmp"
export APPTAINERENV_UV_PROJECT_ENVIRONMENT="/workspace/.venv-container"

# base uv command stored as an array
UV_CMD=(uv run)

# handle Debug Logic Safely
if [ "${DEBUG:-0}" == "1" ]; then
  # Safeguard: debugpy requires a python script to attach to. It cannot debug "bash".
  if [ $# -eq 0 ]; then
    echo "Error: DEBUG=1 requires a Python script."
    exit 1
  fi

  # allow port assignment (defaults to 5678)
  PORT="${DEBUG_PORT:-5678}"
  echo ">>> Debugger enabled! Waiting for VS Code to attach on port $PORT..."

  # uses `uv run --with debugpy` to ensure it's always available
  UV_CMD+=(--with debugpy python -m debugpy --listen "0.0.0.0:$PORT" --wait-for-client)
fi

# apptainer flags
OPTS=(
  --nv --cleanenv --contain --workdir "$PROJECT_ROOT/.tmp"
  --bind "$PROJECT_ROOT:/workspace"
  --bind "$HOST_HF_PATH:/huggingface_cache"
  --pwd /workspace
)

# execute cleanly using bash arrays
if [ $# -eq 0 ]; then
  # interactive shell (only reached if DEBUG!=1 due to safeguard above)
  exec apptainer exec "${OPTS[@]}" "$IMAGE" "${UV_CMD[@]}" bash
else
  # run the requested script/command
  exec apptainer exec "${OPTS[@]}" "$IMAGE" "${UV_CMD[@]}" "$@"
fi
