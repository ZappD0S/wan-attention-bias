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

# The name of your virtual environment directory
VENV_NAME=".venv-container"
# The mount point inside the container
CONTAINER_ROOT="/workspace"
# get the absolute path of the directory this script lives in
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${CONTAINER_IMAGE:-$PROJECT_ROOT/containers/cuda_ubuntu.sif}"

HOST_UV_CACHE="${UV_CACHE_DIR:-$PROJECT_ROOT/.uv_cache}"
HOST_HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
HOST_PYTHON_INSTALL="$PROJECT_ROOT/.uv_python"
HOST_TMP="${CONTAINER_TMPDIR:-$PROJECT_ROOT/.tmp}"

# ensure host directories exist
mkdir -p "$HOST_UV_CACHE" "$HOST_HF_HOME" "$HOST_PYTHON_INSTALL" "$HOST_TMP"

# export environment variables for the container
export ${ENV_PREFIX}_UV_CACHE_DIR="/.cache/uv"
export ${ENV_PREFIX}_HF_HOME="/.cache/huggingface"
export ${ENV_PREFIX}_UV_PYTHON_INSTALL_DIR="/.cache/uv_python"

export ${ENV_PREFIX}_HF_TOKEN="$HF_TOKEN"
export ${ENV_PREFIX}_UV_PROJECT_ENVIRONMENT="$CONTAINER_ROOT/$VENV_NAME"

# apptainer flags
OPTS=(
  --nv --cleanenv --contain
  --workdir "$HOST_TMP"
  --bind "$PROJECT_ROOT:$CONTAINER_ROOT"
  --bind "$HOST_UV_CACHE:/.cache/uv"
  --bind "$HOST_HF_HOME:/.cache/huggingface"
  --bind "$HOST_PYTHON_INSTALL:/.cache/uv_python"
  --pwd "$CONTAINER_ROOT"
)

if [ -d "$STORAGE_DIR" ]; then
  OPTS+=(--bind "$STORAGE_DIR:${STORAGE_BIND_PATH:-/storage}")
fi

# base uv command stored as an array
UV_CMD=(uv run)

# handle debug logic safely
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

# execute cleanly using bash arrays
if [ $# -eq 0 ]; then
  # interactive shell (only reached if DEBUG!=1 due to safeguard above)
  exec "$RUNTIME_CMD" exec "${OPTS[@]}" "$IMAGE" bash
else
  # run the requested script/command
  exec "$RUNTIME_CMD" exec "${OPTS[@]}" "$IMAGE" "${UV_CMD[@]}" "$@"
fi
