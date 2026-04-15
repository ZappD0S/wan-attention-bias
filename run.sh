#!/bin/bash

set -e

if [ -f .env ]; then
  # export $(grep -v '^#' .env | xargs) is a common shortcut,
  # but this while-read loop is more robust for values with spaces.
  while IFS='=' read -r key value || [ -n "$key" ]; do
    [[ "$key" =~ ^#.*$ ]] && continue
    [[ -z "$key" ]] && continue
    # Clean whitespace and quotes
    key=$(echo "$key" | xargs)
    value=$(echo "$value" | xargs)
    export "$key"="$value"
  done <.env
fi

# The name of your virtual environment directory
VENV_NAME=".venv-container"
# The mount point inside the container
CONTAINER_ROOT="/workspace"

# get the absolute path of the directory this script lives in
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE="$PROJECT_ROOT/containers/arch_ml.sif"

# ensure host directories exist (using absolute paths)
mkdir -p \
  "$PROJECT_ROOT/.uv_cache" \
  "$PROJECT_ROOT/.uv_python" \
  "$PROJECT_ROOT/.tmp" \
  "$PROJECT_ROOT/.cache/huggingface"

# export environment variables for the container
export APPTAINERENV_HF_HOME="${APPTAINERENV_HF_HOME:-/workspace/.cache/huggingface}"
export APPTAINERENV_UV_CACHE_DIR="$CONTAINER_ROOT/.uv_cache"
export APPTAINERENV_UV_PYTHON_INSTALL_DIR="$CONTAINER_ROOT/.uv_python"
export APPTAINERENV_TMPDIR="$CONTAINER_ROOT/.tmp"
export APPTAINERENV_UV_PROJECT_ENVIRONMENT="$CONTAINER_ROOT/.venv-container"

HOST_VENV_PATH="$PROJECT_ROOT/$VENV_NAME"

if [ -d "$HOST_VENV_PATH" ]; then
  # Find all 'lib' directories inside any 'nvidia' package (e.g. npp, cublas, cudnn)
  HOST_LIBS=$(find "$HOST_VENV_PATH" -type d -path "*/site-packages/nvidia/*/lib" 2>/dev/null | tr '\n' ':')

  if [ -n "$HOST_LIBS" ]; then
    # Translate host paths to container paths by swapping PROJECT_ROOT for CONTAINER_ROOT
    # Export for Apptainer (stripping the trailing colon)
    export APPTAINERENV_LD_LIBRARY_PATH="${HOST_LIBS//$PROJECT_ROOT/$CONTAINER_ROOT}"
    export APPTAINERENV_LD_LIBRARY_PATH="${APPTAINERENV_LD_LIBRARY_PATH%:}"
  fi
fi

# apptainer flags
OPTS=(
  --nv --cleanenv --contain --workdir "$PROJECT_ROOT/.tmp"
  --bind "$PROJECT_ROOT:/workspace"
  # --bind "$HOST_HF_PATH:/huggingface_cache"
  --pwd /workspace
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
  exec apptainer exec "${OPTS[@]}" "$IMAGE" bash
else
  # run the requested script/command
  exec apptainer exec "${OPTS[@]}" "$IMAGE" "${UV_CMD[@]}" "$@"
fi
