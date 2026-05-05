#!/bin/bash

set -e

[ -f .env ] && set -a && source .env && set +a

# The name of your virtual environment directory
VENV_NAME=".venv-container"
# The mount point inside the container
CONTAINER_ROOT="/workspace"
# get the absolute path of the directory this script lives in
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="$PROJECT_ROOT/containers/arch_ml.sif"

HOST_UV_CACHE="${UV_CACHE_DIR:-$PROJECT_ROOT/.uv_cache}"
HOST_HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
HOST_PYTHON_INSTALL="$PROJECT_ROOT/.uv_python"
HOST_TMP="$PROJECT_ROOT/.tmp"

# ensure host directories exist
mkdir -p "$HOST_UV_CACHE" "$HOST_HF_HOME" "$HOST_PYTHON_INSTALL" "$HOST_TMP"

# export environment variables for the container
export APPTAINERENV_UV_CACHE_DIR="/.cache/uv"
export APPTAINERENV_HF_HOME="/.cache/huggingface"
export APPTAINERENV_UV_PYTHON_INSTALL_DIR="/.cache/uv_python"

export APPTAINERENV_HF_TOKEN="$HF_TOKEN"
export APPTAINERENV_UV_PROJECT_ENVIRONMENT="$CONTAINER_ROOT/$VENV_NAME"

HOST_VENV_PATH="$PROJECT_ROOT/$VENV_NAME"
if [ -d "$HOST_VENV_PATH" ]; then
  # Find all 'lib' directories inside any 'nvidia' package (e.g. npp, cublas, cudnn)
  HOST_LIBS=$(find "$HOST_VENV_PATH" -type d -path "*/site-packages/nvidia/*/lib" 2>/dev/null | tr '\n' ':')
  if [ -n "$HOST_LIBS" ]; then
    # Translate host paths to container paths by swapping PROJECT_ROOT for CONTAINER_ROOT
    # Export for Apptainer (stripping the trailing colon)
    CONTAINER_VENV_LIBS="${HOST_LIBS//$PROJECT_ROOT/$CONTAINER_ROOT}"
    export APPTAINERENV_LD_LIBRARY_PATH="${CONTAINER_VENV_LIBS}\$LD_LIBRARY_PATH"
  fi

  #  find all nvidia bin dirs (needed because nvcc calls ptxas, fatbinary, etc.)
  HOST_BINS=$(find "$HOST_VENV_PATH" -type d -path "*/site-packages/nvidia/*/bin" 2>/dev/null | tr '\n' ':')
  if [ -n "$HOST_BINS" ]; then
    # Translate host paths to container paths
    CONTAINER_VENV_BINS="${HOST_BINS//$PROJECT_ROOT/$CONTAINER_ROOT}"
    # Use PREPEND_PATH (stripping the trailing colon)
    export APPTAINERENV_PREPEND_PATH="${CONTAINER_VENV_BINS%:}"
  fi

  # set CUDA_HOME to the specific package containing nvcc
  HOST_NVCC=$(find "$HOST_VENV_PATH" -type f -name nvcc -path "*/site-packages/nvidia/*/bin/nvcc" 2>/dev/null | head -1)
  if [ -n "$HOST_NVCC" ]; then
    # Strip /bin/nvcc to get the CUDA_HOME root
    HOST_CUDA_HOME="$(dirname "$(dirname "$HOST_NVCC")")"
    CONTAINER_CUDA_HOME="${HOST_CUDA_HOME//$PROJECT_ROOT/$CONTAINER_ROOT}"
    export APPTAINERENV_CUDA_HOME="$CONTAINER_CUDA_HOME"
  fi
fi

# apptainer flags
OPTS=(
  --nv --cleanenv --contain --workdir "$HOST_TMP"
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
  exec apptainer exec "${OPTS[@]}" "$IMAGE" bash
else
  # run the requested script/command
  exec apptainer exec "${OPTS[@]}" "$IMAGE" "${UV_CMD[@]}" "$@"
fi
