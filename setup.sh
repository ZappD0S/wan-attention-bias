#!/bin/bash
set -e

# install everything EXCEPT cuda-dependent packages
uv sync --no-group cuda

# detect venv location
VENV=$(uv run sh -c 'echo $VIRTUAL_ENV')

# find nvcc and derive CUDA root
NVCC_PATH=$(find "$VENV" -name "nvcc" 2>/dev/null | head -1)
if [ -z "$NVCC_PATH" ]; then
  echo "ERROR: nvcc not found in $VENV — is cuda-toolkit in your dependencies?"
  exit 1
fi

CUDA_ROOT="$(cd "$(dirname "$NVCC_PATH")/.." && pwd)"

# ensure libcudart.so symlink exists (linker needs unversioned name)
CUDART=$(find "$CUDA_ROOT/lib" -name "libcudart.so.*" | head -1)
if [ -z "$CUDART" ]; then
  echo "ERROR: libcudart not found under $CUDA_ROOT/lib"
  exit 1
fi
ln -sf "$CUDART" "$CUDA_ROOT/lib/libcudart.so"

export CUDA_HOME="$CUDA_ROOT"
export CUDA_PATH="$CUDA_ROOT"
export PATH="$CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CUDA_ROOT/lib:${LIBRARY_PATH:-}"

# sanity check
echo "CUDA_HOME=$CUDA_HOME"
echo "$(nvcc --version | grep release)"

# install cuda-dependent packages
uv sync --group cuda
