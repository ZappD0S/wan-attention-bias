#!/bin/bash
# run.sh

IMAGE="containers/arch_ml.sif"

WEIGHTS_PATH=$(realpath weights)
HOST_HF_PATH="${HF_HOME:-$HOME/.cache/huggingface}"

mkdir -p .uv_cache .tmp

export APPTAINERENV_HF_HOME="/huggingface_cache"
export APPTAINERENV_UV_CACHE_DIR="/workspace/.uv_cache"
export APPTAINERENV_TMPDIR="/workspace/.tmp"

apptainer exec \
  --nv \
  --cleanenv \
  --contain \
  --workdir .tmp \
  --bind "$PWD:/workspace" \
  --bind "$HOST_HF_PATH:/huggingface_cache" \
  --bind "$WEIGHTS_PATH:/workspace/weights" \
  --pwd /workspace \
  "$IMAGE" \
  uv run "$@"
