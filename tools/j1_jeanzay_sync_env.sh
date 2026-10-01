#!/bin/bash
# Build the locked J1 environment on a Jean Zay login node (GPU-free, sm_80).
# Usage: bash tools/j1_jeanzay_sync_env.sh
# Overrides: J1_REPO (default: this checkout), J1_TOOLS (default: $WORK/j1_tools).
set -eo pipefail

UV_VERSION=0.12.17
UV_TARBALL=uv-x86_64-unknown-linux-gnu.tar.gz
UV_TARBALL_SHA256=fa82fd8dde8e8eefdecada6aa0889666556cfceb690d06e0c3bca49eb3070a63

REPO=${J1_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
TOOLS=${J1_TOOLS:-$WORK/j1_tools}

# Pinned uv binary, verified against the release tarball digest.
if [[ "$("$TOOLS/bin/uv" --version 2>/dev/null)" != "uv $UV_VERSION"* ]]; then
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT
  curl -fsSL -o "$tmp/$UV_TARBALL" \
    "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/$UV_TARBALL"
  echo "$UV_TARBALL_SHA256  $tmp/$UV_TARBALL" | sha256sum --strict -c -
  tar -xzf "$tmp/$UV_TARBALL" -C "$tmp"
  mkdir -p "$TOOLS/bin"
  install -m 755 "$tmp/${UV_TARBALL%.tar.gz}/uv" "$tmp/${UV_TARBALL%.tar.gz}/uvx" "$TOOLS/bin/"
fi

# CUDA 12.8 toolkit for the SAM2 source build; GPUs hidden, A100 kernels only.
module load cuda/12.8.0
set -u
export CUDA_VISIBLE_DEVICES='' TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=4
# Interpreter from .python-version, uv-managed: the system Python lacks headers and SAM2 fails to build.
export UV_PYTHON_PREFERENCE=only-managed UV_PYTHON_DOWNLOADS=automatic
export UV_CACHE_DIR=$TOOLS/uv-cache UV_PYTHON_INSTALL_DIR=$TOOLS/python
export PATH=$TOOLS/bin:$PATH

cd "$REPO"
echo "start $(date -u +%FT%TZ) host $(hostname) commit $(git rev-parse HEAD)"
uv --version
nvcc --version | tail -1
sha256sum pyproject.toml uv.lock
uv sync --locked
# A second locked sync must be a no-op.
uv sync --locked

# GPU-free probe: versions and SAM2 cubin architectures; no model is loaded.
uv run --no-sync --locked python - "$(cat .python-version)" <<'EOF'
import importlib.metadata
import platform
import sys

import flash_attn
import torch
from torch.nn.attention.flex_attention import flex_attention  # noqa: F401

assert platform.python_version() == sys.argv[1], platform.python_version()
print("python", platform.python_version(), "torch", torch.__version__, "cuda", torch.version.cuda,
      "flash_attn", flash_attn.__version__, "sam2", importlib.metadata.version("sam2"))
EOF
sam2_so=$(uv run --no-sync --locked python -c 'import sam2, pathlib; print(next(pathlib.Path(sam2.__file__).parent.glob("_C*.so")))')
cuobjdump --list-elf "$sam2_so" | grep -o 'sm_[0-9]*' | sort | uniq -c
git status --short --ignore-submodules=none
echo "end $(date -u +%FT%TZ)"
