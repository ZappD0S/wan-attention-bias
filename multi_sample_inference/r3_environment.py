"""Observe the R3 runtime without loading checkpoint or model tensors."""

from __future__ import annotations

import importlib.metadata
import platform
import socket
import subprocess
from pathlib import Path

import torch

from .r3_contracts import sha256_file


def observe_attention_runtime():
    """Read installed attention helper state without caller-supplied claims."""
    from wan.modules import attention  # noqa: PLC0415

    try:
        from torch.nn.attention.flex_attention import flex_attention  # noqa: PLC0415
    except ImportError:
        flex_attention = None
    flash_attn = getattr(attention, "flash_attn", None)
    return {
        "flash_attention_2_available": attention.FLASH_ATTN_2_AVAILABLE,
        "flash_attention_3_available": attention.FLASH_ATTN_3_AVAILABLE,
        "flash_attention_version": getattr(flash_attn, "__version__", None),
        "flex_attention_available": callable(flex_attention),
        "flex_attention_version": torch.__version__ if callable(flex_attention) else None,
    }


def observe_runtime_environment(attention_runtime):
    """Observe the v4+ host/package/hardware binding without loading a model."""
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    rows = [line.split(", ", 3) for line in query.splitlines() if line.strip()]
    if not rows or any(len(row) != 4 for row in rows):
        raise RuntimeError("R3 could not observe the declared GPU inventory")
    names = {row[1] for row in rows}
    drivers = {row[2] for row in rows}
    capabilities = {row[3] for row in rows}
    if len(names) != 1 or len(drivers) != 1 or len(capabilities) != 1:
        raise RuntimeError("R3 requires a homogeneous GPU inventory")
    repo = Path(__file__).resolve().parents[1]
    return {
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "flash_attention_version": attention_runtime["flash_attention_version"],
        "flex_attention_version": attention_runtime["flex_attention_version"],
        "sam2_version": importlib.metadata.version("sam2"),
        "driver_version": next(iter(drivers)),
        "gpu_model": next(iter(names)),
        "gpu_uuids": [row[0] for row in rows],
        "gpu_compute_capability": next(iter(capabilities)),
        "pyproject_sha256": sha256_file(repo / "pyproject.toml"),
        "uv_lock_sha256": sha256_file(repo / "uv.lock"),
    }
