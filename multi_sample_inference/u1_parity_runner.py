"""Frozen checkpoint-backed U1 one-layer parity execution."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import socket
import subprocess
from pathlib import Path

from .experiment_pipeline import checkpoint_identity, repository_identity
from .parity_contracts import (
    bind_verified_checkpoint_record,
    tensor_identity,
    write_immutable_record,
)
from .upstream_parity import run_one_layer_parity

SNAPSHOT_REVISION = "6b73f84e66371cdfe870c72acd6826e1d61cf279"
EXPECTED_INVENTORY_SHA256 = "e6b7adbd6f6e5dfcb7fa06e09d5d2edfcb179e80ff25f1c743567d1948701dd4"
EXPECTED_CHECKPOINT_CONTENT_SHA256 = (
    "80c954fbb46c39139a300a7dd41ed5a8b1c6b11c5f952c1bfb3c763d6e73ebb7"
)
EXPECTED_PROTOCOL_SHA256 = "8b83d1a801ffa8cbdac3abc5d74c47476d4ed49a9b0848f240562811b6e5dbbf"
EXPECTED_UV_LOCK_SHA256 = "d31fb115afdeb800de705fd080316598f1ba7e9de98c42b67cd0fdcea5d8a361"
PARENT_REVISION = "e073adaa742c4e3528a1e488e6413f12d02ae220"
WAN_REVISION = "9f52e9abceb49c5cbf4a7f435a84ef0621c0029f"
LAYER_INDEX = 0
ATOL = 1e-5
RTOL = 0.016
CASES = (("conditional", 2026091401), ("negative", 2026091402))
GRID = (21, 30, 52)
SEQUENCE_LENGTH = 32760
MODEL_DIM = 5120
CONTEXT_LENGTH = 769


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_equal(actual, expected, label):
    if actual != expected:
        raise RuntimeError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def _frozen_source_identity(repo):
    parent = repository_identity(repo)
    wan = repository_identity(repo / "wan2.1")
    _require_equal(parent["revision"], PARENT_REVISION, "parent revision")
    _require_equal(wan["revision"], WAN_REVISION, "Wan revision")
    if not parent["dirty"] or not parent["dirty_fingerprint"]:
        raise RuntimeError("U1 execution must record the resumed dirty parent worktree")
    if wan["dirty"]:
        raise RuntimeError("Wan submodule must match its frozen revision without local changes")
    return {"parent": parent, "wan": wan}


def _verify_frozen_files(repo, checkpoint_dir, inventory_path):
    protocol_path = repo / "docs" / "u1_parity_protocol.md"
    _require_equal(
        _sha256_file(protocol_path), EXPECTED_PROTOCOL_SHA256, "parity protocol SHA-256"
    )
    _require_equal(
        _sha256_file(repo / "uv.lock"), EXPECTED_UV_LOCK_SHA256, "uv.lock SHA-256"
    )
    identity = checkpoint_identity(
        repo / "docs" / "u1_parity_protocol.md",
        {
            "path": str(checkpoint_dir),
            "inventory": str(inventory_path),
            "identifier": f"Wan-AI/Wan2.1-I2V-14B-480P@{SNAPSHOT_REVISION}",
        },
        verify_contents=True,
    )
    _require_equal(
        identity["inventory"]["sha256"],
        EXPECTED_INVENTORY_SHA256,
        "checkpoint inventory SHA-256",
    )
    _require_equal(
        identity["content_sha256"],
        EXPECTED_CHECKPOINT_CONTENT_SHA256,
        "checkpoint content SHA-256",
    )
    _require_equal(checkpoint_dir.name, SNAPSHOT_REVISION, "snapshot directory revision")
    return identity


def _runtime_environment(torch, flash_attn):
    from wan.modules.attention import (  # noqa: PLC0415
        FLASH_ATTN_2_AVAILABLE,
        FLASH_ATTN_3_AVAILABLE,
    )

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("frozen U1 execution requires CUDA_VISIBLE_DEVICES=0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.set_device(0)
    _require_equal(torch.__version__, "2.10.0+cu128", "PyTorch version")
    _require_equal(torch.version.cuda, "12.8", "PyTorch CUDA runtime")
    _require_equal(flash_attn.__version__, "2.8.3", "flash-attn version")
    if not FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        raise RuntimeError("frozen U1 execution requires FA2 and forbids FA3 substitution")
    _require_equal(
        torch.cuda.get_device_name(0),
        "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
        "GPU model",
    )
    _require_equal(torch.cuda.get_device_capability(0), (12, 0), "GPU capability")
    hostname = socket.gethostname()
    if hostname not in {"bootes", "bootes.alias"}:
        raise RuntimeError(f"frozen U1 execution requires Bootes, got {hostname!r}")
    driver = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
            "--id=0",
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    return {
        "device": "cuda:0",
        "dtype": "torch.bfloat16",
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "attention_kernel": "flash-attn-2.8.3-fa2",
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": "12.0",
        "driver_version": driver,
        "hostname": hostname,
        "python_version": platform.python_version(),
        "sam2_installed": str(importlib.util.find_spec("sam2") is not None).lower(),
    }


def _load_models(torch, checkpoint_dir, device):
    from wan.modules.custom_model import CustomWanModel  # noqa: PLC0415
    from wan.modules.model import WanModel  # noqa: PLC0415

    common = {
        "torch_dtype": torch.bfloat16,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }
    print(json.dumps({"event": "load-start", "route": "upstream"}), flush=True)
    upstream = WanModel.from_pretrained(str(checkpoint_dir), **common)
    upstream.eval().requires_grad_(False).to(device)
    print(json.dumps({"event": "load-complete", "route": "upstream"}), flush=True)
    custom = CustomWanModel.from_pretrained(str(checkpoint_dir), **common)
    custom.eval().requires_grad_(False).to(device)
    print(json.dumps({"event": "load-complete", "route": "custom-none"}), flush=True)
    if type(upstream) is not WanModel or type(custom) is not CustomWanModel:
        raise TypeError("checkpoint loaders returned unexpected Wan model types")
    if tensor_identity(upstream.freqs) != tensor_identity(custom.freqs):
        raise ValueError("checkpoint-loaded upstream/custom rotary frequencies differ")
    return upstream, custom


def _random_tensor(torch, shape, generator, *, dtype, device, scale=1.0):
    value = torch.randn(shape, generator=generator, dtype=torch.float32)
    if scale != 1.0:
        value.mul_(scale)
    return value.to(device=device, dtype=dtype)


def _case_inputs(torch, upstream, case_seed, device):
    generator = torch.Generator(device="cpu").manual_seed(case_seed)
    x = _random_tensor(
        torch,
        (1, SEQUENCE_LENGTH, MODEL_DIM),
        generator,
        dtype=torch.bfloat16,
        device=device,
    )
    e = _random_tensor(
        torch,
        (1, 6, MODEL_DIM),
        generator,
        dtype=torch.float32,
        device=device,
        scale=0.1,
    )
    context = _random_tensor(
        torch,
        (1, CONTEXT_LENGTH, MODEL_DIM),
        generator,
        dtype=torch.bfloat16,
        device=device,
    )
    grid_sizes = torch.tensor([GRID], dtype=torch.int64)
    seq_lens = torch.tensor([SEQUENCE_LENGTH], dtype=torch.int64)
    freqs = upstream.freqs.detach().clone().to(device)
    face_masks = torch.zeros((1, GRID[1], GRID[2]), dtype=torch.bool, device=device)
    face_masks[:, 8:22, 13:39] = True
    face_masks = face_masks.flatten(1)
    from wan.utils.attention_contracts import build_static_simil_masks  # noqa: PLC0415

    simil_masks = build_static_simil_masks(face_masks, GRID[0])
    inputs = {
        "x": x,
        "e": e,
        "seq_lens": seq_lens,
        "grid_sizes": grid_sizes,
        "freqs": freqs,
        "context": context,
        "context_lens": None,
    }
    bias_kwargs = {
        "bias_method": "none",
        "bias": False,
        "self_attention_masking": False,
        "simil_masks_type": "fixed",
        "face_masks": face_masks,
        "normalized_timestep": torch.tensor(0.5, device=device),
    }
    return inputs, simil_masks, bias_kwargs


def run(checkpoint_dir, inventory_path, output_dir):
    import flash_attn  # noqa: PLC0415
    import torch  # noqa: PLC0415

    repo = Path(__file__).resolve().parents[1]
    checkpoint_dir = Path(checkpoint_dir).resolve()
    inventory_path = Path(inventory_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.is_relative_to(repo):
        raise ValueError("U1 evidence output must be outside the source repository")
    output_dir.mkdir(parents=True, exist_ok=True)

    source = _frozen_source_identity(repo)
    checkpoint = _verify_frozen_files(repo, checkpoint_dir, inventory_path)
    environment = _runtime_environment(torch, flash_attn)
    print(
        json.dumps(
            {
                "event": "preflight-complete",
                "checkpoint_content_sha256": checkpoint["content_sha256"],
                "environment": environment,
                "source": source,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    device = torch.device("cuda:0")
    upstream, custom = _load_models(torch, checkpoint_dir, device)
    binding = {
        "checkpoint_path": str(checkpoint_dir),
        "snapshot_revision": SNAPSHOT_REVISION,
        "inventory_sha256": checkpoint["inventory"]["sha256"],
        "checkpoint_content_sha256": checkpoint["content_sha256"],
        "loaders": {
            "upstream": "wan.modules.model.WanModel.from_pretrained",
            "custom": "wan.modules.custom_model.CustomWanModel.from_pretrained",
        },
        "model_types": {
            "upstream": "wan.modules.model.WanModel",
            "custom": "wan.modules.custom_model.CustomWanModel",
        },
    }
    layer = {
        "parent_revision": source["parent"]["revision"],
        "parent_dirty_fingerprint": source["parent"]["dirty_fingerprint"],
        "wan_revision": source["wan"]["revision"],
        "snapshot_revision": SNAPSHOT_REVISION,
        "checkpoint_inventory_sha256": checkpoint["inventory"]["sha256"],
        "checkpoint_content_sha256": checkpoint["content_sha256"],
        "layer_index": LAYER_INDEX,
        "layer_type": "i2v_cross_attn",
    }
    protocol = {
        "path": str(repo / "docs" / "u1_parity_protocol.md"),
        "sha256": EXPECTED_PROTOCOL_SHA256,
    }
    records = []
    for case, seed in CASES:
        print(json.dumps({"event": "case-start", "case": case, "seed": seed}), flush=True)
        inputs, simil_masks, bias_kwargs = _case_inputs(torch, upstream, seed, device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            record = run_one_layer_parity(
                upstream,
                custom,
                layer_index=LAYER_INDEX,
                inputs=inputs,
                simil_masks=simil_masks,
                custom_bias_kwargs=bias_kwargs,
                layer_provenance=layer,
                environment=environment,
                atol=ATOL,
                rtol=RTOL,
            )
        record = bind_verified_checkpoint_record(record, binding)
        record["case"] = {"name": case, "seed": seed}
        record["source"] = source
        record["protocol"] = protocol
        record_path = output_dir / f"{case}.one-layer-parity.json"
        write_immutable_record(record_path, record)
        records.append(
            {
                "case": case,
                "path": str(record_path),
                "sha256": _sha256_file(record_path),
                "comparison": record["comparison"],
            }
        )
        print(
            json.dumps(
                {"event": "case-complete", "case": case, "comparison": record["comparison"]},
                sort_keys=True,
            ),
            flush=True,
        )
        if not record["verified_checkpoint_parity"]:
            raise RuntimeError(f"U1 one-layer parity failed for {case}")
        del inputs, simil_masks, bias_kwargs, record
        torch.cuda.empty_cache()

    summary = {
        "schema_version": 1,
        "record_kind": "u1-one-layer-parity-summary",
        "accepted": True,
        "claim_scope": "two-case-checkpoint-bound-one-layer-numerical-parity-only",
        "checkpoint": checkpoint,
        "source": source,
        "environment": environment,
        "protocol": protocol,
        "records": records,
    }
    write_immutable_record(output_dir / "summary.json", summary)
    return summary


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=repo / "docs" / "u1_checkpoint_inventory.json",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = run(args.checkpoint_dir, args.inventory, args.output_dir)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
