"""Run the explicitly authorized R3 backend-only CUDA canaries."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import time
from pathlib import Path

import torch

from .experiment_pipeline import (
    _validate_v4_source_binding,
    repository_identity,
)
from .r3_contracts import (
    load_protocol_bundle,
    sha256_file,
    stage_execution_blockers,
    write_immutable_json,
)
from .r3_environment import observe_attention_runtime, observe_runtime_environment
from .r3_preflight import validate_backend_runtime, validate_v4_runtime_environment

STAGE_ID = "backend-kernel-canary"


def _tensor_record(tensor):
    value = tensor.detach().contiguous()
    finite = bool(torch.isfinite(value).all().item())
    cpu = value.cpu().contiguous()
    payload = cpu.view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "device": value.device.type,
        "finite": finite,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _timed_cuda_call(function):
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = function()
    torch.cuda.synchronize()
    return output, time.perf_counter() - started


def _run_flash_attention_canary():
    from flash_attn import flash_attn_func  # noqa: PLC0415

    q = torch.randn((1, 128, 4, 64), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    output, elapsed = _timed_cuda_call(
        lambda: flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
    )
    record = _tensor_record(output)
    if not record["finite"]:
        raise RuntimeError("FlashAttention 2 canary produced nonfinite output")
    return {"elapsed_seconds": elapsed, "output": record}


def _run_flex_attention_canary():
    from torch.nn.attention.flex_attention import flex_attention  # noqa: PLC0415

    compiled = torch.compile(flex_attention, fullgraph=True, dynamic=False)
    q = torch.randn((1, 4, 128, 64), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    output, elapsed = _timed_cuda_call(lambda: compiled(q, k, v))
    record = _tensor_record(output)
    if not record["finite"]:
        raise RuntimeError("flex-attention canary produced nonfinite output")
    return {
        "compiled": True,
        "elapsed_seconds_including_first_compile": elapsed,
        "output": record,
    }


def _run_sam2_canary():
    from sam2.utils.misc import get_connected_components  # noqa: PLC0415

    mask = torch.zeros((1, 1, 16, 16), device="cuda", dtype=torch.uint8)
    mask[:, :, 1:3, 1:3] = 1
    mask[:, :, 10, 10] = 1
    (labels, counts), elapsed = _timed_cuda_call(
        lambda: get_connected_components(mask)
    )
    label_values = sorted(labels.unique().cpu().tolist())
    foreground_counts = sorted(set(counts[mask.bool()].cpu().tolist()))
    if len([value for value in label_values if value != 0]) != 2:
        raise RuntimeError("SAM2 canary did not find exactly two components")
    if foreground_counts != [1, 4] or bool((counts[~mask.bool()] != 0).any().item()):
        raise RuntimeError("SAM2 canary returned unexpected component areas")
    records = {"labels": _tensor_record(labels), "counts": _tensor_record(counts)}
    if not all(record["finite"] for record in records.values()):
        raise RuntimeError("SAM2 canary produced nonfinite output")
    return {
        "elapsed_seconds": elapsed,
        "label_values": label_values,
        "foreground_component_areas": foreground_counts,
        "outputs": records,
    }


def run_backend_canary(protocol_path, matrix_path, output_path):
    protocol_path = Path(protocol_path).resolve()
    matrix_path = Path(matrix_path).resolve()
    output_path = Path(output_path).resolve()
    repo = Path(__file__).resolve().parents[1]
    if output_path.is_relative_to(repo):
        raise ValueError("backend-canary output must be outside the source repository")

    bundle = load_protocol_bundle(protocol_path, matrix_path)
    protocol = bundle["protocol"]
    if protocol["schema_version"] != 5:
        raise ValueError("backend canary requires the bounded v5 authorization amendment")
    blockers = stage_execution_blockers(protocol, STAGE_ID)
    if blockers:
        raise RuntimeError("R3 backend-canary preflight blocked: " + ", ".join(blockers))

    repositories = {
        "parent": repository_identity(repo),
        "wan": repository_identity(repo / "wan2.1"),
        "lama": repository_identity(repo / "lama"),
    }
    _validate_v4_source_binding(repo, repositories, protocol)

    attention_runtime = observe_attention_runtime()
    declarations = protocol["runtime_declarations"]
    validate_backend_runtime(
        {
            "backend_versions": {
                "flash_attention_2": declarations["flash_attention_version"],
                "flex_attention": declarations["flex_attention_version"],
            }
        },
        attention_runtime,
    )
    environment = observe_runtime_environment(attention_runtime)
    validate_v4_runtime_environment(protocol, environment)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for the authorized backend canary")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("backend canary requires exactly one visible CUDA device")
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    properties = torch.cuda.get_device_properties(0)

    started_at = dt.datetime.now(dt.UTC).isoformat()
    with torch.inference_mode():
        kernels = {
            "flash_attention_2": _run_flash_attention_canary(),
            "flex_attention": _run_flex_attention_canary(),
            "sam2_connected_components": _run_sam2_canary(),
        }
    finished_at = dt.datetime.now(dt.UTC).isoformat()

    record = {
        "schema_version": 1,
        "record_kind": "r3-backend-kernel-canary",
        "stage_id": STAGE_ID,
        "status": "passed",
        "started_at": started_at,
        "finished_at": finished_at,
        "bindings": {
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": bundle["protocol_sha256"],
            "matrix_id": bundle["matrix"]["matrix_id"],
            "matrix_sha256": bundle["matrix_sha256"],
        },
        "repositories": repositories,
        "environment": environment,
        "attention_runtime": attention_runtime,
        "visible_cuda_device": {
            "index": 0,
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": properties.total_memory,
        },
        "seed": 0,
        "kernels": kernels,
        "scope": {
            "checkpoint_or_model_loaded": False,
            "generation_performed": False,
            "distributed_execution_performed": False,
        },
    }
    write_immutable_json(output_path, record)
    return {
        "status": "passed",
        "stage_id": STAGE_ID,
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "protocol_sha256": bundle["protocol_sha256"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = run_backend_canary(args.protocol, args.matrix, args.output)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
