"""Run the explicitly authorized R3 checkpoint-load/source-hook canary."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import time
from pathlib import Path

import torch

from .experiment_pipeline import (
    _validate_v4_source_binding,
    checkpoint_identity,
    repository_identity,
)
from .parity_contracts import state_identity, tensor_identity
from .r3_contracts import (
    load_protocol_bundle,
    sha256_file,
    stage_execution_blockers,
    write_immutable_json,
)
from .r3_environment import observe_attention_runtime, observe_runtime_environment
from .r3_preflight import validate_backend_runtime, validate_v4_runtime_environment
from .r3_runtime import R3RuntimeCollector

STAGE_ID = "checkpoint-load-hook-canary"


def _require_equal(actual, expected, label):
    if actual != expected:
        raise RuntimeError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def _validate_prerequisite_evidence(repo, declaration):
    path = (repo / declaration["path"]).resolve()
    if not path.is_relative_to(repo) or not path.is_file():
        raise RuntimeError("R3 prerequisite backend evidence is missing")
    _require_equal(sha256_file(path), declaration["sha256"], "prerequisite evidence SHA-256")
    record = json.loads(path.read_text())
    _require_equal(record.get("stage_id"), declaration["stage_id"], "prerequisite stage")
    _require_equal(record.get("status"), declaration["status"], "prerequisite status")
    _require_equal(
        record.get("bindings", {}).get("protocol_sha256"),
        declaration["protocol_sha256"],
        "prerequisite protocol SHA-256",
    )
    _require_equal(
        record.get("scope"),
        {
            "checkpoint_or_model_loaded": False,
            "generation_performed": False,
            "distributed_execution_performed": False,
        },
        "prerequisite scope",
    )
    return {"path": declaration["path"], "sha256": declaration["sha256"]}


def _verify_checkpoint(repo, protocol):
    amendment = protocol["execution_amendment"]
    binding = amendment["checkpoint_binding"]
    contract = amendment["hook_canary_contract"]
    inventory_path = (repo / contract["checkpoint_inventory_path"]).resolve()
    identity = checkpoint_identity(
        repo / "docs/r3_protocol_v7.json",
        {
            "path": binding["path"],
            "inventory": str(inventory_path),
            "identifier": binding["identifier"],
        },
        verify_contents=True,
    )
    _require_equal(
        identity["inventory"]["sha256"], binding["inventory_sha256"], "checkpoint inventory SHA-256"
    )
    _require_equal(
        identity["content_sha256"], binding["content_sha256"], "checkpoint content SHA-256"
    )
    _require_equal(
        Path(identity["path"]).name, binding["snapshot_revision"], "checkpoint snapshot revision"
    )
    return identity


def _random_tensor(shape, generator, *, dtype, device, scale=1.0):
    value = torch.randn(shape, generator=generator, dtype=torch.float32)
    if scale != 1.0:
        value.mul_(scale)
    return value.to(device=device, dtype=dtype)


def _build_inputs(contract, freqs, device):
    frozen = contract["synthetic_input"]
    generator = torch.Generator(device="cpu").manual_seed(frozen["seed"])
    model_dim = frozen["model_dim"]
    sequence_length = frozen["sequence_length"]
    return {
        "x": _random_tensor(
            (1, sequence_length, model_dim),
            generator,
            dtype=torch.bfloat16,
            device=device,
        ),
        "e": _random_tensor(
            (1, 6, model_dim),
            generator,
            dtype=torch.float32,
            device=device,
            scale=0.1,
        ),
        "seq_lens": torch.tensor([sequence_length], dtype=torch.int64),
        "grid_sizes": torch.tensor([frozen["grid_size"]], dtype=torch.int64),
        "freqs": freqs.detach().clone().to(device),
        "context": _random_tensor(
            (1, frozen["context_length"], model_dim),
            generator,
            dtype=torch.bfloat16,
            device=device,
        ),
        "context_lens": None,
    }


def _input_identities(inputs):
    return {
        key: ({"value": None} if value is None else tensor_identity(value))
        for key, value in inputs.items()
    }


def _clone_inputs(inputs):
    return {
        key: (value.detach().clone() if hasattr(value, "detach") else value)
        for key, value in inputs.items()
    }


def _validate_observations(records, contract):
    events = [record.get("event") for record in records]
    _require_equal(events, contract["expected_observer_events"], "observer event sequence")
    dispatches = [record for record in records if record.get("event") == "attention-dispatch"]
    _require_equal(
        [record.get("attention_site") for record in dispatches],
        contract["expected_attention_sites"],
        "attention site sequence",
    )
    if any(
        record.get("backend") != contract["expected_attention_backend"]
        or record.get("backend_version") != "2.8.3"
        or record.get("block") != 0
        or record.get("rank") != 0
        for record in dispatches
    ):
        raise RuntimeError("observed attention dispatch differs from the frozen hook contract")
    return records


def _run_layer_neutrality_canary(layer, inputs, contract):
    from wan.utils.runtime_evidence import (  # noqa: PLC0415
        install_runtime_observer,
        runtime_observation_scope,
    )

    input_identity_before = _input_identities(inputs)
    baseline_inputs = _clone_inputs(inputs)
    observed_inputs = _clone_inputs(inputs)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        baseline = layer(**baseline_inputs)
    torch.cuda.synchronize()
    baseline_elapsed = time.perf_counter() - started

    collector = R3RuntimeCollector()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with (
        torch.inference_mode(),
        torch.autocast("cuda", dtype=torch.bfloat16),
        install_runtime_observer(collector, rank=0),
        runtime_observation_scope(block=0),
    ):
        observed = layer(**observed_inputs)
    torch.cuda.synchronize()
    observed_elapsed = time.perf_counter() - started

    if not bool(torch.isfinite(baseline).all().item() and torch.isfinite(observed).all().item()):
        raise RuntimeError("checkpoint hook canary produced nonfinite output")
    baseline_identity = tensor_identity(baseline)
    observed_identity = tensor_identity(observed)
    _require_equal(observed_identity, baseline_identity, "observer/no-observer output identity")
    if not torch.equal(observed, baseline):
        raise RuntimeError("source-hook observer changed the selected-layer output")
    _require_equal(_input_identities(inputs), input_identity_before, "source inputs after canary")
    _require_equal(
        _input_identities(baseline_inputs), input_identity_before, "baseline inputs after canary"
    )
    _require_equal(
        _input_identities(observed_inputs), input_identity_before, "observed inputs after canary"
    )
    observations = _validate_observations(collector.snapshot(), contract)
    return {
        "baseline_elapsed_seconds": baseline_elapsed,
        "observed_elapsed_seconds": observed_elapsed,
        "output": baseline_identity,
        "bitwise_exact": True,
        "inputs_preserved": True,
        "observations": observations,
    }


def _load_selected_layer(checkpoint, contract, device):
    from wan.modules.model import WanAttentionBlock, WanModel  # noqa: PLC0415

    load_started = time.perf_counter()
    print(json.dumps({"event": "model-load-start", "route": "upstream"}), flush=True)
    model = WanModel.from_pretrained(
        checkpoint["path"],
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval().requires_grad_(False)
    load_elapsed = time.perf_counter() - load_started
    if type(model) is not WanModel:
        raise TypeError("checkpoint loader returned an unexpected Wan model type")
    layer = model.blocks[contract["selected_layer_index"]]
    if type(layer) is not WanAttentionBlock:
        raise TypeError("selected checkpoint layer has an unexpected concrete type")
    freqs = model.freqs.detach().clone()
    architecture = {
        "model_type": model.model_type,
        "dim": model.dim,
        "ffn_dim": model.ffn_dim,
        "num_heads": model.num_heads,
        "num_layers": model.num_layers,
        "cross_attn_norm": model.cross_attn_norm,
        "qk_norm": model.qk_norm,
        "window_size": list(model.window_size),
    }
    layer_state = state_identity(layer.state_dict())
    _require_equal(
        layer_state["sha256"],
        contract["expected_selected_layer_state_sha256"],
        "selected layer checkpoint state SHA-256",
    )
    del model
    gc.collect()
    layer.to(device)
    print(
        json.dumps(
            {"event": "model-load-complete", "elapsed_seconds": load_elapsed},
            sort_keys=True,
        ),
        flush=True,
    )
    return layer, freqs, architecture, layer_state, load_elapsed


def run_checkpoint_hook_canary(protocol_path, matrix_path, output_path):
    protocol_path = Path(protocol_path).resolve()
    matrix_path = Path(matrix_path).resolve()
    output_path = Path(output_path).resolve()
    repo = Path(__file__).resolve().parents[1]
    if output_path.is_relative_to(repo):
        raise ValueError("checkpoint-hook canary output must be outside the source repository")

    bundle = load_protocol_bundle(protocol_path, matrix_path)
    protocol = bundle["protocol"]
    if protocol["schema_version"] != 7:
        raise ValueError("checkpoint-hook canary requires the bounded v7 amendment")
    blockers = stage_execution_blockers(protocol, STAGE_ID)
    if blockers:
        raise RuntimeError("R3 checkpoint-hook preflight blocked: " + ", ".join(blockers))

    amendment = protocol["execution_amendment"]
    prerequisite = _validate_prerequisite_evidence(repo, amendment["prerequisite_evidence"])
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
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("checkpoint-hook canary requires exactly one visible CUDA device")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(0)

    checkpoint = _verify_checkpoint(repo, protocol)
    print(
        json.dumps(
            {
                "event": "checkpoint-verified",
                "content_sha256": checkpoint["content_sha256"],
                "inventory_sha256": checkpoint["inventory"]["sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    contract = amendment["hook_canary_contract"]
    started_at = dt.datetime.now(dt.UTC).isoformat()
    layer, freqs, architecture, layer_state_before, load_elapsed = _load_selected_layer(
        checkpoint, contract, device
    )

    torch.manual_seed(contract["synthetic_input"]["seed"])
    torch.cuda.manual_seed_all(contract["synthetic_input"]["seed"])
    inputs = _build_inputs(contract, freqs, device)
    canary = _run_layer_neutrality_canary(layer, inputs, contract)
    layer_state_after = state_identity(layer.state_dict())
    _require_equal(layer_state_after, layer_state_before, "selected layer state after canary")
    finished_at = dt.datetime.now(dt.UTC).isoformat()

    record = {
        "schema_version": 1,
        "record_kind": "r3-checkpoint-load-hook-canary",
        "stage_id": STAGE_ID,
        "status": "passed",
        "started_at": started_at,
        "finished_at": finished_at,
        "bindings": {
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": bundle["protocol_sha256"],
            "matrix_id": bundle["matrix"]["matrix_id"],
            "matrix_sha256": bundle["matrix_sha256"],
            "prerequisite_evidence_sha256": prerequisite["sha256"],
            "checkpoint_inventory_sha256": checkpoint["inventory"]["sha256"],
            "checkpoint_content_sha256": checkpoint["content_sha256"],
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
        "checkpoint_load": {
            "identifier": amendment["checkpoint_binding"]["identifier"],
            "loader": contract["loader"],
            "model_type": contract["model_type"],
            "torch_dtype": contract["torch_dtype"],
            "elapsed_seconds": load_elapsed,
            "architecture": architecture,
            "selected_layer_index": contract["selected_layer_index"],
            "selected_layer_type": contract["selected_layer_type"],
            "selected_layer_state_sha256": layer_state_before["sha256"],
            "selected_layer_state_preserved": True,
        },
        "hook_neutrality": canary,
        "scope": {
            "checkpoint_verified": True,
            "upstream_wan_dit_loaded": True,
            "selected_layer_forward_performed": True,
            "text_encoder_loaded": False,
            "clip_loaded": False,
            "vae_loaded": False,
            "custom_model_loaded": False,
            "full_model_forward_performed": False,
            "generation_performed": False,
            "scheduler_or_decoding_performed": False,
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
    result = run_checkpoint_hook_canary(args.protocol, args.matrix, args.output)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
