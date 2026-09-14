"""Pure record and numerical contracts for checkpoint-backed Wan comparisons.

These helpers never choose tolerances or claim verified checkpoint parity. A
caller supplies a predeclared tolerance, while record construction recomputes
all numerical evidence and labels the unresolved checkpoint binding explicitly.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path

REQUIRED_ENVIRONMENT_KEYS = {
    "device",
    "dtype",
    "torch_version",
    "cuda_version",
    "attention_kernel",
}
REQUIRED_LAYER_PROVENANCE_KEYS = {
    "parent_revision",
    "wan_revision",
    "checkpoint_content_sha256",
    "layer_index",
    "layer_type",
}
REQUIRED_PARITY_INPUT_KEYS = {
    "x",
    "e",
    "seq_lens",
    "grid_sizes",
    "freqs",
    "context",
    "context_lens",
}
UNVERIFIED_PROVENANCE_STATUS = "unverified-external-checkpoint-binding"
VERIFIED_PROVENANCE_STATUS = "verified-local-checkpoint-inventory-binding"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_HASH_CHUNK_BYTES = 16 * 1024 * 1024
_COMPARISON_CHUNK_ELEMENTS = 1024 * 1024


def canonical_json_bytes(value):
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _require_sha256(value, label):
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase hexadecimal SHA-256 digest")
    return value


def _shape(value):
    shape = getattr(value, "shape", None)
    if shape is None:
        raise TypeError("parity values must expose shape")
    return [int(item) for item in shape]


def _dtype(value):
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        raise TypeError("parity values must expose dtype")
    return str(dtype)


def _update_tensor_digest(digest, value):
    current = value
    if hasattr(current, "detach"):
        current = current.detach()
    if hasattr(current, "cpu"):
        current = current.cpu()
    if hasattr(current, "contiguous"):
        current = current.contiguous()

    if type(current).__module__.partition(".")[0] == "torch":
        import torch  # noqa: PLC0415

        if not isinstance(current, torch.Tensor):
            raise TypeError("unsupported object from the torch module")
        # Hash the raw storage representation. This avoids unsupported NumPy
        # conversions and multi-gigabyte Python lists for bfloat16 tensors.
        current = current.reshape(-1).view(torch.uint8).numpy().reshape(-1)

    try:
        buffer = memoryview(current)
        if not buffer.c_contiguous:
            raise TypeError
        buffer = buffer.cast("B")
    except TypeError:
        if not hasattr(current, "tobytes"):
            raise TypeError(
                "parity values must provide a contiguous byte representation"
            ) from None
        buffer = memoryview(current.tobytes(order="C"))

    for offset in range(0, len(buffer), _HASH_CHUNK_BYTES):
        digest.update(buffer[offset : offset + _HASH_CHUNK_BYTES])


def tensor_identity(value):
    """Return a shape/dtype-aware SHA-256 identity for an array or tensor."""
    descriptor = {"shape": _shape(value), "dtype": _dtype(value)}
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes(descriptor))
    _update_tensor_digest(digest, value)
    return descriptor | {"sha256": digest.hexdigest()}


def _validate_tensor_identity(identity, label):
    if not isinstance(identity, dict) or set(identity) != {"shape", "dtype", "sha256"}:
        raise ValueError(f"{label} must be a complete tensor identity")
    shape = identity["shape"]
    if not isinstance(shape, list) or any(type(item) is not int or item < 0 for item in shape):
        raise ValueError(f"{label}.shape must contain nonnegative integers")
    if not isinstance(identity["dtype"], str) or not identity["dtype"]:
        raise ValueError(f"{label}.dtype must be a nonempty string")
    _require_sha256(identity["sha256"], f"{label}.sha256")


def state_identity(state):
    if not hasattr(state, "items"):
        raise TypeError("layer state must be a mapping")
    tensors = {}
    for key, value in sorted(state.items()):
        if not isinstance(key, str) or not key:
            raise ValueError("layer state keys must be nonempty strings")
        tensors[key] = tensor_identity(value)
    if not tensors:
        raise ValueError("layer state must not be empty")
    return {
        "sha256": hashlib.sha256(canonical_json_bytes(tensors)).hexdigest(),
        "tensors": tensors,
    }


def _validate_state_evidence(state):
    if not isinstance(state, dict) or set(state) != {"sha256", "tensors"}:
        raise ValueError("state evidence must contain sha256 and tensors")
    tensors = state["tensors"]
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError("state evidence tensors must be a nonempty mapping")
    for key, identity in tensors.items():
        if not isinstance(key, str) or not key:
            raise ValueError("state evidence tensor keys must be nonempty strings")
        _validate_tensor_identity(identity, f"state.tensors.{key}")
    expected = hashlib.sha256(canonical_json_bytes(tensors)).hexdigest()
    _require_sha256(state["sha256"], "state.sha256")
    if state["sha256"] != expected:
        raise ValueError("state evidence digest does not match tensor identities")


def require_identical_state(upstream_state, custom_state):
    """Fail before execution unless keys, shapes, dtypes, and values match."""
    upstream = state_identity(upstream_state)
    custom = state_identity(custom_state)
    if upstream["tensors"].keys() != custom["tensors"].keys():
        raise ValueError("upstream and custom layer state keys differ")
    for key in upstream["tensors"]:
        if upstream["tensors"][key] != custom["tensors"][key]:
            raise ValueError(f"upstream and custom layer state differ at {key}")
    return {"sha256": upstream["sha256"], "tensors": upstream["tensors"]}


def _empty_metrics():
    return {
        "finite": False,
        "max_abs_error": None,
        "max_rel_error": None,
        "passed": False,
    }


def _torch_comparison(upstream, custom, tolerance):
    import torch  # noqa: PLC0415

    upstream = upstream.detach().reshape(-1)
    custom = custom.detach().reshape(-1)
    if upstream.numel() == 0:
        return _empty_metrics()
    work_dtype = torch.float64 if upstream.dtype == torch.float64 else torch.float32
    max_abs = 0.0
    max_rel = 0.0
    zero_mismatch = False
    passed = True
    for start in range(0, upstream.numel(), _COMPARISON_CHUNK_ELEMENTS):
        stop = start + _COMPARISON_CHUNK_ELEMENTS
        upstream_chunk = upstream[start:stop]
        custom_chunk = custom[start:stop]
        if not bool(
            torch.isfinite(upstream_chunk).all().item()
            and torch.isfinite(custom_chunk).all().item()
        ):
            return _empty_metrics()
        upstream_abs = upstream_chunk.to(work_dtype).abs()
        difference = (custom_chunk.to(work_dtype) - upstream_chunk.to(work_dtype)).abs()
        max_abs = max(max_abs, float(difference.max().item()))
        nonzero = upstream_abs != 0
        if bool(nonzero.any().item()):
            max_rel = max(
                max_rel,
                float((difference[nonzero] / upstream_abs[nonzero]).max().item()),
            )
        zero_mismatch = zero_mismatch or bool(
            ((~nonzero) & (difference != 0)).any().item()
        )
        passed = passed and bool(
            (
                difference
                <= tolerance["atol"] + tolerance["rtol"] * upstream_abs
            ).all().item()
        )
    return {
        "finite": True,
        "max_abs_error": max_abs,
        "max_rel_error": None if zero_mismatch else max_rel,
        "passed": passed,
    }


def _array_comparison(upstream, custom, tolerance):
    import numpy as np  # noqa: PLC0415

    upstream = np.asarray(upstream).reshape(-1)
    custom = np.asarray(custom).reshape(-1)
    if upstream.size == 0:
        return _empty_metrics()
    max_abs = 0.0
    max_rel = 0.0
    zero_mismatch = False
    passed = True
    for start in range(0, upstream.size, _COMPARISON_CHUNK_ELEMENTS):
        stop = start + _COMPARISON_CHUNK_ELEMENTS
        upstream_chunk = upstream[start:stop].astype("float64", copy=False)
        custom_chunk = custom[start:stop].astype("float64", copy=False)
        if not (np.isfinite(upstream_chunk).all() and np.isfinite(custom_chunk).all()):
            return _empty_metrics()
        upstream_abs = np.abs(upstream_chunk)
        difference = np.abs(custom_chunk - upstream_chunk)
        max_abs = max(max_abs, float(difference.max()))
        nonzero = upstream_abs != 0
        if nonzero.any():
            max_rel = max(max_rel, float((difference[nonzero] / upstream_abs[nonzero]).max()))
        zero_mismatch = zero_mismatch or bool(((~nonzero) & (difference != 0)).any())
        passed = passed and bool(
            (difference <= tolerance["atol"] + tolerance["rtol"] * upstream_abs).all()
        )
    return {
        "finite": True,
        "max_abs_error": max_abs,
        "max_rel_error": None if zero_mismatch else max_rel,
        "passed": passed,
    }


def validate_tolerance(atol, rtol):
    for name, value in (("atol", atol), ("rtol", rtol)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be an explicitly supplied number")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    return {"atol": float(atol), "rtol": float(rtol)}


def compare_outputs(upstream_output, custom_output, *, atol, rtol):
    """Compare outputs in bounded chunks under caller-supplied tolerances."""
    tolerance = validate_tolerance(atol, rtol)
    shape_match = _shape(upstream_output) == _shape(custom_output)
    dtype_match = _dtype(upstream_output) == _dtype(custom_output)
    result = {
        "shape_match": shape_match,
        "dtype_match": dtype_match,
        **_empty_metrics(),
    }
    if not shape_match or not dtype_match:
        return result

    try:
        import torch  # noqa: PLC0415
    except ImportError:
        torch = None
    upstream_is_torch = torch is not None and isinstance(upstream_output, torch.Tensor)
    custom_is_torch = torch is not None and isinstance(custom_output, torch.Tensor)
    if upstream_is_torch != custom_is_torch:
        raise TypeError("parity outputs must use the same tensor backend")
    metrics = (
        _torch_comparison(upstream_output, custom_output, tolerance)
        if upstream_is_torch
        else _array_comparison(upstream_output, custom_output, tolerance)
    )
    return result | metrics


def validate_runtime_provenance(layer, environment):
    missing_layer = REQUIRED_LAYER_PROVENANCE_KEYS - set(layer)
    if missing_layer:
        raise ValueError(f"layer provenance is missing: {sorted(missing_layer)}")
    missing_environment = REQUIRED_ENVIRONMENT_KEYS - set(environment)
    if missing_environment:
        raise ValueError(
            f"runtime environment provenance is missing: {sorted(missing_environment)}"
        )
    for key in REQUIRED_ENVIRONMENT_KEYS:
        if not isinstance(environment[key], str) or not environment[key]:
            raise ValueError(f"runtime environment {key} must be a nonempty string")
    for key in ("parent_revision", "wan_revision", "layer_type"):
        if not isinstance(layer[key], str) or not layer[key]:
            raise ValueError(f"layer provenance {key} must be a nonempty string")
    if type(layer["layer_index"]) is not int or layer["layer_index"] < 0:
        raise ValueError("layer_index must be a nonnegative integer")
    _require_sha256(layer["checkpoint_content_sha256"], "checkpoint_content_sha256")


def _validate_input_evidence(inputs):
    if not isinstance(inputs, dict) or set(inputs) != REQUIRED_PARITY_INPUT_KEYS:
        raise ValueError("input identities are incomplete or unexpected")
    for key, identity in inputs.items():
        if key == "context_lens":
            if identity != {"value": None}:
                raise ValueError("inputs.context_lens must record the required None value")
        else:
            _validate_tensor_identity(identity, f"inputs.{key}")


def _validate_diagnostic_masks(diagnostic_masks):
    required = {"input", "generated", "used", "equivalence_target"}
    if not isinstance(diagnostic_masks, dict) or set(diagnostic_masks) != required:
        raise ValueError("diagnostic mask evidence is incomplete or unexpected")
    if diagnostic_masks["equivalence_target"] is not False:
        raise ValueError("diagnostic masks are not an equivalence target")
    for key in ("input", "generated", "used"):
        _validate_tensor_identity(diagnostic_masks[key], f"diagnostic_masks.{key}")


def _validate_implementation_sources(sources):
    if not isinstance(sources, dict) or set(sources) != {"upstream", "custom"}:
        raise ValueError("implementation sources must identify upstream and custom modules")
    for route, source in sources.items():
        if not isinstance(source, dict) or set(source) != {"module", "sha256"}:
            raise ValueError(f"implementation_sources.{route} is incomplete")
        if not isinstance(source["module"], str) or not source["module"]:
            raise ValueError(f"implementation_sources.{route}.module must be nonempty")
        _require_sha256(source["sha256"], f"implementation_sources.{route}.sha256")


def build_parity_record(
    *,
    layer,
    environment,
    tolerance,
    state,
    architecture,
    implementation_sources,
    inputs,
    upstream_output,
    custom_output,
    comparison,
    diagnostic_masks,
    provenance_status,
):
    """Build an immutable-ready, explicitly unverified one-layer record."""
    validate_runtime_provenance(layer, environment)
    validated_tolerance = validate_tolerance(tolerance.get("atol"), tolerance.get("rtol"))
    recomputed = compare_outputs(
        upstream_output,
        custom_output,
        atol=validated_tolerance["atol"],
        rtol=validated_tolerance["rtol"],
    )
    if comparison != recomputed:
        raise ValueError("supplied comparison disagrees with outputs and tolerance")
    _validate_state_evidence(state)
    _validate_input_evidence(inputs)
    _validate_diagnostic_masks(diagnostic_masks)
    if not isinstance(architecture, dict) or not architecture:
        raise ValueError("matched architecture evidence must be a nonempty mapping")
    canonical_json_bytes(architecture)
    _validate_implementation_sources(implementation_sources)
    if provenance_status != UNVERIFIED_PROVENANCE_STATUS:
        raise ValueError("verified checkpoint parity records require a separate verified boundary")
    return {
        "schema_version": 1,
        "record_kind": "unverified-upstream-custom-none-one-layer-comparison",
        "claim_scope": "one-layer-numerical-comparison-only",
        "verified_checkpoint_parity": False,
        "provenance_status": provenance_status,
        "routes": {"reference": "upstream", "candidate": "custom-none"},
        "layer": dict(layer),
        "environment": dict(environment),
        "tolerance": validated_tolerance,
        "architecture": architecture,
        "implementation_sources": implementation_sources,
        "state": state,
        "inputs": inputs,
        "outputs": {
            "upstream": tensor_identity(upstream_output),
            "custom_none": tensor_identity(custom_output),
        },
        "diagnostic_masks": diagnostic_masks,
        "comparison": recomputed,
    }


def bind_verified_checkpoint_record(record, checkpoint_binding):
    """Attach verified local loading evidence to a one-layer record.

    Model loading and inventory verification must occur in the controlling
    runner. This boundary validates that their immutable evidence agrees with
    the record before upgrading its narrowly scoped provenance status.
    """
    if not isinstance(record, dict):
        raise TypeError("parity record must be a mapping")
    if record.get("provenance_status") != UNVERIFIED_PROVENANCE_STATUS:
        raise ValueError("only an unverified one-layer record can be checkpoint-bound")
    if record.get("record_kind") != "unverified-upstream-custom-none-one-layer-comparison":
        raise ValueError("unexpected parity record kind")
    required = {
        "checkpoint_path",
        "snapshot_revision",
        "inventory_sha256",
        "checkpoint_content_sha256",
        "loaders",
        "model_types",
    }
    if not isinstance(checkpoint_binding, dict) or set(checkpoint_binding) != required:
        raise ValueError("checkpoint binding evidence is incomplete or unexpected")
    checkpoint_path = Path(checkpoint_binding["checkpoint_path"])
    if not checkpoint_path.is_absolute() or not checkpoint_path.is_dir():
        raise ValueError("checkpoint binding path must be an existing absolute directory")
    if not isinstance(checkpoint_binding["snapshot_revision"], str) or not checkpoint_binding[
        "snapshot_revision"
    ]:
        raise ValueError("checkpoint binding snapshot_revision must be nonempty")
    inventory_sha256 = _require_sha256(
        checkpoint_binding["inventory_sha256"], "inventory_sha256"
    )
    content_sha256 = _require_sha256(
        checkpoint_binding["checkpoint_content_sha256"],
        "checkpoint_binding.checkpoint_content_sha256",
    )
    layer = record.get("layer", {})
    if checkpoint_binding["snapshot_revision"] != layer.get("snapshot_revision"):
        raise ValueError("checkpoint binding snapshot revision disagrees with parity record")
    if inventory_sha256 != layer.get("checkpoint_inventory_sha256"):
        raise ValueError("checkpoint binding inventory identity disagrees with parity record")
    if content_sha256 != layer.get("checkpoint_content_sha256"):
        raise ValueError("checkpoint binding content identity disagrees with parity record")
    expected_loaders = {
        "upstream": "wan.modules.model.WanModel.from_pretrained",
        "custom": "wan.modules.custom_model.CustomWanModel.from_pretrained",
    }
    if checkpoint_binding["loaders"] != expected_loaders:
        raise ValueError("checkpoint binding loaders are not the concrete Wan loaders")
    expected_types = {
        "upstream": "wan.modules.model.WanModel",
        "custom": "wan.modules.custom_model.CustomWanModel",
    }
    if checkpoint_binding["model_types"] != expected_types:
        raise ValueError("checkpoint binding model types are not the concrete Wan models")

    bound = dict(record)
    bound["record_kind"] = "checkpoint-bound-upstream-custom-none-one-layer-comparison"
    bound["provenance_status"] = VERIFIED_PROVENANCE_STATUS
    bound["verified_checkpoint_binding"] = True
    bound["verified_checkpoint_parity"] = bool(record["comparison"]["passed"])
    bound["checkpoint_binding"] = dict(checkpoint_binding)
    return bound


def write_immutable_record(path, record):
    """Create a comparison record once; never overwrite runtime evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_json_bytes(record)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        if path.read_bytes() != content:
            raise FileExistsError(f"immutable parity record conflict: {path}") from error
        return False
    with os.fdopen(fd, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    return True
