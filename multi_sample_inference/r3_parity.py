"""Bounded offline tensor artifacts and numerical comparison for R3 parity."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import numpy as np

from .parity_contracts import canonical_json_bytes, tensor_identity
from .r3_contracts import (
    CPU_FIXTURE_EVIDENCE,
    GENUINE_RUNTIME_EVIDENCE,
    sha256_file,
    write_immutable_json,
)

_ROUTE_NAMES = {"upstream", "custom-none"}
_COMMON_BINDINGS = {
    "input_sha256",
    "checkpoint_content_sha256",
    "source_sha256",
    "environment_sha256",
    "diffusion_seed",
    "inference_settings_sha256",
}
_ROUTE_BINDINGS = _COMMON_BINDINGS | {"config_sha256", "route_source_sha256"}


def write_parity_tensor_artifact(
    path,
    value,
    *,
    pair_id,
    route,
    job_id,
    bindings,
    evidence_class=CPU_FIXTURE_EVIDENCE,
):
    """Write one immutable NumPy tensor plus immutable trusted-binding metadata."""
    if evidence_class not in {CPU_FIXTURE_EVIDENCE, GENUINE_RUNTIME_EVIDENCE}:
        raise ValueError("parity artifact evidence class is unsupported")
    if route not in _ROUTE_NAMES:
        raise ValueError("parity artifact route must be upstream or custom-none")
    if not isinstance(pair_id, str) or not pair_id or not isinstance(job_id, str) or not job_id:
        raise ValueError("parity artifact pair_id and job_id must be nonempty strings")
    if not isinstance(bindings, dict) or set(bindings) != _ROUTE_BINDINGS:
        raise ValueError("parity artifact bindings are incomplete or unexpected")
    for key in _ROUTE_BINDINGS - {"diffusion_seed"}:
        digest = bindings[key]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"parity artifact binding {key} must be a SHA-256 digest")
    if type(bindings["diffusion_seed"]) is not int:
        raise ValueError("parity artifact diffusion_seed must be an integer")
    current = value
    if type(current).__module__.partition(".")[0] == "torch":
        import torch  # noqa: PLC0415

        supported = {torch.float16, torch.float32, torch.float64}
        if current.dtype not in supported:
            raise ValueError(
                f"parity artifact does not support PyTorch dtype {current.dtype}; "
                "supported dtypes are float16, float32, and float64"
            )
        # CPU transfer and contiguous materialization preserve dtype and values;
        # NumPy then shares that storage rather than casting it.
        current = current.detach().cpu().contiguous()
        array = current.numpy()
    else:
        array = np.ascontiguousarray(np.asarray(current))
    if (
        array.size == 0
        or array.dtype.hasobject
        or not np.issubdtype(array.dtype, np.floating)
    ):
        raise ValueError(
            "parity artifact must be a nonempty real floating-point tensor"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise FileExistsError(f"immutable parity tensor artifact exists: {path}") from error
    with os.fdopen(fd, "wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    metadata = {
        "schema_version": 2,
        "artifact_kind": "r3-full-generator-parity-tensor",
        "evidence_class": evidence_class,
        "pair_id": pair_id,
        "route": route,
        "job_id": job_id,
        "bindings": bindings,
        "tensor": tensor_identity(array),
        "artifact_sha256": sha256_file(path),
    }
    write_immutable_json(path.with_suffix(path.suffix + ".json"), metadata)
    return metadata


def validate_parity_artifact(declaration):
    """Validate one immutable artifact against its trusted declaration."""
    path = Path(declaration["path"])
    metadata_path = Path(declaration["metadata_path"])
    if not path.is_file() or not metadata_path.is_file():
        raise ValueError("parity tensor artifact or metadata is missing")
    import json  # noqa: PLC0415

    metadata = json.loads(metadata_path.read_text())
    required = {
        "schema_version",
        "artifact_kind",
        "evidence_class",
        "pair_id",
        "route",
        "job_id",
        "bindings",
        "tensor",
        "artifact_sha256",
    }
    if not isinstance(metadata, dict) or set(metadata) != required:
        raise ValueError("parity tensor metadata is malformed")
    if (
        metadata["schema_version"] != 2
        or metadata["artifact_kind"] != "r3-full-generator-parity-tensor"
        or metadata["evidence_class"]
        not in {CPU_FIXTURE_EVIDENCE, GENUINE_RUNTIME_EVIDENCE}
    ):
        raise ValueError("parity tensor metadata identity is unsupported")
    if metadata["artifact_sha256"] != sha256_file(path):
        raise ValueError("parity tensor artifact changed after declaration")
    for key in ("pair_id", "route", "job_id", "bindings", "evidence_class"):
        if metadata[key] != declaration[key]:
            raise ValueError(f"parity tensor metadata {key} differs from declaration")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError("parity tensor artifact dtype is not real floating-point")
    if tensor_identity(array) != metadata["tensor"]:
        raise ValueError("parity tensor identity differs from metadata")
    return metadata, array


def _validate_tolerance(value, label):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"parity {label} must be finite and nonnegative")


def _measure_parity(upstream, custom, *, atol, rtol, chunk_elements):
    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    relative_error_unbounded = False
    finite = True
    passed = True
    for offset in range(0, upstream.size, chunk_elements):
        upstream_chunk = np.asarray(
            upstream[offset : offset + chunk_elements], dtype=np.float64
        )
        custom_chunk = np.asarray(
            custom[offset : offset + chunk_elements], dtype=np.float64
        )
        chunk_finite = bool(
            np.isfinite(upstream_chunk).all() and np.isfinite(custom_chunk).all()
        )
        finite = finite and chunk_finite
        if not chunk_finite:
            passed = False
            continue
        difference = np.abs(custom_chunk - upstream_chunk)
        threshold = atol + rtol * np.abs(upstream_chunk)
        passed = passed and bool(np.all(difference <= threshold))
        if not difference.size:
            continue
        maximum_absolute_error = max(
            maximum_absolute_error, float(difference.max())
        )
        nonzero_reference = upstream_chunk != 0
        if np.any(nonzero_reference):
            maximum_relative_error = max(
                maximum_relative_error,
                float(
                    (
                        difference[nonzero_reference]
                        / np.abs(upstream_chunk[nonzero_reference])
                    ).max()
                ),
            )
        relative_error_unbounded = relative_error_unbounded or bool(
            np.any((~nonzero_reference) & (difference != 0))
        )
    return {
        "finite": finite,
        "maximum_absolute_error": maximum_absolute_error if finite else None,
        "maximum_relative_error": (
            None if not finite or relative_error_unbounded else maximum_relative_error
        ),
        "atol": float(atol),
        "rtol": float(rtol),
        "passed": passed,
    }


def compare_parity_artifacts(pair, *, atol, rtol, chunk_elements=1_000_000):
    """Compare trusted paired tensors in bounded memory and derive pass/fail."""
    if not isinstance(pair, dict) or set(pair) != {"pair_id", "artifacts"}:
        raise ValueError("trusted parity pair declaration is malformed")
    artifacts = pair["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != _ROUTE_NAMES:
        raise ValueError("trusted parity pair must contain upstream and custom-none artifacts")
    _validate_tolerance(atol, "atol")
    _validate_tolerance(rtol, "rtol")
    if type(chunk_elements) is not int or chunk_elements <= 0:
        raise ValueError("parity comparison chunk_elements must be positive")

    loaded = {}
    for route in sorted(_ROUTE_NAMES):
        declaration = artifacts[route]
        if declaration.get("route") != route or declaration.get("pair_id") != pair["pair_id"]:
            raise ValueError("parity route or pair identity is mismatched")
        loaded[route] = validate_parity_artifact(declaration)
    upstream_meta, upstream = loaded["upstream"]
    custom_meta, custom = loaded["custom-none"]
    for key in _COMMON_BINDINGS:
        if upstream_meta["bindings"][key] != custom_meta["bindings"][key]:
            raise ValueError(f"paired parity inputs differ for {key}")
    if upstream_meta["bindings"]["route_source_sha256"] == custom_meta["bindings"]["route_source_sha256"]:
        raise ValueError("upstream and custom-none route source identities must remain distinct")
    if upstream.shape != custom.shape or upstream.dtype != custom.dtype:
        raise ValueError("paired parity tensor shape or dtype differs")

    measurements = _measure_parity(
        upstream.reshape(-1),
        custom.reshape(-1),
        atol=atol,
        rtol=rtol,
        chunk_elements=chunk_elements,
    )
    evidence_classes = {
        upstream_meta["evidence_class"], custom_meta["evidence_class"]
    }
    comparison_evidence_class = (
        GENUINE_RUNTIME_EVIDENCE
        if evidence_classes == {GENUINE_RUNTIME_EVIDENCE}
        else CPU_FIXTURE_EVIDENCE
    )
    return {
        "schema_version": 2,
        "record_kind": "r3-offline-full-generator-parity-comparison",
        "evidence_class": comparison_evidence_class,
        "pair_id": pair["pair_id"],
        "artifacts": {
            route: {
                "job_id": loaded[route][0]["job_id"],
                "artifact_sha256": loaded[route][0]["artifact_sha256"],
                "tensor": loaded[route][0]["tensor"],
            }
            for route in sorted(_ROUTE_NAMES)
        },
        "measurements": measurements,
        "comparison_sha256": hashlib.sha256(
            canonical_json_bytes(
                {
                    "upstream": upstream_meta["artifact_sha256"],
                    "custom-none": custom_meta["artifact_sha256"],
                    "atol": atol,
                    "rtol": rtol,
                }
            )
        ).hexdigest(),
        "r3_acceptance": False,
    }
