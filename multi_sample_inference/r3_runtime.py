"""Bounded collectors and collective adapters for opt-in R3 source hooks."""

from __future__ import annotations

import hashlib
import re

from .parity_contracts import canonical_json_bytes

_SHA256 = re.compile(r"[0-9a-f]{64}")

_HASH_CHUNK_BYTES = 16 * 1024 * 1024
_TENSOR_SUFFIX = "_tensor"


def _tensor_identity_and_finiteness(value):
    current = value.detach() if hasattr(value, "detach") else value
    module = type(current).__module__.partition(".")[0]
    dtype = str(current.dtype)
    # R3 uses storage dtype names across tensor libraries. For the supported
    # PyTorch floating types this preserves the dtype exactly while matching
    # the dtype recorded by the lossless NumPy parity artifact.
    if module == "torch" and dtype.startswith("torch."):
        dtype = dtype.removeprefix("torch.")
    descriptor = {
        "shape": [int(item) for item in current.shape],
        "dtype": dtype,
    }
    digest = hashlib.sha256(canonical_json_bytes(descriptor))
    if module == "torch":
        import torch  # noqa: PLC0415

        contiguous = current.contiguous()
        flat = contiguous.reshape(-1)
        finite = True
        elements_per_chunk = max(1, _HASH_CHUNK_BYTES // max(1, contiguous.element_size()))
        for offset in range(0, flat.numel(), elements_per_chunk):
            chunk = flat[offset : offset + elements_per_chunk]
            finite = finite and bool(torch.isfinite(chunk).all().item())
            raw = chunk.contiguous().view(torch.uint8).cpu().numpy()
            digest.update(memoryview(raw).cast("B"))
    else:
        import numpy as np  # noqa: PLC0415

        contiguous = np.ascontiguousarray(current)
        finite = bool(np.isfinite(contiguous).all())
        raw = memoryview(contiguous).cast("B")
        for offset in range(0, len(raw), _HASH_CHUNK_BYTES):
            digest.update(raw[offset : offset + _HASH_CHUNK_BYTES])
    return descriptor | {"sha256": digest.hexdigest()}, finite


class R3RuntimeCollector:
    """Convert ephemeral tensor callbacks into fixed-size evidence records."""

    def __init__(self, *, parity_artifact=None):
        self.records = []
        self._parity_artifact = parity_artifact
        self.parity_metadata = None

    def __call__(self, event):
        if not isinstance(event, dict) or not isinstance(event.get("event"), str):
            raise ValueError("runtime observer event is malformed")
        record = {}
        event_tensor_cache = {}
        for key, value in event.items():
            if (
                key == "output_tensor"
                and event["event"] == "final-latent"
                and self._parity_artifact is not None
                and event.get("rank") == 0
            ):
                if self.parity_metadata is not None:
                    raise ValueError("parity collector received multiple final tensors")
                from .r3_parity import write_parity_tensor_artifact  # noqa: PLC0415

                declaration = self._parity_artifact
                identity, finite = _tensor_identity_and_finiteness(value)
                self.parity_metadata = write_parity_tensor_artifact(
                    declaration["path"],
                    value,
                    pair_id=declaration["pair_id"],
                    route=declaration["route"],
                    job_id=declaration["job_id"],
                    bindings=declaration["bindings"],
                    evidence_class=declaration["evidence_class"],
                )
                if identity != self.parity_metadata["tensor"]:
                    raise ValueError("parity artifact identity differs from observed tensor")
                record["output_identity"] = identity
                record["output_finite"] = finite
                continue
            if key.endswith(_TENSOR_SUFFIX):
                cache_key = id(value)
                cached = event_tensor_cache.get(cache_key)
                if cached is None:
                    cached = _tensor_identity_and_finiteness(value)
                    event_tensor_cache[cache_key] = cached
                identity, finite = cached
                stem = key[: -len(_TENSOR_SUFFIX)]
                record[f"{stem}_identity"] = identity
                record[f"{stem}_finite"] = finite
            else:
                record[key] = value
        record["ordinal"] = len(self.records)
        self.records.append(record)

    def snapshot(self):
        return [dict(record) for record in self.records]


def _validate_rank_local_stream(records, rank):
    if not isinstance(records, list) or any(
        not isinstance(record, dict) or record.get("rank") != rank
        for record in records
    ):
        raise ValueError("R3 observation gather returned malformed or wrong-rank data")
    if [record.get("ordinal") for record in records] != list(range(len(records))):
        raise ValueError("R3 rank-local observation ordinals are incomplete or reordered")


def gather_rank_observations(local_records, *, rank, world_size, gather_object):
    """Call one gather on every rank and return rank-local streams in rank order."""
    if type(rank) is not int or type(world_size) is not int or not 0 <= rank < world_size:
        raise ValueError("invalid rank/world_size for R3 observation gathering")
    _validate_rank_local_stream(local_records, rank)
    gathered = [None] * world_size if rank == 0 else None
    gather_object(local_records, gathered, dst=0)
    if rank != 0:
        return None
    if any(not isinstance(records, list) for records in gathered):
        raise ValueError("R3 observation gather returned malformed rank data")
    for gathered_rank, records in enumerate(gathered):
        _validate_rank_local_stream(records, gathered_rank)
    return [record for records in gathered for record in records]


def _valid_tensor_identity(identity):
    return (
        isinstance(identity, dict)
        and set(identity) == {"shape", "dtype", "sha256"}
        and isinstance(identity["shape"], list)
        and identity["shape"]
        and all(type(value) is int and value > 0 for value in identity["shape"])
        and isinstance(identity["dtype"], str)
        and bool(identity["dtype"])
        and isinstance(identity["sha256"], str)
        and _SHA256.fullmatch(identity["sha256"]) is not None
    )


def _validate_gathered_stream_order(records, expected_ranks):
    previous_rank = -1
    by_rank = {rank: [] for rank in expected_ranks}
    for record in records:
        rank = record.get("rank") if isinstance(record, dict) else None
        if rank not in expected_ranks or rank < previous_rank:
            raise ValueError("source-hook rank streams are missing, wrong, or reordered")
        previous_rank = rank
        by_rank[rank].append(record)
    for rank, local_records in by_rank.items():
        _validate_rank_local_stream(local_records, rank)


def _validate_lifecycle(records, expected_ranks, expected_seed):
    by_event = {}
    for event in (
        "observer-installed",
        "observer-completed",
        "initial-latent",
        "final-latent",
    ):
        selected = [item for item in records if item.get("event") == event]
        ranks = [item.get("rank") for item in selected]
        if len(ranks) != len(expected_ranks) or set(ranks) != expected_ranks:
            raise ValueError(f"{event} rank coverage is incomplete or duplicated")
        by_event[event] = selected
    if any(item.get("event") == "observer-failed" for item in records):
        raise ValueError("source-hook observer recorded a runtime failure")

    initial = by_event["initial-latent"]
    if any(
        item.get("latent_finite") is not True
        or not _valid_tensor_identity(item.get("latent_identity"))
        or item.get("seed") != expected_seed
        for item in initial
    ) or len({item["latent_identity"]["sha256"] for item in initial}) != 1:
        raise ValueError("initial-latent identities are missing, nonfinite, or differ by rank")
    if any(
        item.get("output_finite") is not True
        or not _valid_tensor_identity(item.get("output_identity"))
        for item in by_event["final-latent"]
    ):
        raise ValueError("final-latent identities are missing or nonfinite")


def _validate_cfg(records, expected_ranks, sampling_steps):
    selected = [item for item in records if item.get("event") == "cfg-branch-output"]
    coordinates = [
        (item.get("rank"), item.get("step"), item.get("branch"))
        for item in selected
    ]
    expected = {
        (rank, step, branch)
        for rank in expected_ranks
        for step in range(sampling_steps)
        for branch in ("conditional", "negative")
    }
    if len(coordinates) != len(set(coordinates)) or set(coordinates) != expected:
        raise ValueError("CFG rank/step/branch coverage is incomplete or duplicated")
    if any(
        item.get("output_finite") is not True
        or not _valid_tensor_identity(item.get("output_identity"))
        for item in selected
    ):
        raise ValueError("CFG output identities are missing or nonfinite")


def build_self_attention_dispatch_contract(
    *,
    execution_route,
    method,
    self_attention_masking,
    timestep_bias_schedule,
    blocks_bias_schedule,
    backend_versions,
):
    """Freeze the inputs that determine the unmodified self-attention route."""
    if execution_route not in {"upstream", "custom"}:
        raise ValueError("R3 dispatch execution route is unsupported")
    if not isinstance(method, str) or not method:
        raise ValueError("R3 dispatch method is missing")
    if type(self_attention_masking) is not bool:
        raise ValueError("R3 self-attention routing flag must be boolean")
    for name, schedule in (
        ("timestep_bias_schedule", timestep_bias_schedule),
        ("blocks_bias_schedule", blocks_bias_schedule),
    ):
        if not isinstance(schedule, list) or not schedule or any(
            type(value) is not bool for value in schedule
        ):
            raise ValueError(f"R3 {name} must be a nonempty boolean list")
    if not isinstance(backend_versions, dict) or set(backend_versions) != {
        "flash_attention_2",
        "flex_attention",
    }:
        raise ValueError("R3 dispatch backend version declarations are incomplete")
    if any(
        value is not None and (not isinstance(value, str) or not value)
        for value in backend_versions.values()
    ):
        raise ValueError("R3 dispatch backend version declaration is invalid")
    return {
        "execution_route": execution_route,
        "method": method,
        "self_attention_masking": self_attention_masking,
        "timestep_bias_schedule": list(timestep_bias_schedule),
        "blocks_bias_schedule": list(blocks_bias_schedule),
        "backend_versions": dict(backend_versions),
    }


def expected_self_attention_backend(dispatch_contract, *, step, branch, block):
    if branch not in {"conditional", "negative"}:
        raise ValueError("R3 dispatch branch is unsupported")
    if not 0 <= step < len(dispatch_contract["timestep_bias_schedule"]):
        raise ValueError("R3 dispatch step is out of range")
    if not 0 <= block < len(dispatch_contract["blocks_bias_schedule"]):
        raise ValueError("R3 dispatch block is out of range")
    uses_flex = (
        dispatch_contract["execution_route"] == "custom"
        and branch == "conditional"
        and dispatch_contract["method"] != "none"
        and dispatch_contract["self_attention_masking"]
        and dispatch_contract["timestep_bias_schedule"][step]
        and dispatch_contract["blocks_bias_schedule"][block]
    )
    return "flex_attention" if uses_flex else "flash_attention_2"


def validate_backend_request(dispatch_contract, requested_backend):
    if requested_backend not in {"flash_attention_2", "flex_attention"}:
        raise ValueError("R3 requested attention backend is unsupported")
    conditional_backends = {
        expected_self_attention_backend(
            dispatch_contract, step=step, branch="conditional", block=block
        )
        for step in range(len(dispatch_contract["timestep_bias_schedule"]))
        for block in range(len(dispatch_contract["blocks_bias_schedule"]))
    }
    compatible = (
        "flex_attention" in conditional_backends
        if requested_backend == "flex_attention"
        else "flex_attention" not in conditional_backends
    )
    if not compatible:
        raise ValueError(
            "requested attention coverage is incompatible with the concrete route and schedules"
        )
    return True


def validate_worker_dispatch_binding(task):
    """Reject task/config substitutions before an opt-in R3 model is loaded."""
    evidence = task.get("r3_evidence")
    schema_version = (
        evidence.get("protocol_schema_version")
        if isinstance(evidence, dict)
        else None
    )
    if type(schema_version) is not int or schema_version < 3:
        return True
    requested = evidence.get("requested")
    if not isinstance(requested, dict):
        raise ValueError("R3 worker requested binding is missing")
    config = task.get("config")
    settings = task.get("inference_settings")
    if not isinstance(config, dict) or not isinstance(settings, dict):
        raise ValueError("R3 worker configuration is missing")
    expected = build_self_attention_dispatch_contract(
        execution_route=requested.get("execution_route"),
        method=config.get("bias_method"),
        self_attention_masking=config.get("self_attention_masking"),
        timestep_bias_schedule=config.get("timestep_bias_schedule"),
        blocks_bias_schedule=config.get("blocks_bias_schedule"),
        backend_versions=requested.get("dispatch_contract", {}).get(
            "backend_versions"
        ),
    )
    if expected != requested.get("dispatch_contract"):
        raise ValueError("R3 worker route or schedule differs from its immutable binding")
    if len(expected["timestep_bias_schedule"]) != settings.get("sampling_steps"):
        raise ValueError("R3 worker timestep schedule cardinality differs from inference settings")
    expected_cardinality = evidence.get("expected", {})
    if len(expected["blocks_bias_schedule"]) != expected_cardinality.get("num_layers"):
        raise ValueError("R3 worker block schedule cardinality differs from expected layers")
    validate_backend_request(expected, requested.get("attention_backend"))
    return True


def _dispatch_coordinates(expected_ranks, sampling_steps, num_layers):
    return {
        (rank, step, branch, block)
        for rank in expected_ranks
        for step in range(sampling_steps)
        for branch in ("conditional", "negative")
        for block in range(num_layers)
    }


def _validate_pre_model_dispatches(records, expected_ranks, expected_backend_version, require_masks):
    auxiliary = [item for item in records if item.get("event") == "pre-model-attention-dispatch"]
    if not auxiliary:
        return
    if not require_masks:
        raise ValueError("pre-model attention is not part of the pristine route")
    for rank in expected_ranks:
        stream = [item for item in records if item["rank"] == rank]
        initial = next(item["ordinal"] for item in stream if item["event"] == "initial-latent")
        scoped = [item["ordinal"] for item in stream if item["event"] == "attention-dispatch"]
        if not scoped:
            raise ValueError("pre-model attention lacks subsequent model dispatch")
        for item in stream:
            if item.get("event") != "pre-model-attention-dispatch":
                continue
            if (set(item) != {"event", "ordinal", "rank", "backend", "backend_version", "scope"}
                    or item["scope"] != "pre-model"
                    or not initial < item["ordinal"] < min(scoped)
                    or item["backend"] != "flash_attention_2"
                    or item["backend_version"] != expected_backend_version):
                raise ValueError("pre-model attention is malformed, unbound, or outside its scope")


def _validate_dispatches(
    records,
    expected_ranks,
    sampling_steps,
    num_layers,
    requested_backend,
    expected_backend_version,
    dispatch_contract,
):
    dispatches = [item for item in records if item.get("event") == "attention-dispatch"]
    expected = _dispatch_coordinates(expected_ranks, sampling_steps, num_layers)
    actual = {
        (item.get("rank"), item.get("step"), item.get("branch"), item.get("block"))
        for item in dispatches
    }
    if actual != expected or any(
        item.get("attention_site") not in {"self", "cross"}
        or not isinstance(item.get("backend"), str)
        or not item["backend"]
        or not isinstance(item.get("backend_version"), str)
        or not item["backend_version"]
        for item in dispatches
    ):
        raise ValueError(
            "attention dispatch evidence lacks complete runtime coordinates, sites, or versions"
        )
    self_dispatches = [
        item for item in dispatches if item["attention_site"] == "self"
    ]
    self_coordinates = [
        (item["rank"], item["step"], item["branch"], item["block"])
        for item in self_dispatches
    ]
    if len(self_coordinates) != len(set(self_coordinates)) or set(
        self_coordinates
    ) != expected:
        raise ValueError(
            "self-attention dispatch coordinates are incomplete or duplicated"
        )
    if dispatch_contract is None:
        target_coordinates = {
            (item["rank"], item["step"], item["block"])
            for item in self_dispatches
            if item["branch"] == "conditional"
            and item["backend"] == requested_backend
        }
        expected_targets = {
            (rank, step, block)
            for rank in expected_ranks
            for step in range(sampling_steps)
            for block in range(num_layers)
        }
        if target_coordinates != expected_targets:
            raise ValueError("requested attention backend was not actually dispatched at every rank/step/block")
        if expected_backend_version is not None and any(
            item["backend_version"] != expected_backend_version
            for item in self_dispatches
            if item["branch"] == "conditional"
            and item["backend"] == requested_backend
        ):
            raise ValueError(
                "requested attention backend version differs from the protocol declaration"
            )
        return

    validate_backend_request(dispatch_contract, requested_backend)
    versions = dispatch_contract["backend_versions"]
    if any(not isinstance(value, str) or not value for value in versions.values()):
        raise ValueError("R3 dispatch backend versions must be declared before validation")
    for item in self_dispatches:
        expected_backend = expected_self_attention_backend(
            dispatch_contract,
            step=item["step"],
            branch=item["branch"],
            block=item["block"],
        )
        if item["backend"] != expected_backend:
            raise ValueError("self-attention dispatch differs from the trusted route and schedule")
        if item["backend_version"] != versions[expected_backend]:
            raise ValueError("self-attention dispatch version differs from the protocol declaration")
    if any(
        item["backend"] not in versions
        or item["backend_version"] != versions[item["backend"]]
        for item in dispatches
    ):
        raise ValueError("attention dispatch backend or version is undeclared or mismatched")


def _validate_trackers(records, expected, mask_configuration):
    coordinates = [
        (item.get("rank"), item.get("step"), item.get("branch"), item.get("block"))
        for item in records
        if item.get("event") == "tracker-call"
    ]
    if mask_configuration == "fixed:fixed":
        if coordinates:
            raise ValueError("fixed masks recorded a tracker call")
        return
    if mask_configuration not in {"hard:dynamic", "soft:dynamic"}:
        raise ValueError("source-hook mask configuration is unsupported")
    if len(coordinates) != len(set(coordinates)) or set(coordinates) != expected:
        raise ValueError("dynamic tracker coordinates are incomplete or duplicated")


def _consumer_map(records, event, expected):
    selected = [
        item
        for item in records
        if item.get("event") == event and item.get("branch") == "conditional"
    ]
    coordinates = [
        (item.get("rank"), item.get("step"), item.get("branch"), item.get("block"))
        for item in selected
    ]
    if len(coordinates) != len(set(coordinates)) or set(coordinates) != expected:
        raise ValueError(f"{event} coordinates are incomplete or duplicated")
    if any(
        item.get("used_mask_finite") is not True
        or not _valid_tensor_identity(item.get("used_mask_identity"))
        for item in selected
    ):
        raise ValueError(f"{event} contains missing or nonfinite masks")
    if event == "self-mask-consumer" and any(
        item.get("generated_mask_finite") is not True
        or not _valid_tensor_identity(item.get("generated_mask_identity"))
        for item in selected
    ):
        raise ValueError("generated mask evidence is missing or nonfinite")
    return {
        coordinate: item["used_mask_identity"]
        for coordinate, item in zip(coordinates, selected, strict=True)
    }


def _validate_mask_consumers(records, expected_ranks, sampling_steps, num_layers):
    expected = {
        (rank, step, "conditional", block)
        for rank in expected_ranks
        for step in range(sampling_steps)
        for block in range(num_layers)
    }
    self_masks = _consumer_map(records, "self-mask-consumer", expected)
    cross_masks = _consumer_map(records, "cross-mask-consumer", expected)
    if self_masks != cross_masks:
        raise ValueError("self/cross used-mask identities differ")
    per_step_block = {}
    for (rank, step, branch, block), identity in self_masks.items():
        del rank, branch
        per_step_block.setdefault((step, block), set()).add(identity["sha256"])
    if any(len(hashes) != 1 for hashes in per_step_block.values()):
        raise ValueError("used-mask identity differs across ranks")


def validate_source_observations(
    records,
    *,
    rank_count,
    sampling_steps,
    num_layers,
    requested_backend,
    expected_seed,
    expected_backend_version=None,
    require_masks=True,
    mask_configuration=None,
    dispatch_contract=None,
    pre_model_backend_version=None,
):
    """Validate lifecycle and exact source coordinates without claiming GPU success."""
    cardinalities = (rank_count, sampling_steps, num_layers)
    if (
        not isinstance(records, list)
        or not records
        or any(type(value) is not int or value <= 0 for value in cardinalities)
    ):
        raise ValueError("source-hook observations or expected cardinalities are invalid")
    if not isinstance(requested_backend, str) or not requested_backend:
        raise ValueError("source-hook requested backend is missing")
    if expected_backend_version is not None and (
        not isinstance(expected_backend_version, str) or not expected_backend_version
    ):
        raise ValueError("source-hook expected backend version is invalid")
    # Pre-model attention always runs on flash, so a flex route binds it to the flash version.
    if pre_model_backend_version is None:
        pre_model_backend_version = expected_backend_version
    elif not isinstance(pre_model_backend_version, str) or not pre_model_backend_version:
        raise ValueError("source-hook pre-model backend version is invalid")
    if type(expected_seed) is not int:
        raise ValueError("source-hook expected seed is invalid")
    if dispatch_contract is not None:
        rebuilt = build_self_attention_dispatch_contract(**dispatch_contract)
        if rebuilt != dispatch_contract:
            raise ValueError("source-hook dispatch contract is not canonical")
        if len(dispatch_contract["timestep_bias_schedule"]) != sampling_steps or len(
            dispatch_contract["blocks_bias_schedule"]
        ) != num_layers:
            raise ValueError("source-hook schedules differ from expected cardinalities")

    expected_ranks = set(range(rank_count))
    _validate_gathered_stream_order(records, expected_ranks)
    _validate_lifecycle(records, expected_ranks, expected_seed)
    _validate_cfg(records, expected_ranks, sampling_steps)
    _validate_pre_model_dispatches(records, expected_ranks, pre_model_backend_version, require_masks)
    _validate_dispatches(
        records,
        expected_ranks,
        sampling_steps,
        num_layers,
        requested_backend,
        expected_backend_version,
        dispatch_contract,
    )
    if require_masks:
        expected_dispatches = _dispatch_coordinates(
            expected_ranks, sampling_steps, num_layers
        )
        _validate_trackers(records, expected_dispatches, mask_configuration)
        _validate_mask_consumers(
            records, expected_ranks, sampling_steps, num_layers
        )
    return True
