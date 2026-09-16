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


def _dispatch_coordinates(expected_ranks, sampling_steps, num_layers):
    return {
        (rank, step, branch, block)
        for rank in expected_ranks
        for step in range(sampling_steps)
        for branch in ("conditional", "negative")
        for block in range(num_layers)
    }


def _validate_dispatches(
    records,
    expected_ranks,
    sampling_steps,
    num_layers,
    requested_backend,
    expected_backend_version,
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
    if type(expected_seed) is not int:
        raise ValueError("source-hook expected seed is invalid")

    expected_ranks = set(range(rank_count))
    _validate_gathered_stream_order(records, expected_ranks)
    _validate_lifecycle(records, expected_ranks, expected_seed)
    _validate_cfg(records, expected_ranks, sampling_steps)
    _validate_dispatches(
        records,
        expected_ranks,
        sampling_steps,
        num_layers,
        requested_backend,
        expected_backend_version,
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
