import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import multi_sample_inference.r3_runtime as runtime
from multi_sample_inference.r3_parity import (
    compare_parity_artifacts,
    validate_parity_artifact,
    write_parity_tensor_artifact,
)
from multi_sample_inference.r3_runtime import (
    R3RuntimeCollector,
    gather_rank_observations,
)
from multi_sample_inference.r3_runtime import (
    validate_source_observations as _validate_source_observations,
)

ROOT = Path(__file__).parents[1]


def validate_source_observations(records, **kwargs):
    kwargs.setdefault("expected_seed", 101)
    return _validate_source_observations(records, **kwargs)


def _load_source_observer_module():
    path = ROOT / "wan2.1/wan/utils/runtime_evidence.py"
    spec = importlib.util.spec_from_file_location("wan_runtime_evidence", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_records(
    *, tracker=False, rank=0, seed=101, parity_artifact=None, final_tensor=None
):
    hooks = _load_source_observer_module()
    collector = R3RuntimeCollector(parity_artifact=parity_artifact)
    tensor = np.ones((1, 2), dtype=np.float32)
    with hooks.install_runtime_observer(collector, rank=rank):
        hooks.emit_runtime_observation(
            "initial-latent", seed=seed, latent_tensor=tensor
        )
        for branch in ("conditional", "negative"):
            with hooks.runtime_observation_scope(step=0, branch=branch, block=0):
                hooks.emit_runtime_observation(
                    "cfg-branch-output", output_tensor=tensor
                )
                hooks.emit_runtime_observation(
                    "attention-dispatch",
                    attention_site="self",
                    backend="flash_attention_2",
                    backend_version="test-fa2",
                )
                if tracker:
                    hooks.emit_runtime_observation(
                        "tracker-call", tracker="hard-similarity-mask"
                    )
                hooks.emit_runtime_observation(
                    "self-mask-consumer",
                    used_mask_tensor=tensor,
                    generated_mask_tensor=tensor,
                )
                hooks.emit_runtime_observation(
                    "cross-mask-consumer", used_mask_tensor=tensor
                )
        hooks.emit_runtime_observation(
            "final-latent",
            output_tensor=tensor if final_tensor is None else final_tensor,
        )
    return collector.snapshot()


def test_source_observer_is_noop_uninstalled_and_restores_lifecycle():
    hooks = _load_source_observer_module()
    assert hooks.emit_runtime_observation("ignored") is False
    with hooks.runtime_observation_scope(step=0):
        assert hooks.emit_runtime_observation("also-ignored") is False
    collector = R3RuntimeCollector()
    with hooks.install_runtime_observer(collector, rank=3):
        assert hooks.runtime_observer_installed()
    assert not hooks.runtime_observer_installed()
    assert [item["event"] for item in collector.records] == [
        "observer-installed",
        "observer-completed",
    ]
    assert all(item["rank"] == 3 for item in collector.records)


def test_source_observer_records_callback_errors_and_rejects_coordinate_overlap():
    hooks = _load_source_observer_module()
    collector = R3RuntimeCollector()
    with (
        pytest.raises(RuntimeError, match="callback failure"),
        hooks.install_runtime_observer(collector, rank=0),
    ):
        raise RuntimeError("callback failure")
    assert not hooks.runtime_observer_installed()
    assert [item["event"] for item in collector.records] == [
        "observer-installed",
        "observer-failed",
    ]
    assert collector.records[-1]["error_type"] == "RuntimeError"

    with (
        pytest.raises(ValueError, match="already set"),
        hooks.install_runtime_observer(R3RuntimeCollector(), rank=0),
        hooks.runtime_observation_scope(step=0),
        hooks.runtime_observation_scope(step=1),
    ):
        pass
    assert not hooks.runtime_observer_installed()


def test_collector_caches_only_within_one_event_and_validates_coordinates(monkeypatch):
    calls = 0
    original = runtime._tensor_identity_and_finiteness

    def counted(value):
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(runtime, "_tensor_identity_and_finiteness", counted)
    records = _source_records()
    assert validate_source_observations(
        records,
        rank_count=1,
        sampling_steps=1,
        num_layers=1,
        mask_configuration="fixed:fixed",
        requested_backend="flash_attention_2",
    )
    # Eight tensor-bearing events are hashed independently. The self-consumer's
    # used/generated aliases share one snapshot only within that event.
    assert calls == 8
    duplicate = copy.deepcopy(records)
    duplicate.append(
        copy.deepcopy(next(item for item in records if item["event"] == "cfg-branch-output"))
    )
    duplicate[-1]["ordinal"] = len(duplicate) - 1
    with pytest.raises(ValueError, match="incomplete or duplicated"):
        validate_source_observations(
            duplicate,
            rank_count=1,
            sampling_steps=1,
            num_layers=1,
            mask_configuration="fixed:fixed",
            requested_backend="flash_attention_2",
        )


def test_collector_detects_in_place_mutation_between_events():
    collector = R3RuntimeCollector()
    tensor = np.ones((1, 2), dtype=np.float32)
    collector(
        {
            "event": "self-mask-consumer",
            "rank": 0,
            "step": 0,
            "branch": "conditional",
            "block": 0,
            "used_mask_tensor": tensor,
            "generated_mask_tensor": tensor,
        }
    )
    tensor[...] = 2
    collector(
        {
            "event": "cross-mask-consumer",
            "rank": 0,
            "step": 0,
            "branch": "conditional",
            "block": 0,
            "used_mask_tensor": tensor,
        }
    )
    assert (
        collector.records[0]["used_mask_identity"]
        != collector.records[1]["used_mask_identity"]
    )

    collector({"event": "cfg-branch-output", "rank": 0, "output_tensor": tensor})
    tensor[...] = 3
    collector({"event": "final-latent", "rank": 0, "output_tensor": tensor})
    assert (
        collector.records[2]["output_identity"]
        != collector.records[3]["output_identity"]
    )


def test_tracker_events_prove_fixed_zero_and_dynamic_coordinate_coverage():
    fixed = _source_records(tracker=False)
    assert validate_source_observations(
        fixed,
        rank_count=1,
        sampling_steps=1,
        num_layers=1,
        mask_configuration="fixed:fixed",
        requested_backend="flash_attention_2",
    )
    dynamic = _source_records(tracker=True)
    assert validate_source_observations(
        dynamic,
        rank_count=1,
        sampling_steps=1,
        num_layers=1,
        mask_configuration="hard:dynamic",
        requested_backend="flash_attention_2",
    )
    with pytest.raises(ValueError, match="fixed masks recorded"):
        validate_source_observations(
            dynamic,
            rank_count=1,
            sampling_steps=1,
            num_layers=1,
            mask_configuration="fixed:fixed",
            requested_backend="flash_attention_2",
        )


def test_source_validation_rejects_missing_final_and_unversioned_or_wrong_dispatch():
    records = _source_records()
    missing_final = [
        copy.deepcopy(item) for item in records if item["event"] != "final-latent"
    ]
    for ordinal, item in enumerate(missing_final):
        item["ordinal"] = ordinal
    with pytest.raises(ValueError, match="final-latent rank coverage"):
        validate_source_observations(
            missing_final,
            rank_count=1,
            sampling_steps=1,
            num_layers=1,
            mask_configuration="fixed:fixed",
            requested_backend="flash_attention_2",
        )

    unversioned = copy.deepcopy(records)
    next(
        item for item in unversioned if item["event"] == "attention-dispatch"
    )["backend_version"] = None
    with pytest.raises(ValueError, match="coordinates, sites, or versions"):
        validate_source_observations(
            unversioned,
            rank_count=1,
            sampling_steps=1,
            num_layers=1,
            mask_configuration="fixed:fixed",
            requested_backend="flash_attention_2",
        )
    assert validate_source_observations(
        records,
        rank_count=1,
        sampling_steps=1,
        num_layers=1,
        mask_configuration="fixed:fixed",
        requested_backend="flash_attention_2",
        expected_backend_version="test-fa2",
    )
    with pytest.raises(ValueError, match="version differs"):
        validate_source_observations(
            records,
            rank_count=1,
            sampling_steps=1,
            num_layers=1,
            mask_configuration="fixed:fixed",
            requested_backend="flash_attention_2",
            expected_backend_version="other-fa2",
        )
    with pytest.raises(ValueError, match="was not actually dispatched"):
        validate_source_observations(
            records,
            rank_count=1,
            sampling_steps=1,
            num_layers=1,
            mask_configuration="fixed:fixed",
            requested_backend="flex_attention",
        )


def test_cross_attention_dispatch_cannot_satisfy_requested_self_backend():
    records = copy.deepcopy(_source_records())
    dispatches = [
        item for item in records if item["event"] == "attention-dispatch"
    ]
    for item in dispatches:
        item["backend"] = "flex_attention"
        item["backend_version"] = "test-flex"
        cross = copy.deepcopy(item)
        cross["attention_site"] = "cross"
        cross["backend"] = "flash_attention_2"
        cross["backend_version"] = "test-fa2"
        records.append(cross)
    for ordinal, item in enumerate(records):
        item["ordinal"] = ordinal

    with pytest.raises(ValueError, match="was not actually dispatched"):
        validate_source_observations(
            records,
            rank_count=1,
            sampling_steps=1,
            num_layers=1,
            mask_configuration="fixed:fixed",
            requested_backend="flash_attention_2",
            expected_backend_version="test-fa2",
        )
    assert validate_source_observations(
        records,
        rank_count=1,
        sampling_steps=1,
        num_layers=1,
        mask_configuration="fixed:fixed",
        requested_backend="flex_attention",
        expected_backend_version="test-flex",
    )


def test_source_validation_binds_initial_seed_on_every_rank():
    records = _source_records(seed=102)
    with pytest.raises(ValueError, match="initial-latent identities"):
        validate_source_observations(
            records,
            rank_count=1,
            sampling_steps=1,
            num_layers=1,
            mask_configuration="fixed:fixed",
            requested_backend="flash_attention_2",
        )

    streams = [_source_records(rank=0), _source_records(rank=1, seed=102)]

    def gather(_local, gathered, dst):
        assert dst == 0
        gathered[:] = streams

    gathered = gather_rank_observations(
        streams[0], rank=0, world_size=2, gather_object=gather
    )
    with pytest.raises(ValueError, match="initial-latent identities"):
        validate_source_observations(
            gathered,
            rank_count=2,
            sampling_steps=1,
            num_layers=1,
            mask_configuration="fixed:fixed",
            requested_backend="flash_attention_2",
        )


def test_two_rank_collector_gather_composes_with_source_validation():
    streams = [_source_records(rank=rank) for rank in range(2)]
    calls = []

    def other_rank_gather(local, gathered, dst):
        calls.append((local, dst))
        assert gathered is None

    assert (
        gather_rank_observations(
            streams[1], rank=1, world_size=2, gather_object=other_rank_gather
        )
        is None
    )

    def rank_zero_gather(local, gathered, dst):
        calls.append((local, dst))
        gathered[:] = streams

    gathered = gather_rank_observations(
        streams[0], rank=0, world_size=2, gather_object=rank_zero_gather
    )
    assert len(calls) == 2
    assert validate_source_observations(
        gathered,
        rank_count=2,
        sampling_steps=1,
        num_layers=1,
        mask_configuration="fixed:fixed",
        requested_backend="flash_attention_2",
    )


@pytest.mark.parametrize("fault", ["duplicate", "reordered", "wrong-rank"])
def test_rank_gather_rejects_invalid_local_streams(fault):
    streams = [_source_records(rank=rank) for rank in range(2)]
    if fault == "duplicate":
        streams[1][1]["ordinal"] = streams[1][0]["ordinal"]
    elif fault == "reordered":
        streams[1][0], streams[1][1] = streams[1][1], streams[1][0]
    else:
        streams[1][0]["rank"] = 0

    def gather(_local, gathered, dst):
        assert dst == 0
        gathered[:] = streams

    with pytest.raises(ValueError, match=r"ordinals|wrong-rank"):
        gather_rank_observations(
            streams[0], rank=0, world_size=2, gather_object=gather
        )


def test_rank_gather_rejects_invalid_rank_world_size_and_malformed_data():
    stream = _source_records(rank=0)

    def malformed(_local, gathered, dst):
        assert dst == 0
        gathered[:] = [stream, None]

    with pytest.raises(ValueError, match="malformed rank data"):
        gather_rank_observations(
            stream, rank=0, world_size=2, gather_object=malformed
        )
    for rank, world_size in ((-1, 2), (2, 2), (0, 0), (True, 2)):
        with pytest.raises(ValueError, match="invalid rank/world_size"):
            gather_rank_observations(
                stream,
                rank=rank,
                world_size=world_size,
                gather_object=malformed,
            )


def _artifact_bindings(route, *, input_sha="1" * 64):
    return {
        "input_sha256": input_sha,
        "checkpoint_content_sha256": "2" * 64,
        "source_sha256": "3" * 64,
        "environment_sha256": "4" * 64,
        "diffusion_seed": 101,
        "inference_settings_sha256": "5" * 64,
        "config_sha256": ("6" if route == "upstream" else "7") * 64,
        "route_source_sha256": ("8" if route == "upstream" else "9") * 64,
    }


def _artifact_declaration(tmp_path, route, values, *, input_sha="1" * 64):
    path = tmp_path / f"{route}.npy"
    bindings = _artifact_bindings(route, input_sha=input_sha)
    write_parity_tensor_artifact(
        path,
        np.asarray(values, dtype=np.float32),
        pair_id="pair-one",
        route=route,
        job_id=f"job-{route}",
        bindings=bindings,
    )
    return {
        "path": str(path),
        "metadata_path": str(path) + ".json",
        "pair_id": "pair-one",
        "route": route,
        "evidence_class": "non-scientific-cpu-fixture",
        "job_id": f"job-{route}",
        "bindings": bindings,
    }


def _collector_artifact_declaration(tmp_path, route, name=None):
    path = tmp_path / f"{name or route}.npy"
    return {
        "path": str(path),
        "metadata_path": str(path) + ".json",
        "pair_id": "pair-one",
        "route": route,
        "job_id": f"job-{route}",
        "evidence_class": "non-scientific-cpu-fixture",
        "bindings": _artifact_bindings(route),
    }


@pytest.mark.parametrize(
    ("dtype", "canonical_dtype"),
    [
        (torch.float16, "float16"),
        (torch.float32, "float32"),
        (torch.float64, "float64"),
    ],
)
def test_torch_final_collector_artifact_composes_with_source_validation(
    tmp_path, dtype, canonical_dtype
):
    declaration = _collector_artifact_declaration(
        tmp_path, "upstream", canonical_dtype
    )
    tensor = torch.ones((1, 2), dtype=dtype)
    records = _source_records(
        parity_artifact=declaration,
        final_tensor=tensor,
    )
    final = next(item for item in records if item["event"] == "final-latent")
    metadata, array = validate_parity_artifact(declaration)
    assert final["output_identity"] == metadata["tensor"]
    assert metadata["tensor"]["dtype"] == canonical_dtype
    assert array.dtype == np.dtype(canonical_dtype)
    assert validate_source_observations(
        records,
        rank_count=1,
        sampling_steps=1,
        num_layers=1,
        mask_configuration="fixed:fixed",
        requested_backend="flash_attention_2",
    )


def test_torch_bfloat16_parity_artifact_fails_without_lossy_conversion(tmp_path):
    declaration = _collector_artifact_declaration(tmp_path, "upstream", "bfloat16")
    collector = R3RuntimeCollector(parity_artifact=declaration)
    with pytest.raises(ValueError, match=r"does not support PyTorch dtype torch\.bfloat16"):
        collector(
            {
                "event": "final-latent",
                "rank": 0,
                "output_tensor": torch.ones((1, 2), dtype=torch.bfloat16),
            }
        )
    assert not Path(declaration["path"]).exists()


def test_torch_parity_pair_compares_and_rejects_tampered_artifact(tmp_path):
    declarations = {
        route: _collector_artifact_declaration(tmp_path, route)
        for route in ("upstream", "custom-none")
    }
    values = {
        "upstream": torch.tensor([1.0, 2.0], dtype=torch.float32),
        "custom-none": torch.tensor([1.0, 2.01], dtype=torch.float32),
    }
    for route, declaration in declarations.items():
        R3RuntimeCollector(parity_artifact=declaration)(
            {
                "event": "final-latent",
                "rank": 0,
                "output_tensor": values[route],
            }
        )
    pair = {"pair_id": "pair-one", "artifacts": declarations}
    comparison = compare_parity_artifacts(pair, atol=0.02, rtol=0.0)
    assert comparison["evidence_class"] == "non-scientific-cpu-fixture"
    assert comparison["measurements"]["passed"] is True
    assert comparison["r3_acceptance"] is False

    Path(declarations["custom-none"]["path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed after declaration"):
        compare_parity_artifacts(pair, atol=0.02, rtol=0.0)


def test_parity_collector_stream_validates_finiteness_and_rejects_nonfinite(tmp_path):
    def declaration(name):
        return _collector_artifact_declaration(tmp_path, "upstream", name)

    finite_records = _source_records(parity_artifact=declaration("finite"))
    final = next(item for item in finite_records if item["event"] == "final-latent")
    assert final["output_finite"] is True
    assert validate_source_observations(
        finite_records,
        rank_count=1,
        sampling_steps=1,
        num_layers=1,
        mask_configuration="fixed:fixed",
        requested_backend="flash_attention_2",
    )

    nonfinite_records = _source_records(
        parity_artifact=declaration("nonfinite"),
        final_tensor=np.asarray([[np.inf, 1.0]], dtype=np.float32),
    )
    final = next(item for item in nonfinite_records if item["event"] == "final-latent")
    assert final["output_finite"] is False
    with pytest.raises(ValueError, match="final-latent identities are missing or nonfinite"):
        validate_source_observations(
            nonfinite_records,
            rank_count=1,
            sampling_steps=1,
            num_layers=1,
            mask_configuration="fixed:fixed",
            requested_backend="flash_attention_2",
        )


def test_offline_parity_comparator_uses_paired_artifacts_and_measured_tolerance(tmp_path):
    pair = {
        "pair_id": "pair-one",
        "artifacts": {
            "upstream": _artifact_declaration(tmp_path, "upstream", [1.0, 2.0]),
            "custom-none": _artifact_declaration(
                tmp_path, "custom-none", [1.0, 2.01]
            ),
        },
    }
    failed = compare_parity_artifacts(pair, atol=0.001, rtol=0.0, chunk_elements=1)
    assert failed["measurements"]["passed"] is False
    assert failed["measurements"]["maximum_absolute_error"] > 0.009
    assert failed["r3_acceptance"] is False
    passed = compare_parity_artifacts(pair, atol=0.02, rtol=0.0, chunk_elements=1)
    assert passed["measurements"]["passed"] is True


def test_offline_parity_uses_upstream_reference_and_json_safe_nonfinite_results(
    tmp_path,
):
    asymmetric_dir = tmp_path / "asymmetric"
    pair = {
        "pair_id": "pair-one",
        "artifacts": {
            "upstream": _artifact_declaration(asymmetric_dir, "upstream", [1.0]),
            "custom-none": _artifact_declaration(
                asymmetric_dir, "custom-none", [2.0]
            ),
        },
    }
    asymmetric = compare_parity_artifacts(pair, atol=0.0, rtol=0.6)
    assert asymmetric["measurements"]["passed"] is False
    assert asymmetric["measurements"]["maximum_relative_error"] == 1.0

    zero_dir = tmp_path / "zero-reference"
    zero_pair = {
        "pair_id": "pair-one",
        "artifacts": {
            "upstream": _artifact_declaration(zero_dir, "upstream", [0.0]),
            "custom-none": _artifact_declaration(
                zero_dir, "custom-none", [0.1]
            ),
        },
    }
    zero = compare_parity_artifacts(zero_pair, atol=0.0, rtol=1.0)
    assert zero["measurements"]["passed"] is False
    assert zero["measurements"]["maximum_relative_error"] is None

    nonfinite_dir = tmp_path / "nonfinite"
    nonfinite_pair = {
        "pair_id": "pair-one",
        "artifacts": {
            "upstream": _artifact_declaration(
                nonfinite_dir, "upstream", [np.inf]
            ),
            "custom-none": _artifact_declaration(
                nonfinite_dir, "custom-none", [np.inf]
            ),
        },
    }
    nonfinite = compare_parity_artifacts(nonfinite_pair, atol=0.0, rtol=0.0)
    assert nonfinite["measurements"] | {
        "finite": False,
        "passed": False,
        "maximum_absolute_error": None,
        "maximum_relative_error": None,
    } == nonfinite["measurements"]
    json.dumps(nonfinite, allow_nan=False)


def test_parity_artifact_rejects_complex_values(tmp_path):
    with pytest.raises(ValueError, match="real floating"):
        write_parity_tensor_artifact(
            tmp_path / "complex.npy",
            np.asarray([1 + 2j]),
            pair_id="pair-one",
            route="upstream",
            job_id="job-upstream",
            bindings=_artifact_bindings("upstream"),
        )


def test_offline_parity_rejects_mismatched_inputs_and_artifact_tampering(tmp_path):
    upstream = _artifact_declaration(tmp_path, "upstream", [1.0])
    custom = _artifact_declaration(
        tmp_path, "custom-none", [1.0], input_sha="a" * 64
    )
    pair = {
        "pair_id": "pair-one",
        "artifacts": {"upstream": upstream, "custom-none": custom},
    }
    with pytest.raises(ValueError, match="inputs differ"):
        compare_parity_artifacts(pair, atol=0.0, rtol=0.0)
    with pytest.raises(FileExistsError, match="immutable parity"):
        write_parity_tensor_artifact(
            upstream["path"],
            np.ones(1, dtype=np.float32),
            pair_id="pair-one",
            route="upstream",
            job_id="job-upstream",
            bindings=upstream["bindings"],
        )
    Path(upstream["path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed after declaration"):
        compare_parity_artifacts(pair, atol=0.0, rtol=0.0)
