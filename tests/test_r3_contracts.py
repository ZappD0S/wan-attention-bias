import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from multi_sample_inference import experiment_pipeline as pipeline
from multi_sample_inference.r3_contracts import (
    CPU_FIXTURE_EVIDENCE,
    RUNTIME_RECORD_KIND,
    build_worker_observation,
    canonical_json_bytes,
    enumerate_matrix_cases,
    load_protocol_bundle,
    validate_runtime_evidence,
    validate_worker_observation,
    write_immutable_json,
)
from multi_sample_inference.r3_preflight import preflight

ROOT = Path(__file__).parents[1]
PROTOCOL = ROOT / "docs/r3_protocol.json"
MATRIX = ROOT / "docs/r3_test_matrix.json"
PROTOCOL_V2 = ROOT / "docs/r3_protocol_v2.json"
MATRIX_V2 = ROOT / "docs/r3_test_matrix_v2.json"
SOURCE = ROOT / "tests/fixtures/smoke_experiment.json"
DIGEST = "a" * 64


def _binding():
    return {
        "path": "/tmp/not-written-by-unit-test.json",
        "bindings": {
            "protocol_sha256": DIGEST,
            "matrix_sha256": "b" * 64,
            "manifest_sha256": "c" * 64,
            "checkpoint_content_sha256": "d" * 64,
            "source_sha256": "e" * 64,
            "config_sha256": "f" * 64,
        },
        "case_id": "custom-generator-test",
        "expected": {"rank_count": 1, "sampling_steps": 2, "num_layers": 40},
        "requested": {
            "method": "none",
            "mask_configuration": "fixed:fixed",
            "mask_sharing": "current",
            "self_attention_masking": False,
            "solver": "unipc",
            "attention_backend": "flash_attention_2",
            "rank_mode": "single",
            "diffusion_seed": 101,
            "cfg": 5.0,
            "execution_route": "custom",
        },
    }


def _runtime_fixture(bundle, case, *, rank_count=1, sampling_steps=40, num_layers=40):
    expected_job = {
        "bindings": {
            "manifest_sha256": "1" * 64,
            "checkpoint_content_sha256": "2" * 64,
            "source_sha256": "3" * 64,
            "config_sha256": "4" * 64,
        },
        "case_id": case["case_id"],
        "expected": {
            "rank_count": rank_count,
            "sampling_steps": sampling_steps,
            "num_layers": num_layers,
        },
    }
    record = {
        "schema_version": 1,
        "record_kind": RUNTIME_RECORD_KIND,
        "evidence_class": CPU_FIXTURE_EVIDENCE,
        "bindings": {
            "protocol_sha256": bundle["protocol_sha256"],
            "matrix_sha256": bundle["matrix_sha256"],
            **expected_job["bindings"],
            "environment_sha256": "5" * 64,
        },
        "case_id": case["case_id"],
        "requested": case["selection"],
        "expected": expected_job["expected"],
        "checks": {
            check_id: {"status": "not-run", "evidence": None}
            for check_id in case["required_observations"]
        },
        "r3_acceptance": False,
    }
    return record, expected_job


def _declared_protocol_bundle(tmp_path):
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_bytes(MATRIX.read_bytes())
    protocol = copy.deepcopy(json.loads(PROTOCOL.read_text()))
    protocol["matrix"]["sha256"] = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    protocol["runtime_declarations"].update(
        {
            "checkpoint_content_sha256": "2" * 64,
            "hardware_identifier": "test-hardware",
            "torch_version": "test-torch",
            "cuda_version": "test-cuda",
            "flash_attention_version": "test-fa2",
            "flex_attention_version": "test-flex",
            "intended_fsdp_rank_count": 2,
        }
    )
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    return load_protocol_bundle(protocol_path, matrix_path)


def _amended_v2_paths(tmp_path):
    matrix_path = tmp_path / "matrix-v2.json"
    matrix_path.write_bytes(MATRIX_V2.read_bytes())
    protocol = copy.deepcopy(json.loads(PROTOCOL_V2.read_text()))
    protocol["matrix"]["sha256"] = hashlib.sha256(
        matrix_path.read_bytes()
    ).hexdigest()
    protocol["approvals"] = {
        "gpu_execution": "approved",
        "hardware_environment": "approved",
    }
    protocol["runtime_declarations"].update(
        {
            "checkpoint_content_sha256": "2" * 64,
            "hardware_identifier": "synthetic-contract-test-hardware",
            "torch_version": "synthetic-torch",
            "cuda_version": "synthetic-cuda",
            "flash_attention_version": "synthetic-fa2",
            "flex_attention_version": "synthetic-flex",
            "intended_fsdp_rank_count": 2,
        }
    )
    protocol["numerical_tolerances"]["full_generator_parity"] = {
        "atol": 0.01,
        "rtol": 0.02,
        "amendment_required": False,
    }
    protocol_path = tmp_path / "protocol-v2-amended.json"
    protocol_path.write_text(json.dumps(protocol))
    return protocol_path, matrix_path


def _portable_r3_source(tmp_path):
    source = copy.deepcopy(json.loads(SOURCE.read_text()))
    source["checkpoint"]["path"] = str((SOURCE.parent / source["checkpoint"]["path"]).resolve())
    source["checkpoint"]["inventory"] = str((SOURCE.parent / source["checkpoint"]["inventory"]).resolve())
    source["r3_evidence"] = {
        "protocol": str(PROTOCOL),
        "matrix": str(MATRIX),
        "attention_backend": "flash_attention_2",
        "rank_mode": "single",
    }
    for scene in source["scenes"]:
        scene["reference_image"] = str((SOURCE.parent / scene["reference_image"]).resolve())
        for actor in scene["actors"]:
            actor["isolated_image"] = str((SOURCE.parent / actor["isolated_image"]).resolve())
        scene["segmentation_masks"] = {
            actor_id: str((SOURCE.parent / path).resolve())
            for actor_id, path in scene["segmentation_masks"].items()
        }
    path = tmp_path / "source.json"
    path.write_text(json.dumps(source))
    return path


def test_frozen_protocol_enumerates_full_matrix_but_execution_is_blocked():
    bundle = load_protocol_bundle(PROTOCOL, MATRIX)
    cases = enumerate_matrix_cases(bundle["matrix"])
    assert bundle["summary"] == {"case_count": 592, "invalid_check_count": 13}
    assert len(cases) == len({case["case_id"] for case in cases})
    custom = [case for case in cases if case["family"] == "custom-generator"]
    assert {case["selection"]["attention_backend"] for case in custom} == {
        "flash_attention_2",
        "flex_attention",
    }
    fixed = [case for case in custom if case["selection"]["mask_configuration"] == "fixed:fixed"]
    assert all("fixed_mask_no_tracker" in case["required_observations"] for case in fixed)
    result = preflight(PROTOCOL, MATRIX)
    assert result["execution_ready"] is False
    assert result["gpu_or_checkpoint_access"] == "not-performed"
    with pytest.raises(RuntimeError, match="execution preflight blocked"):
        preflight(PROTOCOL, MATRIX, execution=True)


def test_v2_template_preserves_v1_lineage_and_remains_execution_blocked():
    bundle = load_protocol_bundle(PROTOCOL_V2, MATRIX_V2)
    assert bundle["summary"] == {"case_count": 592, "invalid_check_count": 13}
    assert bundle["protocol"]["lineage"] == {
        "protocol_id": "r3-gpu-contracts-v1",
        "sha256": "1b45951b65fc757615dfb91c9e07859084457a2dba34c18a4f9c4e3f83f3488b",
    }
    result = preflight(PROTOCOL_V2, MATRIX_V2)
    assert result["schema_version"] == 2
    assert result["execution_ready"] is False
    with pytest.raises(RuntimeError, match="execution preflight blocked"):
        preflight(PROTOCOL_V2, MATRIX_V2, execution=True)


def test_amended_v2_can_authorize_contract_validation_before_evidence_acceptance(
    tmp_path,
):
    protocol_path, matrix_path = _amended_v2_paths(tmp_path)
    result = preflight(protocol_path, matrix_path, execution=True)
    assert result["execution_ready"] is True
    assert result["contract_validation_execution_ready"] is True
    assert result["evidence_acceptance_ready"] is False
    assert result["blockers"] == []
    assert all(
        blocker.startswith("gpu-validation:runtime-hook:")
        for blocker in result["evidence_acceptance_blockers"]
    )
    assert len(result["evidence_acceptance_blockers"]) == 6
    assert result["gpu_or_checkpoint_access"] == "not-performed"


def test_protocol_rejects_malformed_checkpoint_digest(tmp_path):
    protocol = json.loads(PROTOCOL_V2.read_text())
    protocol["runtime_declarations"]["checkpoint_content_sha256"] = "not-a-digest"
    protocol_path = tmp_path / "malformed-checkpoint-protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="checkpoint_content_sha256"):
        load_protocol_bundle(protocol_path, MATRIX_V2)


def test_source_expansion_rejects_protocol_checkpoint_mismatch(tmp_path):
    protocol_path, matrix_path = _amended_v2_paths(tmp_path)
    source_path = _portable_r3_source(tmp_path)
    source = json.loads(source_path.read_text())
    source["r3_evidence"]["protocol"] = str(protocol_path)
    source["r3_evidence"]["matrix"] = str(matrix_path)
    source_path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="checkpoint declaration differs"):
        pipeline.expand_source(source_path, tmp_path / "mismatch", write=False)


def test_v2_parity_pair_binds_distinct_upstream_and_custom_none_jobs(tmp_path):
    source_path = _portable_r3_source(tmp_path)
    source = json.loads(source_path.read_text())
    custom = copy.deepcopy(source["conditions"][0])
    custom.update(
        {"id": "custom-none-pair", "method": "none", "prompt_representation": "joint"}
    )
    upstream = copy.deepcopy(custom)
    upstream.update({"id": "upstream-pair", "method": "upstream"})
    source["conditions"] = [custom, upstream]
    source["inference"]["frame_num"] = 81
    source["inference"]["negative_prompt"] = "frozen negative prompt"
    source["r3_evidence"] = {
        "protocol": str(PROTOCOL_V2),
        "matrix": str(MATRIX_V2),
        "attention_backend": "flash_attention_2",
        "rank_mode": "single",
        "parity_pair": {
            "pair_id": "pair-one",
            "upstream_condition_id": "upstream-pair",
            "custom_none_condition_id": "custom-none-pair",
        },
    }
    source_path.write_text(json.dumps(source))
    manifests = pipeline.expand_source(source_path, tmp_path / "v2", write=False)
    pair_manifests = [item for item in manifests if "parity_artifact" in item["r3_evidence"]]
    assert {item["r3_evidence"]["parity_artifact"]["route"] for item in pair_manifests} == {
        "upstream",
        "custom-none",
    }
    tasks = [pipeline.worker_task_blueprint(item) for item in pair_manifests]
    assert all(task["r3_evidence"]["source_observation_path"] for task in tasks)
    assert all(
        task["r3_evidence"]["parity_artifact"]["evidence_class"]
        == "genuine-checkpoint-bound-gpu-runtime"
        for task in tasks
    )
    assert len(
        {
            task["r3_evidence"]["parity_artifact"]["bindings"]["route_source_sha256"]
            for task in tasks
        }
    ) == 2
    assert all(
        task["r3_evidence"]["parity_artifact"]["bindings"]["environment_sha256"]
        is None
        for task in tasks
    )


def test_rank_zero_worker_observation_records_only_returned_facts(tmp_path):
    binding = _binding()
    masks = np.zeros((1, 2, 40, 2, 1, 1, 1), dtype=np.float32)
    generated = np.ones_like(masks)
    record = build_worker_observation(
        binding,
        {
            "simil_masks": masks,
            "used_simil_masks": masks,
            "generated_simil_masks": generated,
            "diffusion_seed": 101,
        },
        observed_rank=0,
    )
    assert record["r3_acceptance"] is False
    assert "initial_latent_identity_by_rank" in record["unobserved_execution_blockers"]
    assert validate_worker_observation(record, binding)
    path = tmp_path / "observation.json"
    assert write_immutable_json(path, record)
    assert write_immutable_json(path, record) is False
    changed = copy.deepcopy(record)
    changed["observed"]["diffusion_seed"] = 102
    with pytest.raises(FileExistsError, match="immutable R3 evidence conflict"):
        write_immutable_json(path, changed)


def test_worker_observation_rejects_partial_mismatched_or_nonfinite_masks():
    binding = _binding()
    wrong_steps = np.zeros((1, 1, 40, 2, 1, 1, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="cardinality"):
        build_worker_observation(
            binding,
            {
                "simil_masks": wrong_steps,
                "used_simil_masks": wrong_steps,
                "generated_simil_masks": wrong_steps,
                "diffusion_seed": 101,
            },
            observed_rank=0,
        )
    for malformed in (
        np.zeros((1, 2, 40), dtype=np.float32),
        np.zeros((1, 2, 40, 0), dtype=np.float32),
        np.zeros((1, 2, 40, 2, 1, 0, 1), dtype=np.float32),
    ):
        with pytest.raises(ValueError, match=r"positive \[batch, step, block"):
            build_worker_observation(
                binding,
                {
                    "simil_masks": malformed,
                    "used_simil_masks": malformed,
                    "generated_simil_masks": malformed,
                    "diffusion_seed": 101,
                },
                observed_rank=0,
            )
    masks = np.zeros((1, 2, 40, 2, 1, 1, 1), dtype=np.float32)
    generated = masks.copy()
    generated[0, 0, 0, 0, 0, 0, 0] = np.inf
    record = build_worker_observation(
        binding,
        {
            "simil_masks": masks,
            "used_simil_masks": masks,
            "generated_simil_masks": generated,
            "diffusion_seed": 101,
        },
        observed_rank=0,
    )
    with pytest.raises(ValueError, match="nonfinite"):
        validate_worker_observation(record, binding)


def test_cpu_fixture_and_not_run_checks_cannot_satisfy_runtime_acceptance():
    bundle = load_protocol_bundle(PROTOCOL, MATRIX)
    case = next(
        case
        for case in enumerate_matrix_cases(bundle["matrix"])
        if case["family"] == "upstream-generator"
        and case["selection"]["rank_mode"] == "single"
    )
    record, expected_job = _runtime_fixture(bundle, case)
    assert validate_runtime_evidence(record, bundle=bundle, expected_job=expected_job)
    with pytest.raises(ValueError, match="CPU fixtures cannot satisfy"):
        validate_runtime_evidence(
            record,
            bundle=bundle,
            expected_job=expected_job,
            accept=True,
        )


def test_runtime_evidence_rejects_duplicate_rank_step_block_records():
    bundle = load_protocol_bundle(PROTOCOL, MATRIX)
    case = next(
        case
        for case in enumerate_matrix_cases(bundle["matrix"])
        if case["family"] == "custom-generator"
        and case["selection"]["mask_configuration"] == "hard:dynamic"
    )
    identity = {"shape": [1], "dtype": "float32", "sha256": "5" * 64}
    mask_item = {
        "rank": 0,
        "step": 0,
        "block": 0,
        "used": identity,
        "generated": identity,
        "self_used_sha256": identity["sha256"],
        "cross_used_sha256": identity["sha256"],
        "finite": True,
    }
    checks = {
        check_id: {"status": "not-run", "evidence": None}
        for check_id in case["required_observations"]
    }
    checks["initial_latent_rank_identity"] = {
        "status": "passed",
        "evidence": {"rank_identities": [{"rank": 0, "identity": identity}]},
    }
    checks["used_generated_mask_identity"] = {
        "status": "passed",
        "evidence": {"records": [mask_item, copy.deepcopy(mask_item)]},
    }
    record, expected_job = _runtime_fixture(
        bundle, case, sampling_steps=1, num_layers=2
    )
    record["checks"] = checks
    with pytest.raises(ValueError, match="duplicated"):
        validate_runtime_evidence(record, bundle=bundle, expected_job=expected_job)


def test_runtime_evidence_is_bound_to_protocol_matrix_and_trusted_job():
    bundle = load_protocol_bundle(PROTOCOL, MATRIX)
    case = next(
        case
        for case in enumerate_matrix_cases(bundle["matrix"])
        if case["family"] == "upstream-generator"
        and case["selection"]["rank_mode"] == "single"
    )
    record, expected_job = _runtime_fixture(bundle, case)

    unbound_bundle = copy.deepcopy(bundle)
    unbound_bundle["protocol"]["status"] = "completed"
    with pytest.raises(ValueError, match="not bound to its validated files"):
        validate_runtime_evidence(
            record, bundle=unbound_bundle, expected_job=expected_job
        )

    wrong_case = copy.deepcopy(expected_job)
    wrong_case["case_id"] = "missing-from-validated-matrix"
    with pytest.raises(ValueError, match="validated matrix"):
        validate_runtime_evidence(record, bundle=bundle, expected_job=wrong_case)

    wrong_config = copy.deepcopy(record)
    wrong_config["bindings"]["config_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="config_sha256 differs"):
        validate_runtime_evidence(
            wrong_config, bundle=bundle, expected_job=expected_job
        )

    reduced = copy.deepcopy(record)
    reduced["expected"]["num_layers"] = 1
    with pytest.raises(ValueError, match="cardinalities differ"):
        validate_runtime_evidence(reduced, bundle=bundle, expected_job=expected_job)


def test_runtime_evidence_rejects_protocol_checkpoint_environment_and_rank_mismatch(
    tmp_path,
):
    bundle = _declared_protocol_bundle(tmp_path)
    cases = enumerate_matrix_cases(bundle["matrix"])
    single = next(
        case
        for case in cases
        if case["family"] == "upstream-generator"
        and case["selection"]["attention_backend"] == "flash_attention_2"
        and case["selection"]["rank_mode"] == "single"
    )
    record, expected_job = _runtime_fixture(bundle, single)

    wrong_checkpoint_job = copy.deepcopy(expected_job)
    wrong_checkpoint_job["bindings"]["checkpoint_content_sha256"] = "9" * 64
    wrong_checkpoint_record = copy.deepcopy(record)
    wrong_checkpoint_record["bindings"]["checkpoint_content_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="checkpoint differs from the protocol"):
        validate_runtime_evidence(
            wrong_checkpoint_record,
            bundle=bundle,
            expected_job=wrong_checkpoint_job,
        )

    environment = {
        "hardware_identifier": "wrong-hardware",
        "torch_version": "test-torch",
        "cuda_version": "test-cuda",
        "attention_backend": "flash_attention_2",
        "backend_version": "test-fa2",
    }
    record["checks"]["environment_binding"] = {
        "status": "passed",
        "evidence": environment,
    }
    record["bindings"]["environment_sha256"] = hashlib.sha256(
        canonical_json_bytes(environment)
    ).hexdigest()
    with pytest.raises(ValueError, match="environment differs from the protocol"):
        validate_runtime_evidence(record, bundle=bundle, expected_job=expected_job)

    intended = next(
        case
        for case in cases
        if case["family"] == "upstream-generator"
        and case["selection"]["rank_mode"] == "intended_fsdp"
    )
    intended_record, intended_job = _runtime_fixture(bundle, intended, rank_count=1)
    with pytest.raises(ValueError, match="rank mode differs"):
        validate_runtime_evidence(
            intended_record, bundle=bundle, expected_job=intended_job
        )


@pytest.mark.parametrize(
    "claim",
    [
        {"finite": True, "passed": True, "atol": 0.0, "rtol": 0.0},
        {
            "upstream_input_sha256": "6" * 64,
            "custom_input_sha256": "7" * 64,
            "upstream_output_sha256": "8" * 64,
            "custom_output_sha256": "8" * 64,
            "maximum_absolute_error": 0.0,
            "maximum_relative_error": 0.0,
        },
        {
            "upstream_input_sha256": "6" * 64,
            "custom_input_sha256": "6" * 64,
            "upstream_output_sha256": "7" * 64,
            "custom_output_sha256": "8" * 64,
            "maximum_absolute_error": 1.0,
            "maximum_relative_error": 1.0,
            "atol": 0.0,
            "rtol": 0.0,
        },
    ],
    ids=["missing-pair", "mismatched-pair", "out-of-tolerance"],
)
def test_runtime_parity_claims_fail_closed_until_paired_schema_exists(claim):
    bundle = load_protocol_bundle(PROTOCOL, MATRIX)
    case = next(
        case
        for case in enumerate_matrix_cases(bundle["matrix"])
        if case["family"] == "upstream-custom-none-parity"
        and case["selection"]["rank_mode"] == "single"
    )
    record, expected_job = _runtime_fixture(bundle, case)
    record["checks"]["full_generator_parity"] = {
        "status": "passed",
        "evidence": claim,
    }
    with pytest.raises(ValueError, match="parity acceptance is unavailable"):
        validate_runtime_evidence(record, bundle=bundle, expected_job=expected_job)


def test_manifest_r3_declaration_is_optional_but_fail_closed_when_present(tmp_path):
    legacy = pipeline.expand_source(SOURCE, tmp_path / "legacy", write=False)[0]
    assert "r3_evidence" not in legacy
    assert "r3_worker_observation_path" not in legacy["outputs"]

    null_source = json.loads(SOURCE.read_text())
    null_source["r3_evidence"] = None
    with pytest.raises(ValueError, match="r3_evidence must declare"):
        pipeline._validate_source(null_source, SOURCE)

    source_path = _portable_r3_source(tmp_path)
    manifest = pipeline.expand_source(source_path, tmp_path / "r3", write=False)[0]
    assert manifest["r3_evidence"]["worker_observation_required"] is True
    task = pipeline.worker_task_blueprint(manifest)
    assert task["r3_evidence"]["bindings"]["manifest_sha256"] == pipeline._manifest_hash(manifest)
    with pytest.raises(ValueError, match="observation is missing"):
        pipeline._verify_declared_r3_worker_evidence(manifest)

    manifest["smoke_only"] = False
    manifest_path = tmp_path / "r3-job.json"
    manifest_path.write_bytes(pipeline._canonical(manifest))
    with pytest.raises(ValueError, match="R3 execution preflight blocked"):
        pipeline.execute_job(manifest_path)


def test_source_expansion_rejects_present_null_parity_pair_but_allows_absence(
    tmp_path,
):
    source_path = _portable_r3_source(tmp_path)
    source = json.loads(source_path.read_text())
    source["r3_evidence"]["protocol"] = str(PROTOCOL_V2)
    source["r3_evidence"]["matrix"] = str(MATRIX_V2)
    source["r3_evidence"]["parity_pair"] = None
    source_path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="parity_pair requires a valid v2 declaration"):
        pipeline.expand_source(source_path, tmp_path / "present-null", write=False)

    del source["r3_evidence"]["parity_pair"]
    source_path.write_text(json.dumps(source))
    manifests = pipeline.expand_source(source_path, tmp_path / "absent", write=False)
    assert manifests
    assert all("parity_artifact" not in item["r3_evidence"] for item in manifests)


@pytest.mark.parametrize(
    "tamper",
    [
        "missing-observation",
        "mutated-observation",
        "substituted-observation",
        "missing-result-evidence",
        "mutated-result-evidence",
        "substituted-status",
    ],
)
def test_completed_r3_job_rejects_evidence_and_status_tampering(tmp_path, tamper):
    source_path = _portable_r3_source(tmp_path)
    manifest = pipeline.expand_source(source_path, tmp_path / "run", write=False)[0]
    binding = pipeline.worker_task_blueprint(manifest)["r3_evidence"]
    masks = np.zeros((1, 2, 40, 2, 1, 1, 1), dtype=np.float32)
    observation = build_worker_observation(
        binding,
        {
            "simil_masks": masks,
            "used_simil_masks": masks,
            "generated_simil_masks": np.ones_like(masks),
            "diffusion_seed": binding["requested"]["diffusion_seed"],
        },
        observed_rank=0,
    )
    observation_path = Path(manifest["outputs"]["r3_worker_observation_path"])
    observation_path.parent.mkdir(parents=True, exist_ok=True)
    observation_path.write_bytes(pipeline._canonical(observation))
    video_path = Path(manifest["outputs"]["video_path"])
    video_path.write_bytes(b"video")
    declared = pipeline._verify_declared_r3_worker_evidence(manifest)
    attempt_id = "attempt-one"
    result = {
        "schema_version": 1,
        "job_id": manifest["job_id"],
        "manifest_sha256": pipeline._manifest_hash(manifest),
        "attempt_id": attempt_id,
        "state": "completed",
        "video_path": str(video_path),
        "video_sha256": pipeline._hash_file(video_path),
        "r3_worker_observation": declared,
    }
    result_path = Path(manifest["outputs"]["result_path"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_bytes(pipeline._canonical(result))
    event = {
        "schema_version": 1,
        "job_id": manifest["job_id"],
        "manifest_sha256": pipeline._manifest_hash(manifest),
        "attempt_id": attempt_id,
        "state": "completed",
        "timestamp": "2026-09-15T00:00:00+00:00",
        "video_sha256": pipeline._hash_file(video_path),
    }
    pipeline._append_status(manifest["outputs"]["status_path"], event)
    assert pipeline.job_status(manifest) == "completed"

    if tamper == "missing-observation":
        observation_path.unlink()
    elif tamper == "mutated-observation":
        observation["observed"]["used_masks"]["sha256"] = "9" * 64
        observation_path.write_bytes(pipeline._canonical(observation))
    elif tamper == "substituted-observation":
        observation["bindings"]["manifest_sha256"] = "9" * 64
        observation_path.write_bytes(pipeline._canonical(observation))
    elif tamper == "missing-result-evidence":
        result.pop("r3_worker_observation")
        result_path.write_bytes(pipeline._canonical(result))
    elif tamper == "mutated-result-evidence":
        result["r3_worker_observation"]["worker_observation"]["sha256"] = "9" * 64
        result_path.write_bytes(pipeline._canonical(result))
    else:
        status_path = Path(manifest["outputs"]["status_path"])
        event["attempt_id"] = "substituted-attempt"
        status_path.write_bytes(pipeline._canonical(event))
    assert pipeline.job_status(manifest) == "invalid-completed"


def test_intended_fsdp_manifest_is_blocked_until_rank_count_is_declared(tmp_path):
    source_path = _portable_r3_source(tmp_path)
    source = json.loads(source_path.read_text())
    source["r3_evidence"]["rank_mode"] = "intended_fsdp"
    source["inference"]["rank_count"] = 2
    source_path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="rank count is undeclared"):
        pipeline.validate_source(source_path)
