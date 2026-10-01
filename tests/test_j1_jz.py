"""CPU-only contracts for the J1 Jean Zay ``r3-gpu-contracts-jz-v*`` branch."""

import copy
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from multi_sample_inference import j1_stage, r3_jz
from multi_sample_inference import r3_generator_pair as pair
from multi_sample_inference.j1_step0_probe import (
    compare_cross_architecture,
    report_final_latents,
    verify_reference_capture,
)
from multi_sample_inference.r3_backend_canary import run_backend_canary
from multi_sample_inference.r3_contracts import (
    V19_PROTOCOL_SHA256,
    execution_blockers,
    load_protocol_bundle,
    sha256_file,
    stage_execution_blockers,
    validate_protocol,
)
from multi_sample_inference.r3_divergence_probe import _Capture, _write_manifest
from multi_sample_inference.r3_preflight import validate_v4_runtime_environment
from tools import j1_freeze

ROOT = Path(__file__).parents[1]
DRAFT = ROOT / "docs/r3_protocol_jz_v1.json"
MATRIX = ROOT / "docs/r3_test_matrix_v3.json"
SLURM_ENV = {"SLURM_JOB_ID": "444270", "SLURM_JOB_PARTITION": "gpu_p5",
             "SLURM_JOB_ACCOUNT": "xvh@a100", "SLURM_JOB_NODELIST": "jean-zay-iam07"}


def _draft():
    return json.loads(DRAFT.read_text())


def _validate(protocol):
    matrix = json.loads(MATRIX.read_text())
    return validate_protocol(protocol, matrix, matrix_sha256=sha256_file(MATRIX))


def test_v19_constant_matches_frozen_file():
    assert sha256_file(ROOT / "docs/r3_protocol_v19.json") == V19_PROTOCOL_SHA256


def test_draft_validates_and_every_stage_is_blocked():
    bundle = load_protocol_bundle(DRAFT, MATRIX)
    protocol = bundle["protocol"]
    assert protocol["schema_version"] == 19
    assert protocol["lineage"] == {"protocol_id": "r3-gpu-contracts-v19", "sha256": V19_PROTOCOL_SHA256}
    environment = protocol["execution_amendment"]["environment_binding"]
    assert environment["driver_version"] == "595.71.05"
    assert "hostname" not in environment and "gpu_uuids" not in environment
    assert protocol["runtime_declarations"]["hardware_identifier"] == (
        r3_jz.node_class_hardware_identifier(environment))
    for stage in r3_jz.J1_STAGE_IDS:
        blockers = stage_execution_blockers(protocol, stage)
        assert "jz-draft-not-frozen" in blockers
        assert f"stage-not-authorized:{stage}" in blockers
        assert "explicit-user-authorization" in blockers
    assert "unknown-stage:single-rank-contract-cases" in stage_execution_blockers(
        protocol, "single-rank-contract-cases")
    assert "staged-execution-gates" in execution_blockers(protocol)


def _amendment(protocol):
    return protocol["execution_amendment"]


@pytest.mark.parametrize("change", [
    lambda p: p["lineage"].update(sha256="0" * 64),
    lambda p: p["lineage"].update(protocol_id="r3-gpu-contracts-v18"),
    lambda p: p.update(schema_version=20),
    lambda p: p.update(frozen_at="2026-10-01T00:00:00Z"),
    lambda p: p.update(approvals={"gpu_execution": "approved", "hardware_environment": "approved"}),
    lambda p: p["runtime_declarations"].update(hardware_identifier="0" * 64),
    lambda p: _amendment(p).update(authorization_state="approved-bounded-stage"),
    lambda p: _amendment(p)["environment_binding"].update(gpu_model="NVIDIA H100 80GB HBM3"),
    lambda p: _amendment(p)["environment_binding"].update(gpu_compute_capability="9.0"),
    lambda p: _amendment(p)["environment_binding"].update(driver_version="580.173.02"),
    lambda p: _amendment(p)["environment_binding"].update(hostname_pattern=".*"),
    lambda p: _amendment(p)["environment_binding"].update(visible_gpu_count=2),
    lambda p: _amendment(p)["environment_binding"].update(hostname="jean-zay-iam07"),
    lambda p: _amendment(p)["environment_binding"]["scheduler"].update(partition="gpu_p6"),
    lambda p: _amendment(p)["checkpoint_binding"].update(path="/tmp/checkpoint"),
    lambda p: _amendment(p)["checkpoint_binding"].update(content_sha256="0" * 64),
    lambda p: _amendment(p)["route_process_bindings"]["official-pristine"].update(wan_root="/tmp/wan"),
    lambda p: _amendment(p)["route_process_bindings"]["local-custom"]["checkout"].update(
        commit="0" * 40),
    lambda p: _amendment(p)["stage_gates"][0].update(authorization="approved"),
    lambda p: _amendment(p)["stage_gates"].pop(),
    lambda p: _amendment(p)["authorization_record"]["backend-kernel-canary"].update(
        output="/tmp/other.json"),
    lambda p: _amendment(p)["authorization_record"]["single-rank-generator-canary"].update(
        command="sbatch other.sh"),
    lambda p: _amendment(p)["authorization_record"]["step0-cross-architecture-probe"]["source"].update(
        sha256="0" * 64),
    lambda p: _amendment(p)["authorization_record"]["step0-cross-architecture-probe"][
        "reference_capture"]["report"].update(sha256="0" * 64),
    lambda p: _amendment(p)["source_binding"].update(parent_revision_at_freeze="a" * 40),
    lambda p: _amendment(p)["production_component_hashes"].update({"r3_jz.py": "0" * 64}),
    lambda p: _amendment(p)["production_component_hashes"].pop("j1_step0_probe.py"),
    lambda p: _amendment(p)["cross_architecture_decision"].update(relative_l2_max=0.03),
    lambda p: _amendment(p)["prerequisite_evidence"].append({"stage_id": "backend-kernel-canary"}),
    lambda p: _amendment(p).update(extra_key=True),
    lambda p: _amendment(p)["provisioning_binding"].update(nvcc_version="12.8.93"),
])
def test_draft_rejects_any_unenumerated_change(change):
    protocol = _draft()
    change(protocol)
    with pytest.raises(ValueError):
        _validate(protocol)


def test_unknown_jz_identity_is_rejected():
    protocol = _draft()
    protocol["protocol_id"] = "r3-gpu-contracts-jz-v9"
    with pytest.raises(ValueError):
        _validate(protocol)


FAKE_STATE = {
    "revision": "a" * 40,
    "production": "b" * 64,
    "components": dict.fromkeys((*j1_freeze.V19_COMPONENTS, *r3_jz.JZ_EXTRA_COMPONENTS), "c" * 64),
}


def _frozen(protocol_id="r3-gpu-contracts-jz-v1", **values):
    return j1_freeze.build_protocol(protocol_id, frozen_at="2026-10-02T09:00:00Z",
                                    revision=FAKE_STATE["revision"],
                                    production=FAKE_STATE["production"],
                                    components=FAKE_STATE["components"], **values)


def test_jz_v1_frozen_approves_only_the_canary_without_input_hashes():
    protocol = _frozen()
    assert _validate(protocol)
    amendment = _amendment(protocol)
    assert [gate["authorization"] for gate in amendment["stage_gates"]] == [
        "approved", "not-approved", "not-approved"]
    assert amendment["authorization_state"] == "approved-bounded-stage"
    assert stage_execution_blockers(protocol, "backend-kernel-canary") == []
    assert "stage-not-authorized:step0-cross-architecture-probe" in stage_execution_blockers(
        protocol, r3_jz.STEP0_STAGE)
    step0 = amendment["authorization_record"][r3_jz.STEP0_STAGE]
    assert step0["source"]["sha256"] is None and step0["reference_capture"]["manifest_sha256"] is None
    assert "source" not in amendment["authorization_record"]["backend-kernel-canary"]
    for change in (
        lambda p: _amendment(p)["source_binding"].update(parent_revision_at_freeze=None),
        lambda p: _amendment(p)["stage_gates"][1].update(authorization="approved"),
        lambda p: _amendment(p)["authorization_record"][r3_jz.STEP0_STAGE]["source"].update(
            sha256="d" * 64),
        lambda p: _amendment(p)["authorization_record"][r3_jz.STEP0_STAGE][
            "reference_capture"].update(manifest_sha256="e" * 64),
        lambda p: p.update(approvals={"gpu_execution": "required-not-approved",
                                      "hardware_environment": "required-not-approved"}),
        lambda p: p.update(frozen_at="yesterday"),
    ):
        changed = copy.deepcopy(protocol)
        change(changed)
        with pytest.raises(ValueError):
            _validate(changed)


def _probe_version(monkeypatch):
    monkeypatch.setitem(r3_jz.JZ_VERSIONS, "r3-gpu-contracts-jz-v2", {
        "lineage": ("r3-gpu-contracts-jz-v1", "f" * 64),
        "draft_allowed": False,
        "approved_stages": ("backend-kernel-canary", r3_jz.STEP0_STAGE),
    })
    evidence = [{"stage_id": "backend-kernel-canary",
                 "path": "docs/j1_evidence/jean-zay-backend-kernel-canary.json",
                 "sha256": "1" * 64, "protocol_sha256": "2" * 64, "status": "passed"}]
    protocol = _frozen("r3-gpu-contracts-jz-v2", source_sha256="d" * 64,
                       reference_manifest_sha256="e" * 64, prerequisite_evidence=evidence)
    protocol["lineage"] = {"protocol_id": "r3-gpu-contracts-jz-v1", "sha256": "f" * 64}
    return protocol


def test_probe_version_binds_its_own_source_and_reference(monkeypatch):
    protocol = _probe_version(monkeypatch)
    assert _validate(protocol)
    assert stage_execution_blockers(protocol, r3_jz.STEP0_STAGE) == []
    record = _amendment(protocol)["authorization_record"][r3_jz.STEP0_STAGE]
    assert record["source"]["path"].endswith("/j1_runs/inputs/jz_v2/source-j1-pair.json")
    assert record["command"].endswith("docs/r3_protocol_jz_v2.json")
    for change in (
        lambda p: _amendment(p)["authorization_record"][r3_jz.STEP0_STAGE][
            "reference_capture"].update(manifest_sha256=None),
        lambda p: _amendment(p)["authorization_record"][r3_jz.STEP0_STAGE]["source"].update(
            sha256=None),
        lambda p: _amendment(p).update(prerequisite_evidence=[]),
        lambda p: p.update(frozen_at=None),
    ):
        changed = copy.deepcopy(protocol)
        change(changed)
        with pytest.raises(ValueError):
            _validate(changed)


def _diff(left, right, path=()):
    if isinstance(left, dict) and isinstance(right, dict):
        assert set(left) == set(right)
        return [item for key in left for item in _diff(left[key], right[key], (*path, key))]
    if isinstance(left, list) and isinstance(right, list):
        assert len(left) == len(right)
        return [item for index, pair in enumerate(zip(left, right, strict=True))
                for item in _diff(*pair, (*path, index))]
    return [] if left == right else ["/".join(map(str, path))]


def test_jz_source_rewrites_only_paths_and_protocol_references():
    bootes = json.loads(j1_freeze.BOOTES_V16_SOURCE.read_bytes())
    data = j1_freeze.derive_jz_source("r3-gpu-contracts-jz-v1")
    assert data == j1_freeze.derive_jz_source("r3-gpu-contracts-jz-v1")
    derived = json.loads(data)
    assert sorted(_diff(bootes, derived)) == sorted([
        "checkpoint/path", "checkpoint/inventory", "scenes/0/reference_image",
        "scenes/0/actors/0/isolated_image", "scenes/0/actors/1/isolated_image",
        "scenes/0/segmentation_masks/actor-left", "scenes/0/segmentation_masks/actor-right",
        "r3_evidence/protocol", "r3_evidence/matrix",
    ])
    assert derived["r3_evidence"]["protocol"] == f"{r3_jz.JZ_REPO_ROOT}/docs/r3_protocol_jz_v1.json"
    assert derived["checkpoint"]["path"] == r3_jz.JZ_CHECKPOINT_PATH
    assert derived["inference"]["sampling_steps"] == 2 and derived["video_seeds"][0]["value"] == 101
    assets = _amendment(_draft())["authorization_record"]["single-rank-generator-canary"]["assets"]
    scene = derived["scenes"][0]
    assert scene["reference_image"] == assets["reference.png"]["path"]
    assert scene["segmentation_masks"]["actor-left"] == assets["mask-left.png"]["path"]
    with pytest.raises(ValueError):
        j1_freeze.derive_jz_source("r3-gpu-contracts-jz-v1", j1_freeze.BOOTES_V16_SOURCE.read_bytes() + b" ")


def _freeze_repo(tmp_path, monkeypatch):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/r3_test_matrix_v3.json").write_bytes(MATRIX.read_bytes())
    (tmp_path / "docs/r3_protocol_jz_v1.json").write_bytes(DRAFT.read_bytes())
    monkeypatch.setattr(j1_freeze, "collect_repo_state", lambda _repo: FAKE_STATE)
    return tmp_path


def test_freeze_writes_canary_version_and_refuses_refreeze(tmp_path, monkeypatch):
    repo = _freeze_repo(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        j1_freeze.freeze("r3-gpu-contracts-jz-v1", reference_manifest_sha256="e" * 64, repo=repo)
    result = j1_freeze.freeze("r3-gpu-contracts-jz-v1", repo=repo, frozen_at="2026-10-02T09:00:00Z")
    assert result["approved_stage"] == "backend-kernel-canary" and result["source_sha256"] is None
    assert result["command"] == (
        "sbatch tools/j1_slurm_stage.sh backend-kernel-canary docs/r3_protocol_jz_v1.json")
    written = json.loads((repo / "docs/r3_protocol_jz_v1.json").read_text())
    assert written == _frozen()
    with pytest.raises(ValueError, match="refusing to overwrite"):
        j1_freeze.freeze("r3-gpu-contracts-jz-v1", repo=repo)


def test_freeze_of_probe_version_requires_reference_and_writes_source(tmp_path, monkeypatch):
    _probe_version(monkeypatch)
    repo = _freeze_repo(tmp_path, monkeypatch)
    evidence = repo / "docs/j1_evidence/jean-zay-backend-kernel-canary.json"
    evidence.parent.mkdir()
    evidence.write_text(json.dumps({"stage_id": "backend-kernel-canary", "status": "passed",
                                    "bindings": {"protocol_sha256": "2" * 64}}))
    prerequisite = [("backend-kernel-canary", "docs/j1_evidence/jean-zay-backend-kernel-canary.json")]
    with pytest.raises(ValueError, match="reference capture"):
        j1_freeze.freeze("r3-gpu-contracts-jz-v2", prerequisites=prerequisite, repo=repo)
    monkeypatch.setitem(r3_jz.JZ_VERSIONS, "r3-gpu-contracts-jz-v2", r3_jz.JZ_VERSIONS[
        "r3-gpu-contracts-jz-v2"] | {"lineage": ("r3-gpu-contracts-jz-v1", sha256_file(DRAFT))})
    source = tmp_path / "source.json"
    result = j1_freeze.freeze("r3-gpu-contracts-jz-v2", reference_manifest_sha256="e" * 64,
                              prerequisites=prerequisite, write_source=source, repo=repo)
    assert result["source_sha256"] == sha256_file(source)
    assert json.loads(source.read_text())["r3_evidence"]["protocol"].endswith("r3_protocol_jz_v2.json")


def _observations(protocol):
    binding = _amendment(protocol)["environment_binding"]
    observed = {key: binding[key] for key in r3_jz.NODE_CLASS_EXACT_KEYS}
    return observed | {"hostname": "jean-zay-iam07",
                       "gpu_uuids": ["GPU-11111111-2222-3333-4444-555555555555"]}


def test_node_class_runtime_accepts_declared_class(monkeypatch):
    protocol = _draft()
    for key, value in SLURM_ENV.items():
        monkeypatch.setenv(key, value)
    assert validate_v4_runtime_environment(protocol, _observations(protocol))
    evidence = r3_jz.validate_node_class_observation(
        _amendment(protocol)["environment_binding"], _observations(protocol), SLURM_ENV)
    assert evidence["hostname"] == "jean-zay-iam07" and evidence["slurm_job_id"] == "444270"


@pytest.mark.parametrize("observe,environ,error", [
    (lambda o: o.update(hostname="jean-zay-iam1"), {}, RuntimeError),
    (lambda o: o.update(hostname="bootes.alias"), {}, RuntimeError),
    (lambda o: o.update(gpu_uuids=[*o["gpu_uuids"], "GPU-other"]), {}, RuntimeError),
    (lambda o: o.update(gpu_uuids=[]), {}, RuntimeError),
    (lambda o: o.update(driver_version="580.173.02"), {}, RuntimeError),
    (lambda o: o.update(gpu_model="NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"), {},
     RuntimeError),
    (lambda o: o.update(python_version="3.11.15"), {}, RuntimeError),
    (lambda o: o.pop("gpu_uuids"), {}, ValueError),
    (lambda o: o.update(extra="x"), {}, ValueError),
    (lambda o: None, {"SLURM_JOB_PARTITION": "gpu_p6"}, RuntimeError),
    (lambda o: None, {"SLURM_JOB_ACCOUNT": "xvh@h100"}, RuntimeError),
    (lambda o: None, {"SLURM_JOB_ID": ""}, RuntimeError),
])
def test_node_class_runtime_rejects_mismatch(observe, environ, error):
    protocol = _draft()
    observations = _observations(protocol)
    observe(observations)
    with pytest.raises(error):
        r3_jz.validate_node_class_observation(
            _amendment(protocol)["environment_binding"], observations, SLURM_ENV | environ)


def test_legacy_exact_host_mode_is_unchanged(monkeypatch):
    for key in SLURM_ENV:
        monkeypatch.delenv(key, raising=False)
    protocol = json.loads((ROOT / "docs/r3_protocol_v19.json").read_text())
    expected = copy.deepcopy(_amendment(protocol)["environment_binding"])
    assert validate_v4_runtime_environment(protocol, expected)
    with pytest.raises(RuntimeError):
        validate_v4_runtime_environment(protocol, expected | {"hostname": "jean-zay-iam07"})


def test_runners_reject_draft_before_gpu_access(tmp_path):
    protocol = _draft()
    records = _amendment(protocol)["authorization_record"]
    with pytest.raises(RuntimeError, match="preflight blocked"):
        run_backend_canary(DRAFT, MATRIX, records["backend-kernel-canary"]["output"])
    pair_record = records["single-rank-generator-canary"]
    with pytest.raises(ValueError, match="J1 pair preflight blocked"):
        pair.run_generator_pair(DRAFT, MATRIX, pair_record["source"]["path"], pair_record["output"])
    with pytest.raises(ValueError, match="J1 step-0 preflight blocked"):
        j1_stage.run_stage(DRAFT, r3_jz.STEP0_STAGE)
    with pytest.raises(ValueError):
        run_backend_canary(DRAFT, MATRIX, tmp_path / "unbound.json")
    copied = tmp_path / "r3_protocol_jz_v1.json"
    copied.write_bytes(DRAFT.read_bytes())
    with pytest.raises(ValueError):
        r3_jz.require_jz_protocol_path(ROOT, copied, MATRIX, protocol)
    assert j1_stage.bound_output(DRAFT, "backend-kernel-canary") == (
        records["backend-kernel-canary"]["output"])


def test_slurm_wrapper_is_bound_by_every_stage_command():
    protocol = _draft()
    for stage, record in _amendment(protocol)["authorization_record"].items():
        assert record["command"].endswith(f"tools/j1_slurm_stage.sh {stage} docs/r3_protocol_jz_v1.json")
    script = (ROOT / "tools/j1_slurm_stage.sh").read_text()
    for directive in ("--partition=gpu_p5", "--account=xvh@a100", "--gres=gpu:1", "--no-requeue",
                      "HF_HUB_OFFLINE=1", "PET_MASTER_PORT", "--print-output"):
        assert directive in script


def _capture(directory, route, tensors, values=None):
    capture = _Capture(directory)
    for name, tensor in tensors.items():
        capture.save(name, tensor)
    for name, value in (values or {}).items():
        capture.save(name, value)
    _write_manifest(capture, {"route": route}, "complete")
    return directory


def _base_tensors():
    generator = torch.Generator().manual_seed(0)
    return {
        "encoder/clip/0/input/0": torch.randn(3, 1, 4, 4, generator=generator),
        "encoder/t5/261f3817a9e7e746/0": torch.randn(12, 8, generator=generator).bfloat16(),
        "dit/conditional/output": torch.ones(1000, dtype=torch.float64),
        "dit/conditional/freqs": torch.polar(torch.ones(4, dtype=torch.float64),
                                             torch.arange(4, dtype=torch.float64)),
        "dit/negative/zero": torch.zeros(5),
    }


def _gate(tmp_path, change=None, route="official-pristine", extra=None):
    reference = _capture(tmp_path / "reference", "official-pristine", _base_tensors(),
                         {"dit/conditional/input/seq_len": 32760})
    tensors = _base_tensors()
    values = {"dit/conditional/input/seq_len": 32760}
    if change is not None:
        change(tensors, values)
    tensors |= extra or {}
    candidate = _capture(tmp_path / "candidate", route, tensors, values)
    return compare_cross_architecture(reference, candidate, route=route)


def _set(name, value):
    def change(tensors, _values):
        tensors[name] = value
    return change


def test_gate_passes_identical_capture(tmp_path):
    report = _gate(tmp_path)
    assert report["passed"] and report["failed_stages"] == []
    rules = {stage["name"]: stage["rule"] for stage in report["stages"]}
    assert rules["encoder/clip/0/input/0"] == "bitwise"
    assert rules["dit/conditional/output"] == "relative-l2"
    assert rules["dit/negative/zero"] == "zero-reference-norm-bitwise"
    assert rules["dit/conditional/input/seq_len"] == "value-equality"


@pytest.mark.parametrize("delta,passed", [(0.0199, True), (0.0201, False)])
def test_gate_relative_l2_boundary(tmp_path, delta, passed):
    report = _gate(tmp_path, _set("dit/conditional/output", torch.full((1000,), 1 + delta,
                                                                       dtype=torch.float64)))
    stage = next(s for s in report["stages"] if s["name"] == "dit/conditional/output")
    assert stage["relative_l2"] == pytest.approx(delta)
    assert report["passed"] is passed and stage["passed"] is passed


def test_gate_complex_stage_uses_real_view(tmp_path):
    rotated = torch.polar(torch.ones(4, dtype=torch.float64), torch.arange(4, dtype=torch.float64) + 1e-3)
    report = _gate(tmp_path, _set("dit/conditional/freqs", rotated))
    stage = next(s for s in report["stages"] if s["name"] == "dit/conditional/freqs")
    assert stage["rule"] == "relative-l2" and stage["passed"]


@pytest.mark.parametrize("change", [
    _set("encoder/clip/0/input/0", _base_tensors()["encoder/clip/0/input/0"] + 1e-7),
    _set("dit/negative/zero", torch.full((5,), 1e-12)),
    _set("dit/conditional/output", torch.ones(1000, dtype=torch.float32)),
    _set("dit/conditional/output", torch.ones(999, dtype=torch.float64)),
    _set("dit/conditional/output", torch.cat([torch.ones(999, dtype=torch.float64),
                                              torch.tensor([float("nan")], dtype=torch.float64)])),
    lambda tensors, _values: tensors.pop("encoder/t5/261f3817a9e7e746/0"),
    lambda _tensors, values: values.update({"dit/conditional/input/seq_len": 32761}),
])
def test_gate_fails_closed(tmp_path, change):
    report = _gate(tmp_path, change)
    assert report["passed"] is False and report["failed_stages"]


def test_gate_allows_only_the_enumerated_local_only_stage(tmp_path):
    allowed = {"encoder/t5/72f6646e35e0e69d/0": torch.ones(2)}
    assert _gate(tmp_path / "a", route="local-custom", extra=allowed)["passed"]
    report = _gate(tmp_path / "b", route="official-pristine", extra=allowed)
    assert not report["passed"] and report["unexpected_candidate_only_stages"]
    assert not _gate(tmp_path / "c", route="local-custom", extra={"encoder/t5/other/0": torch.ones(2)})[
        "passed"]
    with pytest.raises(ValueError):
        _gate(tmp_path / "d", route="unknown-route")


def test_gate_rejects_tampered_tensor_file(tmp_path):
    reference = _capture(tmp_path / "reference", "official-pristine", _base_tensors())
    candidate = _capture(tmp_path / "candidate", "official-pristine", _base_tensors())
    manifest = json.loads((candidate / "probe-manifest.json").read_text())
    record = next(r for r in manifest["records"] if r["name"] == "dit/conditional/output")
    torch.save(torch.full((1000,), 2.0, dtype=torch.float64), candidate / record["file"])
    with pytest.raises(ValueError, match="differs from its manifest record"):
        compare_cross_architecture(reference, candidate, route="official-pristine")


def test_reference_capture_is_bound_to_manifest_and_report(tmp_path):
    directory = _capture(tmp_path / "reference", "official-pristine", _base_tensors(),
                         {"dit/conditional/input/seq_len": 32760})
    manifest = json.loads((directory / "probe-manifest.json").read_text())
    stages = []
    for record in manifest["records"]:
        if "file" in record:
            identity = {key: record[key] for key in ("shape", "dtype", "sha256")}
            stages.append({"name": record["name"], "reference": identity})
        else:
            stages.append({"name": record["name"], "reference_value": record["value"]})
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"stages": stages}))
    binding = {"path": str(directory), "manifest_sha256": sha256_file(directory / "probe-manifest.json"),
               "report": {"path": str(report), "sha256": sha256_file(report)}}
    assert verify_reference_capture(tmp_path, binding)["route"] == "official-pristine"
    with pytest.raises(ValueError):
        verify_reference_capture(tmp_path, binding | {"manifest_sha256": "0" * 64})
    stages[2]["reference"]["sha256"] = "0" * 64
    report.write_text(json.dumps({"stages": stages}))
    with pytest.raises(ValueError):
        verify_reference_capture(tmp_path, binding | {
            "report": {"path": str(report), "sha256": sha256_file(report)}})


def test_final_latent_report_is_threshold_free(tmp_path):
    reference = np.ones((2, 3), dtype=np.float32)
    candidate = reference.copy()
    candidate[0, 0] = 1.5
    np.save(tmp_path / "reference.npy", reference)
    np.save(tmp_path / "candidate.npy", candidate)
    report = report_final_latents(tmp_path / "reference.npy", tmp_path / "candidate.npy")
    assert report["threshold"] is None and report["comparable"]
    assert report["max_absolute_error"] == pytest.approx(0.5)
    assert report["relative_l2"] == pytest.approx(0.5 / np.sqrt(6))
    assert 0.9 < report["cosine_similarity"] < 1.0


def test_gpu_memory_sampler_keeps_peak(monkeypatch):
    outputs = iter(["1000\n", "5000\n", "3000\n"])

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess([], 0, stdout=next(outputs, "2000\n"))

    monkeypatch.setattr(r3_jz.subprocess, "run", fake_run)
    sampler = r3_jz.GpuMemorySampler(interval_seconds=0.01)
    for _ in range(3):
        sampler._sample()
    assert sampler.record()["peak_memory_used_mib"] == 5000
