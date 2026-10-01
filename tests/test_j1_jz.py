"""CPU-only contracts for the J1 Jean Zay ``r3-gpu-contracts-jz-v*`` branch."""

import ast
import copy
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from multi_sample_inference import j1_stage, r3_jz
from multi_sample_inference import r3_contract_cases as contract_cases
from multi_sample_inference import r3_generator_pair as pair
from multi_sample_inference.j1_step0_probe import (
    compare_cross_architecture,
    report_final_latents,
    verify_reference_capture,
)
from multi_sample_inference.r3_backend_canary import run_backend_canary
from multi_sample_inference.r3_contracts import (
    STAGE4_V19_SOURCES,
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
# The committed jz-v1 is frozen (canary approved); the draft is rebuilt in memory.
FROZEN_V1 = ROOT / "docs/r3_protocol_jz_v1.json"
MATRIX = ROOT / "docs/r3_test_matrix_v3.json"
SLURM_ENV = {"SLURM_JOB_ID": "444270", "SLURM_JOB_PARTITION": "gpu_p5",
             "SLURM_JOB_ACCOUNT": "xvh@a100", "SLURM_JOB_NODELIST": "jean-zay-iam07"}


def _draft():
    components = dict.fromkeys((*j1_freeze.V19_COMPONENTS, *r3_jz.JZ_EXTRA_COMPONENTS))
    return j1_freeze.build_protocol("r3-gpu-contracts-jz-v1", frozen_at=None, revision=None,
                                    production=None, components=components)


def _validate(protocol):
    matrix = json.loads(MATRIX.read_text())
    return validate_protocol(protocol, matrix, matrix_sha256=sha256_file(MATRIX))


def test_v19_constant_matches_frozen_file():
    assert sha256_file(ROOT / "docs/r3_protocol_v19.json") == V19_PROTOCOL_SHA256


def test_draft_validates_and_every_stage_is_blocked():
    protocol = _draft()
    assert _validate(protocol)
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


def _probe_version(_monkeypatch=None):
    evidence = [{"stage_id": "backend-kernel-canary",
                 "path": "docs/j1_evidence/jean-zay-backend-kernel-canary.json",
                 "sha256": "1" * 64, "protocol_sha256": r3_jz.JZ_V2_PROTOCOL_SHA256,
                 "status": "passed"}]
    return _frozen("r3-gpu-contracts-jz-v3", source_sha256="d" * 64,
                   reference_manifest_sha256="e" * 64, prerequisite_evidence=evidence)


def test_probe_version_binds_its_own_source_and_reference(monkeypatch):
    protocol = _probe_version(monkeypatch)
    assert _validate(protocol)
    assert stage_execution_blockers(protocol, r3_jz.STEP0_STAGE) == []
    record = _amendment(protocol)["authorization_record"][r3_jz.STEP0_STAGE]
    assert record["source"]["path"].endswith("/j1_runs/inputs/jz_v3/source-j1-pair.json")
    assert record["command"].endswith("docs/r3_protocol_jz_v3.json")
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
    (tmp_path / "docs/r3_protocol_jz_v1.json").write_text(json.dumps(_draft(), indent=2) + "\n")
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
                                    "bindings": {"protocol_sha256": r3_jz.JZ_V2_PROTOCOL_SHA256}}))
    prerequisite = [("backend-kernel-canary", "docs/j1_evidence/jean-zay-backend-kernel-canary.json")]
    with pytest.raises(ValueError, match="reference capture"):
        j1_freeze.freeze("r3-gpu-contracts-jz-v3", prerequisites=prerequisite, repo=repo)
    source = tmp_path / "source.json"
    result = j1_freeze.freeze("r3-gpu-contracts-jz-v3", reference_manifest_sha256="e" * 64,
                              prerequisites=prerequisite, write_source=source, repo=repo)
    assert result["source_sha256"] == sha256_file(source)
    assert json.loads(source.read_text())["r3_evidence"]["protocol"].endswith("r3_protocol_jz_v3.json")


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


def test_expected_protocol_does_not_alias_binding_constants():
    protocol = _draft()
    _amendment(protocol)["cross_architecture_decision"]["relative_l2_max"] = 0.5
    _amendment(protocol)["environment_binding"]["gpu_model"] = "tampered"
    fresh = _draft()
    assert _amendment(fresh)["cross_architecture_decision"]["relative_l2_max"] == 2e-2
    assert _amendment(fresh)["environment_binding"]["gpu_model"] == "NVIDIA A100-SXM4-80GB"


def test_committed_jz_v1_is_frozen_for_the_canary_only():
    protocol = load_protocol_bundle(FROZEN_V1, MATRIX)["protocol"]
    binding = _amendment(protocol)["source_binding"]
    assert protocol == j1_freeze.build_protocol(
        "r3-gpu-contracts-jz-v1", frozen_at=protocol["frozen_at"],
        revision=binding["parent_revision_at_freeze"],
        production=binding["parent_production_content_sha256"],
        components=_amendment(protocol)["production_component_hashes"])
    assert stage_execution_blockers(protocol, "backend-kernel-canary") == []
    for stage in (r3_jz.STEP0_STAGE, "single-rank-generator-canary"):
        assert f"stage-not-authorized:{stage}" in stage_execution_blockers(protocol, stage)


def test_runners_reject_unapproved_stages_before_gpu_access(tmp_path):
    protocol = load_protocol_bundle(FROZEN_V1, MATRIX)["protocol"]
    records = _amendment(protocol)["authorization_record"]
    pair_record = records["single-rank-generator-canary"]
    with pytest.raises(ValueError, match="J1 pair preflight blocked"):
        pair.run_generator_pair(FROZEN_V1, MATRIX, pair_record["source"]["path"], pair_record["output"])
    with pytest.raises(ValueError, match="J1 step-0 preflight blocked"):
        j1_stage.run_stage(FROZEN_V1, r3_jz.STEP0_STAGE)
    with pytest.raises(ValueError):
        run_backend_canary(FROZEN_V1, MATRIX, tmp_path / "unbound.json")
    copied = tmp_path / "r3_protocol_jz_v1.json"
    copied.write_bytes(FROZEN_V1.read_bytes())
    with pytest.raises(ValueError):
        r3_jz.require_jz_protocol_path(ROOT, copied, MATRIX, protocol)
    assert j1_stage.bound_output(FROZEN_V1, "backend-kernel-canary") == (
        records["backend-kernel-canary"]["output"])


def test_slurm_wrapper_is_bound_by_every_stage_command():
    protocol = _draft()
    for stage, record in _amendment(protocol)["authorization_record"].items():
        assert record["command"].endswith(f"tools/j1_slurm_stage.sh {stage} docs/r3_protocol_jz_v1.json")
    script = (ROOT / "tools/j1_slurm_stage.sh").read_text()
    for directive in ("--partition=gpu_p5", "--account=xvh@a100", "--gres=gpu:1", "--no-requeue",
                      "HF_HUB_OFFLINE=1", "PET_MASTER_PORT", "--print-output"):
        assert directive in script
    # An untracked log inside the checkout would fail the clean-worktree gate.
    assert "#SBATCH --output=/lustre/fswork/projects/rech/xvh/ukl39yh/j1_runs/slurm-logs/" in script
    # Compute nodes lack system git; the source/checkout gates shell out to it.
    assert "module load arch/a100 cuda/12.8.0 git/2.53.0" in script


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


def test_jz_v1_lineage_constant_matches_committed_frozen_file():
    assert sha256_file(FROZEN_V1) == r3_jz.JZ_V1_PROTOCOL_SHA256


def test_jz_v2_is_a_canary_retry_with_a_fresh_attempt_directory():
    protocol = _frozen("r3-gpu-contracts-jz-v2")
    assert _validate(protocol)
    assert protocol["lineage"] == {"protocol_id": "r3-gpu-contracts-jz-v1",
                                   "sha256": r3_jz.JZ_V1_PROTOCOL_SHA256}
    amendment = _amendment(protocol)
    assert [gate["authorization"] for gate in amendment["stage_gates"]] == [
        "approved", "not-approved", "not-approved"]
    canary = amendment["authorization_record"]["backend-kernel-canary"]
    assert canary["output"].endswith("/backend-canary/attempt-2/backend-kernel-canary.json")
    assert canary["command"] == (
        "sbatch tools/j1_slurm_stage.sh backend-kernel-canary docs/r3_protocol_jz_v2.json")
    assert stage_execution_blockers(protocol, "backend-kernel-canary") == []
    assert amendment["prerequisite_evidence"] == []


@pytest.mark.parametrize("change", [
    lambda p: p["lineage"].update(sha256="0" * 64),
    lambda p: p["lineage"].update(protocol_id="r3-gpu-contracts-v19"),
    lambda p: _amendment(p)["authorization_record"]["backend-kernel-canary"].update(
        output="/lustre/fswork/projects/rech/xvh/ukl39yh/j1_runs/backend-canary/attempt-1/"
               "backend-kernel-canary.json"),
    lambda p: p.update(frozen_at=None),
])
def test_jz_v2_rejects_lineage_attempt_and_draft_changes(change):
    protocol = _frozen("r3-gpu-contracts-jz-v2")
    change(protocol)
    with pytest.raises(ValueError):
        _validate(protocol)


def test_jz_v2_requires_the_unchanged_frozen_predecessor_file(monkeypatch):
    protocol = _frozen("r3-gpu-contracts-jz-v2")
    monkeypatch.setattr(r3_jz, "JZ_V1_PROTOCOL_SHA256", "0" * 64)
    monkeypatch.setitem(r3_jz.JZ_VERSIONS, "r3-gpu-contracts-jz-v2", r3_jz.JZ_VERSIONS[
        "r3-gpu-contracts-jz-v2"] | {"lineage": ("r3-gpu-contracts-jz-v1", "0" * 64)})
    protocol["lineage"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="frozen predecessor"):
        _validate(protocol)


def test_jz_v3_pins_canary_evidence_to_the_jz_v2_protocol():
    protocol = _probe_version()
    assert _validate(protocol)
    assert protocol["lineage"] == {"protocol_id": "r3-gpu-contracts-jz-v2",
                                   "sha256": r3_jz.JZ_V2_PROTOCOL_SHA256}
    records = _amendment(protocol)["authorization_record"]
    # The passed canary keeps its used attempt-2 path, so the canary runner refuses a rerun.
    assert records["backend-kernel-canary"]["output"].endswith("/attempt-2/backend-kernel-canary.json")
    assert records[r3_jz.STEP0_STAGE]["output"].endswith("/step0-probe/attempt-1")
    assert "stage-not-authorized:single-rank-generator-canary" in stage_execution_blockers(
        protocol, "single-rank-generator-canary")
    _amendment(protocol)["prerequisite_evidence"][0]["protocol_sha256"] = r3_jz.JZ_V1_PROTOCOL_SHA256
    with pytest.raises(ValueError, match="pinned frozen version"):
        _validate(protocol)


def test_jz_v2_constant_matches_committed_frozen_file():
    assert sha256_file(ROOT / "docs/r3_protocol_jz_v2.json") == r3_jz.JZ_V2_PROTOCOL_SHA256


def test_committed_jz_v1_to_v3_validate_unchanged_without_offload():
    for version, digest in ((1, r3_jz.JZ_V1_PROTOCOL_SHA256), (2, r3_jz.JZ_V2_PROTOCOL_SHA256),
                            (3, r3_jz.JZ_V3_PROTOCOL_SHA256)):
        path = ROOT / f"docs/r3_protocol_jz_v{version}.json"
        assert sha256_file(path) == digest
        protocol = load_protocol_bundle(path, MATRIX)["protocol"]
        assert "generation_settings" not in _amendment(protocol)
        assert r3_jz.protocol_offload_model(protocol) is False
    for name in ("r3_protocol_v16.json", "r3_protocol_v19.json"):
        protocol = load_protocol_bundle(ROOT / "docs" / name, MATRIX)["protocol"]
        assert r3_jz.protocol_offload_model(protocol) is False


CANARY_EVIDENCE = {
    "stage_id": "backend-kernel-canary",
    "path": "docs/j1_evidence/jean-zay-backend-kernel-canary.json",
    "sha256": sha256_file(ROOT / "docs/j1_evidence/jean-zay-backend-kernel-canary.json"),
    "protocol_sha256": r3_jz.JZ_V2_PROTOCOL_SHA256,
    "status": "passed",
}


def _jz_v4(**values):
    values = {"source_sha256": "d" * 64, "prerequisite_evidence": [CANARY_EVIDENCE]} | values
    return _frozen("r3-gpu-contracts-jz-v4", **values)


def test_jz_v4_approves_the_offload_pair_after_the_jz_v2_canary():
    protocol = _jz_v4()
    assert _validate(protocol)
    assert protocol["lineage"] == {"protocol_id": "r3-gpu-contracts-jz-v3",
                                   "sha256": r3_jz.JZ_V3_PROTOCOL_SHA256}
    amendment = _amendment(protocol)
    assert amendment["generation_settings"] == {"offload_model": True}
    assert r3_jz.protocol_offload_model(protocol) is True
    gates = {gate["id"]: gate for gate in amendment["stage_gates"]}
    assert gates[r3_jz.STEP0_STAGE]["authorization"] == "not-approved"
    assert gates["single-rank-generator-canary"]["prerequisites"] == ["backend-kernel-canary"]
    assert stage_execution_blockers(protocol, "single-rank-generator-canary") == []
    assert "stage-not-authorized:step0-cross-architecture-probe" in stage_execution_blockers(
        protocol, r3_jz.STEP0_STAGE)
    records = amendment["authorization_record"]
    pair_record = records["single-rank-generator-canary"]
    assert "report_only_bootes_v16_reference" not in pair_record
    assert pair_record["output"].endswith("/generator-pair/attempt-1")
    assert pair_record["source"] == {
        "path": f"{r3_jz.JZ_RUN_ROOT}/inputs/jz_v4/source-j1-pair.json", "sha256": "d" * 64}
    assert pair_record["command"] == ("sbatch --qos=qos_gpu_a100-t3 --time=04:00:00 "
                                      "tools/j1_slurm_stage.sh single-rank-generator-canary "
                                      "docs/r3_protocol_jz_v4.json")
    assert records["backend-kernel-canary"]["output"].endswith("/attempt-2/backend-kernel-canary.json")
    assert records[r3_jz.STEP0_STAGE]["reference_capture"]["manifest_sha256"] is None


@pytest.mark.parametrize("change", [
    lambda p: _amendment(p).update(generation_settings={"offload_model": False}),
    lambda p: _amendment(p).pop("generation_settings"),
    lambda p: _amendment(p)["stage_gates"][1].update(authorization="approved"),
    lambda p: _amendment(p)["stage_gates"][2].update(
        prerequisites=["backend-kernel-canary", r3_jz.STEP0_STAGE]),
    lambda p: _amendment(p)["authorization_record"]["single-rank-generator-canary"].update(
        report_only_bootes_v16_reference=r3_jz.JZ_BOOTES_V16_REFERENCE),
    lambda p: _amendment(p)["authorization_record"][r3_jz.STEP0_STAGE][
        "reference_capture"].update(manifest_sha256="e" * 64),
    lambda p: _amendment(p)["prerequisite_evidence"][0].update(
        protocol_sha256=r3_jz.JZ_V1_PROTOCOL_SHA256),
    lambda p: _amendment(p).update(prerequisite_evidence=[]),
    lambda p: p["lineage"].update(sha256=r3_jz.JZ_V2_PROTOCOL_SHA256),
])
def test_jz_v4_rejects_unbound_changes(change):
    protocol = _jz_v4()
    change(protocol)
    with pytest.raises(ValueError):
        _validate(protocol)


def test_jz_v4_requires_its_pair_source_hash():
    with pytest.raises(ValueError):
        _jz_v4(source_sha256=None)


def test_worker_threads_offload_only_from_the_validated_protocol():
    """fsdp_worker needs CUDA at import, so check its offload threading structurally."""
    tree = ast.parse((ROOT / "multi_sample_inference/fsdp_worker.py").read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and getattr(node.func, "id", None) in {"run_generator", "run_inference"}]
    assert len(calls) == 4
    for call in calls:
        keyword = next(k for k in call.keywords if k.arg == "offload_model")
        assert isinstance(keyword.value, ast.Name) and keyword.value.id == "offload_model"
    validator = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "_validate_r3_before_model_load")
    returns = [ast.unparse(node.value) for node in ast.walk(validator)
               if isinstance(node, ast.Return)]
    assert set(returns) == {"False", "protocol_offload_model(bundle['protocol'])"}
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    assert "offload_model = _validate_r3_before_model_load(task, args.manifest_file)" in ast.unparse(main)


def test_freeze_of_jz_v4_binds_source_and_canary_without_reference(tmp_path, monkeypatch):
    repo = _freeze_repo(tmp_path, monkeypatch)
    evidence = repo / CANARY_EVIDENCE["path"]
    evidence.parent.mkdir()
    evidence.write_bytes((ROOT / CANARY_EVIDENCE["path"]).read_bytes())
    prerequisite = [("backend-kernel-canary", CANARY_EVIDENCE["path"])]
    with pytest.raises(ValueError, match="reference capture"):
        j1_freeze.freeze("r3-gpu-contracts-jz-v4", reference_manifest_sha256="e" * 64,
                         prerequisites=prerequisite, repo=repo)
    source = tmp_path / "source.json"
    result = j1_freeze.freeze("r3-gpu-contracts-jz-v4", prerequisites=prerequisite,
                              write_source=source, repo=repo, frozen_at="2026-10-02T09:00:00Z")
    assert result["approved_stage"] == "single-rank-generator-canary"
    assert result["source_sha256"] == sha256_file(source)
    assert json.loads(source.read_text())["r3_evidence"]["protocol"].endswith("r3_protocol_jz_v4.json")
    written = json.loads((repo / "docs/r3_protocol_jz_v4.json").read_text())
    assert written == _jz_v4(source_sha256=sha256_file(source))


# --- jz-v5: R3 stage 4 as sequential contract-case chunks on the A100 -------------------

V5 = "r3-gpu-contracts-jz-v5"
CHUNKS = r3_jz.JZ_CONTRACT_CHUNK_STAGES
PAIR_EVIDENCE = {
    "stage_id": "single-rank-generator-canary",
    "path": "docs/j1_evidence/jean-zay-generator-pair.json",
    "sha256": sha256_file(ROOT / "docs/j1_evidence/jean-zay-generator-pair.json"),
    "protocol_sha256": r3_jz.JZ_V4_PROTOCOL_SHA256,
    "status": "passed",
}


def _jz_v5(**values):
    values = {"prerequisite_evidence": [CANARY_EVIDENCE, PAIR_EVIDENCE]} | values
    return _frozen(V5, **values)


def _chunk(protocol, index):
    return _amendment(protocol)["authorization_record"][CHUNKS[index - 1]]


def _bootes_sources():
    return {name: json.loads((ROOT / r3_jz.bootes_v19_stage4_source_path(name)).read_text())
            for name in STAGE4_V19_SOURCES}


def test_jz_v4_constant_and_committed_file_validate_unchanged():
    path = ROOT / "docs/r3_protocol_jz_v4.json"
    assert sha256_file(path) == r3_jz.JZ_V4_PROTOCOL_SHA256
    protocol = load_protocol_bundle(path, MATRIX)["protocol"]
    assert set(_amendment(protocol)["authorization_record"]) == set(r3_jz.J1_STAGE_IDS)
    record = _amendment(protocol)["authorization_record"]["single-rank-generator-canary"]
    assert record["source"]["sha256"] == r3_jz.JZ_V4_PAIR_SOURCE_SHA256


def test_committed_bootes_v19_sources_match_the_frozen_v19_declaration():
    v19 = json.loads((ROOT / "docs/r3_protocol_v19.json").read_text())
    declared = v19["execution_amendment"]["authorization_record"]["sources"]
    for name, digest in STAGE4_V19_SOURCES.items():
        assert sha256_file(ROOT / r3_jz.bootes_v19_stage4_source_path(name)) == digest
        assert declared[name]["sha256"] == digest


def test_planned_order_reproduces_the_bootes_v18_runtime_order():
    """Job ids and cases are path-free, so the pure plan matches the real v18 expansion."""
    ordered, pairs = contract_cases.planned_contract_order(
        _bootes_sources(), json.loads(MATRIX.read_text()))
    v18 = json.loads((ROOT / "docs/r3_evidence/bootes-contract-cases-v18-failed-attempt.json")
                     .read_text())
    observed = [{key: job[key] for key in ("source", "job_id", "case_id")} for job in v18["jobs"]]
    assert ordered[:len(observed)] == observed
    assert ordered[len(observed)]["job_id"] == v18["failed_job"]
    jobs, parity = contract_cases.expected_single_rank_cases(json.loads(MATRIX.read_text()))
    assert len(ordered) == 200 and {job["case_id"] for job in ordered} == jobs
    assert len({job["job_id"] for job in ordered}) == 200
    assert set(pairs) == {"dpmpp-flash", "unipc-flash"} and len(parity) == len(pairs)


def test_jz_v5_freezes_four_sequential_chunks_after_the_canary_and_pair():
    protocol = _jz_v5()
    assert _validate(protocol)
    assert protocol["lineage"] == {"protocol_id": "r3-gpu-contracts-jz-v4",
                                   "sha256": r3_jz.JZ_V4_PROTOCOL_SHA256}
    assert protocol["claim_boundary"] == r3_jz.JZ_V5_CLAIM_BOUNDARY
    amendment = _amendment(protocol)
    assert amendment["generation_settings"] == {"offload_model": True}
    assert amendment["prerequisite_evidence"] == [CANARY_EVIDENCE, PAIR_EVIDENCE]
    records = amendment["authorization_record"]
    assert set(records) == set(r3_jz.JZ_ALL_STAGE_IDS)
    # Passed stages keep their used attempts, so their runners refuse a rerun.
    assert records["backend-kernel-canary"]["output"].endswith("/attempt-2/backend-kernel-canary.json")
    pair_record = records["single-rank-generator-canary"]
    assert pair_record["output"].endswith("/generator-pair/attempt-1")
    assert pair_record["source"] == {
        "path": f"{r3_jz.JZ_RUN_ROOT}/inputs/jz_v4/source-j1-pair.json",
        "sha256": r3_jz.JZ_V4_PAIR_SOURCE_SHA256}
    gates = {gate["id"]: gate for gate in amendment["stage_gates"]}
    assert gates[r3_jz.STEP0_STAGE]["authorization"] == "not-approved"
    planned, previous = [], None
    for index, stage in enumerate(CHUNKS, start=1):
        gate, record = gates[stage], _chunk(protocol, index)
        assert gate["authorization"] == "approved"
        assert gate["prerequisites"] == ["backend-kernel-canary", "single-rank-generator-canary",
                                         *([CHUNKS[index - 2]] if index > 1 else [])]
        assert gate["waived_prerequisites"] == [r3_jz.JZ_CONTRACT_HOOK_CANARY_WAIVER]
        assert stage_execution_blockers(protocol, stage) == []
        assert record["chunk"]["index"] == index and len(record["chunk"]["jobs"]) == 50
        assert record["output"] == f"{r3_jz.JZ_RUN_ROOT}/contract-cases/chunk-{index}/attempt-1"
        assert record["previous_chunk"] == previous
        assert record["job_timeout_seconds"] == 2700 and record["expected_jobs"] == 200
        assert record["sources"] == _chunk(protocol, 1)["sources"]
        dependency = ("" if index == 1 else
                      "--dependency=afterok:$J1_PREVIOUS_CHUNK_JOB_ID --kill-on-invalid-dep=yes ")
        assert record["command"] == (
            f"sbatch --parsable {dependency}--qos=qos_gpu_a100-t3 --time=16:00:00 "
            f"tools/j1_slurm_stage.sh {stage} docs/r3_protocol_jz_v5.json")
        planned += record["chunk"]["jobs"]
        previous = {"stage_id": stage, "record": f"{record['output']}/attempt.json"}
    ordered, pairs = contract_cases.planned_contract_order(
        _bootes_sources(), json.loads(MATRIX.read_text()))
    assert planned == ordered
    first = _chunk(protocol, 1)["chunk"]
    assert first["parity_pairs"] == pairs
    assert {job["job_id"] for job in first["jobs"][:4]} == {
        job_id for pair in pairs.values() for job_id in pair.values()}
    assert all(_chunk(protocol, index)["chunk"]["parity_pairs"] == {} for index in (2, 3, 4))


def _swap_first_jobs(protocol):
    jobs = _chunk(protocol, 2)["chunk"]["jobs"]
    jobs[0], jobs[1] = jobs[1], jobs[0]


def _move_job(protocol):
    _chunk(protocol, 3)["chunk"]["jobs"].append(_chunk(protocol, 4)["chunk"]["jobs"].pop(0))


@pytest.mark.parametrize("change", [
    _swap_first_jobs,
    _move_job,
    lambda p: _chunk(p, 1)["chunk"].update(plan_sha256="0" * 64),
    lambda p: _chunk(p, 2)["chunk"].update(parity_pairs=_chunk(p, 1)["chunk"]["parity_pairs"]),
    lambda p: _chunk(p, 2).update(command=_chunk(p, 2)["command"].replace(
        "--dependency=afterok:$J1_PREVIOUS_CHUNK_JOB_ID ", "")),
    lambda p: _chunk(p, 2).update(previous_chunk=None),
    lambda p: _chunk(p, 3)["sources"]["unipc-flex"].update(sha256="0" * 64),
    lambda p: _chunk(p, 1).update(job_timeout_seconds=1800),
    lambda p: _amendment(p)["stage_gates"][3].update(waived_prerequisites=[]),
    lambda p: _amendment(p)["stage_gates"][4].update(
        prerequisites=["backend-kernel-canary", "single-rank-generator-canary"]),
    lambda p: _amendment(p)["stage_gates"][1].update(authorization="approved"),
    lambda p: _amendment(p)["authorization_record"]["single-rank-generator-canary"][
        "source"].update(sha256="d" * 64),
    lambda p: _amendment(p)["authorization_record"][r3_jz.STEP0_STAGE]["source"].update(
        sha256="d" * 64),
    lambda p: _amendment(p).update(prerequisite_evidence=[CANARY_EVIDENCE]),
    lambda p: _amendment(p)["prerequisite_evidence"][1].update(
        protocol_sha256=r3_jz.JZ_V3_PROTOCOL_SHA256),
    lambda p: _amendment(p).pop("generation_settings"),
    lambda p: p.update(claim_boundary=r3_jz.JZ_CLAIM_BOUNDARY),
    lambda p: p["lineage"].update(sha256=r3_jz.JZ_V3_PROTOCOL_SHA256),
])
def test_jz_v5_rejects_unbound_changes(change):
    protocol = _jz_v5()
    change(protocol)
    with pytest.raises(ValueError):
        _validate(protocol)


def test_jz_v5_chunk_prerequisites_rehash_committed_canary_and_offload_pair(tmp_path):
    protocol = _jz_v5()
    for stage in CHUNKS:
        records = r3_jz.verify_jz_prerequisites(ROOT, protocol, stage)
        assert set(records) == {"backend-kernel-canary", "single-rank-generator-canary"}
    (tmp_path / "docs/j1_evidence").mkdir(parents=True)
    for entry in (CANARY_EVIDENCE, PAIR_EVIDENCE):
        (tmp_path / entry["path"]).write_bytes((ROOT / entry["path"]).read_bytes())
    tampered = json.loads((tmp_path / PAIR_EVIDENCE["path"]).read_text())
    tampered["generation_settings"] = {"offload_model": False}
    (tmp_path / PAIR_EVIDENCE["path"]).write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="changed or is missing"):
        r3_jz.verify_jz_prerequisites(tmp_path, protocol, CHUNKS[0])
    _amendment(protocol)["prerequisite_evidence"][1]["sha256"] = sha256_file(
        tmp_path / PAIR_EVIDENCE["path"])
    with pytest.raises(ValueError, match="invalid stage/status/binding"):
        r3_jz.verify_jz_prerequisites(tmp_path, protocol, CHUNKS[0])


def _chunk_record(protocol, index, protocol_sha256, previous_sha256, **changes):
    auth = _chunk(protocol, index)
    comparisons = {name: {"measurements": {"passed": True, "maximum_absolute_error": 0.0}}
                   for name in auth["chunk"]["parity_pairs"]}
    return {
        "stage_id": CHUNKS[index - 1], "status": "passed", "protocol_sha256": protocol_sha256,
        "matrix_sha256": sha256_file(MATRIX), "r3_acceptance": False,
        "chunk": {"index": index, "count": 4, "plan_sha256": auth["chunk"]["plan_sha256"]},
        "previous_chunk_record_sha256": previous_sha256,
        "jobs": [job | {"video_sha256": "9" * 64} for job in auth["chunk"]["jobs"]],
        "comparisons": comparisons,
    } | changes


def _write_chunk_records(tmp_path, protocol, protocol_sha256, changes=None):
    paths, previous = [], None
    for index in range(1, 5):
        path = tmp_path / f"chunk-{index}.json"
        record = _chunk_record(protocol, index, protocol_sha256, previous)
        path.write_text(json.dumps(record | (changes or {}).get(index, {})))
        paths.append(path)
        previous = sha256_file(path)
    return paths


def test_aggregate_accepts_exactly_the_passed_hash_chained_chunks(tmp_path):
    protocol = _jz_v5()
    path = tmp_path / "r3_protocol_jz_v5.json"
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    digest = sha256_file(path)
    summary = contract_cases.aggregate_jz_contract_chunks(
        path, MATRIX, _write_chunk_records(tmp_path, protocol, digest))
    assert summary["status"] == "passed" and summary["jobs"] == 200
    assert summary["parity_pairs"] == ["dpmpp-flash", "unipc-flash"]
    assert summary["r3_acceptance"] is False
    first_jobs = _chunk(protocol, 1)["chunk"]["jobs"]
    for changes in (
        {3: {"status": "failed"}},
        {2: {"protocol_sha256": "0" * 64}},
        {2: {"previous_chunk_record_sha256": "0" * 64}},
        {4: {"jobs": [job | {"video_sha256": "9" * 64} for job in first_jobs[:50]]}},
        {1: {"comparisons": {"unipc-flash": {"measurements": {"passed": True}}}}},
        {1: {"comparisons": {name: {"measurements": {"passed": False}}
                             for name in ("dpmpp-flash", "unipc-flash")}}},
    ):
        with pytest.raises(ValueError):
            contract_cases.aggregate_jz_contract_chunks(
                path, MATRIX, _write_chunk_records(tmp_path, protocol, digest, changes))
    records = _write_chunk_records(tmp_path, protocol, digest)
    with pytest.raises(ValueError):
        contract_cases.aggregate_jz_contract_chunks(path, MATRIX, records[:3])
    with pytest.raises(ValueError):
        contract_cases.aggregate_jz_contract_chunks(path, MATRIX, [records[1], records[0],
                                                                   *records[2:]])


def _patched_chunk_runner(tmp_path, monkeypatch, protocol):
    """Run the chunk runner's CPU gates against an in-memory jz-v5 rooted in tmp_path."""
    monkeypatch.setattr(contract_cases, "JZ_RUN_ROOT", str(tmp_path))
    monkeypatch.setattr(contract_cases, "require_jz_protocol_path", lambda *args: None)
    bundle = {"protocol": protocol, "matrix": json.loads(MATRIX.read_text()),
              "protocol_sha256": "f" * 64, "matrix_sha256": sha256_file(MATRIX)}
    monkeypatch.setattr(contract_cases, "load_protocol_bundle", lambda *args: bundle)
    inputs = tmp_path / "inputs"
    digests = j1_freeze.write_contract_sources(V5, inputs)
    for index in range(1, 5):
        auth = _chunk(protocol, index)
        auth["output"] = str(tmp_path / f"chunk-{index}")
        auth["sources"] = {name: {"path": str(inputs / f"source-stage4-{name}.json"),
                                  "sha256": digest} for name, digest in digests.items()}
        if auth["previous_chunk"] is not None:
            auth["previous_chunk"]["record"] = str(tmp_path / f"chunk-{index - 1}/attempt.json")
    return f"{r3_jz.JZ_REPO_ROOT}/docs/r3_protocol_jz_v5.json"


def test_chunk_runner_refuses_rerun_and_unpassed_predecessor_before_gpu(tmp_path, monkeypatch):
    protocol = _jz_v5()
    protocol_path = _patched_chunk_runner(tmp_path, monkeypatch, protocol)
    output = tmp_path / "chunk-2"
    output.mkdir()
    with pytest.raises(ValueError, match="attempt already exists"):
        contract_cases.run_jz_contract_chunk(protocol_path, MATRIX, CHUNKS[1], output)
    output.rmdir()
    with pytest.raises(ValueError, match="has no record"):
        contract_cases.run_jz_contract_chunk(protocol_path, MATRIX, CHUNKS[1], output)
    previous = tmp_path / "chunk-1/attempt.json"
    previous.parent.mkdir()
    previous.write_text(json.dumps(_chunk_record(protocol, 1, "f" * 64, None, status="failed")))
    with pytest.raises(ValueError, match="did not pass under this protocol"):
        contract_cases.run_jz_contract_chunk(protocol_path, MATRIX, CHUNKS[1], output)
    assert not output.exists()
    previous.write_text(json.dumps(_chunk_record(protocol, 1, "f" * 64, None)))
    assert contract_cases.verify_previous_chunk(_chunk(protocol, 2), "f" * 64) == sha256_file(previous)
    assert contract_cases.verify_previous_chunk(_chunk(protocol, 1), "f" * 64) is None
    with pytest.raises(ValueError):
        contract_cases.verify_previous_chunk(_chunk(protocol, 2), "e" * 64)
    with pytest.raises(ValueError, match="unknown J1 contract-case chunk"):
        contract_cases.run_jz_contract_chunk(protocol_path, MATRIX, "single-rank-generator-canary",
                                             output)


def test_jz_v4_has_no_contract_chunk_stages():
    path = ROOT / "docs/r3_protocol_jz_v4.json"
    protocol = load_protocol_bundle(path, MATRIX)["protocol"]
    with pytest.raises(ValueError, match="unknown J1 stage"):
        r3_jz.jz_authorization(protocol, CHUNKS[0])
    with pytest.raises(ValueError, match="unknown J1 stage"):
        contract_cases.run_jz_contract_chunk(path, MATRIX, CHUNKS[0], "/nonexistent")


def test_derived_contract_sources_change_only_enumerated_fields(tmp_path):
    digests = j1_freeze.write_contract_sources(V5, tmp_path)
    assert digests == {name: entry["sha256"]
                       for name, entry in _chunk(_jz_v5(), 1)["sources"].items()}

    def leaves(value, prefix=()):
        if isinstance(value, dict):
            return {k: v for key, item in value.items() for k, v in leaves(item, (*prefix, key)).items()}
        if isinstance(value, list):
            return {k: v for index, item in enumerate(value)
                    for k, v in leaves(item, (*prefix, index)).items()}
        return {prefix: value}

    for name, bootes in _bootes_sources().items():
        derived = json.loads((tmp_path / f"source-stage4-{name}.json").read_text())
        before, after = leaves(bootes), leaves(derived)
        assert set(before) == set(after)
        changed = {key for key in before if before[key] != after[key]}
        assert changed == {
            ("checkpoint", "path"), ("checkpoint", "inventory"),
            ("scenes", 0, "reference_image"), ("scenes", 0, "actors", 0, "isolated_image"),
            ("scenes", 0, "actors", 1, "isolated_image"),
            ("scenes", 0, "segmentation_masks", "actor-left"),
            ("scenes", 0, "segmentation_masks", "actor-right"),
            ("r3_evidence", "protocol"), ("r3_evidence", "matrix"),
        }
        assert derived["r3_evidence"]["protocol"] == (
            f"{r3_jz.JZ_REPO_ROOT}/docs/r3_protocol_jz_v5.json")
    with pytest.raises(ValueError, match="already exists"):
        j1_freeze.write_contract_sources(V5, tmp_path)
    with pytest.raises(ValueError, match="binds no contract-case sources"):
        j1_freeze.write_contract_sources("r3-gpu-contracts-jz-v4", tmp_path / "v4")


def test_pair_source_derivation_is_unchanged_for_jz_v4():
    assert hashlib.sha256(j1_freeze.derive_jz_source("r3-gpu-contracts-jz-v4")).hexdigest() == (
        r3_jz.JZ_V4_PAIR_SOURCE_SHA256)


def test_freeze_of_jz_v5_binds_canary_and_pair_evidence(tmp_path, monkeypatch):
    repo = _freeze_repo(tmp_path, monkeypatch)
    (repo / "docs/j1_evidence").mkdir()
    for entry in (CANARY_EVIDENCE, PAIR_EVIDENCE):
        (repo / entry["path"]).write_bytes((ROOT / entry["path"]).read_bytes())
    prerequisites = [("backend-kernel-canary", CANARY_EVIDENCE["path"]),
                     ("single-rank-generator-canary", PAIR_EVIDENCE["path"])]
    with pytest.raises(ValueError, match="writes no pair source"):
        j1_freeze.freeze(V5, prerequisites=prerequisites, write_source=tmp_path / "s.json",
                         repo=repo)
    result = j1_freeze.freeze(V5, prerequisites=prerequisites, repo=repo,
                              frozen_at="2026-10-02T09:00:00Z")
    assert result["approved_stage"] == CHUNKS[-1]
    assert result["source_sha256"] == r3_jz.JZ_V4_PAIR_SOURCE_SHA256
    assert json.loads((repo / "docs/r3_protocol_jz_v5.json").read_text()) == _jz_v5()


def test_submission_helper_chains_bound_commands_with_afterok(tmp_path):
    protocol = _jz_v5()
    commands = {stage: _amendment(protocol)["authorization_record"][stage]["command"]
                for stage in CHUNKS}
    bin_dir, repo = tmp_path / "bin", tmp_path / "repo"
    bin_dir.mkdir()
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (tmp_path / "commands.json").write_text(json.dumps(commands))
    (bin_dir / "uv").write_text(
        "#!/bin/bash\n"
        f"exec python3 -c 'import json,sys; print(json.load(open(\"{tmp_path}/commands.json\"))"
        "[sys.argv[sys.argv.index(\"--stage\") + 1]])' \"$@\"\n")
    (bin_dir / "sbatch").write_text(
        "#!/bin/bash\n"
        f"echo \"$*\" >> {tmp_path}/sbatch.log\n"
        f"echo $((1000 + $(wc -l < {tmp_path}/sbatch.log)))\n")
    for tool in ("uv", "sbatch"):
        (bin_dir / tool).chmod(0o755)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "J1_REPO": str(repo),
           "J1_TOOLS": str(tmp_path / "tools"), "HOME": str(tmp_path)}
    result = subprocess.run(["bash", str(ROOT / "tools/j1_submit_contract_chunks.sh"),
                             "docs/r3_protocol_jz_v5.json"],
                            env=env, capture_output=True, text=True, check=True)
    assert result.stdout.split() == [CHUNKS[0], "1001", CHUNKS[1], "1002",
                                     CHUNKS[2], "1003", CHUNKS[3], "1004"]
    submitted = (tmp_path / "sbatch.log").read_text().splitlines()
    assert submitted[0] == commands[CHUNKS[0]].removeprefix("sbatch ")
    for index in (1, 2, 3):
        assert submitted[index] == commands[CHUNKS[index]].removeprefix("sbatch ").replace(
            "$J1_PREVIOUS_CHUNK_JOB_ID", str(1000 + index))
    code = [line for line in (ROOT / "tools/j1_submit_contract_chunks.sh").read_text().splitlines()
            if not line.lstrip().startswith("#")]
    assert not any("eval" in line.split() for line in code)
