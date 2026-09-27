"""CPU-only regressions for the not-yet-frozen Bootes generator-pair amendment."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from multi_sample_inference import experiment_pipeline as pipeline
from multi_sample_inference import r3_generator_pair as pair
from multi_sample_inference.r3_contracts import (
    execution_blockers,
    load_protocol_bundle,
    sha256_file,
    stage_execution_blockers,
    validate_protocol,
)

ROOT = Path(__file__).parents[1]
MATRIX = ROOT / "docs/r3_test_matrix_v3.json"


def _synthetic_v12():
    protocol = json.loads((ROOT / "docs/r3_protocol_v11.json").read_text())
    amendment = protocol["execution_amendment"]
    amendment["amendment_id"] = "r3-gpu-execution-amendment-v12"
    amendment["source_binding"]["parent_revision_at_freeze"] = "a" * 40
    amendment["source_binding"]["parent_production_content_sha256"] = "b" * 64
    amendment["production_component_hashes"].update({
        "r3_contracts.py": "c" * 64,
        "experiment_pipeline.py": "d" * 64,
        "r3_generator_pair.py": "e" * 64,
    })
    amendment["stage_gates"][2]["authorization"] = "approved"
    amendment["prerequisite_evidence"] = [
        amendment["prerequisite_evidence"],
        {
            "stage_id": "checkpoint-load-hook-canary",
            "path": "docs/r3_evidence/bootes-checkpoint-hook-canary.json",
            "sha256": sha256_file(ROOT / "docs/r3_evidence/bootes-checkpoint-hook-canary.json"),
            "protocol_sha256": sha256_file(ROOT / "docs/r3_protocol_v11.json"),
            "status": "passed",
        },
    ]
    amendment["authorization_record"] = {
        "authorized_stage": "single-rank-generator-canary",
        "authorization_source": "explicit-current-session-user-approval",
        "authorized_operations": [
            "verify-bootes-host-source-environment-checkpoint-and-prerequisites",
            "two-sequential-isolated-single-rank-generator-jobs-and-offline-parity",
        ],
        "prohibited_operations": [
            "repeat-attempt", "additional-jobs", "distributed-execution",
            "later-stage-execution", "scientific-claim",
        ],
        "checkpoint_inventory": "docs/u1_checkpoint_inventory.json",
        "source": {"path": "/local_scratch2/gzappavi/r3_stage3/input/source.json",
                   "sha256": "ebceee5b5b3246bf98bb0c434ad81af7e7b33cefa0da64e87f9de3134f6ab5c2"},
        "output": "/local_scratch2/gzappavi/r3_stage3/generator-pair-attempt",
        "gpu_uuid": "GPU-c247e0e3-654a-7387-8ec6-46791821a52d",
        "min_free_gpu_bytes": 70 * 1024**3,
        "job_timeout_seconds": 1800,
        "assets": {
            "reference.png": {"path": "/local_scratch2/gzappavi/r3_stage3/input/reference.png", "sha256": "3f24239d8b0eef18cb022f46ef81b9a0bbcbe76cab9bffe3bf018b94be6ca17b"},
            "mask-left.png": {"path": "/local_scratch2/gzappavi/r3_stage3/input/mask-left.png", "sha256": "ec534110d2c7c022847e2e9edcd1bb2c3bdeffa807cd43e336382dfce572b7bd"},
            "mask-right.png": {"path": "/local_scratch2/gzappavi/r3_stage3/input/mask-right.png", "sha256": "f7c53ec340a68e501182bb2987750bfaecb7dfc9d57536edbf36a6c629820b56"},
        },
        "stop_after_stage": True,
    }
    auth = amendment["authorization_record"]
    auth["command"] = (
        f"CUDA_VISIBLE_DEVICES={auth['gpu_uuid']} uv run --no-sync --locked "
        "python -m multi_sample_inference.r3_generator_pair "
        "--protocol docs/r3_protocol_v12.json --matrix docs/r3_test_matrix_v3.json "
        f"--source {auth['source']['path']} --output {auth['output']}"
    )
    protocol.update({
        "schema_version": 12,
        "protocol_id": "r3-gpu-contracts-v12",
        "lineage": {"protocol_id": "r3-gpu-contracts-v11",
                    "sha256": sha256_file(ROOT / "docs/r3_protocol_v11.json")},
        "claim_boundary": (
            "Only one Bootes single-rank official-pristine/custom-none generator pair is approved; "
            "stop after the pair attempt, with no repeat, other jobs, distributed execution, "
            "scientific claim, or later stage."
        ),
        "amendment_rule": (
            "Stop after one single-rank-generator-canary pair attempt, including partial failure. "
            "Any subsequent stage or retry requires fresh approval and a versioned amendment."
        ),
    })
    return protocol


def test_synthetic_v12_stage_is_scoped_without_approving_generic_execution():
    protocol = _synthetic_v12()
    matrix = json.loads(MATRIX.read_text())
    assert validate_protocol(protocol, matrix, matrix_sha256=sha256_file(MATRIX))
    assert stage_execution_blockers(protocol, pair.STAGE) == []
    assert "staged-execution-gates" in execution_blockers(protocol)
    assert "stage-not-authorized:single-rank-contract-cases" in stage_execution_blockers(
        protocol, "single-rank-contract-cases"
    )


@pytest.mark.parametrize("change", [
    lambda p: p["lineage"].update(sha256="0" * 64),
    lambda p: p["execution_amendment"]["authorization_record"]["source"].update(sha256="invalid"),
    lambda p: p["execution_amendment"]["authorization_record"].update(command="python -m other"),
    lambda p: p["execution_amendment"]["authorization_record"].update(output="/tmp/unbound"),
    lambda p: p["execution_amendment"]["stage_gates"][3].update(authorization="approved"),
    lambda p: p["execution_amendment"]["prerequisite_evidence"].pop(),
    lambda p: p["execution_amendment"]["authorization_record"].update(stop_after_stage=False),
    lambda p: p["execution_amendment"]["authorization_record"].update(min_free_gpu_bytes=69 * 1024**3),
    lambda p: p["execution_amendment"]["authorization_record"].update(job_timeout_seconds=599),
    lambda p: p["execution_amendment"]["authorization_record"].update(job_timeout_seconds=3601),
    lambda p: p["execution_amendment"]["authorization_record"]["assets"]["mask-left.png"].update(sha256="0" * 64),
    lambda p: p["execution_amendment"]["authorization_record"]["assets"].pop("mask-right.png"),
])
def test_synthetic_v12_rejects_authorization_changes(change):
    protocol = _synthetic_v12()
    change(protocol)
    matrix = json.loads(MATRIX.read_text())
    with pytest.raises(ValueError):
        validate_protocol(protocol, matrix, matrix_sha256=sha256_file(MATRIX))


def _synthetic_v13():
    protocol = json.loads((ROOT / "docs/r3_protocol_v12.json").read_text())
    protocol["schema_version"] = 13
    protocol["protocol_id"] = "r3-gpu-contracts-v13"
    protocol["lineage"] = {"protocol_id": "r3-gpu-contracts-v12",
                           "sha256": sha256_file(ROOT / "docs/r3_protocol_v12.json")}
    amendment = protocol["execution_amendment"]
    amendment["amendment_id"] = "r3-gpu-execution-amendment-v13"
    amendment["source_binding"]["parent_revision_at_freeze"] = "f" * 40
    amendment["source_binding"]["parent_production_content_sha256"] = "a" * 64
    for name in ("r3_contracts.py", "fsdp_worker.py", "r3_generator_pair.py"):
        amendment["production_component_hashes"][name] = "b" * 64
    auth = amendment["authorization_record"]
    auth["source"] = {"path": "/local_scratch2/gzappavi/r3_stage3/input/source-fp32.json",
                      "sha256": "528fc048f0956c85e4a17efe281ccc752a7876e37de5a4a1eef784d5e476df3a"}
    auth["output"] = "/local_scratch2/gzappavi/r3_stage3/generator-pair-fp32-attempt"
    auth["command"] = auth["command"].replace("r3_protocol_v12.json", "r3_protocol_v13.json").replace(
        "source.json", "source-fp32.json").replace(
        "generator-pair-attempt", "generator-pair-fp32-attempt")
    return protocol


def test_v13_retry_is_bounded_and_rejects_changed_inputs():
    protocol = _synthetic_v13()
    matrix = json.loads(MATRIX.read_text())
    assert validate_protocol(protocol, matrix, matrix_sha256=sha256_file(MATRIX))
    assert stage_execution_blockers(protocol, pair.STAGE) == []
    assert "staged-execution-gates" in execution_blockers(protocol)
    for change in (
        lambda p: p["execution_amendment"]["authorization_record"].update(output="/tmp/other"),
        lambda p: p["execution_amendment"]["authorization_record"]["source"].update(sha256="0" * 64),
        lambda p: p["execution_amendment"]["production_component_hashes"].update(experiment_pipeline_py="0" * 64),
        lambda p: p["execution_amendment"]["stage_gates"][3].update(authorization="approved"),
    ):
        changed = copy.deepcopy(protocol)
        change(changed)
        with pytest.raises(ValueError):
            validate_protocol(changed, matrix, matrix_sha256=sha256_file(MATRIX))


def _pair_fixture(protocol):
    auth = protocol["execution_amendment"]["authorization_record"]
    source = {
        "smoke_only": False,
        "scenes": [{"assignments": [{}], "reference_image": "/tmp/reference.png",
                    "segmentation_masks": {"a": "/tmp/mask-left.png", "b": "/tmp/mask-right.png"}}],
        "video_seeds": [{}],
        "conditions": [{"id": "official"}, {"id": "none"}],
        "inference": {"rank_count": 1},
        "r3_evidence": {"parity_pair": {"pair_id": "pair", "upstream_condition_id": "official",
                                        "custom_none_condition_id": "none"}},
    }
    common = {
        "experiment_id": "exp", "scene_id": "scene", "assignment_id": "assignment",
        "video_seed_id": "seed", "actor_order": ["a", "b"],
    }
    manifests = []
    for route, method, condition in (
        ("official-pristine", "upstream", "official"),
        ("local-custom", "none", "none"),
    ):
        label = "upstream" if method == "upstream" else "custom-none"
        manifests.append({
            "job_id": "job-" + condition,
            "identity": common | {"condition_id": condition},
            "intervention": {"method": method},
            "source": copy.deepcopy(auth["source"]), "smoke_only": False,
            "video_seed": {"id": "seed", "value": 4},
            "assignment": {"targets": {}},
            "prompts": {}, "assets": {}, "checkpoint": {}, "inference": {"rank_count": 1},
            "outputs": {"artifact_dir": auth["output"] + "/" + condition,
                        "task_path": auth["output"] + "/" + condition + "/task.pkl"},
            "r3_evidence": {
                "protocol": {"path": "/tmp/synthetic-v12.json", "sha256": "a" * 64},
                "matrix": {"path": str(MATRIX), "sha256": "b" * 64},
                "route_process": protocol["execution_amendment"]["route_process_bindings"][route],
                "case": {"lineage_disposition": {"disposition": "runnable"}},
                "parity_artifact": {"route": label, "pair_id": "pair"},
            },
        })
    return source, manifests


@pytest.mark.parametrize("mutation", [
    lambda s, m: s.update(smoke_only=True),
    lambda s, m: s["scenes"].append({"assignments": [{}]}),
    lambda s, m: m.append(copy.deepcopy(m[0])),
    lambda s, m: m[1].update(job_id=m[0]["job_id"]),
    lambda s, m: m[0]["source"].update(sha256="0" * 64),
    lambda s, m: m[0]["r3_evidence"].update(route_process=m[1]["r3_evidence"]["route_process"]),
    lambda s, m: m[1]["inference"].update(rank_count=2),
])
def test_pair_plan_rejects_extra_job_changed_input_route_and_smoke(monkeypatch, mutation):
    protocol = _synthetic_v12()
    source, manifests = _pair_fixture(protocol)
    monkeypatch.setattr(pair, "validate_checkout_route_binding", lambda _binding: True)
    assert pair.validate_pair_plan(protocol, source, manifests,
                                   Path(protocol["execution_amendment"]["authorization_record"]["output"]))
    mutation(source, manifests)
    with pytest.raises(ValueError):
        pair.validate_pair_plan(protocol, source, manifests,
                                Path(protocol["execution_amendment"]["authorization_record"]["output"]))


def test_pair_worker_command_uses_exact_isolated_single_rank_route(monkeypatch, tmp_path):
    protocol = _synthetic_v12()
    source, manifests = _pair_fixture(protocol)
    del source
    for manifest in manifests:
        root = tmp_path / manifest["identity"]["condition_id"]
        root.mkdir()
        manifest["r3_evidence"]["route_process"] = {
            **manifest["r3_evidence"]["route_process"], "wan_root": str(root)
        }
    monkeypatch.setattr(pipeline, "load_protocol_bundle", lambda *_args: {"protocol": protocol})
    monkeypatch.setattr(
        pipeline, "worker_task_blueprint",
        lambda manifest: {"r3_evidence": {"route_process": manifest["r3_evidence"]["route_process"]}},
    )
    for manifest in manifests:
        prepare, worker = pipeline._pipeline_commands(Path("/tmp/pair-job.json"), manifest)
        assert prepare[2] == "multi_sample_inference.manifest_adapter"
        assert worker[1:3] == ["-I", "-c"]
        assert json.loads(worker[4])[2] == "torch.distributed.run"
        assert json.loads(worker[4])[0] == manifest["r3_evidence"]["route_process"]["wan_root"]
        assert "--nproc_per_node=1" in worker
        assert "multi_sample_inference.fsdp_worker" in worker


def test_bound_assets_rechecked_and_source_paths_fixed(tmp_path):
    protocol = _synthetic_v12()
    auth = protocol["execution_amendment"]["authorization_record"]
    auth["source"]["path"] = str(tmp_path / "input.json")
    source, _manifests = _pair_fixture(protocol)
    for name in auth["assets"]:
        path = tmp_path / name
        path.write_bytes(name.encode())
        auth["assets"][name] = {"path": str(path), "sha256": sha256_file(path)}
    scene = source["scenes"][0]
    scene["reference_image"] = "reference.png"
    scene["actors"] = [
        {"id": actor, "isolated_image": "reference.png"} for actor in ("a", "b")
    ]
    scene["segmentation_masks"] = {"a": "mask-left.png", "b": "mask-right.png"}
    pair._verify_bound_assets(auth, source)
    (tmp_path / "mask-left.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="asset changed"):
        pair._verify_bound_assets(auth, source)
    (tmp_path / "mask-left.png").write_bytes(b"mask-left.png")
    scene["reference_image"] = "mask-left.png"
    with pytest.raises(ValueError, match="reference or isolated images differ"):
        pair._verify_bound_assets(auth, source)
    scene["reference_image"] = "reference.png"
    scene["actors"][0]["isolated_image"] = "mask-left.png"
    with pytest.raises(ValueError, match="reference or isolated images differ"):
        pair._verify_bound_assets(auth, source)


def test_bounded_command_kills_descendant_on_timeout(tmp_path):
    child_pid = tmp_path / "child.pid"
    script = ("import subprocess,time; from pathlib import Path; "
              f"p=subprocess.Popen(['sleep','30']);Path({str(child_pid)!r}).write_text(str(p.pid));"
              "time.sleep(30)")
    with pytest.raises(subprocess.TimeoutExpired):
        pair._run_bounded_command([sys.executable, "-c", script], tmp_path, 2)
    pid = int(child_pid.read_text())
    stat = Path(f"/proc/{pid}/stat")
    try:
        state = stat.read_text().split(") ", 1)[1][0]
    except OSError:
        state = "gone"
    assert state in {"Z", "gone"}


@pytest.mark.parametrize("failure", [RuntimeError("worker failed"),
                                        subprocess.TimeoutExpired("torchrun", 1),
                                        KeyboardInterrupt()])
def test_failed_first_job_never_launches_custom_route(failure, tmp_path):
    launched = []
    completed = []

    def launch(route, _manifest):
        launched.append(route)
        raise failure

    with pytest.raises(type(failure)) as caught:
        pair._run_pair_jobs({"upstream": {}, "custom-none": {}}, completed, launch)
    pair._record_failed_attempt(tmp_path, {"r3_acceptance": False}, completed, caught.value)
    assert launched == ["upstream"] and completed == []
    record = json.loads((tmp_path / "attempt.json").read_text())
    assert record["status"] == "failed" and record["r3_acceptance"] is False


def test_interrupt_terminates_process_group(monkeypatch, tmp_path):
    process = type("InterruptedProcess", (), {"pid": 12345,
                                               "wait": lambda self, timeout: (_ for _ in ()).throw(KeyboardInterrupt())})()
    terminated = []
    monkeypatch.setattr(pair.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(pair, "_terminate_process_group", terminated.append)
    with pytest.raises(KeyboardInterrupt):
        pair._run_bounded_command([sys.executable, "-c", "pass"], tmp_path, 1)
    assert terminated == [process]


def test_failed_job_stops_next_command_and_records_nonpassing_attempt(tmp_path):
    sentinel = tmp_path / "second-command-started"
    commands = [[sys.executable, "-c", "raise SystemExit(3)"],
                [sys.executable, "-c", f"from pathlib import Path;Path({str(sentinel)!r}).touch()"]]
    with pytest.raises(subprocess.CalledProcessError) as failure:
        pair._run_job_commands(commands, tmp_path, 600)
    assert failure.value.returncode == 3
    assert not sentinel.exists()
    pair._record_failed_attempt(tmp_path, {"r3_acceptance": False}, [], failure.value)
    record = json.loads((tmp_path / "attempt.json").read_text())
    assert record["status"] == "failed" and record["jobs"] == []
    assert record["r3_acceptance"] is False
    assert pair._record_failed_attempt(tmp_path, {"r3_acceptance": False}, [], failure.value) is None
    with pytest.raises(FileExistsError):
        pair._record_failed_attempt(tmp_path, {"r3_acceptance": True}, [], failure.value)


def test_runner_rejects_unfrozen_protocol_without_gpu_access(tmp_path):
    with pytest.raises(ValueError, match="exact frozen v12"):
        pair.run_generator_pair(ROOT / "docs/r3_protocol_v11.json", MATRIX,
                                tmp_path / "source.json", tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_historical_v11_is_immutable_and_stage_three_blocked():
    bundle = load_protocol_bundle(ROOT / "docs/r3_protocol_v11.json", MATRIX)
    assert bundle["protocol_sha256"] == "374d5037111ec04b7e304ce07e44c261827fdf66c13e9d363b3960209f21f34b"
    assert "stage-not-authorized:single-rank-generator-canary" in stage_execution_blockers(
        bundle["protocol"], pair.STAGE
    )
