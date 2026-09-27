"""One-shot Bootes R3 generator pair; never a general M1 execution entry point."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

from .experiment_pipeline import (
    _canonical,
    _hash_file,
    _pipeline_commands,
    _read_json,
    _validate_v4_source_binding,
    _verify_declared_r3_worker_evidence,
    expand_source,
    repository_identity,
    verify_job_inputs,
    worker_task_blueprint,
)
from .r3_checkout_binding import validate_checkout_route_binding
from .r3_contracts import (
    GENUINE_RUNTIME_EVIDENCE,
    load_protocol_bundle,
    sha256_file,
    stage_execution_blockers,
    write_immutable_json,
)
from .r3_preflight import validate_backend_runtime, validate_v4_runtime_environment

STAGE = "single-rank-generator-canary"


def _require(value, message):
    if not value:
        raise ValueError(message)


def validate_pair_plan(protocol, source, manifests, output):
    """Reject Cartesian expansions, route swaps and undeclared job/output changes."""
    authorization = protocol["execution_amendment"]["authorization_record"]
    pair = source.get("r3_evidence", {}).get("parity_pair")
    _require(source.get("smoke_only") is False and len(source.get("scenes", [])) == 1
             and len(source.get("video_seeds", [])) == 1
             and len(source["scenes"][0].get("assignments", [])) == 1
             and isinstance(pair, dict) and len(source.get("conditions", [])) == 2
             and source.get("inference", {}).get("rank_count") == 1,
             "R3 pair requires one non-smoke scene/assignment/seed and exactly two conditions")
    _require(len(manifests) == 2, "R3 pair source must expand to exactly two jobs")
    routes = protocol["execution_amendment"]["route_process_bindings"]
    by_route = {}
    for manifest in manifests:
        r3 = manifest["r3_evidence"]
        method = manifest["intervention"]["method"]
        route = "upstream" if method == "upstream" else "custom-none"
        binding_name = "official-pristine" if route == "upstream" else "local-custom"
        _require(method in {"upstream", "none"} and route not in by_route
                 and r3.get("route_process") == routes[binding_name]
                 and r3.get("parity_artifact", {}).get("route") == route
                 and r3["parity_artifact"]["pair_id"] == pair["pair_id"]
                 and manifest["source"]["sha256"] == authorization["source"]["sha256"]
                 and manifest["source"]["path"] == authorization["source"]["path"]
                 and manifest["smoke_only"] is False
                 and manifest["inference"]["rank_count"] == 1
                 and Path(manifest["outputs"]["artifact_dir"]).is_relative_to(output),
                 "R3 pair job route/source/output differs from the bounded declaration")
        expected_condition = pair["upstream_condition_id" if route == "upstream"
                                  else "custom_none_condition_id"]
        _require(manifest["identity"]["condition_id"] == expected_condition
                 and r3["case"]["lineage_disposition"]["disposition"] == "runnable",
                 "R3 pair condition or case differs from the declared pair")
        by_route[route] = manifest
    _require(set(by_route) == {"upstream", "custom-none"}, "R3 pair is incomplete")
    upstream, custom = by_route["upstream"], by_route["custom-none"]
    _require(upstream["job_id"] != custom["job_id"], "R3 pair job identities must be distinct")
    for key in ("experiment_id", "scene_id", "assignment_id", "video_seed_id", "actor_order"):
        _require(upstream["identity"][key] == custom["identity"][key],
                 f"R3 pair {key} differs")
    for key in ("source", "video_seed", "assignment", "prompts", "assets", "checkpoint"):
        _require(upstream[key] == custom[key], f"R3 pair {key} differs")
    _require(upstream["inference"] == custom["inference"]
             and upstream["r3_evidence"]["protocol"] == custom["r3_evidence"]["protocol"],
             "R3 pair inference or protocol differs")
    for route in ("upstream", "custom-none"):
        validate_checkout_route_binding(by_route[route]["r3_evidence"]["route_process"])
    return by_route


def _verify_bound_assets(authorization, source):
    """Rehash exactly the three source-selected synthetic assets before each launch."""
    bound = authorization["assets"]
    scene = source["scenes"][0]
    source_root = Path(authorization["source"]["path"]).parent

    def resolve(path):
        candidate = Path(path)
        return (candidate if candidate.is_absolute() else source_root / candidate).resolve()

    reference_path = Path(bound["reference.png"]["path"])
    _require(resolve(scene["reference_image"]) == reference_path
             and len(scene["actors"]) == 2
             and all(resolve(actor["isolated_image"]) == reference_path
                     for actor in scene["actors"]),
             "R3 pair reference or isolated images differ from bound asset")
    masks = scene["segmentation_masks"]
    _require(isinstance(masks, dict) and len(masks) == 2
             and {resolve(path) for path in masks.values()}
             == {Path(bound[name]["path"]) for name in ("mask-left.png", "mask-right.png")},
             "R3 pair segmentation masks differ from bound assets")
    for name, asset in bound.items():
        _require(sha256_file(asset["path"]) == asset["sha256"],
                 f"R3 pair bound synthetic asset changed: {name}")


def _terminate_process_group(process):
    """Stop torchrun descendants even when the launcher exits on SIGTERM."""
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=3)
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def _run_bounded_command(command, repo, timeout):
    # Wan imports precede worker checkout validation; never let either pair process
    # create ignored bytecode that makes the next clean-checkout gate fail.
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.Popen(command, cwd=repo, env=environment, start_new_session=True)
    try:
        exit_code = process.wait(timeout=timeout)
    except BaseException:
        _terminate_process_group(process)
        raise
    # Also clear surviving descendants if a launcher exits before its workers.
    _terminate_process_group(process)
    if exit_code != 0:
        raise subprocess.CalledProcessError(exit_code, command)


def _run_job_commands(commands, repo, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    for command in commands:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(command, timeout_seconds)
        _run_bounded_command(command, repo, remaining)


def _run_pair_jobs(by_route, completed, launch):
    for route in ("upstream", "custom-none"):
        completed.append(launch(route, by_route[route]))


def _record_failed_attempt(output, base, completed, error):
    result = base | {"status": "failed", "jobs": completed,
                     "error": f"{type(error).__name__}: {error}"}
    write_immutable_json(output / "attempt.json", result)


def _verify_prerequisites(repo, declarations):
    _require(isinstance(declarations, list) and len(declarations) == 2,
             "R3 pair needs two prerequisite records")
    for index, stage in enumerate(("backend-kernel-canary", "checkpoint-load-hook-canary")):
        declaration = declarations[index]
        path = (repo / declaration["path"]).resolve()
        _require(path.is_relative_to(repo) and path.is_file()
                 and sha256_file(path) == declaration["sha256"],
                 f"R3 prerequisite {stage} changed or is missing")
        record = _read_json(path)
        _require(declaration["stage_id"] == stage and declaration["status"] == "passed"
                 and record.get("stage_id") == stage and record.get("status") == "passed"
                 and record.get("bindings", {}).get("protocol_sha256")
                 == declaration["protocol_sha256"]
                 and record.get("scope", {}).get("generation_performed") is False
                 and record.get("scope", {}).get("distributed_execution_performed") is False,
                 f"R3 prerequisite {stage} has invalid stage/scope/binding")


def _preflight_runtime(protocol, repo, protocol_path, gpu):
    # Delay Torch imports until all CPU/source/route checks have succeeded.
    import torch  # noqa: PLC0415

    from .r3_checkpoint_hook_canary import _verify_checkpoint  # noqa: PLC0415
    from .r3_environment import (  # noqa: PLC0415
        observe_attention_runtime,
        observe_runtime_environment,
    )

    attention = observe_attention_runtime()
    declarations = protocol["runtime_declarations"]
    validate_backend_runtime({"backend_versions": {
        "flash_attention_2": declarations["flash_attention_version"],
        "flex_attention": declarations["flex_attention_version"],
    }}, attention)
    validate_v4_runtime_environment(protocol, observe_runtime_environment(attention))
    checkpoint = _verify_checkpoint(repo, protocol, protocol_path)
    _require(checkpoint["content_sha256"] == declarations["checkpoint_content_sha256"],
             "R3 pair checkpoint declaration differs from rehashed contents")
    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1
             and os.environ.get("CUDA_VISIBLE_DEVICES") == gpu,
             "R3 pair requires only its selected visible GPU")
    free, _total = torch.cuda.mem_get_info(0)
    _require(free >= protocol["execution_amendment"]["authorization_record"]["min_free_gpu_bytes"],
             "R3 pair GPU free memory is below the declared floor")
    return checkpoint


def run_generator_pair(protocol_path, matrix_path, source_path, output):  # noqa: PLR0915
    repo = Path(__file__).resolve().parents[1]
    protocol_path, matrix_path = Path(protocol_path).resolve(), Path(matrix_path).resolve()
    source_path, output = Path(source_path).resolve(), Path(output).resolve()
    _require(protocol_path in {repo / f"docs/r3_protocol_v{version}.json"
                               for version in (12, 13, 14, 15)}
             and matrix_path == repo / "docs/r3_test_matrix_v3.json",
             "R3 pair requires its exact frozen v12–v15 protocol and matrix paths")
    bundle = load_protocol_bundle(protocol_path, matrix_path)
    protocol = bundle["protocol"]
    _require(protocol["schema_version"] in {12, 13, 14, 15}
             and protocol_path.name == f"r3_protocol_v{protocol['schema_version']}.json",
             "R3 pair protocol version/path mismatch")
    auth = protocol["execution_amendment"]["authorization_record"]
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == auth["gpu_uuid"]
             and source_path == Path(auth["source"]["path"])
             and output == Path(auth["output"])
             and output.is_relative_to("/local_scratch2/gzappavi")
             and not output.is_relative_to(repo)
             and not output.exists(), "R3 pair input/GPU/output differs or attempt already exists")
    _require(sha256_file(source_path) == auth["source"]["sha256"],
             "R3 pair input source changed")
    blockers = stage_execution_blockers(protocol, STAGE)
    _require(not blockers, "R3 pair preflight blocked: " + ", ".join(blockers))
    _verify_prerequisites(repo, protocol["execution_amendment"]["prerequisite_evidence"])
    repositories = {name: repository_identity(path) for name, path in (
        ("parent", repo), ("wan", repo / "wan2.1"), ("lama", repo / "lama"))}
    _validate_v4_source_binding(repo, repositories, protocol)
    source = _read_json(source_path)
    manifests = expand_source(source_path, output, write=False)
    by_route = validate_pair_plan(protocol, source, manifests, output)
    _verify_bound_assets(auth, source)
    # All expensive checkpoint/hardware checks precede any model or task launch.
    checkpoint = _preflight_runtime(protocol, repo, protocol_path, auth["gpu_uuid"])
    # Exclusive creation is the irreversible one-attempt marker. Never reuse this directory.
    output.mkdir(parents=True, exist_ok=False)
    base = {
        "stage_id": STAGE,
        "protocol_sha256": bundle["protocol_sha256"],
        "matrix_sha256": bundle["matrix_sha256"],
        "source_sha256": auth["source"]["sha256"],
        "checkpoint_content_sha256": checkpoint["content_sha256"],
        "gpu_uuid": auth["gpu_uuid"],
        "started_at": dt.datetime.now(dt.UTC).isoformat(),
        "r3_acceptance": False,
    }
    completed = []
    def launch(route, manifest):
        manifest_path = output / "jobs" / f"{manifest['job_id']}.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        write_immutable_json(manifest_path, manifest)
        verify_job_inputs(manifest)
        _verify_bound_assets(auth, source)
        _require(_canonical(_read_json(manifest_path)) == _canonical(manifest),
                 "R3 pair manifest changed")
        commands = _pipeline_commands(manifest_path, manifest)
        _require(len(commands) == 2 and commands[0][2] == "multi_sample_inference.manifest_adapter"
                 and commands[1][1] == "-I" and commands[1][0] == sys.executable
                 and json.loads(commands[1][4]) == [
                     manifest["r3_evidence"]["route_process"]["wan_root"],
                     str(repo), "torch.distributed.run",
                 ]
                 and "--nproc_per_node=1" in commands[1]
                 and "multi_sample_inference.fsdp_worker" in commands[1],
                 "R3 pair worker commands differ from isolated single-rank route")
        _run_job_commands(commands, repo, auth["job_timeout_seconds"])
        video = Path(manifest["outputs"]["video_path"])
        _require(video.is_file() and video.stat().st_size > 0,
                 "R3 pair worker did not produce a nonempty video")
        evidence = _verify_declared_r3_worker_evidence(manifest)
        from .r3_parity import validate_parity_artifact  # noqa: PLC0415

        declaration = worker_task_blueprint(manifest)["r3_evidence"]["parity_artifact"]
        metadata, _tensor = validate_parity_artifact(declaration)
        _require(metadata["evidence_class"] == GENUINE_RUNTIME_EVIDENCE,
                 "R3 pair worker did not produce genuine parity evidence")
        return {"route": route, "job_id": manifest["job_id"],
                "video_sha256": _hash_file(video), "evidence": evidence}

    try:
        _run_pair_jobs(by_route, completed, launch)
        pair = source["r3_evidence"]["parity_pair"]
        artifacts = {}
        for route, manifest in by_route.items():
            artifacts[route] = worker_task_blueprint(manifest)["r3_evidence"]["parity_artifact"]
        tolerance = protocol["numerical_tolerances"]["full_generator_parity"]
        from .r3_parity import compare_parity_artifacts  # noqa: PLC0415

        comparison = compare_parity_artifacts(
            {"pair_id": pair["pair_id"], "artifacts": artifacts},
            atol=tolerance["atol"], rtol=tolerance["rtol"],
        )
        _require(comparison["measurements"]["passed"] is True,
                 "R3 pair offline numerical parity failed")
        result = base | {"status": "passed", "jobs": completed, "comparison": comparison}
    except BaseException as error:
        _record_failed_attempt(output, base, completed, error)
        raise
    write_immutable_json(output / "attempt.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(run_generator_pair(args.protocol, args.matrix, args.source, args.output),
                     sort_keys=True))


if __name__ == "__main__":
    main()
