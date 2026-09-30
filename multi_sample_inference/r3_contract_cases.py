"""One-shot Bootes R3 stage 4: every runnable single-rank contract case.

Never a general M1 execution entry point. The runner binds four frozen sources
(one per solver x requested backend), expands them to exactly the runnable
single-rank generator cases of the v3 matrix, runs each job in its isolated
route process, validates every job's source-hook evidence, compares the two
declared upstream/custom-none parity pairs, and stops at the first failure.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import os
import sys
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
from .r3_generator_pair import (
    _preflight_runtime,
    _run_job_commands,
    _verify_bound_assets,
    _verify_prerequisites,
)

STAGE = "single-rank-contract-cases"
SCHEMA_VERSION = 18
JOB_FAMILIES = {"custom-generator", "upstream-generator"}
PARITY_FAMILY = "upstream-custom-none-parity"
PARITY_CONDITIONS = {"upstream": "upstream-joint", "custom-none": "custom-none-joint"}


def _require(value, message):
    if not value:
        raise ValueError(message)


def expected_single_rank_cases(matrix):
    """Return runnable single-rank job case ids and aggregate parity selections."""
    runnable = [case for case in matrix["lineage_cases"]
                if case["disposition"] == "runnable"
                and case["selection"].get("rank_mode") == "single"]
    jobs = {case["case_id"] for case in runnable if case["family"] in JOB_FAMILIES}
    parity = {case["case_id"]: case["selection"] for case in runnable
              if case["family"] == PARITY_FAMILY}
    return jobs, parity


def _route_name(manifest):
    return "upstream" if manifest["intervention"]["method"] == "upstream" else "custom-none"


def validate_case_plan(protocol, matrix, sources, manifests, output):
    """Bind every expanded job to exactly one runnable single-rank matrix case."""
    amendment = protocol["execution_amendment"]
    declared_sources = amendment["authorization_record"]["sources"]
    routes = amendment["route_process_bindings"]
    job_cases, parity_cases = expected_single_rank_cases(matrix)
    _require(set(sources) == set(declared_sources) == set(manifests),
             "R3 contract-case sources differ from the bounded declaration")
    seen_cases, seen_jobs, plan, pairs = set(), set(), [], {}
    for name in sorted(sources):
        path, source = sources[name]
        declared = declared_sources[name]
        _require(str(path) == declared["path"]
                 and source.get("smoke_only") is False
                 and len(source.get("scenes", [])) == 1
                 and len(source.get("video_seeds", [])) == 1
                 and len(source["scenes"][0].get("assignments", [])) == 1
                 and source.get("inference", {}).get("rank_count") == 1,
                 f"R3 contract-case source {name} is not one bound single-rank scene")
        pair = source.get("r3_evidence", {}).get("parity_pair")
        for manifest in manifests[name]:
            r3 = manifest["r3_evidence"]
            route = _route_name(manifest)
            binding = "official-pristine" if route == "upstream" else "local-custom"
            case = r3["case"]
            _require(r3.get("route_process") == routes[binding]
                     and manifest["source"]["path"] == declared["path"]
                     and manifest["source"]["sha256"] == declared["sha256"]
                     and manifest["smoke_only"] is False
                     and manifest["inference"]["rank_count"] == 1
                     and Path(manifest["outputs"]["artifact_dir"]).is_relative_to(output / name)
                     and case["lineage_disposition"]["disposition"] == "runnable"
                     and case["lineage_disposition"]["selection"]["rank_mode"] == "single",
                     "R3 contract-case job route/source/output/case differs from the declaration")
            _require(case["case_id"] in job_cases and case["case_id"] not in seen_cases
                     and manifest["job_id"] not in seen_jobs,
                     "R3 contract-case job is unexpected or duplicated")
            seen_cases.add(case["case_id"])
            seen_jobs.add(manifest["job_id"])
            parity = r3.get("parity_artifact")
            if parity is not None:
                _require(isinstance(pair, dict) and parity["route"] == route
                         and parity["pair_id"] == pair["pair_id"]
                         and manifest["identity"]["condition_id"] == PARITY_CONDITIONS[route]
                         and route not in pairs.get(name, {}),
                         "R3 contract-case parity job differs from its declared pair")
                pairs.setdefault(name, {})[route] = manifest
            plan.append((name, manifest))
        _require((pair is None and name not in pairs)
                 or (pair is not None and set(pairs.get(name, {})) == set(PARITY_CONDITIONS)),
                 f"R3 contract-case source {name} parity pair is incomplete")
    _require(seen_cases == job_cases,
             "R3 contract-case plan does not cover exactly the runnable single-rank cases")
    covered = {(sources[name][1]["inference"]["solver"],
                sources[name][1]["r3_evidence"]["attention_backend"]) for name in pairs}
    _require(covered == {(selection["solver"], selection["attention_backend"])
                         for selection in parity_cases.values()},
             "R3 contract-case parity pairs do not match the aggregate parity cases")
    for binding in routes.values():
        validate_checkout_route_binding(binding)
    return plan, pairs


def order_jobs(plan, pairs):
    """Run known parity pairs first, then round-robin across method/mask/backend groups.

    Stop-on-first-failure makes early diversity valuable: a systematic defect in
    any method or mask configuration surfaces within the first few jobs.
    """
    first = [(name, pairs[name][route]) for name in sorted(pairs)
             for route in ("upstream", "custom-none")]
    first_ids = {manifest["job_id"] for _name, manifest in first}
    groups = {}
    for name, manifest in plan:
        if manifest["job_id"] in first_ids:
            continue
        selection = manifest["r3_evidence"]["case"]["lineage_disposition"]["selection"]
        key = (selection.get("method", "upstream"), selection.get("mask_configuration", ""),
               selection["attention_backend"])
        groups.setdefault(key, []).append((name, manifest))
    # Greedily order groups so each next group adds the most unseen axis values.
    remaining, keys, seen = sorted(groups), [], [set(), set(), set()]
    def gain(key):
        return sum(key[axis] not in seen[axis] for axis in range(3))

    while remaining:
        if not any(gain(key) for key in remaining):
            seen = [set(), set(), set()]
        best = min(remaining, key=lambda key: (-gain(key), key))
        remaining.remove(best)
        keys.append(best)
        for axis in range(3):
            seen[axis].add(best[axis])
    rest = [job for batch in itertools.zip_longest(*(groups[key] for key in keys))
            for job in batch if job is not None]
    ordered = first + rest
    _require(len(ordered) == len(plan) and len({m["job_id"] for _n, m in ordered}) == len(plan),
             "R3 contract-case ordering lost or duplicated a job")
    return ordered


def _verify_contract_prerequisites(repo, declarations):
    _require(isinstance(declarations, list) and len(declarations) == 3,
             "R3 contract cases need three prerequisite records")
    _verify_prerequisites(repo, declarations[:2])
    declaration = declarations[2]
    path = (repo / declaration["path"]).resolve()
    _require(path.is_relative_to(repo) and path.is_file()
             and sha256_file(path) == declaration["sha256"],
             "R3 prerequisite single-rank-generator-canary changed or is missing")
    record = _read_json(path)
    _require(declaration["stage_id"] == "single-rank-generator-canary"
             and declaration["status"] == "passed"
             and record.get("stage_id") == declaration["stage_id"]
             and record.get("status") == "passed"
             and record.get("protocol_sha256") == declaration["protocol_sha256"]
             and record.get("comparison", {}).get("measurements", {}).get("passed") is True,
             "R3 prerequisite single-rank-generator-canary did not pass")


def _launch(repo, output, auth, sources, name, manifest):
    manifest_path = output / "jobs" / f"{manifest['job_id']}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_immutable_json(manifest_path, manifest)
    verify_job_inputs(manifest)
    _verify_bound_assets(auth | {"source": auth["sources"][name]}, sources[name][1])
    _require(_canonical(_read_json(manifest_path)) == _canonical(manifest),
             "R3 contract-case manifest changed")
    commands = _pipeline_commands(manifest_path, manifest)
    _require(len(commands) == 2 and commands[0][2] == "multi_sample_inference.manifest_adapter"
             and commands[1][1] == "-I" and commands[1][0] == sys.executable
             and json.loads(commands[1][4]) == [
                 manifest["r3_evidence"]["route_process"]["wan_root"],
                 str(repo), "torch.distributed.run",
             ]
             and "--nproc_per_node=1" in commands[1]
             and "multi_sample_inference.fsdp_worker" in commands[1],
             "R3 contract-case worker commands differ from isolated single-rank route")
    _run_job_commands(commands, repo, auth["job_timeout_seconds"])
    video = Path(manifest["outputs"]["video_path"])
    _require(video.is_file() and video.stat().st_size > 0,
             "R3 contract-case worker did not produce a nonempty video")
    evidence = _verify_declared_r3_worker_evidence(manifest)
    if manifest["r3_evidence"].get("parity_artifact") is not None:
        from .r3_parity import validate_parity_artifact  # noqa: PLC0415

        declaration = worker_task_blueprint(manifest)["r3_evidence"]["parity_artifact"]
        metadata, _tensor = validate_parity_artifact(declaration)
        _require(metadata["evidence_class"] == GENUINE_RUNTIME_EVIDENCE,
                 "R3 contract-case worker did not produce genuine parity evidence")
    selection = manifest["r3_evidence"]["case"]["lineage_disposition"]["selection"]
    return {"source": name, "job_id": manifest["job_id"],
            "case_id": manifest["r3_evidence"]["case"]["case_id"], "selection": selection,
            "video_sha256": _hash_file(video), "evidence": evidence}


def _compare_pair(protocol, source, pair):
    from .r3_parity import compare_parity_artifacts  # noqa: PLC0415

    artifacts = {route: worker_task_blueprint(manifest)["r3_evidence"]["parity_artifact"]
                 for route, manifest in pair.items()}
    tolerance = protocol["numerical_tolerances"]["full_generator_parity"]
    comparison = compare_parity_artifacts(
        {"pair_id": source["r3_evidence"]["parity_pair"]["pair_id"], "artifacts": artifacts},
        atol=tolerance["atol"], rtol=tolerance["rtol"],
    )
    _require(comparison["measurements"]["passed"] is True,
             "R3 contract-case offline numerical parity failed")
    return comparison


def run_contract_cases(protocol_path, matrix_path, output):
    repo = Path(__file__).resolve().parents[1]
    protocol_path, matrix_path = Path(protocol_path).resolve(), Path(matrix_path).resolve()
    output = Path(output).resolve()
    _require(protocol_path == repo / f"docs/r3_protocol_v{SCHEMA_VERSION}.json"
             and matrix_path == repo / "docs/r3_test_matrix_v3.json",
             "R3 contract cases require their exact frozen protocol and matrix paths")
    bundle = load_protocol_bundle(protocol_path, matrix_path)
    protocol, matrix = bundle["protocol"], bundle["matrix"]
    _require(protocol["schema_version"] == SCHEMA_VERSION, "R3 contract-case protocol mismatch")
    auth = protocol["execution_amendment"]["authorization_record"]
    _require(auth["authorized_stage"] == STAGE
             and os.environ.get("CUDA_VISIBLE_DEVICES") == auth["gpu_uuid"]
             and output == Path(auth["output"])
             and output.is_relative_to("/local_scratch2/gzappavi")
             and not output.is_relative_to(repo)
             and not output.exists(), "R3 contract-case GPU/output differs or attempt exists")
    sources = {}
    for name, declared in auth["sources"].items():
        path = Path(declared["path"])
        _require(sha256_file(path) == declared["sha256"],
                 f"R3 contract-case source changed: {name}")
        sources[name] = (path, _read_json(path))
    blockers = stage_execution_blockers(protocol, STAGE)
    _require(not blockers, "R3 contract-case preflight blocked: " + ", ".join(blockers))
    _verify_contract_prerequisites(repo, protocol["execution_amendment"]["prerequisite_evidence"])
    repositories = {name: repository_identity(path) for name, path in (
        ("parent", repo), ("wan", repo / "wan2.1"), ("lama", repo / "lama"))}
    _validate_v4_source_binding(repo, repositories, protocol)
    manifests = {name: expand_source(path, output / name, write=False)
                 for name, (path, _source) in sources.items()}
    plan, pairs = validate_case_plan(protocol, matrix, sources, manifests, output)
    ordered = order_jobs(plan, pairs)
    _require(len(ordered) == auth["expected_jobs"],
             "R3 contract-case job count differs from the bounded declaration")
    for name in sources:
        _verify_bound_assets(auth | {"source": auth["sources"][name]}, sources[name][1])
    # All expensive checkpoint/hardware checks precede any model or task launch.
    checkpoint = _preflight_runtime(protocol, repo, protocol_path, auth["gpu_uuid"])
    # Exclusive creation is the irreversible one-attempt marker. Never reuse this directory.
    output.mkdir(parents=True, exist_ok=False)
    base = {
        "stage_id": STAGE,
        "protocol_sha256": bundle["protocol_sha256"],
        "matrix_sha256": bundle["matrix_sha256"],
        "source_sha256": {name: declared["sha256"] for name, declared in auth["sources"].items()},
        "checkpoint_content_sha256": checkpoint["content_sha256"],
        "gpu_uuid": auth["gpu_uuid"],
        "started_at": dt.datetime.now(dt.UTC).isoformat(),
        "planned_jobs": len(ordered),
        "r3_acceptance": False,
    }
    completed, comparisons, current = [], {}, None
    try:
        for index, (name, manifest) in enumerate(ordered):
            current = manifest["job_id"]
            result = _launch(repo, output, auth, sources, name, manifest)
            write_immutable_json(output / "results" / f"{index:03d}-{manifest['job_id']}.json",
                                 result)
            completed.append({key: result[key] for key in
                              ("source", "job_id", "case_id", "video_sha256")})
            done = {job["job_id"] for job in completed}
            for pair_source, pair in pairs.items():
                if pair_source not in comparisons and all(
                        member["job_id"] in done for member in pair.values()):
                    comparisons[pair_source] = _compare_pair(
                        protocol, sources[pair_source][1], pair)
            print(json.dumps({"completed": len(completed), "of": len(ordered),
                              "case_id": result["case_id"]}), flush=True)
        _require(set(comparisons) == set(pairs), "R3 contract-case parity comparison missing")
        result = base | {"status": "passed", "jobs": completed, "comparisons": comparisons,
                         "finished_at": dt.datetime.now(dt.UTC).isoformat()}
    except BaseException as error:
        write_immutable_json(output / "attempt.json", base | {
            "status": "failed", "jobs": completed, "comparisons": comparisons,
            "failed_job": current,
            "error": f"{type(error).__name__}: {error}",
        })
        raise
    write_immutable_json(output / "attempt.json", result)
    return {key: result[key] for key in ("stage_id", "status", "planned_jobs", "protocol_sha256")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(run_contract_cases(args.protocol, args.matrix, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
