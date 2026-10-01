"""J1 step-0 cross-architecture gate and report-only Bootes comparisons.

Under an approved jz protocol, ``run_step0_probe`` captures the step-0 conditioning
and first DiT forwards of the bound pair source with the existing
``r3_divergence_probe`` (official-pristine, then local-custom, each in an isolated
single-rank process) and applies the frozen decision (d) rule against the retained
Bootes pristine capture. Bootes pristine and local-fixed captures were bitwise
identical over all 67 shared stages, so the pristine capture is the single
reference for both A100 routes.

``compare_cross_architecture`` is CPU-only. ``report_final_latents`` and
``report_video_psnr`` compare against Bootes v16 for reporting only; they apply
no threshold and never gate J1.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import json
import math
import sys
from pathlib import Path

from .experiment_pipeline import (
    _read_json,
    _validate_v4_source_binding,
    expand_source,
    repository_identity,
    verify_job_inputs,
)
from .r3_contracts import (
    canonical_json_bytes,
    load_protocol_bundle,
    sha256_file,
    stage_execution_blockers,
    write_immutable_json,
)
from .r3_divergence_probe import MANIFEST_NAME, _load_manifest, build_launch_command
from .r3_jz import (
    JZ_CROSS_ARCHITECTURE_DECISION,
    JZ_RUN_ROOT,
    STEP0_STAGE,
    GpuMemorySampler,
    jz_authorization,
    require_jz_protocol_path,
    verify_jz_prerequisites,
)
from .r3_runtime import _tensor_identity_and_finiteness

RECORD_KIND = "j1-step0-cross-architecture-gate"
_CHUNK_ELEMENTS = 1 << 24


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_reference_capture(repo, reference):
    """Bind the retained Bootes capture to its manifest hash and the frozen report's stage hashes."""
    report_path = (Path(repo) / reference["report"]["path"]).resolve()
    _require(report_path.is_file() and sha256_file(report_path) == reference["report"]["sha256"],
             "J1 Bootes reference report changed or is missing")
    directory = Path(reference["path"])
    manifest_path = directory / MANIFEST_NAME
    _require(manifest_path.is_file() and sha256_file(manifest_path) == reference["manifest_sha256"],
             "J1 Bootes reference capture manifest changed or is missing")
    manifest = _load_manifest(directory)
    _require(manifest.get("route") == "official-pristine",
             "J1 Bootes reference capture is not the pristine route")
    report = _read_json(report_path)
    stages = report["stages"]
    _require([record["name"] for record in manifest["records"]]
             == [stage["name"] for stage in stages],
             "J1 Bootes reference capture stages differ from the bound report")
    for record, stage in zip(manifest["records"], stages, strict=True):
        if "file" in record:
            expected = stage.get("reference", {})
            _require({key: record.get(key) for key in ("shape", "dtype", "sha256")} == expected,
                     f"J1 Bootes reference stage identity differs: {record['name']}")
        else:
            _require(record.get("value") == stage.get("reference_value"),
                     f"J1 Bootes reference value differs: {record['name']}")
    return manifest


def _load_verified(directory, record):
    import torch  # noqa: PLC0415

    tensor = torch.load(Path(directory) / record["file"], map_location="cpu",
                        weights_only=True, mmap=True)
    identity, finite = _tensor_identity_and_finiteness(tensor)
    _require(identity == {key: record[key] for key in ("shape", "dtype", "sha256")},
             f"capture tensor file differs from its manifest record: {record['name']}")
    return tensor, finite


def _relative_l2(candidate, reference):
    """float64 ||a-b||_2 / ||b||_2 in bounded chunks; returns (relative, reference_norm)."""
    import torch  # noqa: PLC0415

    if candidate.is_complex():
        candidate, reference = torch.view_as_real(candidate), torch.view_as_real(reference)
    a, b = candidate.reshape(-1), reference.reshape(-1)
    difference = 0.0
    norm = 0.0
    for offset in range(0, b.numel(), _CHUNK_ELEMENTS):
        chunk_a = a[offset:offset + _CHUNK_ELEMENTS].to(torch.float64)
        chunk_b = b[offset:offset + _CHUNK_ELEMENTS].to(torch.float64)
        difference += float(torch.sum((chunk_a - chunk_b) ** 2))
        norm += float(torch.sum(chunk_b ** 2))
    reference_norm = math.sqrt(norm)
    if reference_norm == 0.0:
        return None, reference_norm
    return math.sqrt(difference) / reference_norm, reference_norm


def _compare_stage(reference_dir, candidate_dir, reference, candidate, decision):  # noqa: PLR0911
    name = reference["name"]
    result = {"name": name}
    if candidate is None:
        return result | {"rule": "presence", "passed": False, "reason": "missing-in-candidate"}
    if "file" not in reference or "file" not in candidate:
        equal = ("file" in reference) == ("file" in candidate) and (
            reference.get("value") == candidate.get("value"))
        return result | {"rule": "value-equality", "passed": equal,
                         "reference_value": reference.get("value"),
                         "candidate_value": candidate.get("value")}
    identity = {key: reference[key] for key in ("shape", "dtype")}
    result |= {"reference_sha256": reference["sha256"], "candidate_sha256": candidate["sha256"],
               **identity}
    if identity != {key: candidate[key] for key in ("shape", "dtype")}:
        return result | {"rule": "shape-dtype", "passed": False,
                         "candidate_shape": candidate["shape"],
                         "candidate_dtype": candidate["dtype"]}
    reference_tensor, reference_finite = _load_verified(reference_dir, reference)
    candidate_tensor, candidate_finite = _load_verified(candidate_dir, candidate)
    if not (reference_finite and candidate_finite):
        return result | {"rule": "finiteness", "passed": False,
                         "reference_finite": reference_finite, "candidate_finite": candidate_finite}
    bitwise = reference["sha256"] == candidate["sha256"]
    if any(fnmatch.fnmatchcase(name, pattern) for pattern in decision["bitwise_stage_patterns"]):
        return result | {"rule": "bitwise", "passed": bitwise}
    relative, reference_norm = _relative_l2(candidate_tensor, reference_tensor)
    if relative is None:
        return result | {"rule": "zero-reference-norm-bitwise", "passed": bitwise,
                         "reference_l2_norm": reference_norm}
    limit = decision["relative_l2_max"]
    return result | {"rule": "relative-l2", "passed": math.isfinite(relative) and relative <= limit,
                     "relative_l2": relative, "relative_l2_max": limit,
                     "reference_l2_norm": reference_norm, "bitwise_identical": bitwise}


def compare_cross_architecture(reference_dir, candidate_dir, *, route,
                               decision=JZ_CROSS_ARCHITECTURE_DECISION):
    """Apply the frozen decision (d) rule; any failing stage fails the whole gate."""
    _require(route in decision["allowed_candidate_only_stages"], "unknown J1 probe route")
    reference = _load_manifest(reference_dir)
    candidate = _load_manifest(candidate_dir)
    _require(candidate.get("route") == route, "J1 candidate capture route differs")
    candidate_records = {record["name"]: record for record in candidate["records"]}
    reference_names = [record["name"] for record in reference["records"]]
    stages = [
        _compare_stage(reference_dir, candidate_dir, record,
                       candidate_records.get(record["name"]), decision)
        for record in reference["records"]
    ]
    candidate_only = sorted(set(candidate_records) - set(reference_names))
    allowed = set(decision["allowed_candidate_only_stages"][route])
    unexpected = [name for name in candidate_only if name not in allowed]
    failed = [stage["name"] for stage in stages if not stage["passed"]]
    relative = [stage["relative_l2"] for stage in stages if "relative_l2" in stage]
    return {
        "record_kind": f"{RECORD_KIND}-comparison",
        "r3_acceptance": False,
        "route": route,
        "reference": {"path": str(Path(reference_dir).resolve()), "route": reference.get("route")},
        "candidate": {"path": str(Path(candidate_dir).resolve()), "route": candidate.get("route")},
        "decision": decision,
        "passed": not failed and not unexpected,
        "failed_stages": failed,
        "candidate_only_stages": candidate_only,
        "unexpected_candidate_only_stages": unexpected,
        "max_relative_l2": max(relative) if relative else None,
        "stages": stages,
    }


def report_final_latents(reference_path, candidate_path):
    """Report-only final-latent metrics against Bootes v16; no threshold is applied."""
    import numpy as np  # noqa: PLC0415

    reference = np.load(reference_path, allow_pickle=False)
    candidate = np.load(candidate_path, allow_pickle=False)
    result = {
        "record_kind": "j1-final-latent-report-only",
        "reference": {"path": str(reference_path), "sha256": sha256_file(reference_path),
                      "shape": list(reference.shape), "dtype": str(reference.dtype)},
        "candidate": {"path": str(candidate_path), "sha256": sha256_file(candidate_path),
                      "shape": list(candidate.shape), "dtype": str(candidate.dtype)},
        "threshold": None,
    }
    if reference.shape != candidate.shape:
        return result | {"comparable": False}
    a = candidate.astype(np.float64).reshape(-1)
    b = reference.astype(np.float64).reshape(-1)
    difference = a - b
    norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return result | {
        "comparable": True,
        "all_finite": bool(np.isfinite(a).all() and np.isfinite(b).all()),
        "max_absolute_error": float(np.max(np.abs(difference))),
        "relative_l2": float(np.linalg.norm(difference)) / norm_b if norm_b else None,
        "cosine_similarity": float(a @ b) / (norm_a * norm_b) if norm_a and norm_b else None,
    }


def report_video_psnr(reference_path, candidate_path):
    """Report-only decoded-video PSNR (8-bit, all frames); no threshold is applied."""
    import imageio.v2 as imageio  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    def frames(path):
        with imageio.get_reader(path, format="ffmpeg") as reader:
            return np.stack([np.asarray(frame) for frame in reader])

    reference, candidate = frames(reference_path), frames(candidate_path)
    result = {
        "record_kind": "j1-video-psnr-report-only",
        "reference": {"path": str(reference_path), "sha256": sha256_file(reference_path),
                      "shape": list(reference.shape)},
        "candidate": {"path": str(candidate_path), "sha256": sha256_file(candidate_path),
                      "shape": list(candidate.shape)},
        "threshold": None,
    }
    if reference.shape != candidate.shape:
        return result | {"comparable": False}
    mse = float(np.mean((candidate.astype(np.float64) - reference.astype(np.float64)) ** 2))
    psnr = math.inf if mse == 0.0 else 10.0 * math.log10(255.0 ** 2 / mse)
    return result | {"comparable": True, "mse": mse,
                     "psnr_db": "inf" if math.isinf(psnr) else psnr}


def run_step0_probe(protocol_path, matrix_path, output):  # noqa: PLR0915
    from .r3_generator_pair import (  # noqa: PLC0415
        _run_bounded_command,
        _verify_bound_assets,
        preflight_jz_runtime,
        validate_pair_plan,
    )

    repo = Path(__file__).resolve().parents[1]
    protocol_path, matrix_path = Path(protocol_path).resolve(), Path(matrix_path).resolve()
    output = Path(output).resolve()
    bundle = load_protocol_bundle(protocol_path, matrix_path)
    protocol = bundle["protocol"]
    require_jz_protocol_path(repo, protocol_path, matrix_path, protocol)
    auth = jz_authorization(protocol, STEP0_STAGE)
    source_path = Path(auth["source"]["path"])
    _require(output == Path(auth["output"]) and output.is_relative_to(JZ_RUN_ROOT)
             and not output.is_relative_to(repo) and not output.exists(),
             "J1 step-0 output differs or attempt already exists")
    blockers = stage_execution_blockers(protocol, STEP0_STAGE)
    _require(not blockers, "J1 step-0 preflight blocked: " + ", ".join(blockers))
    _require(sha256_file(source_path) == auth["source"]["sha256"], "J1 pair source changed")
    verify_jz_prerequisites(repo, protocol, STEP0_STAGE)
    repositories = {name: repository_identity(path) for name, path in (
        ("parent", repo), ("wan", repo / "wan2.1"), ("lama", repo / "lama"))}
    _validate_v4_source_binding(repo, repositories, protocol)
    reference = auth["reference_capture"]
    verify_reference_capture(repo, reference)
    source = _read_json(source_path)
    manifests = expand_source(source_path, output / "pair", write=False)
    by_route = validate_pair_plan(protocol, source, manifests, output, auth)
    _verify_bound_assets(auth, source)
    checkpoint, node_observation = preflight_jz_runtime(protocol, repo, protocol_path, auth)
    from .manifest_adapter import materialize  # noqa: PLC0415

    # Exclusive creation is the irreversible one-attempt marker.
    output.mkdir(parents=True, exist_ok=False)
    base = {
        "record_kind": RECORD_KIND,
        "stage_id": STEP0_STAGE,
        "bindings": {"protocol_id": protocol["protocol_id"],
                     "protocol_sha256": bundle["protocol_sha256"],
                     "matrix_sha256": bundle["matrix_sha256"]},
        "source_sha256": auth["source"]["sha256"],
        "checkpoint_content_sha256": checkpoint["content_sha256"],
        "reference_manifest_sha256": reference["manifest_sha256"],
        "repositories": repositories,
        "j1_node_observation": node_observation,
        "started_at": dt.datetime.now(dt.UTC).isoformat(),
        "scope": {"checkpoint_or_model_loaded": True, "generation_performed": False,
                  "step0_forward_only": True, "distributed_execution_performed": False},
        "r3_acceptance": False,
    }
    captures = {}
    sampler = GpuMemorySampler()
    try:
        try:
            with sampler:
                for route in ("upstream", "custom-none"):
                    manifest = by_route[route]
                    manifest_path = output / "jobs" / f"{manifest['job_id']}.json"
                    write_immutable_json(manifest_path, manifest)
                    verify_job_inputs(manifest)
                    _verify_bound_assets(auth, source)
                    task_path = materialize(manifest_path)
                    binding = manifest["r3_evidence"]["route_process"]
                    capture_dir = output / "captures" / binding["route"]
                    command = build_launch_command(
                        python=sys.executable, route=binding["route"],
                        wan_root=binding["wan_root"], project_root=repo,
                        task_file=task_path, output=capture_dir,
                    )
                    _run_bounded_command(command, repo, auth["job_timeout_seconds"])
                    _load_manifest(capture_dir)
                    captures[binding["route"]] = {
                        "job_id": manifest["job_id"],
                        "path": str(capture_dir),
                        "manifest_sha256": sha256_file(capture_dir / MANIFEST_NAME),
                    }
        finally:
            base["peak_gpu_memory"] = sampler.record()
        comparisons = {}
        for route, capture in captures.items():
            report = compare_cross_architecture(reference["path"], capture["path"], route=route)
            report_path = output / f"comparison-{route}.json"
            write_immutable_json(report_path, report)
            comparisons[route] = {"path": str(report_path), "sha256": sha256_file(report_path),
                                  "passed": report["passed"],
                                  "failed_stages": report["failed_stages"],
                                  "unexpected_candidate_only_stages":
                                      report["unexpected_candidate_only_stages"],
                                  "max_relative_l2": report["max_relative_l2"]}
        passed = len(comparisons) == 2 and all(item["passed"] for item in comparisons.values())
        result = base | {"status": "passed" if passed else "failed", "captures": captures,
                         "comparisons": comparisons,
                         "finished_at": dt.datetime.now(dt.UTC).isoformat()}
    except BaseException as error:
        write_immutable_json(output / "attempt.json", base | {
            "status": "failed", "captures": captures,
            "error": f"{type(error).__name__}: {error}"})
        raise
    write_immutable_json(output / "attempt.json", result)
    if not passed:
        raise RuntimeError("J1 step-0 cross-architecture gate failed")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--reference", required=True, type=Path)
    compare.add_argument("--candidate", required=True, type=Path)
    compare.add_argument("--route", required=True,
                         choices=sorted(JZ_CROSS_ARCHITECTURE_DECISION["allowed_candidate_only_stages"]))
    compare.add_argument("--report", required=True, type=Path)
    latents = commands.add_parser("report-latents")
    latents.add_argument("--reference", required=True, type=Path)
    latents.add_argument("--candidate", required=True, type=Path)
    latents.add_argument("--report", required=True, type=Path)
    video = commands.add_parser("report-video")
    video.add_argument("--reference", required=True, type=Path)
    video.add_argument("--candidate", required=True, type=Path)
    video.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "compare":
        report = compare_cross_architecture(args.reference, args.candidate, route=args.route)
    elif args.command == "report-latents":
        report = report_final_latents(args.reference, args.candidate)
    else:
        report = report_video_psnr(args.reference, args.candidate)
    _require(not Path(args.report).exists(), "report output already exists")
    Path(args.report).write_bytes(canonical_json_bytes(report) + b"\n")
    print(json.dumps({key: report.get(key) for key in (
        "passed", "failed_stages", "max_relative_l2", "relative_l2", "psnr_db")
        if key in report}))


if __name__ == "__main__":
    main()
