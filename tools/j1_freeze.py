"""Derive J1 Jean Zay pair sources and freeze the next jz protocol version (CPU only).

``source`` rewrites the hash-bound Bootes v16 pair source into the Jean Zay source for
one jz version: only checkpoint/inventory/asset paths and the protocol/matrix
references change. ``freeze`` records the repository state (HEAD, production digest,
component hashes), the version's freeze-time inputs and its approved stage into
``docs/r3_protocol_jz_vN.json``, then revalidates it. Approval is a user decision:
run ``freeze`` only after the user approves that version's stage.

Usage:
  uv run --no-sync --locked python -m tools.j1_freeze freeze --protocol-id r3-gpu-contracts-jz-v1
  uv run --no-sync --locked python -m tools.j1_freeze source --protocol-id ID --output PATH
  uv run --no-sync --locked python -m tools.j1_freeze contract-sources --protocol-id ID --output-dir DIR
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

from multi_sample_inference.experiment_pipeline import production_source_content_sha256
from multi_sample_inference.r3_contracts import (
    _read_json,
    _require,
    load_protocol_bundle,
    sha256_file,
    stage_execution_blockers,
)
from multi_sample_inference.r3_jz import (
    JZ_EXTRA_COMPONENTS,
    JZ_MATRIX_PATH,
    JZ_VERSIONS,
    STEP0_STAGE,
    derive_jz_bootes_source,
    expected_jz_protocol,
    jz_contract_sources,
    jz_protocol_relative_path,
    jz_stage_ids,
)

ROOT = Path(__file__).resolve().parents[1]
BOOTES_V16_SOURCE = ROOT / "docs/j1_inputs/bootes-v16-source-unpatchify-parity.json"
BOOTES_V16_SOURCE_SHA256 = "0556b71a022305b39a91469e2b0c40ac1f2a92cf14bf071c3932624df4b5490e"
PRODUCTION_PATHS = ("multi_sample_inference", "pyproject.toml", "uv.lock")
V19_COMPONENTS = (
    "experiment_pipeline.py", "fsdp_worker.py", "r3_checkout_binding.py", "r3_route_isolation.py",
    "r3_pristine_adapter.py", "r3_contracts.py", "r3_checkpoint_hook_canary.py",
    "r3_generator_pair.py", "r3_contract_cases.py",
)


def derive_jz_source(protocol_id, bootes_bytes=None):
    """Return the deterministic Jean Zay pair source bytes for one jz version."""
    if bootes_bytes is None:
        bootes_bytes = BOOTES_V16_SOURCE.read_bytes()
    return derive_jz_bootes_source(bootes_bytes, BOOTES_V16_SOURCE_SHA256,
                                   "r3_protocol_v16.json", protocol_id)


def collect_repo_state(repo=ROOT):
    """HEAD, production digest and component hashes; production paths must be committed."""
    repo = Path(repo).resolve()
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--", *PRODUCTION_PATHS],
                           check=True, capture_output=True, text=True).stdout
    _require(not dirty, "commit production changes before freezing a jz version")
    revision = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
    return {
        "revision": revision,
        "production": production_source_content_sha256(repo, list(PRODUCTION_PATHS)),
        "components": {name: sha256_file(repo / "multi_sample_inference" / name)
                       for name in (*V19_COMPONENTS, *JZ_EXTRA_COMPONENTS)},
    }


def build_protocol(protocol_id, *, frozen_at, revision, production, components,
                   source_sha256=None, reference_manifest_sha256=None, prerequisite_evidence=()):
    """Assemble the only acceptable protocol for these freeze-time values (None = draft)."""
    pinned = JZ_VERSIONS[protocol_id].get("pinned_pair_source")
    if pinned is not None and source_sha256 is None:
        source_sha256 = pinned[1]
    skeleton = {
        "protocol_id": protocol_id,
        "frozen_at": frozen_at,
        "execution_amendment": {
            "source_binding": {"parent_revision_at_freeze": revision,
                               "parent_production_content_sha256": production},
            "production_component_hashes": dict(components),
            "authorization_record": {
                stage: ({"source": {"sha256": source_sha256},
                         "reference_capture": {"manifest_sha256": reference_manifest_sha256}}
                        if stage == STEP0_STAGE else {})
                for stage in jz_stage_ids(protocol_id)
            },
            "prerequisite_evidence": list(prerequisite_evidence),
        },
    }
    return expected_jz_protocol(skeleton)


def _prerequisite_entry(repo, stage_id, relative):
    path = (repo / relative).resolve()
    _require(path.is_relative_to(repo / "docs/j1_evidence") and path.is_file(),
             f"prerequisite evidence must be committed under docs/j1_evidence: {relative}")
    record = _read_json(path)
    _require(record.get("stage_id") == stage_id and record.get("status") == "passed",
             f"prerequisite evidence is not a passed {stage_id} record")
    # Pair records bind their protocol at top level; canary records under ``bindings``.
    protocol_sha256 = (record["bindings"]["protocol_sha256"] if "bindings" in record
                       else record["protocol_sha256"])
    return {"stage_id": stage_id, "path": path.relative_to(repo).as_posix(),
            "sha256": sha256_file(path), "protocol_sha256": protocol_sha256,
            "status": "passed"}


def write_contract_sources(protocol_id, output_dir):
    """Write the four derived Jean Zay contract-case sources; return their SHA-256 values."""
    output_dir = Path(output_dir)
    _require(JZ_VERSIONS[protocol_id].get("contract_chunks"),
             f"{protocol_id} binds no contract-case sources")
    sources = jz_contract_sources(protocol_id)
    targets = {name: output_dir / f"source-stage4-{name}.json" for name in sources}
    _require(not any(path.exists() for path in targets.values()),
             "contract-case source output already exists")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in sources.items():
        targets[name].write_bytes(data)
    return {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()}


def freeze(protocol_id, *, reference_manifest_sha256=None, prerequisites=(), write_source=None,
           frozen_at=None, repo=ROOT):
    repo = Path(repo).resolve()
    _require(protocol_id in JZ_VERSIONS, f"no JZ_VERSIONS entry for {protocol_id}")
    spec = JZ_VERSIONS[protocol_id]
    approved = spec["approved_stages"]
    pinned = spec.get("pinned_pair_source")
    binds_reference = STEP0_STAGE in approved
    binds_source = binds_reference or "single-rank-generator-canary" in approved
    _require((binds_source and pinned is None) or write_source is None,
             "a canary-only or pinned-pair version writes no pair source")
    _require(binds_reference == (reference_manifest_sha256 is not None),
             "only versions approving the step-0 probe bind the Bootes reference capture manifest "
             "SHA-256, and they require it")
    target = repo / jz_protocol_relative_path(protocol_id)
    _require(not target.exists() or _read_json(target).get("frozen_at") is None,
             f"refusing to overwrite frozen {target.name}")
    state = collect_repo_state(repo)
    source_sha256 = None
    if pinned is not None:
        source_sha256 = pinned[1]
    elif binds_source:
        source_bytes = derive_jz_source(protocol_id)
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        if write_source is not None:
            Path(write_source).write_bytes(source_bytes)
    protocol = build_protocol(
        protocol_id,
        frozen_at=frozen_at or dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        revision=state["revision"], production=state["production"],
        components=state["components"], source_sha256=source_sha256,
        reference_manifest_sha256=reference_manifest_sha256,
        prerequisite_evidence=[_prerequisite_entry(repo, stage, path)
                               for stage, path in prerequisites],
    )
    target.write_text(json.dumps(protocol, indent=2) + "\n")
    bundle = load_protocol_bundle(target, repo / JZ_MATRIX_PATH)
    blockers = stage_execution_blockers(bundle["protocol"], approved[-1])
    _require(blockers == [], f"frozen {protocol_id} still blocks {approved[-1]}: {blockers}")
    return {"protocol": str(target.relative_to(repo)), "protocol_sha256": bundle["protocol_sha256"],
            "approved_stage": approved[-1], "parent_revision_at_freeze": state["revision"],
            "parent_production_content_sha256": state["production"],
            "source_sha256": source_sha256, "command": protocol["execution_amendment"][
                "authorization_record"][approved[-1]]["command"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    source = commands.add_parser("source")
    source.add_argument("--protocol-id", required=True, choices=sorted(JZ_VERSIONS))
    source.add_argument("--output", required=True, type=Path)
    contract = commands.add_parser("contract-sources")
    contract.add_argument("--protocol-id", required=True, choices=sorted(JZ_VERSIONS))
    contract.add_argument("--output-dir", required=True, type=Path)
    frozen = commands.add_parser("freeze")
    frozen.add_argument("--protocol-id", required=True, choices=sorted(JZ_VERSIONS))
    frozen.add_argument("--reference-manifest-sha256")
    frozen.add_argument("--prerequisite", action="append", default=[], metavar="STAGE=PATH")
    frozen.add_argument("--write-source", type=Path)
    args = parser.parse_args(argv)
    if args.command == "contract-sources":
        print(json.dumps(write_contract_sources(args.protocol_id, args.output_dir), indent=2))
        return
    if args.command == "source":
        _require(not args.output.exists(), "source output already exists")
        data = derive_jz_source(args.protocol_id)
        args.output.write_bytes(data)
        print(json.dumps({"output": str(args.output), "sha256": hashlib.sha256(data).hexdigest()}))
        return
    prerequisites = [tuple(item.split("=", 1)) for item in args.prerequisite]
    print(json.dumps(freeze(args.protocol_id, reference_manifest_sha256=args.reference_manifest_sha256,
                            prerequisites=prerequisites, write_source=args.write_source), indent=2))


if __name__ == "__main__":
    main()
