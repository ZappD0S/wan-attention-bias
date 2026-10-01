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
"""

from __future__ import annotations

import argparse
import copy
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
    J1_STAGE_IDS,
    JZ_CHECKPOINT_PATH,
    JZ_EXTRA_COMPONENTS,
    JZ_MATRIX_PATH,
    JZ_REFERENCE_ROOT,
    JZ_REPO_ROOT,
    JZ_VERSIONS,
    STEP0_STAGE,
    expected_jz_protocol,
    jz_protocol_relative_path,
)

ROOT = Path(__file__).resolve().parents[1]
BOOTES_V16_SOURCE = ROOT / "docs/j1_inputs/bootes-v16-source-unpatchify-parity.json"
BOOTES_V16_SOURCE_SHA256 = "0556b71a022305b39a91469e2b0c40ac1f2a92cf14bf071c3932624df4b5490e"
_BOOTES_ROOT = "/local_scratch2/gzappavi"
_BOOTES_REPO = f"{_BOOTES_ROOT}/wan_experiments_r3_cpu_20260926"
_BOOTES_INPUT = f"{_BOOTES_ROOT}/r3_stage3/input"
_BOOTES_CHECKPOINT = (
    f"{_BOOTES_ROOT}/hf/hub/models--Wan-AI--Wan2.1-I2V-14B-480P/"
    "snapshots/6b73f84e66371cdfe870c72acd6826e1d61cf279"
)
PRODUCTION_PATHS = ("multi_sample_inference", "pyproject.toml", "uv.lock")
V19_COMPONENTS = (
    "experiment_pipeline.py", "fsdp_worker.py", "r3_checkout_binding.py", "r3_route_isolation.py",
    "r3_pristine_adapter.py", "r3_contracts.py", "r3_checkpoint_hook_canary.py",
    "r3_generator_pair.py", "r3_contract_cases.py",
)


def _rewrite(container, key, old, new):
    _require(container.get(key) == old, f"Bootes v16 source field {key} differs from the bound value")
    container[key] = new


def derive_jz_source(protocol_id, bootes_bytes=None):
    """Return the deterministic Jean Zay pair source bytes for one jz version."""
    if bootes_bytes is None:
        bootes_bytes = BOOTES_V16_SOURCE.read_bytes()
    _require(hashlib.sha256(bootes_bytes).hexdigest() == BOOTES_V16_SOURCE_SHA256,
             "Bootes v16 pair source differs from its bound SHA-256")
    source = copy.deepcopy(json.loads(bootes_bytes))
    _rewrite(source["checkpoint"], "path", _BOOTES_CHECKPOINT, JZ_CHECKPOINT_PATH)
    _rewrite(source["checkpoint"], "inventory", f"{_BOOTES_REPO}/docs/u1_checkpoint_inventory.json",
             f"{JZ_REPO_ROOT}/docs/u1_checkpoint_inventory.json")
    _require(len(source["scenes"]) == 1, "Bootes v16 source must have one scene")
    scene = source["scenes"][0]
    inputs = f"{JZ_REFERENCE_ROOT}/bootes-inputs"
    _rewrite(scene, "reference_image", f"{_BOOTES_INPUT}/reference.png", f"{inputs}/reference.png")
    for actor in scene["actors"]:
        _rewrite(actor, "isolated_image", f"{_BOOTES_INPUT}/reference.png", f"{inputs}/reference.png")
    for actor_id, name in (("actor-left", "mask-left.png"), ("actor-right", "mask-right.png")):
        _rewrite(scene["segmentation_masks"], actor_id, f"{_BOOTES_INPUT}/{name}", f"{inputs}/{name}")
    _rewrite(source["r3_evidence"], "protocol", f"{_BOOTES_REPO}/docs/r3_protocol_v16.json",
             f"{JZ_REPO_ROOT}/{jz_protocol_relative_path(protocol_id)}")
    _rewrite(source["r3_evidence"], "matrix", f"{_BOOTES_REPO}/docs/r3_test_matrix_v3.json",
             f"{JZ_REPO_ROOT}/{JZ_MATRIX_PATH}")
    text = json.dumps(source, indent=2) + "\n"
    _require(_BOOTES_ROOT not in text, "derived Jean Zay source still names a Bootes path")
    return text.encode()


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
                for stage in J1_STAGE_IDS
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
    return {"stage_id": stage_id, "path": path.relative_to(repo).as_posix(),
            "sha256": sha256_file(path), "protocol_sha256": record["bindings"]["protocol_sha256"],
            "status": "passed"}


def freeze(protocol_id, *, reference_manifest_sha256=None, prerequisites=(), write_source=None,
           frozen_at=None, repo=ROOT):
    repo = Path(repo).resolve()
    _require(protocol_id in JZ_VERSIONS, f"no JZ_VERSIONS entry for {protocol_id}")
    approved = JZ_VERSIONS[protocol_id]["approved_stages"]
    binds_inputs = STEP0_STAGE in approved
    _require(binds_inputs or (reference_manifest_sha256 is None and write_source is None),
             "a canary-only version binds neither the pair source nor the Bootes capture")
    _require(not binds_inputs or reference_manifest_sha256 is not None,
             "this version requires the Bootes reference capture manifest SHA-256")
    target = repo / jz_protocol_relative_path(protocol_id)
    _require(not target.exists() or _read_json(target).get("frozen_at") is None,
             f"refusing to overwrite frozen {target.name}")
    state = collect_repo_state(repo)
    source_sha256 = None
    if binds_inputs:
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
    frozen = commands.add_parser("freeze")
    frozen.add_argument("--protocol-id", required=True, choices=sorted(JZ_VERSIONS))
    frozen.add_argument("--reference-manifest-sha256")
    frozen.add_argument("--prerequisite", action="append", default=[], metavar="STAGE=PATH")
    frozen.add_argument("--write-source", type=Path)
    args = parser.parse_args(argv)
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
