"""Small, stdlib-only manifest CLI for reproducible Wan experiments.

Generation is delegated to ``fsdp_worker`` through ``manifest_adapter``. This
module deliberately does not import torch so validation and bookkeeping work on
CPU-only login machines.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import itertools
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

from .generation_routes import UPSTREAM_FRAME_NUM
from .r3_contracts import (
    GENUINE_RUNTIME_EVIDENCE,
    execution_blockers,
    load_protocol_bundle,
    matrix_case_for_request,
    sha256_file,
    validate_worker_observation,
)
from .r3_runtime import (
    build_self_attention_dispatch_contract,
    validate_backend_request,
    validate_source_observations,
    validate_worker_dispatch_binding,
)

SCHEMA_VERSION = 1
METHODS = {"upstream", "none", "regional_prompting", "concept_weaver", "ediff-i"}
MASK_TYPES = {"fixed", "hard", "soft"}
SOLVERS = {"unipc", "dpm++"}
LABELS = {"true", "false", "uncertain", "missing"}
ANNOTATION_COLUMNS = [
    "job_id", "scene_id", "assignment_id", "condition_id", "video_path",
    "actor_id", "target_action", "target_present", "wrong_action_present",
    "identity_correct", "visible", "joint_both_correct", "quality",
    "uncertain", "missing", "tracking_failure", "notes",
]


def _canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def _hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _resolve_file(source_path, value, label):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = source_path.parent / path
    path = path.resolve()
    _require(path.is_file(), f"{label} is not a file: {path}")
    return path


def _resolve_dir(source_path, value, label):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = source_path.parent / path
    path = path.resolve()
    _require(path.is_dir(), f"{label} is not a directory: {path}")
    return path


def _checkpoint_inventory(checkpoint_dir, inventory_path, verify_contents):
    inventory = _read_json(inventory_path)
    _require(inventory.get("schema_version") == 1, "unsupported checkpoint inventory schema_version")
    entries = inventory.get("files")
    _require(isinstance(entries, list) and entries, "checkpoint inventory files must be a nonempty list")
    paths = [entry.get("path") for entry in entries]
    _require(
        all(isinstance(path, str) and path and not Path(path).is_absolute() and ".." not in Path(path).parts for path in paths),
        "checkpoint inventory paths must be safe relative paths",
    )
    _require(len(paths) == len(set(paths)), "checkpoint inventory paths must be unique")
    actual_paths = sorted(
        path.relative_to(checkpoint_dir).as_posix()
        for path in checkpoint_dir.rglob("*")
        if path.is_file()
    )
    _require(sorted(paths) == actual_paths, "checkpoint inventory file set does not match checkpoint directory")
    for entry in entries:
        path = checkpoint_dir / entry["path"]
        _require(type(entry.get("size")) is int and entry["size"] >= 0, "checkpoint inventory sizes must be nonnegative integers")
        _require(
            isinstance(entry.get("sha256"), str) and len(entry["sha256"]) == 64,
            "checkpoint inventory entries require SHA-256 digests",
        )
        _require(path.stat().st_size == entry["size"], f"checkpoint file size changed: {path}")
        if verify_contents:
            _require(_hash_file(path) == entry["sha256"], f"checkpoint file content changed: {path}")
    return inventory


def checkpoint_identity(source_path, checkpoint, verify_contents=True):
    """Resolve and verify an approved content inventory for a local checkpoint."""
    checkpoint_dir = _resolve_dir(source_path, checkpoint.get("path", ""), "checkpoint.path")
    inventory_path = _resolve_file(source_path, checkpoint.get("inventory", ""), "checkpoint.inventory")
    _require(not inventory_path.is_relative_to(checkpoint_dir), "checkpoint inventory must be stored outside the checkpoint directory")
    inventory = _checkpoint_inventory(checkpoint_dir, inventory_path, verify_contents)
    return {
        "path": str(checkpoint_dir),
        "identifier": checkpoint["identifier"],
        "inventory": {"path": str(inventory_path), "sha256": _hash_file(inventory_path)},
        "content_sha256": _hash_bytes(_canonical(inventory["files"])),
    }


def verify_checkpoint_identity(checkpoint):
    checkpoint_dir = Path(checkpoint["path"])
    inventory_path = Path(checkpoint["inventory"]["path"])
    _require(checkpoint_dir.is_dir(), "local checkpoint directory is missing")
    _require(
        inventory_path.is_file() and _hash_file(inventory_path) == checkpoint["inventory"]["sha256"],
        "checkpoint inventory changed or is missing",
    )
    inventory = _checkpoint_inventory(checkpoint_dir, inventory_path, verify_contents=True)
    _require(
        _hash_bytes(_canonical(inventory["files"])) == checkpoint["content_sha256"],
        "checkpoint content identity changed",
    )


def _git_output(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, stdout=subprocess.PIPE
    ).stdout


def repository_identity(repo):
    """Return HEAD and a content fingerprint when the worktree is dirty."""
    repo = Path(repo).resolve()
    revision = _git_output(repo, "rev-parse", "HEAD").decode().strip()
    status = _git_output(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    if not status:
        return {"revision": revision, "dirty": False, "dirty_fingerprint": None}
    digest = hashlib.sha256()
    digest.update(status)
    digest.update(_git_output(repo, "diff", "--binary", "HEAD", "--"))
    digest.update(_git_output(repo, "diff", "--cached", "--binary", "HEAD", "--"))
    untracked = _git_output(repo, "ls-files", "--others", "--exclude-standard", "-z")
    for raw in sorted(filter(None, untracked.split(b"\0"))):
        relative = raw.decode()
        path = repo / relative
        digest.update(raw + b"\0")
        if path.is_file():
            digest.update(bytes.fromhex(_hash_file(path)))
    return {"revision": revision, "dirty": True, "dirty_fingerprint": digest.hexdigest()}


def production_source_content_sha256(repo, paths):
    """Hash the tracked production files named by a v4 source declaration."""
    repo = Path(repo).resolve()
    raw_paths = _git_output(repo, "ls-files", "-z", "--", *paths)
    relative_paths = sorted(item.decode() for item in raw_paths.split(b"\0") if item)
    _require(relative_paths, "R3 production source inventory is empty")
    inventory = []
    for relative in relative_paths:
        path = repo / relative
        _require(path.is_file(), f"R3 production source path is missing: {relative}")
        inventory.append({"path": relative, "sha256": _hash_file(path)})
    return _hash_bytes(_canonical(inventory))


def _validate_v4_source_binding(repo, repositories, protocol):
    binding = protocol["execution_amendment"]["source_binding"]
    _require(
        all(identity["dirty"] is False for identity in repositories.values()),
        "R3 v4 requires clean parent, Wan, and LaMa worktrees",
    )
    _require(
        repositories["wan"]["revision"] == binding["wan_revision"]
        and repositories["lama"]["revision"] == binding["lama_revision"],
        "R3 v4 Wan or LaMa revision differs from the execution amendment",
    )
    observed_content = production_source_content_sha256(
        repo, binding["parent_production_paths"]
    )
    _require(
        observed_content == binding["parent_production_content_sha256"],
        "R3 v4 parent production content differs from the execution amendment",
    )


def _r3_source_contract(source, source_path):
    if "r3_evidence" not in source:
        return None
    declaration = source["r3_evidence"]
    required = {"protocol", "matrix", "attention_backend", "rank_mode"}
    _require(
        isinstance(declaration, dict)
        and frozenset(declaration)
        in {frozenset(required), frozenset(required | {"parity_pair"})},
        "r3_evidence must declare protocol, matrix, attention_backend, and rank_mode",
    )
    protocol_path = _resolve_file(source_path, declaration["protocol"], "r3_evidence.protocol")
    matrix_path = _resolve_file(source_path, declaration["matrix"], "r3_evidence.matrix")
    bundle = load_protocol_bundle(protocol_path, matrix_path)
    _require(
        declaration["attention_backend"] in {"flash_attention_2", "flex_attention"},
        "r3_evidence.attention_backend is unsupported",
    )
    rank_mode = declaration["rank_mode"]
    rank_count = source.get("inference", {}).get("rank_count")
    if rank_mode == "single":
        _require(rank_count == 1, "R3 single rank_mode requires inference.rank_count=1")
    elif rank_mode == "intended_fsdp":
        intended = bundle["protocol"]["runtime_declarations"]["intended_fsdp_rank_count"]
        _require(intended is not None, "R3 intended FSDP rank count is undeclared and blocks execution")
        _require(rank_count == intended, "inference.rank_count differs from the frozen intended FSDP rank count")
    else:
        raise ValueError("r3_evidence.rank_mode must be single or intended_fsdp")
    parity_pair = declaration.get("parity_pair")
    if "parity_pair" in declaration:
        _require(
            bundle["protocol"]["schema_version"] >= 2
            and isinstance(parity_pair, dict)
            and set(parity_pair)
            == {"pair_id", "upstream_condition_id", "custom_none_condition_id"}
            and isinstance(parity_pair["pair_id"], str)
            and parity_pair["pair_id"],
            "R3 parity_pair requires a valid v2 declaration",
        )
        conditions = {item["id"]: item for item in source.get("conditions", [])}
        upstream = conditions.get(parity_pair["upstream_condition_id"])
        custom = conditions.get(parity_pair["custom_none_condition_id"])
        _require(
            isinstance(upstream, dict)
            and upstream.get("method") == "upstream"
            and isinstance(custom, dict)
            and custom.get("method") == "none"
            and upstream["id"] != custom["id"],
            "R3 parity_pair must identify distinct upstream and custom-none conditions",
        )
    return bundle | {
        "attention_backend": declaration["attention_backend"],
        "rank_mode": rank_mode,
        "parity_pair": parity_pair,
    }


def _validate_source(source, source_path):
    _require(source.get("schema_version") == SCHEMA_VERSION, "unsupported source schema_version")
    _require(isinstance(source.get("experiment_id"), str) and source["experiment_id"], "experiment_id is required")
    checkpoint = source.get("checkpoint", {})
    _require(isinstance(checkpoint.get("identifier"), str) and checkpoint["identifier"], "checkpoint.identifier is required")
    checkpoint_identity(source_path, checkpoint)
    inference = source.get("inference", {})
    for key in ("width", "height", "frame_num", "sampling_steps", "rank_count", "num_layers"):
        _require(type(inference.get(key)) is int and inference[key] > 0, f"inference.{key} must be a positive integer")
    _require((inference["frame_num"] - 1) % 4 == 0, "inference.frame_num must have form 4n+1")
    _require(inference["num_layers"] == 40, "this Wan worker requires inference.num_layers=40")
    if not source.get("smoke_only", False):
        _require((inference["height"], inference["width"]) == (480, 832), "this Wan2.1 480P worker requires height=480 and width=832")
    _require(inference.get("solver") in SOLVERS, "unsupported inference.solver")
    for key in ("cfg", "shift"):
        _require(type(inference.get(key)) in (int, float) and inference[key] > 0, f"inference.{key} must be positive")
    negative_prompt = inference.get("negative_prompt")
    if negative_prompt is not None:
        _require(
            isinstance(negative_prompt, str) and negative_prompt,
            "inference.negative_prompt must be a nonempty string when supplied",
        )
    seeds = source.get("video_seeds", [])
    _require(seeds, "at least one video seed is required")
    seed_ids = [item.get("id") for item in seeds]
    _require(len(seed_ids) == len(set(seed_ids)) and all(seed_ids), "video seed IDs must be nonempty and unique")
    for item in seeds:
        _require(type(item.get("value")) is int and 0 <= item["value"] < 2**63, "video seed values must be signed 64-bit nonnegative integers")
    conditions = source.get("conditions", [])
    _require(conditions, "at least one condition is required")
    condition_ids = [item.get("id") for item in conditions]
    _require(len(condition_ids) == len(set(condition_ids)) and all(condition_ids), "condition IDs must be nonempty and unique")
    for condition in conditions:
        method = condition.get("method")
        _require(method in METHODS, f"unsupported condition method: {method!r}")
        representation = condition.get("prompt_representation")
        _require(representation in {"joint", "split"}, "prompt_representation must be joint or split")
        if method == "concept_weaver":
            _require(representation == "split", "concept_weaver requires split singleton prompts")
        if method == "upstream":
            _require(representation == "joint", "upstream requires one joint prompt sentence")
            _require(
                inference["frame_num"] == UPSTREAM_FRAME_NUM,
                "pinned upstream Wan generation requires "
                f"inference.frame_num={UPSTREAM_FRAME_NUM}",
            )
            _require(
                negative_prompt is not None,
                "upstream requires an explicit inference.negative_prompt",
            )
        mask_type = condition.get("mask_type")
        _require(mask_type in MASK_TYPES, f"unsupported mask_type: {mask_type!r}")
        _require(condition.get("mask_source") in {"fixed", "dynamic"}, "mask_source must be fixed or dynamic")
        _require((mask_type == "fixed") == (condition["mask_source"] == "fixed"), "fixed mask_type and fixed mask_source must be selected together")
        _require(condition.get("image_context_isolation") is False, "this pilot foundation requires image_context_isolation=false")
        _require(type(condition.get("self_attention_masking")) is bool, "self_attention_masking must be boolean")
        _require(condition.get("mask_sharing") in {"current", "first", "previous_generated"}, "unsupported mask_sharing")
        if method == "regional_prompting":
            _require(type(condition.get("beta")) in (int, float) and 0 <= condition["beta"] <= 1, "regional_prompting requires beta in [0,1]")
        if method == "ediff-i":
            _require(type(condition.get("strength")) in (int, float), "ediff-i requires numeric strength")
        for schedule_name, expected_length in (
            ("timestep_bias_schedule", inference["sampling_steps"]),
            ("blocks_bias_schedule", inference["num_layers"]),
        ):
            schedule = condition.get(schedule_name)
            if schedule is not None:
                _require(
                    isinstance(schedule, list)
                    and len(schedule) == expected_length
                    and all(type(value) is bool for value in schedule),
                    f"condition {schedule_name} must be a boolean list of length {expected_length}",
                )
    scenes = source.get("scenes", [])
    _require(scenes, "at least one scene is required")
    scene_ids = [item.get("id") for item in scenes]
    _require(len(scene_ids) == len(set(scene_ids)) and all(scene_ids), "scene IDs must be nonempty and unique")
    _r3_source_contract(source, source_path)
    for scene in scenes:
        actors = scene.get("actors", [])
        _require(len(actors) >= 2, f"scene {scene.get('id')} requires at least two actors")
        actor_ids = [actor.get("id") for actor in actors]
        _require(len(actor_ids) == len(set(actor_ids)) and all(actor_ids), "actor IDs must be nonempty and unique within a scene")
        _resolve_file(source_path, scene.get("reference_image", ""), "reference_image")
        fixed = scene.get("fixed_boxes", {})
        dynamic = scene.get("segmentation_masks", {})
        for actor in actors:
            actor_id = actor["id"]
            _resolve_file(source_path, actor.get("isolated_image", ""), f"isolated image for {actor_id}")
            _require(actor_id in fixed and len(fixed[actor_id]) == 4, f"fixed box missing for {actor_id}")
            box = fixed[actor_id]
            _require(all(type(value) in (int, float) for value in box), f"fixed box must be numeric for {actor_id}")
            left, top, right, bottom = box
            _require(0 <= left < right <= inference["width"] and 0 <= top < bottom <= inference["height"], f"fixed box is outside target dimensions for {actor_id}")
            _resolve_file(source_path, dynamic.get(actor_id, ""), f"segmentation mask for {actor_id}")
        assignments = scene.get("assignments", [])
        assignment_ids = [item.get("id") for item in assignments]
        _require(assignments and len(assignment_ids) == len(set(assignment_ids)) and all(assignment_ids), "assignment IDs must be nonempty and unique within a scene")
        for assignment in assignments:
            targets = assignment.get("targets", {})
            _require(set(targets) == set(actor_ids) and all(isinstance(v, str) and v for v in targets.values()), "assignment targets must name every actor exactly once")
            prompts = assignment.get("prompts", {})
            _require(isinstance(prompts.get("joint_sentence"), str) and prompts["joint_sentence"], "joint_sentence is required")
            _require(len(prompts.get("joint_character_segments", [])) == len(actors), "joint_character_segments must match actor order")
            _require(len(prompts.get("split_sentences", [])) == len(actors), "split_sentences must match actor order")
            _require(len(prompts.get("split_character_segments", [])) == len(actors) and all(len(row) == 1 for row in prompts["split_character_segments"]), "split_character_segments must contain one singleton per actor")
            _require(isinstance(prompts.get("general_prompt"), str) and prompts["general_prompt"], "general_prompt is required")
    return True


def validate_source(path):
    path = Path(path).resolve()
    source = _read_json(path)
    _validate_source(source, path)
    return source


def _manifest_inference(source_inference):
    return {
        "width": source_inference["width"],
        "height": source_inference["height"],
        "frame_num": source_inference["frame_num"],
        "sampling_steps": source_inference["sampling_steps"],
        "cfg": source_inference["cfg"],
        "shift": source_inference["shift"],
        "solver": source_inference["solver"],
        "negative_prompt": source_inference.get("negative_prompt"),
        "rank_count": source_inference["rank_count"],
        "num_layers": source_inference["num_layers"],
    }


def _manifest_intervention(condition, source_inference):
    return {
        "method": condition["method"],
        "mask_type": condition["mask_type"],
        "mask_source": condition["mask_source"],
        "mask_sharing": condition["mask_sharing"],
        "self_attention_masking": condition["self_attention_masking"],
        "image_context_isolation": False,
        "beta": condition.get("beta"),
        "strength": condition.get("strength"),
        "timestep_bias_schedule": condition.get(
            "timestep_bias_schedule", [True] * source_inference["sampling_steps"]
        ),
        "blocks_bias_schedule": condition.get(
            "blocks_bias_schedule", [True] * source_inference["num_layers"]
        ),
    }


def _asset(path):
    return {"path": str(path), "sha256": _hash_file(path)}


def _stable_id(prefix, payload):
    return prefix + "-" + _hash_bytes(_canonical(payload))[:16]


def expand_source(source_path, output_dir, write=True):
    source_path = Path(source_path).resolve()
    source = validate_source(source_path)
    output_dir = Path(output_dir).resolve()
    repo = Path(__file__).resolve().parents[1]
    _require(
        not output_dir.is_relative_to(repo),
        "output_dir must be outside the source repository so manifest creation cannot change its recorded dirty fingerprint",
    )
    repositories = {
        "parent": repository_identity(repo),
        "wan": repository_identity(repo / "wan2.1"),
    }
    # Validation already checked every checkpoint file against the approved inventory.
    checkpoint = checkpoint_identity(source_path, source["checkpoint"], verify_contents=False)
    inference = source["inference"]
    r3_contract = _r3_source_contract(source, source_path)
    r3_environment_sha256 = None
    if r3_contract is not None:
        protocol = r3_contract["protocol"]
        if protocol["schema_version"] >= 4:
            repositories["lama"] = repository_identity(repo / "lama")
            _validate_v4_source_binding(repo, repositories, protocol)
        declarations = protocol["runtime_declarations"]
        declared_checkpoint = declarations["checkpoint_content_sha256"]
        _require(
            declared_checkpoint is None
            or declared_checkpoint == checkpoint["content_sha256"],
            "R3 protocol checkpoint declaration differs from the source checkpoint",
        )
        backend_key = (
            "flash_attention_version"
            if r3_contract["attention_backend"] == "flash_attention_2"
            else "flex_attention_version"
        )
        environment = (
            protocol["execution_amendment"]["environment_binding"]
            if protocol["schema_version"] >= 4
            else {
                "hardware_identifier": declarations["hardware_identifier"],
                "torch_version": declarations["torch_version"],
                "cuda_version": declarations["cuda_version"],
                "attention_backend": r3_contract["attention_backend"],
                "backend_version": declarations[backend_key],
            }
        )
        if all(value is not None for value in environment.values()):
            r3_environment_sha256 = _hash_bytes(_canonical(environment))
    manifests = []
    # Assignments belong to each scene, so the Cartesian product is per scene.
    for scene in source["scenes"]:
        actors = scene["actors"]
        actor_ids = [actor["id"] for actor in actors]
        reference = _resolve_file(source_path, scene["reference_image"], "reference_image")
        isolated = {
            actor["id"]: _asset(_resolve_file(source_path, actor["isolated_image"], "isolated_image"))
            for actor in actors
        }
        segmentation = {
            actor_id: _asset(_resolve_file(source_path, scene["segmentation_masks"][actor_id], "segmentation mask"))
            for actor_id in actor_ids
        }
        for assignment, seed, condition in itertools.product(scene["assignments"], source["video_seeds"], source["conditions"]):
            prompts = assignment["prompts"]
            if condition["prompt_representation"] == "joint":
                sentences = [prompts["joint_sentence"]]
                character_segments = [prompts["joint_character_segments"]]
            else:
                sentences = prompts["split_sentences"]
                character_segments = prompts["split_character_segments"]
            identity = {
                "experiment_id": source["experiment_id"], "scene_id": scene["id"],
                "assignment_id": assignment["id"], "video_seed_id": seed["id"],
                "condition_id": condition["id"],
            }
            job_id = _stable_id("job", identity)
            output_base = output_dir / "outputs" / job_id
            prompt_payload = {
                "representation": condition["prompt_representation"],
                "sentences": sentences,
                "character_segments": character_segments,
                "general_prompt": prompts["general_prompt"],
            }
            mask_payload = {
                "source": condition["mask_source"], "type": condition["mask_type"],
                "actor_order": actor_ids,
                "fixed_boxes": {actor_id: scene["fixed_boxes"][actor_id] for actor_id in actor_ids},
                "segmentation_masks": segmentation,
                "processing": {
                    "variant": "fixed_binary_v1" if condition["mask_source"] == "fixed" else "legacy_boolean_gaussian_v1",
                    "overlap": "remove_before_and_after_filter",
                    "gaussian_sigma": 0.0 if condition["mask_source"] == "fixed" else 5.0,
                    "filter_input_dtype": "bool",
                    "filter_output_dtype": "bool",
                    "post_filter_rule": "nonzero_is_foreground",
                    "threshold": 128,
                },
            }
            intervention = _manifest_intervention(condition, inference)
            manifest = {
                "schema_version": SCHEMA_VERSION, "job_id": job_id,
                "experiment_id": source["experiment_id"], "smoke_only": bool(source.get("smoke_only", False)),
                "source": {"path": str(source_path), "sha256": _hash_file(source_path), "record_id": scene["id"]},
                "repositories": repositories, "checkpoint": checkpoint,
                "identity": identity | {"actor_order": actor_ids},
                "assignment": {"targets": assignment["targets"]},
                "video_seed": {"id": seed["id"], "value": seed["value"]},
                "prompts": prompt_payload | {"sha256": _hash_bytes(_canonical(prompt_payload))},
                "assets": {"reference_image": _asset(reference), "isolated_images": isolated},
                "masks": mask_payload | {"sha256": _hash_bytes(_canonical(mask_payload))},
                "intervention": intervention,
                "inference": _manifest_inference(inference),
                "config_sha256": _hash_bytes(_canonical({"intervention": intervention, "inference": inference})),
                "outputs": {
                    "video_path": str(output_base / "video.mp4"),
                    "artifact_dir": str(output_base),
                    "task_path": str(output_dir / ".tasks" / f"{job_id}.pkl"),
                    "task_metadata_path": str(output_dir / ".tasks" / f"{job_id}.json"),
                    "status_path": str(output_dir / "status" / f"{job_id}.jsonl"),
                    "result_path": str(output_dir / "results" / f"{job_id}.json"),
                },
            }
            if r3_contract is not None:
                if condition["method"] == "upstream":
                    family = "upstream-generator"
                    selection = {
                        "solver": inference["solver"],
                        "attention_backend": r3_contract["attention_backend"],
                        "rank_mode": r3_contract["rank_mode"],
                    }
                else:
                    family = "custom-generator"
                    selection = {
                        "method": condition["method"],
                        "mask_configuration": f"{condition['mask_type']}:{condition['mask_source']}",
                        "mask_sharing": condition["mask_sharing"],
                        "self_attention_masking": condition["self_attention_masking"],
                        "solver": inference["solver"],
                        "attention_backend": r3_contract["attention_backend"],
                        "rank_mode": r3_contract["rank_mode"],
                    }
                case = matrix_case_for_request(r3_contract["matrix"], family, selection)
                dispatch_contract = None
                if r3_contract["protocol"]["schema_version"] >= 3:
                    disposition = case["lineage_disposition"]
                    _require(
                        disposition["disposition"] == "runnable",
                        "R3 matrix case is rejected before materialization: "
                        + disposition["reason"],
                    )
                    declarations = r3_contract["protocol"]["runtime_declarations"]
                    dispatch_contract = build_self_attention_dispatch_contract(
                        execution_route="upstream"
                        if condition["method"] == "upstream"
                        else "custom",
                        method=condition["method"],
                        self_attention_masking=condition["self_attention_masking"],
                        timestep_bias_schedule=intervention[
                            "timestep_bias_schedule"
                        ],
                        blocks_bias_schedule=intervention["blocks_bias_schedule"],
                        backend_versions={
                            "flash_attention_2": declarations[
                                "flash_attention_version"
                            ],
                            "flex_attention": declarations["flex_attention_version"],
                        },
                    )
                    validate_backend_request(
                        dispatch_contract, r3_contract["attention_backend"]
                    )
                observation_path = None
                if condition["method"] != "upstream":
                    observation_path = str(
                        output_base / "r3-rank-zero-worker-observation.json"
                    )
                    manifest["outputs"]["r3_worker_observation_path"] = observation_path
                if r3_contract["protocol"]["schema_version"] >= 2:
                    manifest["outputs"]["r3_source_observation_path"] = str(
                        output_base / "r3-source-observations.json"
                    )
                manifest["r3_evidence"] = {
                    "protocol": {
                        "path": str(r3_contract["protocol_path"]),
                        "sha256": r3_contract["protocol_sha256"],
                    },
                    "matrix": {
                        "path": str(r3_contract["matrix_path"]),
                        "sha256": r3_contract["matrix_sha256"],
                    },
                    "case": case,
                    "worker_observation_required": observation_path is not None,
                    "declared_environment_sha256": r3_environment_sha256,
                    "expected": {
                        "rank_count": inference["rank_count"],
                        "sampling_steps": inference["sampling_steps"],
                        "num_layers": inference["num_layers"],
                    },
                }
                if dispatch_contract is not None:
                    manifest["r3_evidence"]["dispatch_contract"] = dispatch_contract
                parity_pair = r3_contract["parity_pair"]
                if parity_pair is not None and condition["id"] in {
                    parity_pair["upstream_condition_id"],
                    parity_pair["custom_none_condition_id"],
                }:
                    route = (
                        "upstream"
                        if condition["id"] == parity_pair["upstream_condition_id"]
                        else "custom-none"
                    )
                    parity_path = str(output_base / f"r3-parity-{route}.npy")
                    manifest["outputs"]["r3_parity_tensor_path"] = parity_path
                    manifest["r3_evidence"]["parity_artifact"] = {
                        "pair_id": parity_pair["pair_id"],
                        "route": route,
                        "path": parity_path,
                        "metadata_path": parity_path + ".json",
                    }
            manifests.append(manifest)
    if write:
        for manifest in manifests:
            path = output_dir / "jobs" / f"{manifest['job_id']}.json"
            _write_immutable(path, _canonical(manifest))
        index = {"schema_version": SCHEMA_VERSION, "experiment_id": source["experiment_id"], "jobs": [m["job_id"] for m in manifests]}
        _write_immutable(output_dir / "jobs" / "index.json", _canonical(index))
    return manifests


def _write_immutable(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        if path.read_bytes() != content:
            raise FileExistsError(f"immutable content conflict: {path}")
        return False
    with os.fdopen(fd, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    return True


def _manifest_hash(manifest):
    return _hash_bytes(_canonical(manifest))


def task_descriptor_metadata(manifest, task_path, resolved_masks):
    return {
        "schema_version": 2,
        "job_id": manifest["job_id"],
        "manifest_sha256": _manifest_hash(manifest),
        "task_sha256": _hash_file(task_path),
        "resolved_masks": resolved_masks,
    }


def verify_task_descriptor(manifest, task_path, metadata_path):
    task_path, metadata_path = Path(task_path), Path(metadata_path)
    _require(task_path.is_file() and metadata_path.is_file(), "task descriptor or metadata is missing")
    metadata = _read_json(metadata_path)
    _require(metadata.get("schema_version") == 2, "unsupported task metadata schema_version")
    _require(metadata.get("job_id") == manifest["job_id"], "task metadata job_id does not match manifest")
    _require(metadata.get("manifest_sha256") == _manifest_hash(manifest), "task metadata manifest hash does not match")
    _require(metadata.get("task_sha256") == _hash_file(task_path), "task descriptor content hash does not match metadata")
    _require(isinstance(metadata.get("resolved_masks"), dict), "task metadata lacks resolved mask evidence")
    return metadata


def verify_repository_identity(manifest):
    repo = Path(__file__).resolve().parents[1]
    current = {"parent": repository_identity(repo), "wan": repository_identity(repo / "wan2.1")}
    if "lama" in manifest["repositories"]:
        current["lama"] = repository_identity(repo / "lama")
    _require(current == manifest["repositories"], "source repository revision/dirty fingerprint changed after expansion")


def verify_job_inputs(manifest):
    verify_repository_identity(manifest)
    verify_checkpoint_identity(manifest["checkpoint"])
    for asset in [manifest["assets"]["reference_image"], *manifest["assets"]["isolated_images"].values(), *manifest["masks"]["segmentation_masks"].values()]:
        _require(Path(asset["path"]).is_file() and _hash_file(asset["path"]) == asset["sha256"], f"asset changed or missing: {asset['path']}")
    _require(_hash_file(manifest["source"]["path"]) == manifest["source"]["sha256"], "source experiment JSON changed after expansion")


def worker_task_blueprint(manifest):
    """Return the exact path/scalar portion consumed by manifest_adapter/worker."""
    task = {
        "prompt_sentences": manifest["prompts"]["sentences"],
        "prompt_representation": manifest["prompts"].get("representation"),
        "negative_prompt": manifest["inference"].get("negative_prompt"),
        "checkpoint_dir": manifest["checkpoint"]["path"],
        "char_segments_list": manifest["prompts"]["character_segments"],
        "general_prompt": manifest["prompts"]["general_prompt"],
        "video_path": manifest["outputs"]["video_path"],
        "action_output_path": manifest["outputs"]["artifact_dir"],
        "repeat_idx": 0,
        "diffusion_seed": manifest["video_seed"]["value"],
        "inference_settings": {
            "sampling_steps": manifest["inference"]["sampling_steps"],
            "frame_num": manifest["inference"]["frame_num"],
            "target_size": [manifest["inference"]["height"], manifest["inference"]["width"]],
            "shift": manifest["inference"]["shift"],
            "sample_solver": manifest["inference"]["solver"],
            "guide_scale": manifest["inference"]["cfg"],
        },
        "config": {
            "bias_method": manifest["intervention"]["method"],
            "simil_masks_type": manifest["intervention"]["mask_type"],
            "mask_sharing": manifest["intervention"]["mask_sharing"],
            "self_attention_masking": manifest["intervention"]["self_attention_masking"],
            "image_context_isolation": manifest["intervention"]["image_context_isolation"],
            "beta": manifest["intervention"]["beta"],
            "strength": manifest["intervention"]["strength"],
            "timestep_bias_schedule": list(
                manifest["intervention"]["timestep_bias_schedule"]
            ),
            "blocks_bias_schedule": list(
                manifest["intervention"]["blocks_bias_schedule"]
            ),
        },
    }
    if "r3_evidence" in manifest:
        r3 = manifest["r3_evidence"]
        task["r3_evidence"] = {
            "path": manifest["outputs"].get("r3_worker_observation_path"),
            "bindings": {
                "protocol_sha256": r3["protocol"]["sha256"],
                "matrix_sha256": r3["matrix"]["sha256"],
                "manifest_sha256": _manifest_hash(manifest),
                "checkpoint_content_sha256": manifest["checkpoint"]["content_sha256"],
                "source_sha256": manifest["source"]["sha256"],
                "config_sha256": manifest["config_sha256"],
            },
            "case_id": r3["case"]["case_id"],
            "expected": r3["expected"],
            "requested": {
                **r3["case"]["selection"],
                "diffusion_seed": manifest["video_seed"]["value"],
                "cfg": manifest["inference"]["cfg"],
                "execution_route": "upstream"
                if manifest["intervention"]["method"] == "upstream"
                else "custom",
            },
        }
        bundle = load_protocol_bundle(
            r3["protocol"]["path"], r3["matrix"]["path"]
        )
        _require(
            bundle["protocol_sha256"] == r3["protocol"]["sha256"]
            and bundle["matrix_sha256"] == r3["matrix"]["sha256"],
            "R3 manifest protocol or matrix binding differs from disk",
        )
        dispatch_contract = r3.get("dispatch_contract")
        if bundle["protocol"]["schema_version"] >= 3:
            _require(
                isinstance(dispatch_contract, dict),
                "R3 v3 manifest dispatch contract is missing",
            )
            declarations = bundle["protocol"]["runtime_declarations"]
            source_path = Path(manifest["source"]["path"])
            _require(
                source_path.is_file()
                and _hash_file(source_path) == manifest["source"]["sha256"],
                "R3 manifest source binding changed or is missing",
            )
            source = _read_json(source_path)
            source_contract = _r3_source_contract(source, source_path)
            _require(
                source_contract is not None
                and source_contract["protocol_sha256"] == bundle["protocol_sha256"]
                and source_contract["matrix_sha256"] == bundle["matrix_sha256"],
                "R3 manifest protocol or matrix differs from its source",
            )
            source_conditions = [
                condition
                for condition in source["conditions"]
                if condition["id"] == manifest["identity"]["condition_id"]
            ]
            _require(
                len(source_conditions) == 1,
                "R3 manifest condition is not uniquely bound to its source",
            )
            source_condition = source_conditions[0]
            source_inference = source["inference"]
            expected_inference = _manifest_inference(source_inference)
            expected_intervention = _manifest_intervention(
                source_condition, source_inference
            )
            _require(
                manifest["inference"] == expected_inference,
                "R3 manifest inference differs from its source",
            )
            _require(
                manifest["intervention"] == expected_intervention,
                "R3 manifest intervention differs from its source",
            )
            _require(
                _hash_bytes(
                    _canonical(
                        {
                            "intervention": expected_intervention,
                            "inference": source_inference,
                        }
                    )
                )
                == manifest["config_sha256"],
                "R3 manifest configuration differs from its source-bound hash",
            )
            if source_condition["method"] == "upstream":
                expected_family = "upstream-generator"
                expected_selection = {
                    "solver": source_inference["solver"],
                    "attention_backend": source_contract["attention_backend"],
                    "rank_mode": source_contract["rank_mode"],
                }
            else:
                expected_family = "custom-generator"
                expected_selection = {
                    "method": source_condition["method"],
                    "mask_configuration": (
                        f"{source_condition['mask_type']}:"
                        f"{source_condition['mask_source']}"
                    ),
                    "mask_sharing": source_condition["mask_sharing"],
                    "self_attention_masking": source_condition[
                        "self_attention_masking"
                    ],
                    "solver": source_inference["solver"],
                    "attention_backend": source_contract["attention_backend"],
                    "rank_mode": source_contract["rank_mode"],
                }
            selection = r3["case"]["selection"]
            _require(
                r3["case"]["family"] == expected_family
                and selection == expected_selection,
                "R3 manifest case selection differs from its source",
            )
            expected_case = matrix_case_for_request(
                bundle["matrix"], expected_family, expected_selection
            )
            _require(
                expected_case == r3["case"],
                "R3 manifest case differs from the validated matrix",
            )
            expected_cardinalities = {
                "rank_count": source_inference["rank_count"],
                "sampling_steps": source_inference["sampling_steps"],
                "num_layers": source_inference["num_layers"],
            }
            _require(
                r3["expected"] == expected_cardinalities,
                "R3 manifest expected cardinalities differ from its source",
            )
            expected_dispatch = build_self_attention_dispatch_contract(
                execution_route=(
                    "upstream"
                    if source_condition["method"] == "upstream"
                    else "custom"
                ),
                method=source_condition["method"],
                self_attention_masking=source_condition[
                    "self_attention_masking"
                ],
                timestep_bias_schedule=expected_intervention[
                    "timestep_bias_schedule"
                ],
                blocks_bias_schedule=expected_intervention[
                    "blocks_bias_schedule"
                ],
                backend_versions={
                    "flash_attention_2": declarations["flash_attention_version"],
                    "flex_attention": declarations["flex_attention_version"],
                },
            )
            _require(
                dispatch_contract == expected_dispatch,
                "R3 manifest dispatch contract differs from its source",
            )
            validate_backend_request(
                dispatch_contract, selection["attention_backend"]
            )
            task["r3_evidence"]["protocol_schema_version"] = bundle["protocol"][
                "schema_version"
            ]
            task["r3_evidence"]["requested"]["dispatch_contract"] = (
                dispatch_contract
            )
        if "r3_source_observation_path" in manifest["outputs"]:
            task["r3_evidence"]["source_observation_path"] = manifest["outputs"][
                "r3_source_observation_path"
            ]
        parity = r3.get("parity_artifact")
        if parity is not None:
            route_source_sha256 = _hash_bytes(
                _canonical(
                    {
                        "route": parity["route"],
                        "wan_repository": manifest["repositories"]["wan"],
                    }
                )
            )
            task["r3_evidence"]["parity_artifact"] = parity | {
                "job_id": manifest["job_id"],
                "evidence_class": GENUINE_RUNTIME_EVIDENCE,
                "bindings": {
                    "input_sha256": _hash_bytes(
                        _canonical(
                            {
                                "prompts": manifest["prompts"],
                                "assets": manifest["assets"],
                                "video_seed": manifest["video_seed"],
                            }
                        )
                    ),
                    "checkpoint_content_sha256": manifest["checkpoint"][
                        "content_sha256"
                    ],
                    "source_sha256": manifest["source"]["sha256"],
                    "environment_sha256": r3["declared_environment_sha256"],
                    "diffusion_seed": manifest["video_seed"]["value"],
                    "inference_settings_sha256": _hash_bytes(
                        _canonical(manifest["inference"])
                    ),
                    "config_sha256": manifest["config_sha256"],
                    "route_source_sha256": route_source_sha256,
                },
            }
    return task


def validate_r3_worker_task_binding(task, manifest):
    """Bind a v3+ task to its manifest without trusting task version markers."""
    evidence = task.get("r3_evidence")
    task_schema = (
        evidence.get("protocol_schema_version")
        if isinstance(evidence, dict)
        else None
    )
    task_claims_v3_plus = type(task_schema) is int and task_schema >= 3
    task_has_dispatch = (
        isinstance(evidence, dict)
        and isinstance(evidence.get("requested"), dict)
        and "dispatch_contract" in evidence["requested"]
    )
    expected = worker_task_blueprint(manifest)
    expected_evidence = expected.get("r3_evidence")
    expected_schema = (
        expected_evidence.get("protocol_schema_version")
        if isinstance(expected_evidence, dict)
        else None
    )
    manifest_requires_v3_plus = type(expected_schema) is int and expected_schema >= 3
    if not manifest_requires_v3_plus:
        _require(
            not task_claims_v3_plus and not task_has_dispatch,
            "R3 worker task claims v3+ without a v3+ manifest binding",
        )
        return None
    _require(
        task_claims_v3_plus and task_schema == expected_schema,
        "R3 v3+ worker task discriminator is missing or downgraded/substituted",
    )
    _require(
        evidence == expected_evidence
        and task.get("config") == expected.get("config")
        and task.get("inference_settings") == expected.get("inference_settings"),
        "R3 worker task differs from its immutable manifest binding",
    )
    validate_worker_dispatch_binding(task)
    return evidence["requested"]["dispatch_contract"]


def _append_status(path, event):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(_canonical(event).decode())
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle, fcntl.LOCK_UN)


def _events(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _verify_declared_r3_worker_evidence(manifest):
    r3 = manifest.get("r3_evidence")
    if r3 is None:
        return None
    _require(
        r3.get("worker_observation_required") in {True, False},
        "R3 worker evidence declaration is invalid",
    )
    for key in ("protocol", "matrix"):
        declared = r3[key]
        _require(
            Path(declared["path"]).is_file() and sha256_file(declared["path"]) == declared["sha256"],
            f"declared R3 {key} changed or is missing",
        )
    bundle = load_protocol_bundle(r3["protocol"]["path"], r3["matrix"]["path"])
    _require(
        bundle["protocol_sha256"] == r3["protocol"]["sha256"]
        and bundle["matrix_sha256"] == r3["matrix"]["sha256"],
        "declared R3 protocol or matrix identity changed",
    )
    binding = worker_task_blueprint(manifest)["r3_evidence"]
    evidence = {"r3_acceptance": False}
    if r3["worker_observation_required"]:
        path = Path(manifest["outputs"]["r3_worker_observation_path"])
        _require(path.is_file(), "declared R3 worker observation is missing")
        record = _read_json(path)
        validate_worker_observation(record, binding)
        evidence["worker_observation"] = {
            "path": str(path),
            "sha256": _hash_file(path),
            "record_kind": record["record_kind"],
        }
    source_path = manifest["outputs"].get("r3_source_observation_path")
    if source_path is not None:
        _require(Path(source_path).is_file(), "declared R3 source observations are missing")
        source_record = _read_json(source_path)
        _require(
            set(source_record)
            == {
                "schema_version",
                "record_kind",
                "evidence_class",
                "bindings",
                "case_id",
                "expected",
                "requested",
                "observations",
                "r3_acceptance",
            }
            and source_record.get("schema_version")
            == bundle["protocol"]["schema_version"]
            and source_record.get("record_kind") == "r3-source-hook-observations"
            and source_record.get("evidence_class") == GENUINE_RUNTIME_EVIDENCE
            and source_record.get("bindings") == binding["bindings"]
            and source_record.get("case_id") == binding["case_id"]
            and source_record.get("expected") == binding["expected"]
            and source_record.get("requested") == binding["requested"]
            and source_record.get("r3_acceptance") is False,
            "declared R3 source observations differ from the job binding",
        )
        backend_key = (
            "flash_attention_version"
            if binding["requested"]["attention_backend"] == "flash_attention_2"
            else "flex_attention_version"
        )
        validate_source_observations(
            source_record.get("observations"),
            rank_count=binding["expected"]["rank_count"],
            sampling_steps=binding["expected"]["sampling_steps"],
            num_layers=binding["expected"]["num_layers"],
            require_masks=binding["requested"]["execution_route"] == "custom",
            mask_configuration=binding["requested"].get("mask_configuration"),
            requested_backend=binding["requested"]["attention_backend"],
            expected_backend_version=bundle["protocol"]["runtime_declarations"][
                backend_key
            ],
            expected_seed=binding["requested"]["diffusion_seed"],
            dispatch_contract=binding["requested"].get("dispatch_contract"),
        )
        evidence["source_observations"] = {
            "path": source_path,
            "sha256": _hash_file(source_path),
            "record_kind": source_record.get("record_kind"),
        }
    parity = r3.get("parity_artifact")
    if parity is not None:
        from .r3_parity import validate_parity_artifact  # noqa: PLC0415

        declaration = binding.get("parity_artifact")
        _require(declaration is not None, "trusted R3 parity declaration is missing")
        metadata, _array = validate_parity_artifact(declaration)
        parity_path = Path(declaration["path"])
        metadata_path = Path(declaration["metadata_path"])
        evidence["parity_artifact"] = {
            "path": str(parity_path),
            "sha256": metadata["artifact_sha256"],
            "metadata_sha256": _hash_file(metadata_path),
        }
    return evidence


def job_status(manifest):
    events = _events(manifest["outputs"]["status_path"])
    result_path = Path(manifest["outputs"]["result_path"])
    video_path = Path(manifest["outputs"]["video_path"])
    state = events[-1]["state"] if events else "not-started"
    if state == "completed":
        if not result_path.is_file() or not video_path.is_file() or video_path.stat().st_size == 0:
            return "invalid-completed"
        completed_event = events[-1]
        result = _read_json(result_path)
        manifest_sha256 = _manifest_hash(manifest)
        video_sha256 = _hash_file(video_path)
        if (
            result.get("schema_version") != SCHEMA_VERSION
            or result.get("state") != "completed"
            or result.get("job_id") != manifest["job_id"]
            or result.get("manifest_sha256") != manifest_sha256
            or result.get("video_path") != str(video_path)
            or result.get("video_sha256") != video_sha256
            or completed_event.get("schema_version") != SCHEMA_VERSION
            or completed_event.get("job_id") != manifest["job_id"]
            or completed_event.get("manifest_sha256") != manifest_sha256
            or completed_event.get("attempt_id") != result.get("attempt_id")
            or completed_event.get("video_sha256") != video_sha256
        ):
            return "invalid-completed"
        if "r3_evidence" in manifest:
            try:
                evidence = _verify_declared_r3_worker_evidence(manifest)
            except (KeyError, OSError, TypeError, ValueError):
                return "invalid-completed"
            if result.get("r3_worker_observation") != evidence:
                return "invalid-completed"
    return state


def _pipeline_commands(manifest_path, manifest, t5_cpu=False):
    outputs = manifest["outputs"]
    prepare = [sys.executable, "-m", "multi_sample_inference.manifest_adapter", "--manifest", str(manifest_path)]
    mode = "solo" if manifest["inference"]["rank_count"] == 1 else "fsdp"
    worker = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={manifest['inference']['rank_count']}",
        "-m", "multi_sample_inference.fsdp_worker", "--task-file", outputs["task_path"],
        "--manifest-file", str(manifest_path), "--mode", mode,
    ]
    if t5_cpu:
        worker.append("--t5-cpu")
    return [prepare, worker]


def execute_job(manifest_path, t5_cpu=False, devices=None, command_factory=None):
    manifest_path = Path(manifest_path).resolve()
    manifest = _read_json(manifest_path)
    _require(not manifest.get("smoke_only"), "NON-SCIENTIFIC smoke fixtures cannot be run")
    verify_job_inputs(manifest)
    if "r3_evidence" in manifest:
        r3 = manifest["r3_evidence"]
        bundle = load_protocol_bundle(r3["protocol"]["path"], r3["matrix"]["path"])
        _require(
            bundle["protocol_sha256"] == r3["protocol"]["sha256"]
            and bundle["matrix_sha256"] == r3["matrix"]["sha256"],
            "declared R3 protocol or matrix identity changed",
        )
        declared_checkpoint = bundle["protocol"]["runtime_declarations"][
            "checkpoint_content_sha256"
        ]
        _require(
            declared_checkpoint is None
            or declared_checkpoint == manifest["checkpoint"]["content_sha256"],
            "R3 protocol checkpoint declaration differs from the manifest checkpoint",
        )
        blockers = execution_blockers(bundle["protocol"])
        _require(not blockers, "R3 execution preflight blocked: " + ", ".join(blockers))
        if bundle["protocol"]["schema_version"] >= 3:
            disposition = r3["case"].get("lineage_disposition", {})
            _require(
                disposition.get("disposition") == "runnable",
                "R3 matrix case is rejected before launch",
            )
            worker_task_blueprint(manifest)
    if job_status(manifest) == "completed":
        return "skipped-completed"
    lock_path = Path(manifest["outputs"]["status_path"] + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"job already has an active run lock: {lock_path}") from error
    try:
        return _execute_job_locked(manifest_path, manifest, t5_cpu, devices, command_factory)
    finally:
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)


def _execute_job_locked(manifest_path, manifest, t5_cpu, devices, command_factory):
    if job_status(manifest) == "completed":
        return "skipped-completed"
    video_path = Path(manifest["outputs"]["video_path"])
    result_path = Path(manifest["outputs"]["result_path"])
    if video_path.exists() or result_path.exists():
        raise FileExistsError("incomplete/failed output exists; preserve or move it before retrying")
    if devices is not None:
        selected = [part for part in devices.split(",") if part]
        _require(len(selected) == manifest["inference"]["rank_count"], "device count must equal manifest rank_count")
    attempt = uuid.uuid4().hex
    now = lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    base = {"schema_version": SCHEMA_VERSION, "job_id": manifest["job_id"], "manifest_sha256": _manifest_hash(manifest), "attempt_id": attempt}
    _append_status(manifest["outputs"]["status_path"], base | {"state": "running", "timestamp": now()})
    commands = (command_factory or _pipeline_commands)(manifest_path, manifest, t5_cpu)
    env = os.environ.copy()
    if devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = devices
    try:
        for command in commands:
            subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[1], env=env)
        _require(video_path.is_file() and video_path.stat().st_size > 0, "worker exited successfully without a nonempty declared video")
        r3_evidence = _verify_declared_r3_worker_evidence(manifest)
        result = base | {"state": "completed", "video_path": str(video_path), "video_sha256": _hash_file(video_path)}
        if r3_evidence is not None:
            result["r3_worker_observation"] = r3_evidence
        _write_immutable(result_path, _canonical(result))
        _append_status(manifest["outputs"]["status_path"], base | {"state": "completed", "timestamp": now(), "video_sha256": result["video_sha256"]})
        return "completed"
    except Exception as error:
        exit_code = error.returncode if isinstance(error, subprocess.CalledProcessError) else None
        _append_status(manifest["outputs"]["status_path"], base | {"state": "failed", "timestamp": now(), "exit_code": exit_code, "error": f"{type(error).__name__}: {error}"})
        raise


def _annotation_rows(manifest_paths):
    rows = []
    for path in manifest_paths:
        manifest = _read_json(path)
        for actor_id in manifest["identity"]["actor_order"]:
            rows.append({
                "job_id": manifest["job_id"], "scene_id": manifest["identity"]["scene_id"],
                "assignment_id": manifest["identity"]["assignment_id"], "condition_id": manifest["identity"]["condition_id"],
                "video_path": manifest["outputs"]["video_path"], "actor_id": actor_id,
                "target_action": manifest["assignment"]["targets"][actor_id],
                **{key: "" for key in ANNOTATION_COLUMNS[7:]},
            })
    return rows


def export_annotations(manifest_paths, output):
    rows = _annotation_rows(manifest_paths)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_COLUMNS)
        writer.writeheader(); writer.writerows(rows)
    return len(rows)


def import_annotations(template, output, manifest_paths):
    with Path(template).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require(reader.fieldnames == ANNOTATION_COLUMNS, "annotation columns were changed")
        rows = list(reader)
    expected_rows = _annotation_rows(manifest_paths)
    immutable_fields = ANNOTATION_COLUMNS[:7]
    expected = {(row["job_id"], row["actor_id"]): tuple(row[field] for field in immutable_fields) for row in expected_rows}
    actual = {(row["job_id"], row["actor_id"]): tuple(row[field] for field in immutable_fields) for row in rows}
    _require(actual == expected, "annotation actor IDs, targets, paths, or row set differ from the job manifests")
    seen = set()
    per_job = {}
    actor_fields = ("target_present", "wrong_action_present", "identity_correct", "visible")
    clip_fields = ("joint_both_correct", "quality", "uncertain", "missing", "tracking_failure")
    for row in rows:
        key = (row["job_id"], row["actor_id"])
        _require(key not in seen, f"duplicate annotation row: {key}"); seen.add(key)
        for field in actor_fields:
            _require(row[field] in LABELS, f"{field} must be true/false/uncertain/missing")
        for field in ("joint_both_correct", "uncertain", "missing", "tracking_failure"):
            _require(row[field] in LABELS, f"{field} must be true/false/uncertain/missing")
        _require(row["quality"] in {"1", "2", "3", "4", "5", "uncertain", "missing"}, "quality must be 1-5/uncertain/missing")
        values = tuple(row[field] for field in clip_fields)
        _require(row["job_id"] not in per_job or per_job[row["job_id"]] == values, "clip-level labels must agree across actor rows")
        per_job[row["job_id"]] = values
    payload = {"schema_version": SCHEMA_VERSION, "rows": rows, "denominators": {"actor_rows": len(rows), "jobs": len(per_job), "uncertain_rows": sum(r["uncertain"] == "true" for r in rows), "missing_rows": sum(r["missing"] == "true" for r in rows), "tracking_failure_rows": sum(r["tracking_failure"] == "true" for r in rows)}}
    _write_immutable(output, _canonical(payload))
    return payload


def _job_files(path):
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(item for item in path.glob("*.json") if item.name != "index.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate"); validate.add_argument("--source", required=True, type=Path)
    expand = sub.add_parser("expand"); expand.add_argument("--source", required=True, type=Path); expand.add_argument("--output-dir", required=True, type=Path)
    dry = sub.add_parser("dry-run"); dry.add_argument("--source", required=True, type=Path); dry.add_argument("--output-dir", required=True, type=Path)
    status = sub.add_parser("status"); status.add_argument("--jobs", required=True, type=Path)
    run = sub.add_parser("run"); run.add_argument("--job", required=True, type=Path); run.add_argument("--devices"); run.add_argument("--t5-cpu", action="store_true")
    ann_export = sub.add_parser("annotations-export"); ann_export.add_argument("--jobs", required=True, type=Path); ann_export.add_argument("--output", required=True, type=Path)
    ann_import = sub.add_parser("annotations-import"); ann_import.add_argument("--template", required=True, type=Path); ann_import.add_argument("--jobs", required=True, type=Path); ann_import.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "validate":
        source = validate_source(args.source); print(f"valid: {source['experiment_id']}")
    elif args.command in {"expand", "dry-run"}:
        manifests = expand_source(args.source, args.output_dir, write=args.command == "expand")
        print(f"{args.command}: {len(manifests)} jobs")
        for manifest in manifests: print(manifest["job_id"])
    elif args.command == "status":
        for path in _job_files(args.jobs):
            manifest = _read_json(path); print(f"{manifest['job_id']}\t{job_status(manifest)}")
    elif args.command == "run":
        print(execute_job(args.job, t5_cpu=args.t5_cpu, devices=args.devices))
    elif args.command == "annotations-export":
        print(f"exported {export_annotations(_job_files(args.jobs), args.output)} actor rows")
    elif args.command == "annotations-import":
        payload = import_annotations(args.template, args.output, _job_files(args.jobs)); print(f"imported {len(payload['rows'])} actor rows")


if __name__ == "__main__":
    main()
