import copy
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from multi_sample_inference import experiment_pipeline as pipeline
from multi_sample_inference.task_contracts import (
    REQUIRED_WORKER_TASK_KEYS,
    resolve_bool_schedule,
    validate_worker_task,
)

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "tests/fixtures/smoke_experiment.json"


def _expand(tmp_path):
    manifests = pipeline.expand_source(SOURCE, tmp_path)
    paths = sorted((tmp_path / "jobs").glob("job-*.json"))
    return manifests, paths


def _write_portable_source(tmp_path, source):
    source = copy.deepcopy(source)
    source["checkpoint"]["path"] = str((SOURCE.parent / source["checkpoint"]["path"]).resolve())
    source["checkpoint"]["inventory"] = str(
        (SOURCE.parent / source["checkpoint"]["inventory"]).resolve()
    )
    for scene in source["scenes"]:
        scene["reference_image"] = str((SOURCE.parent / scene["reference_image"]).resolve())
        for actor in scene["actors"]:
            actor["isolated_image"] = str(
                (SOURCE.parent / actor["isolated_image"]).resolve()
            )
        scene["segmentation_masks"] = {
            actor_id: str((SOURCE.parent / path).resolve())
            for actor_id, path in scene["segmentation_masks"].items()
        }
    path = tmp_path / "source.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(source))
    return path


def test_cli_validate_expand_and_dry_run_are_cpu_only_and_deterministic(tmp_path):
    validate = subprocess.run(
        [sys.executable, "-m", "multi_sample_inference.experiment_pipeline", "validate", "--source", str(SOURCE)],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    assert "NON-SCIENTIFIC-smoke-only" in validate.stdout
    dry = subprocess.run(
        [sys.executable, "-m", "multi_sample_inference.experiment_pipeline", "dry-run", "--source", str(SOURCE), "--output-dir", str(tmp_path / "dry")],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    assert "12 jobs" in dry.stdout and not (tmp_path / "dry").exists()
    first, paths = _expand(tmp_path / "expanded")
    second = pipeline.expand_source(SOURCE, tmp_path / "expanded")
    assert [item["job_id"] for item in first] == [item["job_id"] for item in second]
    assert len(paths) == 12
    pairs = {(m["identity"]["scene_id"], m["identity"]["assignment_id"], m["video_seed"]["id"]) for m in first}
    assert len(pairs) == 4
    assert all(sum((m["identity"]["scene_id"], m["identity"]["assignment_id"], m["video_seed"]["id"]) == pair for m in first) == 3 for pair in pairs)


def test_concept_weaver_requires_split_singletons_at_source_contract():
    source = json.loads(SOURCE.read_text())
    condition = copy.deepcopy(source["conditions"][0])
    condition.update({"id": "cw", "method": "concept_weaver"})
    source["conditions"] = [condition]
    assert pipeline._validate_source(source, SOURCE)
    source["conditions"][0]["prompt_representation"] = "joint"
    with pytest.raises(ValueError, match="split singleton"):
        pipeline._validate_source(source, SOURCE)


def test_upstream_requires_joint_prompt_and_explicit_negative_prompt():
    source = json.loads(SOURCE.read_text())
    condition = copy.deepcopy(source["conditions"][0])
    condition.update({
        "id": "upstream",
        "method": "upstream",
        "prompt_representation": "joint",
    })
    source["conditions"] = [condition]
    source["inference"]["frame_num"] = 81
    with pytest.raises(ValueError, match=r"explicit inference\.negative_prompt"):
        pipeline._validate_source(source, SOURCE)
    source["inference"]["negative_prompt"] = "predeclared shared negative prompt"
    assert pipeline._validate_source(source, SOURCE)
    source["inference"]["frame_num"] = 5
    with pytest.raises(ValueError, match=r"requires inference\.frame_num=81"):
        pipeline._validate_source(source, SOURCE)
    source["inference"]["frame_num"] = 81
    source["conditions"][0]["prompt_representation"] = "split"
    with pytest.raises(ValueError, match="one joint prompt"):
        pipeline._validate_source(source, SOURCE)


def test_upstream_and_custom_none_are_distinct_matched_manifests(tmp_path):
    source = json.loads(SOURCE.read_text())
    custom = copy.deepcopy(source["conditions"][0])
    custom.update({"id": "custom-none-joint", "prompt_representation": "joint"})
    upstream = copy.deepcopy(custom)
    upstream.update({"id": "upstream-joint", "method": "upstream"})
    source["conditions"] = [custom, upstream]
    source["inference"]["frame_num"] = 81
    source["inference"]["negative_prompt"] = "predeclared shared negative prompt"
    source_path = _write_portable_source(tmp_path, source)
    manifests = pipeline.expand_source(source_path, tmp_path / "run")

    assert len(manifests) == 8
    grouped = {}
    for manifest in manifests:
        key = (
            manifest["identity"]["scene_id"],
            manifest["identity"]["assignment_id"],
            manifest["video_seed"]["id"],
        )
        grouped.setdefault(key, {})[manifest["intervention"]["method"]] = manifest
    assert all(set(pair) == {"none", "upstream"} for pair in grouped.values())
    for pair in grouped.values():
        custom_manifest, upstream_manifest = pair["none"], pair["upstream"]
        assert custom_manifest["job_id"] != upstream_manifest["job_id"]
        assert custom_manifest["prompts"] == upstream_manifest["prompts"]
        assert custom_manifest["assets"] == upstream_manifest["assets"]
        assert custom_manifest["checkpoint"] == upstream_manifest["checkpoint"]
        assert custom_manifest["video_seed"] == upstream_manifest["video_seed"]
        upstream_task = pipeline.worker_task_blueprint(upstream_manifest)
        assert upstream_task["prompt_representation"] == "joint"
        assert upstream_task["negative_prompt"] == "predeclared shared negative prompt"
        assert upstream_task["config"]["bias_method"] == "upstream"


def test_manifest_hashes_worker_schema_and_immutability(tmp_path):
    manifests, paths = _expand(tmp_path)
    manifest = manifests[0]
    assert manifest["assets"]["reference_image"]["sha256"] == pipeline._hash_file(Path(manifest["assets"]["reference_image"]["path"]))
    assert manifest["prompts"]["sha256"] and manifest["masks"]["sha256"] and manifest["config_sha256"]
    blueprint = pipeline.worker_task_blueprint(manifest)
    task = blueprint | {"img": object(), "single_char_imgs": [object(), object()], "masks": object()}
    assert set(task) == REQUIRED_WORKER_TASK_KEYS | {"prompt_representation", "negative_prompt"}
    assert validate_worker_task(task) is task
    assert resolve_bool_schedule([True, False], 2, "steps") == [True, False]
    with pytest.raises(ValueError, match="length 2"):
        resolve_bool_schedule([True], 2, "steps")
    paths[0].write_text("{}\n")
    with pytest.raises(FileExistsError, match="immutable content conflict"):
        pipeline.expand_source(SOURCE, tmp_path)


def test_checkpoint_inventory_and_task_pickle_are_content_verified(tmp_path):
    manifests, _ = _expand(tmp_path / "expanded")
    manifest = manifests[0]

    checkpoint_dir = tmp_path / "checkpoint"
    shutil.copytree(Path(manifest["checkpoint"]["path"]), checkpoint_dir)
    inventory_path = tmp_path / "checkpoint.inventory.json"
    shutil.copyfile(manifest["checkpoint"]["inventory"]["path"], inventory_path)
    checkpoint = copy.deepcopy(manifest["checkpoint"])
    checkpoint["path"] = str(checkpoint_dir)
    checkpoint["inventory"] = {
        "path": str(inventory_path),
        "sha256": pipeline._hash_file(inventory_path),
    }
    pipeline.verify_checkpoint_identity(checkpoint)
    checkpoint_file = checkpoint_dir / "NOT_A_MODEL.txt"
    original_checkpoint_bytes = checkpoint_file.read_bytes()
    checkpoint_file.write_bytes(b"X" * checkpoint_file.stat().st_size)
    with pytest.raises(ValueError, match="content changed"):
        pipeline.verify_checkpoint_identity(checkpoint)
    checkpoint_file.write_bytes(original_checkpoint_bytes)
    inventory_path.write_text("{}\n")
    with pytest.raises(ValueError, match="inventory changed"):
        pipeline.verify_checkpoint_identity(checkpoint)

    task_path = tmp_path / "task.pkl"
    metadata_path = tmp_path / "task.json"
    task_path.write_bytes(b"original task")
    metadata_path.write_bytes(pipeline._canonical(pipeline.task_descriptor_metadata(
        manifest,
        task_path,
        {"processing_variant": "fixed_binary_v1", "actors": {}},
    )))
    pipeline.verify_task_descriptor(manifest, task_path, metadata_path)
    task_path.write_bytes(b"mutated task")
    with pytest.raises(ValueError, match="content hash"):
        pipeline.verify_task_descriptor(manifest, task_path, metadata_path)


def test_subprocess_failure_resume_and_existing_output_are_not_silently_skipped(tmp_path):
    manifests, paths = _expand(tmp_path)
    manifest_path = paths[0]
    manifest = json.loads(manifest_path.read_text())
    manifest["smoke_only"] = False
    manifest_path.write_bytes(pipeline._canonical(manifest))

    def fail_commands(_path, _manifest, _t5_cpu):
        return [[sys.executable, "-c", "import sys; sys.exit(7)"]]

    with pytest.raises(subprocess.CalledProcessError) as error:
        pipeline.execute_job(manifest_path, command_factory=fail_commands)
    assert error.value.returncode == 7
    assert pipeline.job_status(manifest) == "failed"

    def succeed_commands(_path, selected, _t5_cpu):
        code = "from pathlib import Path; Path(%r).parent.mkdir(parents=True, exist_ok=True); Path(%r).write_bytes(b'video')" % (selected["outputs"]["video_path"], selected["outputs"]["video_path"])
        return [[sys.executable, "-c", code]]

    assert pipeline.execute_job(manifest_path, command_factory=succeed_commands) == "completed"
    assert pipeline.job_status(manifest) == "completed"
    assert pipeline.execute_job(manifest_path, command_factory=fail_commands) == "skipped-completed"

    other_path = paths[1]
    other = json.loads(other_path.read_text()); other["smoke_only"] = False
    other_path.write_bytes(pipeline._canonical(other))
    video = Path(other["outputs"]["video_path"]); video.parent.mkdir(parents=True); video.write_bytes(b"partial")
    with pytest.raises(FileExistsError, match="incomplete/failed output"):
        pipeline.execute_job(other_path, command_factory=succeed_commands)


def test_annotation_export_and_import_preserves_actor_and_failure_labels(tmp_path):
    _, paths = _expand(tmp_path / "run")
    template = tmp_path / "annotations.csv"
    assert pipeline.export_annotations(paths[:1], template) == 2
    with template.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["actor_id"] for row in rows} == {"actor-left", "actor-right"}
    assert all(row["target_action"] for row in rows)
    for row in rows:
        row.update({
            "target_present": "true", "wrong_action_present": "false",
            "identity_correct": "true", "visible": "true",
            "joint_both_correct": "true", "quality": "4",
            "uncertain": "false", "missing": "false", "tracking_failure": "false",
        })
    with template.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pipeline.ANNOTATION_COLUMNS)
        writer.writeheader(); writer.writerows(rows)
    result = pipeline.import_annotations(template, tmp_path / "annotations.json", paths[:1])
    assert result["denominators"] == {"actor_rows": 2, "jobs": 1, "uncertain_rows": 0, "missing_rows": 0, "tracking_failure_rows": 0}
    rows[1]["joint_both_correct"] = "false"
    with template.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pipeline.ANNOTATION_COLUMNS)
        writer.writeheader(); writer.writerows(rows)
    with pytest.raises(ValueError, match="clip-level labels"):
        pipeline.import_annotations(template, tmp_path / "bad.json", paths[:1])
