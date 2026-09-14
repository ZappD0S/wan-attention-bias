"""Materialize one immutable JSON job as the existing fsdp_worker task pickle."""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from diffusers.utils.loading_utils import load_image

from .experiment_pipeline import (
    _canonical,
    _read_json,
    task_descriptor_metadata,
    verify_task_descriptor,
    worker_task_blueprint,
)
from .generation_routes import generation_route
from .mask_contracts import process_subject_masks, resolved_mask_evidence
from .task_contracts import validate_worker_task


def _build_masks(manifest):
    height, width = manifest["inference"]["height"], manifest["inference"]["width"]
    actor_order = manifest["identity"]["actor_order"]
    if manifest["masks"]["source"] == "fixed":
        masks = np.zeros((len(actor_order), height, width), dtype=bool)
        for index, actor_id in enumerate(actor_order):
            left, top, right, bottom = manifest["masks"]["fixed_boxes"][actor_id]
            masks[index, int(top):int(bottom), int(left):int(right)] = True
    else:
        masks = np.stack([
            np.array(load_image(manifest["masks"]["segmentation_masks"][actor_id]["path"]).convert("L")) > manifest["masks"]["processing"]["threshold"]
            for actor_id in actor_order
        ])
    if masks.shape != (len(actor_order), height, width):
        raise ValueError(
            f"resolved masks must have shape {(len(actor_order), height, width)}, got {masks.shape}; implicit resizing is unsupported"
        )
    masks = process_subject_masks(masks, manifest["masks"]["processing"])
    return torch.from_numpy(masks)


def build_task(manifest):
    task = worker_task_blueprint(manifest)
    actor_order = manifest["identity"]["actor_order"]
    task["video_path"] = Path(task["video_path"])
    task["action_output_path"] = Path(task["action_output_path"])
    task["img"] = load_image(manifest["assets"]["reference_image"]["path"])
    if generation_route(task) == "upstream":
        task["single_char_imgs"] = []
        task["masks"] = None
    else:
        task["single_char_imgs"] = [
            load_image(manifest["assets"]["isolated_images"][actor_id]["path"])
            for actor_id in actor_order
        ]
        task["masks"] = _build_masks(manifest)
    validate_worker_task(task)
    return task


def materialize(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = _read_json(manifest_path)
    task_path = Path(manifest["outputs"]["task_path"])
    metadata_path = Path(manifest["outputs"]["task_metadata_path"])
    if task_path.exists() or metadata_path.exists():
        try:
            verify_task_descriptor(manifest, task_path, metadata_path)
        except (ValueError, OSError) as error:
            raise FileExistsError("existing task descriptor does not match this manifest") from error
        return task_path
    task = build_task(manifest)
    task_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(task_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(task, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if task["masks"] is None:
            resolved_masks = {
                "processing_variant": "not_materialized_upstream_v1",
                "actors": {},
                "reason": "upstream route does not consume intervention masks",
            }
        else:
            masks = task["masks"].cpu().numpy()
            resolved_masks = resolved_mask_evidence(
                masks,
                manifest["identity"]["actor_order"],
                manifest["masks"]["processing"],
            )
        metadata = task_descriptor_metadata(manifest, task_path, resolved_masks)
        fd = os.open(metadata_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, "wb") as handle:
            handle.write(_canonical(metadata))
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        task_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        raise
    return task_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    print(materialize(args.manifest))


if __name__ == "__main__":
    main()
