import argparse
import importlib.metadata
import os
import pickle
import platform
import socket
import subprocess
import time
from pathlib import Path

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torch.distributed as dist
from diffusers.utils.export_utils import export_to_video
from wan.configs.wan_i2v_14B import i2v_14B

from debug_utils import unscale, write_video_soft_masks
from utils import normalize_video_tensor

from .experiment_pipeline import (
    _read_json,
    validate_r3_worker_task_binding,
    verify_repository_identity,
)
from .generation_routes import generation_route, rank_zero_output, run_generator
from .r3_contracts import (
    GENUINE_RUNTIME_EVIDENCE,
    build_worker_observation,
    load_protocol_bundle,
    sha256_file,
    write_immutable_json,
)
from .r3_preflight import validate_backend_runtime, validate_v4_runtime_environment
from .r3_runtime import (
    R3RuntimeCollector,
    gather_rank_observations,
)
from .task_contracts import (
    build_subject_indices,
    resolve_bool_schedule,
    resolve_inference_settings,
    validate_method_layout,
    validate_worker_task,
)


@torch.inference_mode()
def run_inference(wan_i2v, task):
    validate_worker_task(task)
    settings = resolve_inference_settings(task.get("inference_settings"))
    if generation_route(task) == "upstream":
        return run_generator(wan_i2v, task, settings)

    masks = task["masks"]
    if masks.ndim != 3 or masks.shape[0] == 0:
        raise ValueError("task masks must have shape [subjects, height, width]")
    masks = masks.bool()
    if (masks.sum(dim=0) > 1).any():
        raise ValueError("task subject masks must be disjoint")
    if not masks.flatten(1).any(dim=1).all():
        raise ValueError("every task subject mask must be nonempty")

    num_entities = masks.shape[0]
    subject_indices = build_subject_indices(task["char_segments_list"], num_entities)
    config = task["config"].copy()
    config.pop("diffusion_seed", None)
    config.pop("inference_settings", None)
    for key in settings:
        config.pop(key, None)
    timestep_schedule = resolve_bool_schedule(
        config.pop("timestep_bias_schedule", [True] * settings["sampling_steps"]),
        settings["sampling_steps"],
        "timestep_bias_schedule",
    )
    blocks_schedule = resolve_bool_schedule(
        config.pop("blocks_bias_schedule", [True] * wan_i2v.model.num_layers),
        wan_i2v.model.num_layers,
        "blocks_bias_schedule",
    )
    validate_method_layout(config.get("bias_method"), subject_indices, num_entities)
    use_isolated_images = config.pop("image_context_isolation", True) and config.get(
        "bias_method"
    ) == "concept_weaver"
    if use_isolated_images and len(task["single_char_imgs"]) != num_entities:
        raise ValueError("isolated image count must match subject mask count")

    bias_kwargs = {
        "general_prompt": task["general_prompt"],
        "prompt_data_list": [
            {
                "control_prompts": [
                    ((subject_idx,), {"prompt": seg, "char_descr_list": []})
                    for subject_idx, seg in zip(indices, chars, strict=True)
                ],
                "single_char_img": task["single_char_imgs"][i]
                if use_isolated_images
                else None,
            }
            for i, (indices, chars) in enumerate(
                zip(subject_indices, task["char_segments_list"], strict=True)
            )
        ],
        "timestep_bias_schedule": torch.tensor(timestep_schedule, dtype=torch.bool),
        "blocks_bias_schedule": torch.tensor(blocks_schedule, dtype=torch.bool),
        "face_masks": masks.to(wan_i2v.device),
        "wlw_matrix": torch.from_numpy(
            np.repeat(
                np.eye(num_entities)[..., np.newaxis],
                settings["frame_num"],
                axis=-1,
            )
        ).to(wan_i2v.param_dtype),
    } | config

    return run_generator(wan_i2v, task, settings, bias_kwargs=bias_kwargs)


def save_outputs(video, extra_data, task, max_retries=5):
    """Restored logic to save both the video and the attention masks."""

    def _robust_run(path, func, *args):
        attempts = 0
        while attempts < max_retries:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                return func(*args)
            except FileNotFoundError:
                attempts += 1
                if attempts == max_retries:
                    raise
                time.sleep(1)

    # 1. Save the main Video
    video_norm = normalize_video_tensor(video.cpu().numpy())
    _robust_run(task["video_path"], export_to_video, list(video_norm), task["video_path"], 16)

    # 2. Extract and Save Soft Attention Masks (from extra_data)
    if "simil_masks" in extra_data:
        simil_masks = extra_data["simil_masks"]

        assert simil_masks.shape[0] == 1
        simil_masks = simil_masks[0]

        h, w = video.shape[-2:]
        action_output_path = task["action_output_path"]
        repeat_idx = task["repeat_idx"]

        for step_idx in [0, -1]:
            # average the masks across all dit blocks
            # Result shape: [num_char, T, H, W]
            step_simil_masks = simil_masks[step_idx].float().mean(dim=0)

            # upscale latent-space masks to match the real video resolution
            frame_num = resolve_inference_settings(
                task.get("inference_settings")
            )["frame_num"]
            step_simil_masks = unscale(step_simil_masks, (frame_num, h, w))

            for char_idx, mask in enumerate(step_simil_masks):
                mask_filename = f"video_{repeat_idx}_soft_masks_char_{char_idx}_step_{step_idx}.mp4"
                mask_path = action_output_path / mask_filename

                _robust_run(
                    mask_path,
                    write_video_soft_masks,
                    video_norm,
                    mask_path,
                    mask.cpu().numpy(),
                    16,
                )

    if "r3_evidence" in task and task["r3_evidence"]["path"] is not None:
        record = build_worker_observation(
            task["r3_evidence"],
            extra_data,
            observed_rank=dist.get_rank(),
        )
        write_immutable_json(task["r3_evidence"]["path"], record)


def _observed_attention_runtime():
    """Read helper state without accepting caller-supplied availability claims."""
    from wan.modules import attention  # noqa: PLC0415

    try:
        from torch.nn.attention.flex_attention import flex_attention  # noqa: PLC0415
    except ImportError:
        flex_attention = None
    flash_attn = getattr(attention, "flash_attn", None)
    return {
        "flash_attention_2_available": attention.FLASH_ATTN_2_AVAILABLE,
        "flash_attention_3_available": attention.FLASH_ATTN_3_AVAILABLE,
        "flash_attention_version": getattr(flash_attn, "__version__", None),
        "flex_attention_available": callable(flex_attention),
        "flex_attention_version": torch.__version__ if callable(flex_attention) else None,
    }


def _observed_v4_environment(attention_runtime):
    """Observe the v4 execution environment without loading checkpoint tensors."""
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    rows = [line.split(", ", 3) for line in query.splitlines() if line.strip()]
    if not rows or any(len(row) != 4 for row in rows):
        raise RuntimeError("R3 v4 could not observe the declared GPU inventory")
    names = {row[1] for row in rows}
    drivers = {row[2] for row in rows}
    capabilities = {row[3] for row in rows}
    if len(names) != 1 or len(drivers) != 1 or len(capabilities) != 1:
        raise RuntimeError("R3 v4 requires a homogeneous GPU inventory")
    repo = Path(__file__).resolve().parents[1]
    return {
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "flash_attention_version": attention_runtime["flash_attention_version"],
        "flex_attention_version": attention_runtime["flex_attention_version"],
        "sam2_version": importlib.metadata.version("sam2"),
        "driver_version": next(iter(drivers)),
        "gpu_model": next(iter(names)),
        "gpu_uuids": [row[0] for row in rows],
        "gpu_compute_capability": next(iter(capabilities)),
        "pyproject_sha256": sha256_file(repo / "pyproject.toml"),
        "uv_lock_sha256": sha256_file(repo / "uv.lock"),
    }


def _validate_r3_before_model_load(task, manifest_path):
    evidence = task.get("r3_evidence")
    schema_version = (
        evidence.get("protocol_schema_version")
        if isinstance(evidence, dict)
        else None
    )
    task_has_v3_material = (
        isinstance(evidence, dict)
        and (
            (type(schema_version) is int and schema_version >= 3)
            or (
                isinstance(evidence.get("requested"), dict)
                and "dispatch_contract" in evidence["requested"]
            )
        )
    )
    if manifest_path is None:
        if task_has_v3_material:
            raise ValueError("R3 v3+ worker requires its immutable manifest binding")
        return
    manifest = _read_json(manifest_path)
    if task_has_v3_material:
        verify_repository_identity(manifest)
    dispatch_contract = validate_r3_worker_task_binding(task, manifest)
    if dispatch_contract is None:
        return
    attention_runtime = _observed_attention_runtime()
    validate_backend_runtime(dispatch_contract, attention_runtime)
    if schema_version == 4:
        r3 = manifest["r3_evidence"]
        bundle = load_protocol_bundle(
            r3["protocol"]["path"], r3["matrix"]["path"]
        )
        validate_v4_runtime_environment(
            bundle["protocol"], _observed_v4_environment(attention_runtime)
        )


def _build_generator(task, *, local_rank, rank, t5_fsdp, dit_fsdp, t5_cpu):
    if generation_route(task) == "upstream":
        from wan.image2video import WanI2V  # noqa: PLC0415
    else:
        from wan.regional_prompt.image2video import WanI2V  # noqa: PLC0415

    return WanI2V(
        config=i2v_14B,
        checkpoint_dir=task["checkpoint_dir"],
        device_id=local_rank,
        rank=rank,
        t5_fsdp=t5_fsdp,
        dit_fsdp=dit_fsdp,
        use_usp=False,
        t5_cpu=t5_cpu,
        init_on_cpu=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-file", required=True, type=Path)
    parser.add_argument("--manifest-file", type=Path)
    parser.add_argument("--mode", choices=["fsdp", "solo"], required=True)
    parser.add_argument("--t5-cpu", action="store_true")
    args = parser.parse_args()

    try:
        local_rank = int(os.environ["LOCAL_RANK"])
    except KeyError as e:
        raise RuntimeError("Must launch via torchrun.") from e

    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))

    with open(args.task_file, "rb") as f:
        task = pickle.load(f)
    validate_worker_task(task)
    _validate_r3_before_model_load(task, args.manifest_file)

    is_fsdp = args.mode == "fsdp"
    # If we are sharding, we only shard T5 if it's NOT on the CPU
    t5_fsdp = is_fsdp and not args.t5_cpu

    wan_i2v = _build_generator(
        task,
        local_rank=local_rank,
        rank=dist.get_rank(),
        t5_fsdp=t5_fsdp,
        dit_fsdp=is_fsdp,
        t5_cpu=args.t5_cpu,
    )

    collector = None
    if "r3_evidence" in task:
        from wan.utils.runtime_evidence import install_runtime_observer  # noqa: PLC0415

        collector = R3RuntimeCollector(
            parity_artifact=task["r3_evidence"].get("parity_artifact")
        )
        with install_runtime_observer(collector, rank=dist.get_rank()):
            outputs = run_inference(wan_i2v, task)
        gathered = gather_rank_observations(
            collector.snapshot(),
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            gather_object=dist.gather_object,
        )
    else:
        outputs = run_inference(wan_i2v, task)
        gathered = None

    if gathered is not None and dist.get_rank() == 0:
        source_path = task["r3_evidence"].get("source_observation_path")
        if source_path is not None:
            write_immutable_json(
                source_path,
                {
                    "schema_version": task["r3_evidence"].get(
                        "protocol_schema_version", 2
                    ),
                    "record_kind": "r3-source-hook-observations",
                    "evidence_class": GENUINE_RUNTIME_EVIDENCE,
                    "bindings": task["r3_evidence"]["bindings"],
                    "case_id": task["r3_evidence"]["case_id"],
                    "expected": task["r3_evidence"]["expected"],
                    "requested": task["r3_evidence"]["requested"],
                    "observations": gathered,
                    "r3_acceptance": False,
                },
            )

    output = rank_zero_output(outputs, dist.get_rank())
    if output is not None:
        video, extra_data = output
        if gathered is not None:
            extra_data["r3_source_observations"] = gathered
        save_outputs(video, extra_data, task)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
