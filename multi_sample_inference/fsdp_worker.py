import argparse
import copy
import os
import pickle
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
    write_immutable_json,
)
from .r3_environment import observe_attention_runtime, observe_runtime_environment
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
    attention_runtime = observe_attention_runtime()
    validate_backend_runtime(dispatch_contract, attention_runtime)
    if schema_version >= 4:
        r3 = manifest["r3_evidence"]
        bundle = load_protocol_bundle(
            r3["protocol"]["path"], r3["matrix"]["path"]
        )
        validate_v4_runtime_environment(
            bundle["protocol"], observe_runtime_environment(attention_runtime)
        )


def _validate_r3_route_process_before_model_load(task):
    evidence = task.get("r3_evidence")
    if not isinstance(evidence, dict) or evidence.get("protocol_schema_version", 0) < 9:
        return
    from .r3_checkout_binding import validate_checkout_route_binding  # noqa: PLC0415
    from .r3_route_isolation import assert_loaded_wan_modules  # noqa: PLC0415

    binding = evidence.get("route_process")
    validate_checkout_route_binding(binding)
    expected_route = (
        "official-pristine"
        if generation_route(task) == "upstream"
        else "local-custom"
    )
    if binding["route"] != expected_route:
        raise ValueError("R3 worker route differs from its isolated-process binding")
    assert_loaded_wan_modules(binding["wan_root"])


def _scope_local_dispatch_observer(collector):
    """Keep pre-model encoder attention distinct from denoising-model dispatches."""
    initial_seen = False
    model_started = False

    def observe(record):
        nonlocal initial_seen, model_started
        event = record.get("event")
        if event == "initial-latent":
            initial_seen = True
        elif event == "attention-dispatch":
            coordinates = ("step", "branch", "block", "attention_site")
            if any(key in record for key in coordinates):
                model_started = True
            elif initial_seen and not model_started:
                record = record | {"event": "pre-model-attention-dispatch", "scope": "pre-model"}
        collector(record)

    return observe


def _runtime_observer_context(wan_i2v, task, collector, rank):
    evidence = task["r3_evidence"]
    if evidence.get("protocol_schema_version", 0) >= 9 and generation_route(task) == "upstream":
        from .r3_pristine_adapter import (  # noqa: PLC0415
            install_pristine_runtime_observer,
        )

        requested = evidence["requested"]
        dispatch = requested["dispatch_contract"]
        return install_pristine_runtime_observer(
            wan_i2v,
            collector,
            rank=rank,
            diffusion_seed=requested["diffusion_seed"],
            sampling_steps=evidence["expected"]["sampling_steps"],
            expected_backend=requested["attention_backend"],
            expected_backend_version=dispatch["backend_versions"][
                requested["attention_backend"]
            ],
        )
    from wan.utils.runtime_evidence import install_runtime_observer  # noqa: PLC0415

    if evidence.get("protocol_schema_version", 0) >= 12:
        collector = _scope_local_dispatch_observer(collector)
    return install_runtime_observer(collector, rank=rank)


def _build_generator(task, *, local_rank, rank, t5_fsdp, dit_fsdp, t5_cpu):
    route = generation_route(task)
    if route == "upstream":
        from wan.image2video import WanI2V  # noqa: PLC0415
    else:
        from wan.regional_prompt.image2video import WanI2V  # noqa: PLC0415

    evidence = task.get("r3_evidence") or {}
    parity = evidence.get("parity_artifact") or {}
    match_pristine_weights = (
        evidence.get("protocol_schema_version", 0) >= 12
        and parity.get("route") == "custom-none"
    )
    config = i2v_14B
    if match_pristine_weights:
        if (route != "custom" or task["config"]["bias_method"] != "none"
                or evidence.get("route_process", {}).get("route") != "local-custom"
                or rank != 0 or t5_fsdp or dit_fsdp or i2v_14B.param_dtype != torch.bfloat16):
            raise ValueError("R3 FP32 parity weight load requires single-rank local custom-none BF16 runtime")
        # The official checkpoint stores F32 DiT weights; the pristine constructor
        # preserves them. Copy only this generator's configuration so normal jobs
        # retain their BF16 weight loading and neither Wan checkout is modified.
        config = copy.copy(i2v_14B)
        config.param_dtype = torch.float32

    generator = WanI2V(
        config=config,
        checkpoint_dir=task["checkpoint_dir"],
        device_id=local_rank,
        rank=rank,
        t5_fsdp=t5_fsdp,
        dit_fsdp=dit_fsdp,
        use_usp=False,
        t5_cpu=t5_cpu,
        init_on_cpu=True,
    )
    if match_pristine_weights:
        floating_dtypes = {param.dtype for param in generator.model.parameters()
                           if param.is_floating_point()}
        if floating_dtypes != {torch.float32}:
            raise ValueError("R3 parity DiT weights did not retain checkpoint FP32 dtype")
        # The local generator also uses param_dtype for activation/autocast;
        # preserve its original BF16 runtime while keeping the DiT weights F32.
        generator.param_dtype = i2v_14B.param_dtype
    return generator


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
    _validate_r3_route_process_before_model_load(task)

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
        collector = R3RuntimeCollector(
            parity_artifact=task["r3_evidence"].get("parity_artifact")
        )
        with _runtime_observer_context(
            wan_i2v, task, collector, dist.get_rank()
        ):
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
