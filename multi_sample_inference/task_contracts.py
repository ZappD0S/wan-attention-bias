"""Validation and mapping helpers for inference task adapters."""

REQUIRED_WORKER_TASK_KEYS = {
    "prompt_sentences",
    "checkpoint_dir",
    "img",
    "single_char_imgs",
    "char_segments_list",
    "masks",
    "config",
    "general_prompt",
    "video_path",
    "action_output_path",
    "repeat_idx",
    "diffusion_seed",
    "inference_settings",
}

DEFAULT_INFERENCE_SETTINGS = {
    "sampling_steps": 40,
    "frame_num": 81,
    "target_size": (480, 832),
    "shift": 5.0,
    "sample_solver": "unipc",
    "guide_scale": 5.0,
}


def resolve_inference_settings(overrides=None):
    settings = DEFAULT_INFERENCE_SETTINGS | (overrides or {})
    if not isinstance(settings["sampling_steps"], int) or settings["sampling_steps"] <= 0:
        raise ValueError("sampling_steps must be a positive integer")
    if not isinstance(settings["frame_num"], int) or settings["frame_num"] <= 0:
        raise ValueError("frame_num must be a positive integer")
    if (settings["frame_num"] - 1) % 4:
        raise ValueError("frame_num must have the form 4n+1")
    target_size = tuple(settings["target_size"])
    if len(target_size) != 2 or any(not isinstance(v, int) or v <= 0 for v in target_size):
        raise ValueError("target_size must contain two positive integers")
    settings["target_size"] = target_size
    if settings["sample_solver"] not in {"unipc", "dpm++"}:
        raise ValueError("sample_solver must be unipc or dpm++")
    return settings


def validate_worker_task(task):
    missing = REQUIRED_WORKER_TASK_KEYS - set(task)
    if missing:
        raise ValueError(f"worker task is missing required keys: {sorted(missing)}")
    if not isinstance(task["diffusion_seed"], int) or not 0 <= task["diffusion_seed"] < 2**63:
        raise ValueError("diffusion_seed must be a nonnegative signed 64-bit integer")
    resolve_inference_settings(task["inference_settings"])
    return task


def resolve_bool_schedule(value, expected_length, name):
    if not isinstance(value, list) or len(value) != expected_length:
        raise ValueError(f"{name} must be a boolean list of length {expected_length}")
    if any(type(item) is not bool for item in value):
        raise ValueError(f"{name} must contain only booleans")
    return value


def validate_method_layout(method, subject_indices, num_entities):
    """Reject method/layout combinations that cannot bind every entity."""
    if method == "concept_weaver" and (
        len(subject_indices) != num_entities
        or any(indices != [index] for index, indices in enumerate(subject_indices))
    ):
        raise ValueError(
            "concept_weaver requires one singleton sentence per entity in entity order"
        )


def build_subject_indices(char_segments_list, num_entities):
    """Map prompt character segments to persistent mask/image entity indices.

    Supported layouts are one joint sentence containing every entity, and one
    singleton sentence per entity. The latter keeps sentence i bound to entity i.
    """
    if num_entities <= 0:
        raise ValueError("at least one subject mask is required")
    counts = [len(chars) for chars in char_segments_list]
    if len(counts) == 1 and counts[0] == num_entities:
        return [list(range(num_entities))]
    if len(counts) == num_entities and all(count == 1 for count in counts):
        return [[i] for i in range(num_entities)]
    raise ValueError(
        "unsupported prompt/entity layout: expected one joint sentence with one "
        "character segment per entity, or one singleton sentence per entity"
    )
