"""Dependency-light routing contract for upstream and custom Wan generators."""

CUSTOM_METHODS = {"none", "regional_prompting", "concept_weaver", "ediff-i"}
UPSTREAM_METHOD = "upstream"
UPSTREAM_FRAME_NUM = 81


def generation_route(task):
    """Return the concrete generator route and reject ambiguous task layouts."""
    method = task.get("config", {}).get("bias_method")
    if method == UPSTREAM_METHOD:
        if task.get("prompt_representation") != "joint":
            raise ValueError("upstream generation requires prompt_representation='joint'")
        sentences = task.get("prompt_sentences")
        if not isinstance(sentences, list) or len(sentences) != 1:
            raise ValueError("upstream generation requires exactly one rendered prompt sentence")
        negative_prompt = task.get("negative_prompt")
        if not isinstance(negative_prompt, str) or not negative_prompt:
            raise ValueError("upstream generation requires an explicit nonempty negative_prompt")
        return UPSTREAM_METHOD
    if method not in CUSTOM_METHODS:
        raise ValueError(f"unsupported worker generation method: {method!r}")
    representation = task.get("prompt_representation")
    if representation is not None and representation not in {"joint", "split"}:
        raise ValueError("prompt_representation must be joint or split")
    negative_prompt = task.get("negative_prompt")
    if negative_prompt is not None and (
        not isinstance(negative_prompt, str) or not negative_prompt
    ):
        raise ValueError("negative_prompt must be a nonempty string when supplied")
    return "custom"


def _common_generate_kwargs(task, settings, offload_model=False):
    # offload_model is a protocol binding (J1 jz-v4+ only); never a task or env choice.
    if type(offload_model) is not bool:
        raise TypeError("offload_model must be a bool from a validated protocol binding")
    return {
        "max_area": settings["target_size"][0] * settings["target_size"][1],
        "sampling_steps": settings["sampling_steps"],
        "frame_num": settings["frame_num"],
        "shift": settings["shift"],
        "sample_solver": settings["sample_solver"],
        "guide_scale": settings["guide_scale"],
        "seed": task["diffusion_seed"],
        "offload_model": offload_model,
    }


def _none_on_non_output_rank(generator, output):
    if output is not None:
        return False
    rank = getattr(generator, "rank", None)
    if type(rank) is int and rank != 0:
        return True
    raise TypeError("Wan generator returned None on the output rank")


def rank_zero_output(outputs, rank):
    """Return validated output only on rank zero for worker save control flow."""
    if type(rank) is not int or rank < 0:
        raise ValueError("rank must be a nonnegative integer")
    if rank != 0:
        return None
    if outputs is None:
        raise RuntimeError("rank zero generator did not return output")
    return outputs


def run_generator(generator, task, settings, *, bias_kwargs=None, offload_model=False):
    """Call the selected API while keeping upstream and custom routes distinct."""
    route = generation_route(task)
    if route == UPSTREAM_METHOD and settings.get("frame_num") != UPSTREAM_FRAME_NUM:
        raise ValueError(f"pinned upstream Wan generation requires frame_num={UPSTREAM_FRAME_NUM}")
    kwargs = _common_generate_kwargs(task, settings, offload_model)
    negative_prompt = task.get("negative_prompt")
    if negative_prompt is not None:
        kwargs["n_prompt"] = negative_prompt

    if route == UPSTREAM_METHOD:
        if bias_kwargs is not None:
            raise ValueError("upstream generation does not accept custom bias inputs")
        video = generator.generate(task["prompt_sentences"][0], task["img"], **kwargs)
        if _none_on_non_output_rank(generator, video):
            return None
        return video, {"execution_route": UPSTREAM_METHOD}

    if bias_kwargs is None:
        raise ValueError("custom generation requires explicit bias inputs")
    output = generator.generate(task["prompt_sentences"], task["img"], bias_kwargs, **kwargs)
    if _none_on_non_output_rank(generator, output):
        return None
    if not isinstance(output, tuple) or len(output) != 2:
        raise TypeError("custom Wan generator must return (video, extra_data)")
    video, extra_data = output
    if not isinstance(extra_data, dict):
        raise TypeError("custom Wan generator extra_data must be a dictionary")
    return video, extra_data
