import copy

import pytest

from multi_sample_inference.generation_routes import (
    generation_route,
    rank_zero_output,
    run_generator,
)


class FakeGenerator:
    def __init__(self, output, *, rank=0):
        self.output = output
        self.rank = rank
        self.calls = []

    def generate(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.output


def _task(method):
    return {
        "config": {"bias_method": method},
        "prompt_sentences": ["one rendered joint sentence"],
        "prompt_representation": "joint",
        "negative_prompt": "same explicit negative prompt",
        "img": object(),
        "diffusion_seed": 123,
    }


def _settings(frame_num=5):
    return {
        "sampling_steps": 2,
        "frame_num": frame_num,
        "target_size": (16, 16),
        "shift": 5.0,
        "sample_solver": "unipc",
        "guide_scale": 5.0,
    }


def test_upstream_receives_one_string_and_no_custom_bias_inputs():
    video = object()
    generator = FakeGenerator(video)
    task = _task("upstream")

    assert run_generator(generator, task, _settings(frame_num=81)) == (
        video,
        {"execution_route": "upstream"},
    )
    args, kwargs = generator.calls[0]
    assert args == ("one rendered joint sentence", task["img"])
    assert kwargs["n_prompt"] == task["negative_prompt"]
    assert kwargs["seed"] == task["diffusion_seed"]
    assert "bias_kwargs" not in kwargs


def test_upstream_rejects_frame_counts_incompatible_with_pinned_implementation():
    with pytest.raises(ValueError, match="requires frame_num=81"):
        run_generator(FakeGenerator(object()), _task("upstream"), _settings())


def test_custom_none_keeps_list_api_and_uses_the_same_negative_prompt():
    expected = (object(), {"simil_masks": object()})
    generator = FakeGenerator(expected)
    task = _task("none")
    bias_kwargs = {"bias_method": "none"}

    assert run_generator(generator, task, _settings(), bias_kwargs=bias_kwargs) == expected
    args, kwargs = generator.calls[0]
    assert args == (task["prompt_sentences"], task["img"], bias_kwargs)
    assert kwargs["n_prompt"] == task["negative_prompt"]
    assert kwargs["seed"] == task["diffusion_seed"]


def test_non_output_rank_none_is_preserved_and_worker_control_flow_skips_save():
    task = _task("none")
    output = run_generator(
        FakeGenerator(None, rank=1),
        task,
        _settings(),
        bias_kwargs={"bias_method": "none"},
    )
    assert output is None
    assert rank_zero_output(output, 1) is None
    with pytest.raises(RuntimeError, match="rank zero generator"):
        rank_zero_output(None, 0)
    with pytest.raises(TypeError, match="output rank"):
        run_generator(
            FakeGenerator(None, rank=0),
            task,
            _settings(),
            bias_kwargs={"bias_method": "none"},
        )


def test_upstream_fails_closed_on_split_or_missing_negative_prompt():
    task = _task("upstream")
    task["prompt_representation"] = "split"
    with pytest.raises(ValueError, match="requires prompt_representation='joint'"):
        generation_route(task)

    task = _task("upstream")
    task["negative_prompt"] = None
    with pytest.raises(ValueError, match="explicit nonempty negative_prompt"):
        generation_route(task)


def test_legacy_custom_task_keeps_implicit_negative_prompt_behavior():
    task = _task("none")
    task.pop("negative_prompt")
    task.pop("prompt_representation")
    generator = FakeGenerator((object(), {}))

    run_generator(generator, task, _settings(), bias_kwargs={"bias_method": "none"})
    _, kwargs = generator.calls[0]
    assert "n_prompt" not in kwargs


def test_route_rejects_unknown_worker_method():
    task = copy.deepcopy(_task("unknown"))
    with pytest.raises(ValueError, match="unsupported worker generation method"):
        generation_route(task)


@pytest.mark.parametrize("offload_model", [False, True])
def test_both_routes_receive_the_same_protocol_bound_offload(offload_model):
    upstream = FakeGenerator(object())
    custom = FakeGenerator((object(), {}))
    run_generator(upstream, _task("upstream"), _settings(frame_num=81), offload_model=offload_model)
    run_generator(custom, _task("none"), _settings(frame_num=81),
                  bias_kwargs={"bias_method": "none"}, offload_model=offload_model)
    upstream_kwargs, custom_kwargs = upstream.calls[0][1], custom.calls[0][1]
    assert upstream_kwargs["offload_model"] is offload_model
    assert upstream_kwargs == custom_kwargs


def test_offload_defaults_false_and_rejects_non_bool():
    generator = FakeGenerator(object())
    run_generator(generator, _task("upstream"), _settings(frame_num=81))
    assert generator.calls[0][1]["offload_model"] is False
    with pytest.raises(TypeError):
        run_generator(FakeGenerator(object()), _task("upstream"), _settings(frame_num=81),
                      offload_model="true")
