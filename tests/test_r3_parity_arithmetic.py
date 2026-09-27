"""CPU-only execution of the actual custom attention methods without CUDA imports."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[1] / "wan2.1/wan/modules/custom_model.py"


def _method(class_name):
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    return compile(module, str(SOURCE), "exec")


@pytest.mark.parametrize("parity,expected", [(False, torch.bfloat16), (True, torch.float32)])
def test_custom_none_rope_dtype_matches_pristine_only_when_opted_in(parity, expected):
    observed = []

    def rope_apply(value, *_args):
        return value.float()

    def flash_attention(**kwargs):
        observed.append((kwargs["q"].dtype, kwargs["k"].dtype))
        return kwargs["q"]

    namespace = {
        "rope_apply": rope_apply,
        "flash_attention": flash_attention,
        "self_attention_bias_enabled": lambda _kwargs: False,
        "select_used_simil_masks": lambda generated, _active: generated,
        "emit_runtime_observation": lambda *_args, **_kwargs: None,
    }
    exec(_method("CustomWanSelfAttention"), namespace)
    attn = SimpleNamespace(
        num_heads=1, head_dim=2, norm_q=torch.nn.Identity(), norm_k=torch.nn.Identity(),
        q=torch.nn.Identity(), k=torch.nn.Identity(), v=torch.nn.Identity(),
        o=torch.nn.Identity(), window_size=(-1, -1),
        _generate_simil_masks=lambda *_args: torch.ones((1, 1, 2)),
    )
    out, *_ = namespace["forward"](
        attn, torch.ones((1, 2, 2), dtype=torch.bfloat16), torch.tensor([2]),
        torch.tensor([[1, 1, 2]]), None, None,
        {"_r3_pristine_parity_arithmetic": parity},
    )
    assert observed == [(expected, expected)]
    assert out.dtype == expected


@pytest.mark.parametrize("parity,expected", [(False, torch.bfloat16), (True, torch.float32)])
def test_custom_none_block_self_input_matches_pristine_only_when_opted_in(parity, expected):
    observed = []

    def self_attention(value, *_args, **_kwargs):
        observed.append(("self", value.dtype))
        return torch.zeros_like(value), torch.ones(1), torch.ones(1)

    def ffn(value):
        observed.append(("ffn", value.dtype))
        return torch.zeros_like(value)

    cpu_torch = SimpleNamespace(
        autocast=lambda *_args, **_kwargs: nullcontext(),
        float32=torch.float32,
    )
    namespace = {"torch": cpu_torch, "runtime_observation_scope": lambda **_kwargs: nullcontext()}
    exec(_method("CustomWanAttentionBlock"), namespace)
    block = SimpleNamespace(
        modulation=torch.zeros((1, 6, 2), dtype=torch.float32),
        norm1=torch.nn.Identity(), norm2=torch.nn.Identity(), norm3=torch.nn.Identity(),
        self_attn=self_attention, cross_attn=lambda value, *_args, **_kwargs: torch.zeros_like(value),
        ffn=ffn,
    )
    x = torch.ones((1, 2, 2), dtype=torch.bfloat16)
    e = torch.zeros((1, 6, 2), dtype=torch.float32)
    namespace["forward"](block, x, e, torch.tensor([2]), None, None, None, None,
                         {"_r3_pristine_parity_arithmetic": parity})
    # The FP32 modulation promotes the residual to FP32 before FFN in this fixture.
    assert observed == [("self", expected), ("ffn", torch.float32)]
