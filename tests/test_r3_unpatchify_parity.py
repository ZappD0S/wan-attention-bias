"""CPU check that custom and upstream Wan unpatchify execute the same operation."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

MODULES = Path(__file__).resolve().parents[1] / "wan2.1/wan/modules"


def _unpatchify(filename, class_name):
    source = MODULES / filename
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "unpatchify")
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {"torch": torch, "math": math}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["unpatchify"]


@pytest.mark.parametrize("autocast", [False, True])
def test_custom_unpatchify_is_bitwise_upstream(autocast):
    upstream = _unpatchify("model.py", "WanModel")
    custom = _unpatchify("custom_model.py", "CustomWanModel")
    owner = SimpleNamespace(out_dim=3, patch_size=(1, 2, 2))
    grid_sizes = torch.tensor([[2, 3, 4]])
    torch.manual_seed(0)
    x = torch.randn(1, 2 * 3 * 4 + 5, 1 * 2 * 2 * 3)  # includes sequence padding
    context = (torch.autocast("cpu", dtype=torch.bfloat16) if autocast
               else torch.autocast("cpu", enabled=False))
    with context:
        [expected] = upstream(owner, x, grid_sizes)
        [actual] = custom(owner, x, grid_sizes)
    assert actual.shape == (3, 2, 6, 8)
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)
