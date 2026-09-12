import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


attention = load_module(
    "attention_contracts", "wan2.1/wan/utils/attention_contracts.py"
)
tokens = load_module("token_contracts", "wan2.1/wan/utils/token_contracts.py")
tasks = load_module("task_contracts", "multi_sample_inference/task_contracts.py")
seed_contract = load_module("seed_contract", "wan2.1/wan/utils/seed.py")


def test_self_attention_schedule_and_cfg_guards():
    base = {"bias": True, "bias_method": "regional_prompting", "self_attention_masking": True}
    assert attention.self_attention_bias_enabled(base)
    for override in (
        {"bias": False},  # timestep, block, and negative-CFG all reach this guard
        {"bias_method": "none"},
        {"self_attention_masking": False},
    ):
        assert not attention.self_attention_bias_enabled(base | override)


def test_fixed_masks_are_float_partition_and_do_not_need_tracker():
    faces = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=torch.bool)
    masks = attention.build_static_simil_masks(faces, T=2)
    assert masks.dtype == torch.float32
    assert masks.shape == (1, 3, 8)
    torch.testing.assert_close(masks.sum(dim=1), torch.ones(1, 8))
    with pytest.raises(ValueError):
        attention.build_static_simil_masks(torch.ones(2, 4, dtype=torch.bool), T=1)


@pytest.mark.parametrize("mask_type", ["fixed", "hard", "soft"])
def test_fixed_hard_soft_mask_contract_is_float_partition(mask_type):
    # The generator-specific tracking differs, but every route crosses this validator.
    masks = {
        "fixed": torch.tensor([[[1, 0], [0, 1]]], dtype=torch.bool),
        "hard": torch.tensor([[[1, 0], [0, 1]]], dtype=torch.int64),
        "soft": torch.tensor([[[0.25, 0.75], [0.75, 0.25]]]),
    }[mask_type]
    validated = attention.validate_simil_masks(masks, expected_classes=2)
    assert validated.dtype == torch.float32
    torch.testing.assert_close(validated.sum(dim=1), torch.ones(1, 2))


def test_mask_sharing_uses_and_exports_exact_effective_tensor_identity():
    generated = [torch.tensor([float(i)]) for i in range(3)]
    expected_indices = {
        "current": [0, 1, 2],
        "first": [0, 0, 0],
        "previous_generated": [0, 0, 1],
    }
    for policy, indices in expected_indices.items():
        active = None
        exported_used = []
        for block_index, new_mask in enumerate(generated):
            used = attention.select_used_simil_masks(new_mask, active)
            exported_used.append(used)
            active = attention.update_shared_simil_masks(
                policy, block_index, new_mask, active
            )
        assert all(
            exported is generated[expected]
            for exported, expected in zip(exported_used, indices, strict=True)
        )


def test_mask_sharing_aliases_are_explicit():
    assert attention.normalize_mask_sharing(None) == "current"
    assert attention.normalize_mask_sharing("block0") == "first"
    assert attention.normalize_mask_sharing("previous-generated") == "previous_generated"
    with pytest.raises(ValueError):
        attention.normalize_mask_sharing("average")


def test_regional_binary_matches_boolean_sdpa():
    torch.manual_seed(3)
    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 2, 4, 4)
    v = torch.randn(1, 2, 4, 4)
    binary = torch.tensor(
        [[[True, False, True, False], [False, True, True, False], [True, True, False, False]]]
    )
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=binary.unsqueeze(1))
    actual = attention.regional_attention(q, k, v, binary.float())
    torch.testing.assert_close(actual, expected)


def test_regional_fractional_gate_and_all_masked_rows_are_finite():
    q = torch.zeros(1, 1, 2, 1)
    k = torch.zeros(1, 1, 2, 1)
    v = torch.tensor([[[[0.0], [10.0]]]])
    eligibility = torch.tensor([[[0.25, 1.0], [0.0, 0.0]]])
    output = attention.regional_attention(q, k, v, eligibility)
    torch.testing.assert_close(output[0, 0, 0, 0], torch.tensor(8.0))
    torch.testing.assert_close(output[0, 0, 1], torch.zeros(1))
    assert torch.isfinite(output).all()


def test_ediff_fractional_eligibility_is_finite_positive_bias_not_exclusion():
    q = torch.zeros(1, 1, 1, 1)
    k = torch.zeros(1, 1, 2, 1)
    v = torch.tensor([[[[0.0], [10.0]]]])
    eligibility = torch.tensor([[[0.0, 0.5]]])
    output = attention.ediff_attention(q, k, v, eligibility, 2.0, 0.5)
    assert 5.0 < output.item() < 10.0
    assert torch.isfinite(output).all()
    padded = attention.ediff_attention(
        q,
        k,
        v,
        eligibility,
        2.0,
        0.5,
        valid_tokens=torch.tensor([True, False]),
    )
    torch.testing.assert_close(padded, torch.zeros_like(padded))


def test_subject_mapping_joint_split_and_reordered_actions():
    joint = tasks.build_subject_indices([["A action", "B action"]], 2)
    split = tasks.build_subject_indices([["A action"], ["B action"]], 2)
    assert joint == [[0, 1]]
    assert split == [[0], [1]]
    tasks.validate_method_layout("concept_weaver", split, 2)
    with pytest.raises(ValueError, match="singleton sentence"):
        tasks.validate_method_layout("concept_weaver", joint, 2)
    # BA swaps sentence contents, not persistent entity positions.
    assert tasks.build_subject_indices([["B action"], ["A action"]], 2) == [[0], [1]]
    with pytest.raises(ValueError):
        tasks.build_subject_indices([["A"], ["B", "A"]], 2)


def test_concatenated_token_offsets_include_preceding_sentence_and_eos():
    first = tokens.prefix_for_concatenated_context(torch.tensor([0, 1, 0], dtype=torch.bool), [])
    second = tokens.prefix_for_concatenated_context(
        torch.tensor([1, 0], dtype=torch.bool), [3]
    )
    assert first.tolist() == [False, True, False]
    assert second.tolist() == [False, False, False, True, False]
    assert not (first & second[:3]).any()


def test_seed_resolution_broadcasts_root_value_before_generator_use(monkeypatch):
    state = {"broadcast": False}
    monkeypatch.setattr(seed_contract.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(seed_contract.dist, "get_rank", lambda: 0)

    def broadcast(tensor, src):
        assert src == 0
        state["broadcast"] = True

    monkeypatch.setattr(seed_contract.dist, "broadcast", broadcast)
    def noise_factory(generator):
        assert state["broadcast"]
        return torch.randn((2,), generator=generator)

    resolved, generator, noise = seed_contract.initialize_diffusion_noise(
        1234, "cpu", noise_factory
    )
    assert resolved == 1234 and noise.shape == (2,)
    after_noise = generator.get_state().clone()
    _ = torch.randn((2,), generator=generator)
    assert not torch.equal(after_noise, generator.get_state())


def test_inference_setting_defaults_and_overrides():
    defaults = tasks.resolve_inference_settings()
    assert defaults["sampling_steps"] == 40 and defaults["frame_num"] == 81
    changed = tasks.resolve_inference_settings({"sampling_steps": 4, "frame_num": 9})
    assert changed["sampling_steps"] == 4 and changed["frame_num"] == 9
    with pytest.raises(ValueError):
        tasks.resolve_inference_settings({"frame_num": 8})
