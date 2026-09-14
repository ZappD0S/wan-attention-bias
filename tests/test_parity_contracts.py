import copy

import numpy as np
import pytest
import torch

from multi_sample_inference import parity_contracts
from multi_sample_inference.parity_contracts import (
    UNVERIFIED_PROVENANCE_STATUS,
    VERIFIED_PROVENANCE_STATUS,
    bind_verified_checkpoint_record,
    build_parity_record,
    canonical_json_bytes,
    compare_outputs,
    require_identical_state,
    tensor_identity,
    validate_runtime_provenance,
    validate_tolerance,
    write_immutable_record,
)
from multi_sample_inference.upstream_parity import (
    require_matching_architecture,
    run_one_layer_parity,
)


def _layer_provenance():
    return {
        "parent_revision": "parent-revision",
        "wan_revision": "wan-revision",
        "checkpoint_content_sha256": "a" * 64,
        "checkpoint_inventory_sha256": "d" * 64,
        "snapshot_revision": "pinned-snapshot",
        "layer_index": 3,
        "layer_type": "i2v_cross_attn",
    }


def _environment():
    return {
        "device": "cuda:0",
        "dtype": "torch.bfloat16",
        "torch_version": "test-only",
        "cuda_version": "test-only",
        "attention_kernel": "test-only-fake-kernel",
    }


def test_numerical_comparison_records_supplied_tolerance_without_selecting_one():
    upstream = np.array([0.0, 2.0], dtype=np.float32)
    custom = np.array([0.0, 2.001], dtype=np.float32)
    failed = compare_outputs(upstream, custom, atol=0.0, rtol=0.0001)
    passed = compare_outputs(upstream, custom, atol=0.0, rtol=0.001)
    assert not failed["passed"]
    assert passed["passed"]
    assert passed["finite"] and passed["shape_match"] and passed["dtype_match"]
    assert compare_outputs(upstream, upstream.copy(), atol=0, rtol=0) == {
        "shape_match": True,
        "dtype_match": True,
        "finite": True,
        "max_abs_error": 0.0,
        "max_rel_error": 0.0,
        "passed": True,
    }
    assert validate_tolerance(1e-5, 2e-5) == {"atol": 1e-5, "rtol": 2e-5}
    with pytest.raises(ValueError, match="explicitly supplied"):
        validate_tolerance(None, 0)


def test_torch_comparison_and_bfloat_identity_use_bounded_chunks(monkeypatch):
    monkeypatch.setattr(parity_contracts, "_COMPARISON_CHUNK_ELEMENTS", 2)
    upstream = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16)
    custom = upstream.clone()
    custom[-1] += torch.tensor(0.03125, dtype=torch.bfloat16)

    failed = compare_outputs(upstream, custom, atol=0.0, rtol=0.001)
    passed = compare_outputs(upstream, custom, atol=0.04, rtol=0.0)
    assert failed["finite"] and not failed["passed"]
    assert passed["passed"] and passed["max_abs_error"] == pytest.approx(0.03125)
    assert tensor_identity(upstream) == tensor_identity(upstream.clone())
    assert tensor_identity(upstream) != tensor_identity(custom)


def test_shape_dtype_and_nonfinite_mismatches_fail():
    base = np.array([1.0, 2.0], dtype=np.float32)
    assert not compare_outputs(base, base.reshape(1, 2), atol=1, rtol=1)["passed"]
    assert not compare_outputs(base, base.astype(np.float64), atol=1, rtol=1)["passed"]
    nonfinite = base.copy()
    nonfinite[0] = np.nan
    stats = compare_outputs(base, nonfinite, atol=1, rtol=1)
    assert not stats["passed"] and not stats["finite"]


def test_state_identity_fails_before_numerical_comparison():
    state = {"weight": np.array([1.0, 2.0], dtype=np.float32)}
    identity = require_identical_state(state, copy.deepcopy(state))
    assert identity["sha256"] and set(identity["tensors"]) == {"weight"}
    with pytest.raises(ValueError, match="state keys differ"):
        require_identical_state(state, {"bias": state["weight"]})
    with pytest.raises(ValueError, match="state differ at weight"):
        require_identical_state(state, {"weight": np.array([1.0, 3.0], dtype=np.float32)})


def test_runtime_provenance_requires_lowercase_hex_checkpoint_digest():
    validate_runtime_provenance(_layer_provenance(), _environment())
    incomplete = _environment()
    incomplete.pop("attention_kernel")
    with pytest.raises(ValueError, match="environment provenance is missing"):
        validate_runtime_provenance(_layer_provenance(), incomplete)
    for malformed in ("z" * 64, "A" * 64, "a" * 63):
        provenance = _layer_provenance()
        provenance["checkpoint_content_sha256"] = malformed
        with pytest.raises(ValueError, match="lowercase hexadecimal"):
            validate_runtime_provenance(provenance, _environment())


class WanI2VCrossAttention:
    pass


class CustomWanI2VCrossAttention:
    pass


class _ArchitectureBlock:
    def __init__(self, **overrides):
        values = {
            "dim": 8,
            "ffn_dim": 16,
            "num_heads": 2,
            "window_size": (-1, -1),
            "qk_norm": True,
            "cross_attn_norm": True,
            "eps": 1e-6,
        }
        values.update(overrides)
        for key, value in values.items():
            setattr(self, key, value)
        self.cross_attn = WanI2VCrossAttention()


class _CustomArchitectureBlock(_ArchitectureBlock):
    def __init__(self, **overrides):
        super().__init__(**overrides)
        self.cross_attn = CustomWanI2VCrossAttention()


def test_architecture_contract_rejects_non_state_mismatches():
    evidence = require_matching_architecture(_ArchitectureBlock(), _CustomArchitectureBlock())
    assert evidence["matched_settings"]["cross_attention_type"] == "i2v_cross_attn"
    for field, value in (("eps", 1e-5), ("window_size", (2, 2)), ("num_heads", 4)):
        with pytest.raises(ValueError, match=field):
            require_matching_architecture(
                _ArchitectureBlock(), _CustomArchitectureBlock(**{field: value})
            )


def _record_kwargs():
    upstream = np.array([[1.0, 2.0]], dtype=np.float32)
    custom = upstream.copy()
    state = require_identical_state(
        {"weight": np.array([1.0], dtype=np.float32)},
        {"weight": np.array([1.0], dtype=np.float32)},
    )
    inputs = {
        key: tensor_identity(np.array([index], dtype=np.float32))
        for index, key in enumerate(("x", "e", "seq_lens", "grid_sizes", "freqs", "context"))
    }
    inputs["context_lens"] = {"value": None}
    mask = tensor_identity(np.zeros((1, 1, 1), dtype=np.float32))
    tolerance = {"atol": 0.0, "rtol": 0.0}
    return {
        "layer": _layer_provenance(),
        "environment": _environment(),
        "tolerance": tolerance,
        "state": state,
        "architecture": {"matched_settings": {"dim": 8}},
        "implementation_sources": {
            "upstream": {"module": "wan.modules.model", "sha256": "b" * 64},
            "custom": {"module": "wan.modules.custom_model", "sha256": "c" * 64},
        },
        "inputs": inputs,
        "upstream_output": upstream,
        "custom_output": custom,
        "comparison": compare_outputs(upstream, custom, **tolerance),
        "diagnostic_masks": {
            "input": mask,
            "generated": mask,
            "used": mask,
            "equivalence_target": False,
        },
        "provenance_status": UNVERIFIED_PROVENANCE_STATUS,
    }


def test_record_builder_recomputes_evidence_and_cannot_claim_verified_parity():
    kwargs = _record_kwargs()
    record = build_parity_record(**kwargs)
    assert record["comparison"]["passed"]
    assert record["verified_checkpoint_parity"] is False
    assert record["record_kind"].startswith("unverified-")
    assert record["provenance_status"] == UNVERIFIED_PROVENANCE_STATUS
    canonical_json_bytes(record)

    stale = copy.deepcopy(kwargs)
    stale["custom_output"] = np.array([[1.0, 3.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="comparison disagrees"):
        build_parity_record(**stale)

    verified = kwargs | {"provenance_status": "verified"}
    with pytest.raises(ValueError, match="separate verified boundary"):
        build_parity_record(**verified)


def test_record_builder_rejects_incomplete_state_input_and_mask_evidence():
    kwargs = _record_kwargs()
    invalid_state = kwargs | {"state": {"sha256": "a" * 64, "tensors": {}}}
    with pytest.raises(ValueError, match="nonempty mapping"):
        build_parity_record(**invalid_state)

    inputs = copy.deepcopy(kwargs["inputs"])
    inputs.pop("freqs")
    with pytest.raises(ValueError, match="input identities"):
        build_parity_record(**(kwargs | {"inputs": inputs}))

    masks = copy.deepcopy(kwargs["diagnostic_masks"])
    masks.pop("used")
    with pytest.raises(ValueError, match="mask evidence"):
        build_parity_record(**(kwargs | {"diagnostic_masks": masks}))


def test_production_harness_rejects_name_only_fake_wan_blocks():
    class WanAttentionBlock:
        pass

    class CustomWanAttentionBlock:
        pass

    with pytest.raises(TypeError, match=r"concrete wan\.modules\.model\.WanModel"):
        run_one_layer_parity(
            WanAttentionBlock(),
            CustomWanAttentionBlock(),
            layer_index=0,
            inputs={},
            simil_masks=np.zeros((1,), dtype=np.float32),
            custom_bias_kwargs={},
            layer_provenance=_layer_provenance(),
            environment=_environment(),
            atol=0,
            rtol=0,
        )


def test_verified_binding_requires_concrete_loaders_and_matching_content(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    binding = {
        "checkpoint_path": str(checkpoint.resolve()),
        "snapshot_revision": "pinned-snapshot",
        "inventory_sha256": "d" * 64,
        "checkpoint_content_sha256": "a" * 64,
        "loaders": {
            "upstream": "wan.modules.model.WanModel.from_pretrained",
            "custom": "wan.modules.custom_model.CustomWanModel.from_pretrained",
        },
        "model_types": {
            "upstream": "wan.modules.model.WanModel",
            "custom": "wan.modules.custom_model.CustomWanModel",
        },
    }
    record = build_parity_record(**_record_kwargs())
    bound = bind_verified_checkpoint_record(record, binding)
    assert bound["verified_checkpoint_binding"] is True
    assert bound["verified_checkpoint_parity"] is True
    assert bound["provenance_status"] == VERIFIED_PROVENANCE_STATUS
    assert bound["record_kind"].startswith("checkpoint-bound-")

    wrong_content = binding | {"checkpoint_content_sha256": "e" * 64}
    with pytest.raises(ValueError, match="content identity disagrees"):
        bind_verified_checkpoint_record(record, wrong_content)
    wrong_inventory = binding | {"inventory_sha256": "e" * 64}
    with pytest.raises(ValueError, match="inventory identity disagrees"):
        bind_verified_checkpoint_record(record, wrong_inventory)
    wrong_snapshot = binding | {"snapshot_revision": "other-snapshot"}
    with pytest.raises(ValueError, match="snapshot revision disagrees"):
        bind_verified_checkpoint_record(record, wrong_snapshot)
    wrong_loader = copy.deepcopy(binding)
    wrong_loader["loaders"]["custom"] = "unverified.loader"
    with pytest.raises(ValueError, match="concrete Wan loaders"):
        bind_verified_checkpoint_record(record, wrong_loader)


def test_parity_record_write_is_immutable(tmp_path):
    path = tmp_path / "parity.json"
    record = {"schema_version": 1, "result": "CPU-test-only"}
    assert write_immutable_record(path, record)
    assert not write_immutable_record(path, record)
    with pytest.raises(FileExistsError, match="immutable parity record conflict"):
        write_immutable_record(path, record | {"result": "different"})
