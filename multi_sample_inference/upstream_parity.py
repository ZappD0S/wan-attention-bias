"""Concrete one-layer upstream/custom-none Wan comparison harness.

The boundary selects blocks from genuine checkpoint-loaded Wan model objects,
checks their architecture and runtime facts, and executes exact cloned inputs.
Checkpoint/revision labels remain caller supplied, so records are explicitly
unverified and cannot represent a verified checkpoint-parity claim.
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

from .parity_contracts import (
    UNVERIFIED_PROVENANCE_STATUS,
    build_parity_record,
    compare_outputs,
    require_identical_state,
    tensor_identity,
    validate_runtime_provenance,
    validate_tolerance,
)

SHARED_INPUT_KEYS = ("x", "e", "seq_lens", "grid_sizes", "freqs", "context")
ARCHITECTURE_FIELDS = (
    "dim",
    "ffn_dim",
    "num_heads",
    "window_size",
    "qk_norm",
    "cross_attn_norm",
    "eps",
)
_EXPECTED_MODEL_TYPES = {
    "upstream": ("wan.modules.model", "WanModel"),
    "custom": ("wan.modules.custom_model", "CustomWanModel"),
}
_EXPECTED_BLOCK_TYPES = {
    "upstream": ("wan.modules.model", "WanAttentionBlock"),
    "custom": ("wan.modules.custom_model", "CustomWanAttentionBlock"),
}


def _clone(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "clone"):
        return value.clone()
    if hasattr(value, "copy"):
        return value.copy()
    raise TypeError("layer parity inputs must support cloning")


def _require_declared_type(value, route, expected):
    actual = (type(value).__module__, type(value).__name__)
    if actual != expected:
        raise TypeError(
            f"{route} object must be concrete {expected[0]}.{expected[1]}, got "
            f"{actual[0]}.{actual[1]}"
        )


def _validate_models(upstream_model, custom_model):
    _require_declared_type(upstream_model, "upstream", _EXPECTED_MODEL_TYPES["upstream"])
    _require_declared_type(custom_model, "custom", _EXPECTED_MODEL_TYPES["custom"])

    # Import lazily so dependency-light contract tests do not import the Wan stack.
    from wan.modules.custom_model import CustomWanModel  # noqa: PLC0415
    from wan.modules.model import WanModel  # noqa: PLC0415

    if type(upstream_model) is not WanModel:
        raise TypeError("upstream model type identity does not match the loaded Wan module")
    if type(custom_model) is not CustomWanModel:
        raise TypeError("custom model type identity does not match the loaded Wan module")


def _validate_blocks(upstream_layer, custom_layer):
    _require_declared_type(upstream_layer, "upstream", _EXPECTED_BLOCK_TYPES["upstream"])
    _require_declared_type(custom_layer, "custom", _EXPECTED_BLOCK_TYPES["custom"])

    from wan.modules.custom_model import CustomWanAttentionBlock  # noqa: PLC0415
    from wan.modules.model import WanAttentionBlock  # noqa: PLC0415

    if type(upstream_layer) is not WanAttentionBlock:
        raise TypeError("reference layer must be an unwrapped concrete WanAttentionBlock")
    if type(custom_layer) is not CustomWanAttentionBlock:
        raise TypeError("candidate layer must be an unwrapped concrete CustomWanAttentionBlock")


def _normalized_cross_attention_type(layer):
    name = type(layer.cross_attn).__name__.removeprefix("Custom")
    if name == "WanI2VCrossAttention":
        return "i2v_cross_attn"
    if name == "WanT2VCrossAttention":
        return "t2v_cross_attn"
    raise TypeError(f"unsupported Wan cross-attention implementation: {name}")


def _architecture_settings(layer):
    missing = [field for field in ARCHITECTURE_FIELDS if not hasattr(layer, field)]
    if missing:
        raise ValueError(f"Wan block architecture fields are missing: {missing}")
    settings = {field: getattr(layer, field) for field in ARCHITECTURE_FIELDS}
    settings["window_size"] = list(settings["window_size"])
    settings["cross_attention_type"] = _normalized_cross_attention_type(layer)
    return settings


def require_matching_architecture(upstream_layer, custom_layer):
    """Return architecture evidence or fail before state comparison/execution."""
    upstream = _architecture_settings(upstream_layer)
    custom = _architecture_settings(custom_layer)
    if upstream != custom:
        mismatched = sorted(key for key in upstream if upstream[key] != custom[key])
        raise ValueError(f"upstream and custom layer architecture differs at {mismatched}")
    return {"matched_settings": upstream}


def _validate_model_and_block_settings(model, block, route):
    for field in ARCHITECTURE_FIELDS:
        model_value = getattr(model, field, None)
        block_value = getattr(block, field, None)
        if model_value != block_value:
            raise ValueError(f"{route} model and selected block differ at {field}")


def _implementation_source(value):
    source = inspect.getsourcefile(type(value))
    if source is None:
        raise ValueError("Wan implementation source file cannot be identified")
    return {
        "module": type(value).__module__,
        "sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
    }


def _validate_actual_environment(inputs, environment):
    x = inputs["x"]
    device = getattr(x, "device", None)
    dtype = getattr(x, "dtype", None)
    if device is None or dtype is None:
        raise TypeError("production parity input x must be a torch tensor")
    actual_device = str(device)
    actual_dtype = str(dtype)
    if not actual_device.startswith("cuda"):
        raise ValueError("production one-layer comparison requires CUDA inputs")

    import torch  # noqa: PLC0415

    actual = {
        "device": actual_device,
        "dtype": actual_dtype,
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
    }
    for key, value in actual.items():
        if environment.get(key) != value:
            raise ValueError(
                f"runtime environment {key} conflicts with execution: "
                f"declared {environment.get(key)!r}, actual {value!r}"
            )


def run_one_layer_parity(
    upstream_model,
    custom_model,
    *,
    layer_index,
    inputs,
    simil_masks,
    custom_bias_kwargs,
    layer_provenance,
    environment,
    atol,
    rtol,
):
    """Run a concrete one-layer comparison without claiming checkpoint parity.

    The models must already be loaded. This function verifies concrete Wan type
    identity, selects the declared index itself, and records source/runtime facts.
    It cannot prove that caller-supplied checkpoint/revision labels identify the
    objects, so the resulting record is deliberately marked unverified.
    """
    _validate_models(upstream_model, custom_model)
    if type(layer_index) is not int or layer_index < 0:
        raise ValueError("layer_index must be a nonnegative integer")
    if layer_index >= len(upstream_model.blocks) or layer_index >= len(custom_model.blocks):
        raise ValueError("layer_index is outside one or both Wan models")
    if layer_provenance.get("layer_index") != layer_index:
        raise ValueError("declared layer_index does not match the selected block index")

    upstream_layer = upstream_model.blocks[layer_index]
    custom_layer = custom_model.blocks[layer_index]
    _validate_blocks(upstream_layer, custom_layer)
    _validate_model_and_block_settings(upstream_model, upstream_layer, "upstream")
    _validate_model_and_block_settings(custom_model, custom_layer, "custom")
    architecture = require_matching_architecture(upstream_layer, custom_layer)
    actual_layer_type = architecture["matched_settings"]["cross_attention_type"]
    if layer_provenance.get("layer_type") != actual_layer_type:
        raise ValueError("declared layer_type does not match the selected Wan block")

    missing_inputs = set(SHARED_INPUT_KEYS) - set(inputs)
    if missing_inputs:
        raise ValueError(f"shared layer inputs are missing: {sorted(missing_inputs)}")
    if set(inputs) != set(SHARED_INPUT_KEYS) | {"context_lens"}:
        raise ValueError("shared layer inputs are incomplete or unexpected")
    if inputs["context_lens"] is not None:
        raise ValueError(
            "custom-none has no context_lens input; exact comparison requires context_lens=None"
        )
    if custom_bias_kwargs.get("bias_method") != "none":
        raise ValueError("comparison candidate must use bias_method='none'")
    if custom_bias_kwargs.get("bias") is not False:
        raise ValueError("comparison candidate must set bias=False")
    if custom_bias_kwargs.get("self_attention_masking") is not False:
        raise ValueError("comparison candidate must set self_attention_masking=False")

    validate_runtime_provenance(layer_provenance, environment)
    _validate_actual_environment(inputs, environment)
    tolerance = validate_tolerance(atol, rtol)
    state = require_identical_state(upstream_layer.state_dict(), custom_layer.state_dict())
    input_identities = {key: tensor_identity(inputs[key]) for key in SHARED_INPUT_KEYS}
    input_identities["context_lens"] = {"value": None}

    upstream_output = upstream_layer(
        *[_clone(inputs[key]) for key in SHARED_INPUT_KEYS],
        None,
    )
    custom_result = custom_layer(
        *[_clone(inputs[key]) for key in SHARED_INPUT_KEYS],
        _clone(simil_masks),
        dict(custom_bias_kwargs),
    )
    if not isinstance(custom_result, tuple) or len(custom_result) != 3:
        raise TypeError(
            "CustomWanAttentionBlock must return output, generated masks, and used masks"
        )
    custom_output, generated_masks, used_masks = custom_result
    if generated_masks is None or used_masks is None:
        raise ValueError("custom-none comparison requires generated and used mask evidence")

    comparison = compare_outputs(
        upstream_output,
        custom_output,
        atol=tolerance["atol"],
        rtol=tolerance["rtol"],
    )
    diagnostic_masks = {
        "input": tensor_identity(simil_masks),
        "generated": tensor_identity(generated_masks),
        "used": tensor_identity(used_masks),
        "equivalence_target": False,
    }
    return build_parity_record(
        layer=layer_provenance,
        environment=environment,
        tolerance=tolerance,
        state=state,
        architecture=architecture,
        implementation_sources={
            "upstream": _implementation_source(upstream_layer),
            "custom": _implementation_source(custom_layer),
        },
        inputs=input_identities,
        upstream_output=upstream_output,
        custom_output=custom_output,
        comparison=comparison,
        diagnostic_masks=diagnostic_masks,
        provenance_status=UNVERIFIED_PROVENANCE_STATUS,
    )
