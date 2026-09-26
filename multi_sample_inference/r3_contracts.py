"""Fail-closed, dependency-light contracts for R3 runtime evidence.

This module validates protocol declarations and immutable evidence. It does not
load a checkpoint, select an attention kernel, or run generation. CPU fixtures
may exercise the schema, but only explicitly checkpoint-bound GPU runtime
records can satisfy R3 acceptance.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
from pathlib import Path

from .parity_contracts import canonical_json_bytes, tensor_identity

SCHEMA_VERSION = 1
GENUINE_RUNTIME_EVIDENCE = "genuine-checkpoint-bound-gpu-runtime"
CPU_FIXTURE_EVIDENCE = "non-scientific-cpu-fixture"
WORKER_OBSERVATION_KIND = "r3-rank-zero-worker-observation"
RUNTIME_RECORD_KIND = "r3-runtime-contract-evidence"
V1_PROTOCOL_SHA256 = "1b45951b65fc757615dfb91c9e07859084457a2dba34c18a4f9c4e3f83f3488b"
V1_MATRIX_SHA256 = "381fd9c5829fbe405e17ec77da57a5c3b14e60c325e653571b4c4d3a4043b1a5"
V2_PROTOCOL_SHA256 = "1adb1508be962d140070283a45dc092713e56b3b00029c406947d4989ee058b6"
V2_MATRIX_SHA256 = "ecad41461cd43db148af70d91a798f2342263ceee073ce92d1ae8d12b90bc643"
V3_PROTOCOL_SHA256 = "91a34240c8a9d4d13caa0f6579de523f1d2ca3cb9c387cca17734c18fd2030f5"
V4_PROTOCOL_SHA256 = "004f2f9e01a397ea36db7728a3ffe51ac86fa4b9c1c13de3c5056a2b12e266ef"
V5_PROTOCOL_SHA256 = "d3271493577916eb9f251f489933dc794328e2b0be7653cf0814821506e8e2a9"
V6_PROTOCOL_SHA256 = "f4de93f204c6d11d4b75a1d4c2c45bda44db715bd9426638c6b7e4511816abd6"
V7_PROTOCOL_SHA256 = "63e2781bdf906232014f9c39c9c5fc206ee0ed161aaf63ac80320cd4bbd4ae00"
V8_PROTOCOL_SHA256 = "f8a9c578f47957e8818f70c42d8bad7bfa8947bbbf7cf5eac1917a7e100126fa"
V9_PROTOCOL_SHA256 = "9806d626993b48771ccecdbd3fe61f4f81b296ab3222633f514c09df74771302"
V10_PROTOCOL_SHA256 = "84cc35d0c495b04812e532e02b26c9d7fa7119508e4557718e5b495bff536bfc"
V11_PROTOCOL_SHA256 = "374d5037111ec04b7e304ce07e44c261827fdf66c13e9d363b3960209f21f34b"
R3_STAGE_IDS = (
    "backend-kernel-canary",
    "checkpoint-load-hook-canary",
    "single-rank-generator-canary",
    "single-rank-contract-cases",
    "intended-rank-fsdp",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_REVISION = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")

REQUIRED_CUSTOM_AXES = {
    "method": {"none", "regional_prompting", "concept_weaver", "ediff-i"},
    "mask_configuration": {"fixed:fixed", "hard:dynamic", "soft:dynamic"},
    "mask_sharing": {"current", "first", "previous_generated"},
    "self_attention_masking": {False, True},
    "solver": {"unipc", "dpm++"},
    "attention_backend": {"flash_attention_2", "flex_attention"},
    "rank_mode": {"single", "intended_fsdp"},
}
REQUIRED_INVALID_CHECKS = {
    "invalid-batch",
    "overlapping-masks",
    "empty-mask-after-resolution",
    "unsupported-entity-mapping",
    "over-limit-tokens",
    "invalid-timestep-schedule",
    "invalid-block-schedule",
    "missing-negative-prompt",
    "rank-count-mismatch",
    "evidence-cardinality-mismatch",
    "duplicate-step-block-rank",
    "evidence-identity-mismatch",
    "fixed-mask-tracker-call",
}
REQUIRED_BASE_CHECKS = {
    "environment_binding",
    "checkpoint_binding",
    "full_generator_all_blocks",
    "conditional_cfg_branch",
    "negative_cfg_branch",
    "solver_execution",
    "attention_backend_execution",
    "initial_latent_rank_identity",
    "finiteness",
    "early_rejection",
}
REQUIRED_MASK_CHECKS = {
    "used_generated_mask_identity",
    "self_cross_used_mask_identity",
    "rank_mask_identity",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _require_sha256(value, label):
    _require(
        isinstance(value, str) and _SHA256.fullmatch(value) is not None,
        f"{label} must be a lowercase hexadecimal SHA-256 digest",
    )


def _require_git_revision(value, label):
    _require(
        isinstance(value, str) and _GIT_REVISION.fullmatch(value) is not None,
        f"{label} must be a full lowercase hexadecimal Git object ID",
    )


def _read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _axis_map(family):
    axes = family.get("axes")
    _require(isinstance(axes, list) and axes, "matrix family axes must be a nonempty list")
    result = {}
    for axis in axes:
        _require(
            isinstance(axis, dict) and set(axis) == {"name", "values"},
            "matrix axes must contain only name and values",
        )
        name, values = axis["name"], axis["values"]
        _require(isinstance(name, str) and name and name not in result, "matrix axis names must be unique")
        _require(isinstance(values, list) and values, f"matrix axis {name} values must be nonempty")
        serialized = [canonical_json_bytes(value) for value in values]
        _require(len(serialized) == len(set(serialized)), f"matrix axis {name} values must be unique")
        result[name] = values
    return result


def _case_id(family_id, selection):
    payload = {"family": family_id, "selection": selection}
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()[:20]
    return f"{family_id}-{digest}"


def enumerate_matrix_cases(matrix):
    """Expand the declared Cartesian families into stable case records."""
    cases = []
    for family in matrix["families"]:
        axes = _axis_map(family)
        names = list(axes)
        for values in itertools.product(*(axes[name] for name in names)):
            selection = dict(zip(names, values, strict=True))
            observations = list(family["required_observations"])
            for conditional in family.get("conditional_observations", []):
                if selection.get(conditional["axis"]) in conditional["values"]:
                    observations.extend(conditional["observations"])
            _require(len(observations) == len(set(observations)), "matrix case observations are duplicated")
            cases.append(
                {
                    "case_id": _case_id(family["id"], selection),
                    "family": family["id"],
                    "selection": selection,
                    "required_observations": observations,
                }
            )
    _require(
        len({case["case_id"] for case in cases}) == len(cases),
        "matrix case IDs are not unique",
    )
    return cases


def _v3_case_disposition(case):
    selection = case["selection"]
    backend = selection["attention_backend"]
    if case["family"] in {"upstream-generator", "upstream-custom-none-parity"}:
        runnable = backend == "flash_attention_2"
        reason = (
            "upstream-and-custom-none-use-unmodified-flash-path"
            if runnable
            else "route-cannot-dispatch-flex-attention"
        )
    else:
        can_dispatch_flex = (
            selection["method"] != "none"
            and selection["self_attention_masking"] is True
        )
        runnable = backend == "flash_attention_2" or can_dispatch_flex
        if backend == "flex_attention" and not can_dispatch_flex:
            reason = "method-or-self-routing-cannot-dispatch-flex-attention"
        elif backend == "flex_attention":
            reason = "requires-at-least-one-active-timestep-block-coordinate"
        elif can_dispatch_flex:
            reason = "requires-no-active-flex-coordinate"
        else:
            reason = "configuration-has-flash-only-self-route"
    return {
        "case_id": case["case_id"],
        "family": case["family"],
        "selection": case["selection"],
        "disposition": "runnable" if runnable else "rejected",
        "reason": reason,
    }


def _validate_v3_case_lineage(matrix, cases):
    lineage_cases = matrix.get("lineage_cases")
    _require(
        isinstance(lineage_cases, list) and len(lineage_cases) == len(cases),
        "R3 matrix v3 must explicitly disposition every frozen case",
    )
    expected = [_v3_case_disposition(case) for case in cases]
    _require(
        lineage_cases == expected,
        "R3 matrix v3 case lineage or dispositions differ from the frozen inventory",
    )
    counts = {
        disposition: sum(
            item["disposition"] == disposition for item in lineage_cases
        )
        for disposition in ("runnable", "rejected")
    }
    _require(counts == {"runnable": 404, "rejected": 188}, "unexpected R3 v3 disposition counts")
    return counts


def validate_matrix(matrix):
    _require(isinstance(matrix, dict), "matrix must be a JSON object")
    schema_version = matrix.get("schema_version")
    _require(schema_version in {1, 2, 3}, "unsupported R3 matrix schema_version")
    _require(
        matrix.get("matrix_id") == f"r3-gpu-contract-matrix-v{schema_version}",
        "unexpected R3 matrix_id",
    )
    if schema_version in {2, 3}:
        lineage = matrix.get("lineage")
        expected_lineage = (
            ("r3-gpu-contract-matrix-v1", V1_MATRIX_SHA256)
            if schema_version == 2
            else ("r3-gpu-contract-matrix-v2", V2_MATRIX_SHA256)
        )
        _require(
            isinstance(lineage, dict)
            and lineage.get("matrix_id") == expected_lineage[0]
            and isinstance(lineage.get("sha256"), str),
            f"R3 matrix v{schema_version} lineage is missing",
        )
        _require_sha256(lineage["sha256"], f"R3 matrix v{schema_version} lineage.sha256")
        _require(
            lineage["sha256"] == expected_lineage[1],
            f"R3 matrix v{schema_version} lineage differs from frozen v{schema_version - 1}",
        )
    families = matrix.get("families")
    _require(isinstance(families, list) and families, "matrix families must be nonempty")
    family_map = {}
    for family in families:
        _require(isinstance(family, dict), "matrix families must be objects")
        family_id = family.get("id")
        _require(isinstance(family_id, str) and family_id and family_id not in family_map, "matrix family IDs must be unique")
        observations = family.get("required_observations")
        _require(
            isinstance(observations, list)
            and len(observations) == len(set(observations))
            and set(observations) >= REQUIRED_BASE_CHECKS,
            f"matrix family {family_id} lacks required base observations",
        )
        conditionals = family.get("conditional_observations", [])
        _require(isinstance(conditionals, list), "conditional_observations must be a list")
        axes = _axis_map(family)
        for conditional in conditionals:
            _require(
                isinstance(conditional, dict)
                and set(conditional) == {"axis", "values", "observations"}
                and conditional["axis"] in axes
                and isinstance(conditional["values"], list)
                and {canonical_json_bytes(value) for value in conditional["values"]}
                <= {canonical_json_bytes(value) for value in axes[conditional["axis"]]}
                and isinstance(conditional["observations"], list),
                f"matrix family {family_id} has an invalid conditional observation",
            )
        family_map[family_id] = axes

    _require("custom-generator" in family_map, "matrix lacks custom-generator family")
    custom = family_map["custom-generator"]
    _require(set(custom) == set(REQUIRED_CUSTOM_AXES), "custom-generator axes are incomplete or unexpected")
    for name, required_values in REQUIRED_CUSTOM_AXES.items():
        _require(set(custom[name]) == required_values, f"custom-generator axis {name} is incomplete")
    custom_family = next(item for item in families if item["id"] == "custom-generator")
    _require(set(custom_family["required_observations"]) >= REQUIRED_MASK_CHECKS, "custom-generator lacks mask observations")
    _require(
        any(
            item.get("axis") == "mask_configuration"
            and "fixed:fixed" in item.get("values", [])
            and "fixed_mask_no_tracker" in item.get("observations", [])
            for item in custom_family.get("conditional_observations", [])
        ),
        "custom-generator fixed cases lack the fixed-mask tracker check",
    )

    for family_id in ("upstream-generator", "upstream-custom-none-parity"):
        _require(family_id in family_map, f"matrix lacks {family_id} family")
        axes = family_map[family_id]
        _require(
            set(axes) == {"solver", "attention_backend", "rank_mode"},
            f"{family_id} axes are incomplete or unexpected",
        )
        for name in axes:
            _require(set(axes[name]) == REQUIRED_CUSTOM_AXES[name], f"{family_id} axis {name} is incomplete")
    parity_family = next(item for item in families if item["id"] == "upstream-custom-none-parity")
    _require("full_generator_parity" in parity_family["required_observations"], "parity family lacks full-generator parity observation")

    invalid_checks = matrix.get("invalid_input_checks")
    _require(isinstance(invalid_checks, list), "invalid_input_checks must be a list")
    invalid_ids = {item.get("id") for item in invalid_checks if isinstance(item, dict)}
    _require(invalid_ids >= REQUIRED_INVALID_CHECKS, "matrix lacks required invalid-input checks")
    cases = enumerate_matrix_cases(matrix)
    summary = {"case_count": len(cases), "invalid_check_count": len(invalid_checks)}
    if schema_version == 3:
        counts = _validate_v3_case_lineage(matrix, cases)
        summary |= {
            "runnable_case_count": counts["runnable"],
            "rejected_case_count": counts["rejected"],
        }
    return summary


def _validate_v4_execution_amendment(protocol):
    amendment = protocol.get("execution_amendment")
    _require(
        isinstance(amendment, dict)
        and set(amendment)
        == {
            "amendment_id",
            "authorization_state",
            "source_binding",
            "checkpoint_binding",
            "environment_binding",
            "provisioning_binding",
            "rank_decision",
            "tolerance_decision",
            "stage_gates",
        },
        "R3 protocol v4 execution amendment is incomplete or unexpected",
    )
    _require(
        amendment["amendment_id"] == "r3-gpu-execution-amendment-v4"
        and amendment["authorization_state"]
        in {"pending-explicit-user-approval", "approved"},
        "R3 protocol v4 amendment identity or authorization is invalid",
    )

    source = amendment["source_binding"]
    _require(
        isinstance(source, dict)
        and set(source)
        == {
            "parent_revision_at_freeze",
            "parent_production_paths",
            "parent_production_content_sha256",
            "wan_revision",
            "lama_revision",
            "require_clean_worktrees",
            "manifest_repository_identity",
        }
        and source["parent_production_paths"]
        == ["multi_sample_inference", "pyproject.toml", "uv.lock"]
        and source["require_clean_worktrees"] is True
        and source["manifest_repository_identity"]
        == "exact-head-and-dirty-fingerprint",
        "R3 protocol v4 source binding is incomplete or unexpected",
    )
    for key in ("parent_revision_at_freeze", "wan_revision", "lama_revision"):
        _require_git_revision(
            source.get(key), f"R3 protocol v4 source binding {key}"
        )
    _require_sha256(
        source.get("parent_production_content_sha256"),
        "R3 protocol v4 source binding parent_production_content_sha256",
    )

    checkpoint = amendment["checkpoint_binding"]
    _require(
        isinstance(checkpoint, dict)
        and set(checkpoint)
        == {
            "identifier",
            "snapshot_revision",
            "inventory_sha256",
            "content_sha256",
            "path",
        }
        and all(isinstance(checkpoint[key], str) and checkpoint[key] for key in checkpoint),
        "R3 protocol v4 checkpoint binding is incomplete or unexpected",
    )
    for key in ("inventory_sha256", "content_sha256"):
        _require_sha256(checkpoint[key], f"R3 protocol v4 checkpoint binding {key}")

    environment = amendment["environment_binding"]
    required_environment = {
        "hostname",
        "python_version",
        "torch_version",
        "cuda_runtime_version",
        "flash_attention_version",
        "flex_attention_version",
        "sam2_version",
        "driver_version",
        "gpu_model",
        "gpu_uuids",
        "gpu_compute_capability",
        "pyproject_sha256",
        "uv_lock_sha256",
    }
    _require(
        isinstance(environment, dict)
        and set(environment) == required_environment
        and all(
            isinstance(value, str) and value
            for key, value in environment.items()
            if key != "gpu_uuids"
        )
        and isinstance(environment["gpu_uuids"], list)
        and len(environment["gpu_uuids"]) == 2
        and len(set(environment["gpu_uuids"])) == 2
        and all(isinstance(value, str) and value for value in environment["gpu_uuids"]),
        "R3 protocol v4 environment binding is incomplete or unexpected",
    )
    for key in ("pyproject_sha256", "uv_lock_sha256"):
        _require_sha256(environment[key], f"R3 protocol v4 environment binding {key}")

    hardware_payload = {
        key: environment[key]
        for key in (
            "hostname",
            "driver_version",
            "gpu_model",
            "gpu_uuids",
            "gpu_compute_capability",
        )
    }
    expected_hardware = hashlib.sha256(
        canonical_json_bytes(hardware_payload)
    ).hexdigest()
    declarations = protocol["runtime_declarations"]
    _require(
        declarations["hardware_identifier"] == expected_hardware
        and declarations["checkpoint_content_sha256"] == checkpoint["content_sha256"]
        and declarations["torch_version"] == environment["torch_version"]
        and declarations["cuda_version"] == environment["cuda_runtime_version"]
        and declarations["flash_attention_version"]
        == environment["flash_attention_version"]
        and declarations["flex_attention_version"]
        == environment["flex_attention_version"],
        "R3 protocol v4 amendment differs from its runtime declarations",
    )

    provisioning = amendment["provisioning_binding"]
    _require(
        isinstance(provisioning, dict)
        and set(provisioning)
        == {
            "cuda_toolkit_path",
            "cuda_toolkit_version",
            "nvcc_version",
            "installer_sha256",
            "flash_attention_distribution_version",
            "static_sm120_cubins",
            "inter_gpu_topology",
        }
        and all(
            isinstance(value, str) and value
            for key, value in provisioning.items()
            if key != "static_sm120_cubins"
        )
        and provisioning["static_sm120_cubins"] == ["flash_attention_2", "sam2"],
        "R3 protocol v4 provisioning binding is incomplete or unexpected",
    )
    _require_sha256(
        provisioning["installer_sha256"],
        "R3 protocol v4 provisioning binding installer_sha256",
    )

    rank = amendment["rank_decision"]
    _require(
        isinstance(rank, dict)
        and set(rank) == {"intended_fsdp_rank_count", "basis"}
        and rank["intended_fsdp_rank_count"]
        == declarations["intended_fsdp_rank_count"]
        and isinstance(rank["basis"], str)
        and rank["basis"],
        "R3 protocol v4 rank decision is incomplete or inconsistent",
    )
    tolerance = amendment["tolerance_decision"]
    declared_tolerance = protocol["numerical_tolerances"]["full_generator_parity"]
    _require(
        isinstance(tolerance, dict)
        and set(tolerance)
        == {"atol", "rtol", "basis", "u1_values_not_inherited"}
        and tolerance["atol"] == declared_tolerance["atol"]
        and tolerance["rtol"] == declared_tolerance["rtol"]
        and tolerance["u1_values_not_inherited"] is True
        and isinstance(tolerance["basis"], str)
        and tolerance["basis"],
        "R3 protocol v4 tolerance decision is incomplete or inconsistent",
    )

    stages = amendment["stage_gates"]
    expected_stages = list(R3_STAGE_IDS)
    _require(
        isinstance(stages, list)
        and [stage.get("id") for stage in stages] == expected_stages
        and all(
            isinstance(stage, dict)
            and set(stage)
            == {
                "id",
                "authorization",
                "prerequisites",
                "scope",
                "stop_on",
            }
            and stage["authorization"] == "not-approved"
            and isinstance(stage["prerequisites"], list)
            and isinstance(stage["scope"], str)
            and stage["scope"]
            and isinstance(stage["stop_on"], list)
            and stage["stop_on"]
            and all(isinstance(item, str) and item for item in stage["stop_on"])
            for stage in stages
        ),
        "R3 protocol v4 staged stop gates are incomplete or unexpected",
    )
    for index, stage in enumerate(stages):
        _require(
            stage["prerequisites"] == expected_stages[:index],
            "R3 protocol v4 staged prerequisites are not cumulative",
        )


def _validate_v5_execution_amendment(protocol):
    amendment = protocol.get("execution_amendment")
    expected_keys = {
        "amendment_id",
        "authorization_state",
        "authorization_record",
        "source_binding",
        "checkpoint_binding",
        "environment_binding",
        "provisioning_binding",
        "rank_decision",
        "tolerance_decision",
        "stage_gates",
    }
    _require(
        isinstance(amendment, dict) and set(amendment) == expected_keys,
        "R3 protocol v5 execution amendment is incomplete or unexpected",
    )
    _require(
        amendment["amendment_id"] == "r3-gpu-execution-amendment-v5"
        and amendment["authorization_state"] == "approved-bounded-stage",
        "R3 protocol v5 amendment identity or authorization is invalid",
    )
    authorization = amendment["authorization_record"]
    _require(
        authorization
        == {
            "authorized_stage": "backend-kernel-canary",
            "authorization_source": "explicit-current-session-user-approval",
            "authorized_operations": [
                "transfer-frozen-source-to-bootes",
                "exact-environment-and-source-preflight",
                "fa2-flex-sam2-cuda-canaries",
            ],
            "prohibited_operations": [
                "checkpoint-or-model-load",
                "generation",
                "distributed-execution",
                "later-stage-execution",
            ],
            "stop_after_stage": True,
        },
        "R3 protocol v5 authorization record is incomplete or overbroad",
    )
    stages = amendment["stage_gates"]
    _require(
        isinstance(stages, list)
        and [stage.get("id") for stage in stages] == list(R3_STAGE_IDS)
        and stages[0].get("authorization") == "approved"
        and all(
            stage.get("authorization") == "not-approved" for stage in stages[1:]
        ),
        "R3 protocol v5 must authorize only the backend-kernel canary",
    )

    # Reuse every structural and declaration check from v4 after erasing only
    # the v5 authorization delta. Lineage separately binds the immutable v4.
    v4_amendment = {
        key: value
        for key, value in amendment.items()
        if key != "authorization_record"
    }
    v4_amendment["amendment_id"] = "r3-gpu-execution-amendment-v4"
    v4_amendment["authorization_state"] = "pending-explicit-user-approval"
    v4_amendment["stage_gates"] = [
        stage | {"authorization": "not-approved"} for stage in stages
    ]
    v4_view = protocol | {"execution_amendment": v4_amendment}
    _validate_v4_execution_amendment(v4_view)


def _validate_v6_execution_amendment(protocol):
    amendment = protocol.get("execution_amendment")
    _require(
        isinstance(amendment, dict)
        and amendment.get("amendment_id") == "r3-gpu-execution-amendment-v6"
        and amendment.get("authorization_state") == "approved-bounded-stage",
        "R3 protocol v6 amendment identity or authorization is invalid",
    )
    _require(
        amendment.get("authorization_record")
        == {
            "authorized_stage": "backend-kernel-canary",
            "authorization_source": "explicit-current-session-user-approval",
            "authorized_operations": [
                "transfer-frozen-source-and-environment-to-pollux",
                "exact-environment-and-source-preflight",
                "fa2-flex-sam2-cuda-canaries",
            ],
            "prohibited_operations": [
                "checkpoint-or-model-load",
                "generation",
                "distributed-execution",
                "later-stage-execution",
            ],
            "stop_after_stage": True,
        },
        "R3 protocol v6 authorization record is incomplete or overbroad",
    )

    # V6 changes only the authorized host and its exact source/environment
    # bindings. Reuse the full v5/v4 structural checks after erasing that delta.
    v5_amendment = amendment | {
        "amendment_id": "r3-gpu-execution-amendment-v5",
        "authorization_record": {
            "authorized_stage": "backend-kernel-canary",
            "authorization_source": "explicit-current-session-user-approval",
            "authorized_operations": [
                "transfer-frozen-source-to-bootes",
                "exact-environment-and-source-preflight",
                "fa2-flex-sam2-cuda-canaries",
            ],
            "prohibited_operations": [
                "checkpoint-or-model-load",
                "generation",
                "distributed-execution",
                "later-stage-execution",
            ],
            "stop_after_stage": True,
        },
    }
    _validate_v5_execution_amendment(
        protocol | {"execution_amendment": v5_amendment}
    )


def _validate_v7_execution_amendment(protocol):
    amendment = protocol.get("execution_amendment")
    _require(
        isinstance(amendment, dict)
        and amendment.get("amendment_id") == "r3-gpu-execution-amendment-v7"
        and amendment.get("authorization_state") == "approved-bounded-stage",
        "R3 protocol v7 amendment identity or authorization is invalid",
    )
    _require(
        amendment.get("authorization_record")
        == {
            "authorized_stage": "checkpoint-load-hook-canary",
            "authorization_source": "explicit-current-session-user-approval",
            "authorized_operations": [
                "transfer-bound-checkpoint-to-pollux",
                "verify-exact-checkpoint-inventory-and-content",
                "exact-environment-source-and-prerequisite-preflight",
                "load-upstream-wan-dit-checkpoint-once",
                "single-layer-source-hook-neutrality-canary",
            ],
            "prohibited_operations": [
                "text-encoder-clip-vae-load",
                "custom-model-load",
                "full-model-forward-or-generation",
                "scheduler-or-decoding",
                "distributed-execution",
                "later-stage-execution",
            ],
            "stop_after_stage": True,
        },
        "R3 protocol v7 authorization record is incomplete or overbroad",
    )
    stages = amendment.get("stage_gates")
    _require(
        isinstance(stages, list)
        and [stage.get("id") for stage in stages] == list(R3_STAGE_IDS)
        and [stage.get("authorization") for stage in stages]
        == ["approved", "approved", "not-approved", "not-approved", "not-approved"],
        "R3 protocol v7 must authorize only the cumulative checkpoint/hook stage",
    )
    _require(
        amendment.get("prerequisite_evidence")
        == {
            "stage_id": "backend-kernel-canary",
            "path": "docs/r3_evidence/pollux-backend-kernel-canary.json",
            "sha256": "6d5c5cae32953baf779163233e56baabc73baacaac28c59a5c17e79152894aae",
            "protocol_sha256": V6_PROTOCOL_SHA256,
            "status": "passed",
        },
        "R3 protocol v7 prerequisite evidence is missing or unexpected",
    )
    _require(
        amendment.get("hook_canary_contract")
        == {
            "checkpoint_inventory_path": "docs/u1_checkpoint_inventory.json",
            "route": "upstream",
            "loader": "wan.modules.model.WanModel.from_pretrained",
            "model_type": "wan.modules.model.WanModel",
            "selected_layer_index": 0,
            "selected_layer_type": "wan.modules.model.WanAttentionBlock",
            "torch_dtype": "bfloat16",
            "expected_selected_layer_state_sha256": "eb1b247ba29d7974605b3f8c64c7ab82de411c74be6c4b083768250aa806c8c0",
            "synthetic_input": {
                "seed": 2026091701,
                "sequence_length": 128,
                "grid_size": [2, 8, 8],
                "context_length": 769,
                "model_dim": 5120,
            },
            "expected_observer_events": [
                "observer-installed",
                "attention-dispatch",
                "attention-dispatch",
                "attention-dispatch",
                "observer-completed",
            ],
            "expected_attention_sites": ["self", "cross", "cross"],
            "expected_attention_backend": "flash_attention_2",
            "neutrality": "bitwise-exact-output-input-and-layer-state",
        },
        "R3 protocol v7 hook canary contract is missing or unexpected",
    )

    # V7 changes only the bounded stage authorization and adds its exact
    # prerequisite/canary contract. Reuse every v6/v5/v4 declaration check.
    v6_amendment = {
        key: value
        for key, value in amendment.items()
        if key not in {"prerequisite_evidence", "hook_canary_contract"}
    }
    v6_amendment |= {
        "amendment_id": "r3-gpu-execution-amendment-v6",
        "authorization_record": {
            "authorized_stage": "backend-kernel-canary",
            "authorization_source": "explicit-current-session-user-approval",
            "authorized_operations": [
                "transfer-frozen-source-and-environment-to-pollux",
                "exact-environment-and-source-preflight",
                "fa2-flex-sam2-cuda-canaries",
            ],
            "prohibited_operations": [
                "checkpoint-or-model-load",
                "generation",
                "distributed-execution",
                "later-stage-execution",
            ],
            "stop_after_stage": True,
        },
        "stage_gates": [
            stage | {"authorization": "approved" if index == 0 else "not-approved"}
            for index, stage in enumerate(stages)
        ],
    }
    _validate_v6_execution_amendment(
        protocol | {"execution_amendment": v6_amendment}
    )


def _validate_v8_execution_amendment(protocol):
    amendment = protocol.get("execution_amendment")
    expected_keys = {
        "amendment_id",
        "authorization_state",
        "source_binding",
        "checkpoint_binding",
        "environment_binding",
        "provisioning_binding",
        "rank_decision",
        "tolerance_decision",
        "stage_gates",
        "prerequisite_evidence",
        "hook_canary_contract",
        "checkpoint_hook_evidence",
        "upstream_provenance_binding",
    }
    _require(
        isinstance(amendment, dict) and set(amendment) == expected_keys,
        "R3 protocol v8 execution amendment is incomplete or unexpected",
    )
    _require(
        amendment["amendment_id"] == "r3-gpu-execution-amendment-v8"
        and amendment["authorization_state"] == "pending-explicit-user-approval",
        "R3 protocol v8 must remain execution-unapproved",
    )
    stages = amendment["stage_gates"]
    _require(
        isinstance(stages, list)
        and [stage.get("id") for stage in stages] == list(R3_STAGE_IDS)
        and [stage.get("authorization") for stage in stages]
        == ["approved", "approved", "not-approved", "not-approved", "not-approved"],
        "R3 protocol v8 must preserve completed prerequisites without authorizing a new stage",
    )
    _require(
        amendment["checkpoint_hook_evidence"]
        == {
            "stage_id": "checkpoint-load-hook-canary",
            "path": "docs/r3_evidence/pollux-checkpoint-load-hook-canary.json",
            "sha256": "23a4a8df5fca81c69060c4440b905793da92cc454dcc67638c2b0dd63bdfc26f",
            "protocol_sha256": V7_PROTOCOL_SHA256,
            "status": "passed",
        },
        "R3 protocol v8 checkpoint/hook prerequisite evidence is missing or unexpected",
    )
    _require(
        amendment["upstream_provenance_binding"]
        == {
            "record": {
                "path": "docs/r3_upstream_provenance.json",
                "sha256": "7db6e5cf983c181aed11e4631f98b625a640bd159fed485fbb7c4f7d9266ad33",
            },
            "comparison_digest_sha256": "e54a8ca9757a2ade239c38a848a83a8681327d1ff44c58d343b22410abad422f",
            "decision": "separate-pristine-route-required",
            "official_source": {
                "repository_url": "https://github.com/Wan-Video/Wan2.1.git",
                "commit": "7c81b2f27defa56c7e627a4b6717c8f2292eee58",
                "tree": "91b74dfa24e32dc86350fe2ef2b5f2f2e06f6ee9",
                "require_clean_detached_checkout": True,
            },
            "local_source": {
                "commit": "00bde1e719ccb56c66a01a1f18a70c49b278c202",
                "tree": "ee7dddb233e6acc14a89cf96951cca6536587fee",
            },
            "route_contract": {
                "upstream": "separate-pristine-checkout",
                "custom_none": "pinned-local-wan-checkout",
                "process_isolation": "separate-python-processes-no-shared-wan-modules",
                "pristine_source_mutation": "forbidden",
                "observation_adapter": "required-exact-hash-binding-not-yet-implemented",
                "implementation_state": "required-not-implemented",
            },
        },
        "R3 protocol v8 pristine-upstream provenance or route contract is unexpected",
    )

    # V8 changes the exact parent/environment bindings and freezes the required
    # pristine route, but deliberately grants no new execution. Reuse all v7
    # through v4 structural checks after removing only the v8 additions and
    # restoring the historical v7 authorization record.
    v7_amendment = {
        key: value
        for key, value in amendment.items()
        if key not in {"checkpoint_hook_evidence", "upstream_provenance_binding"}
    }
    v7_amendment |= {
        "amendment_id": "r3-gpu-execution-amendment-v7",
        "authorization_state": "approved-bounded-stage",
        "authorization_record": {
            "authorized_stage": "checkpoint-load-hook-canary",
            "authorization_source": "explicit-current-session-user-approval",
            "authorized_operations": [
                "transfer-bound-checkpoint-to-pollux",
                "verify-exact-checkpoint-inventory-and-content",
                "exact-environment-source-and-prerequisite-preflight",
                "load-upstream-wan-dit-checkpoint-once",
                "single-layer-source-hook-neutrality-canary",
            ],
            "prohibited_operations": [
                "text-encoder-clip-vae-load",
                "custom-model-load",
                "full-model-forward-or-generation",
                "scheduler-or-decoding",
                "distributed-execution",
                "later-stage-execution",
            ],
            "stop_after_stage": True,
        },
    }
    _validate_v7_execution_amendment(
        protocol | {"execution_amendment": v7_amendment}
    )


def _validate_v9_execution_amendment(protocol):
    """Validate a source-bound, unapproved Bootes amendment; never grant execution."""
    amendment = protocol.get("execution_amendment")
    v4_keys = {
        "amendment_id", "authorization_state", "source_binding",
        "checkpoint_binding", "environment_binding", "provisioning_binding",
        "rank_decision", "tolerance_decision", "stage_gates",
    }
    _require(
        isinstance(amendment, dict)
        and set(amendment) == v4_keys | {
            "upstream_provenance_binding", "route_process_bindings",
            "production_component_hashes",
        }
        and amendment["amendment_id"] == "r3-gpu-execution-amendment-v9"
        and amendment["authorization_state"] == "pending-explicit-user-approval",
        "R3 protocol v9 must be a complete, execution-unapproved amendment",
    )
    _require(
        protocol["approvals"] == {
            "gpu_execution": "required-not-approved",
            "hardware_environment": "required-not-approved",
        }
        and all(value != "implemented" for value in protocol["runtime_hooks"].values()),
        "R3 protocol v9 cannot inherit host approval or GPU-validated hooks",
    )
    _require(
        isinstance(amendment["environment_binding"], dict)
        and amendment["environment_binding"].get("hostname") == "bootes.alias",
        "R3 protocol v9 must bind the Bootes host",
    )
    _validate_v4_execution_amendment(protocol | {
        "execution_amendment": {
            key: amendment[key] for key in v4_keys
        } | {"amendment_id": "r3-gpu-execution-amendment-v4"}
    })
    # V8 remains immutable: the pristine decision cannot be relaxed in v9.
    frozen_path = Path(__file__).resolve().parents[1] / "docs/r3_protocol_v8.json"
    _require(
        frozen_path.is_file() and sha256_file(frozen_path) == V8_PROTOCOL_SHA256,
        "R3 protocol v9 frozen v8 provenance is unavailable or changed",
    )
    v8_amendment = _read_json(frozen_path)["execution_amendment"]
    frozen = v8_amendment["upstream_provenance_binding"]
    _require(
        amendment["environment_binding"]["gpu_uuids"]
        != v8_amendment["environment_binding"]["gpu_uuids"]
        and Path(amendment["checkpoint_binding"]["path"]).is_relative_to(
            "/local_scratch2/gzappavi"
        )
        and Path(amendment["provisioning_binding"]["cuda_toolkit_path"]).is_relative_to(
            "/local_scratch2/gzappavi"
        ),
        "R3 protocol v9 cannot reuse Pollux hardware or paths as Bootes evidence",
    )
    provenance = amendment["upstream_provenance_binding"]
    route_record = frozen_path.with_name("r3_pristine_route.json")
    _require(
        route_record.is_file()
        and sha256_file(route_record)
        == "9b3f825ac5ba4e74c41b6b48b177d8f4c4bbbee01809f07749d4afcabf5ffdb4"
        and isinstance(provenance, dict)
        and {key: value for key, value in provenance.items() if key != "route_contract"}
        == {key: value for key, value in frozen.items() if key != "route_contract"}
        and provenance.get("route_contract") == frozen["route_contract"] | {
            "observation_adapter": "implemented-exact-hash-binding",
            "implementation_state": "implemented-and-cpu-validated",
        },
        "R3 protocol v9 pristine provenance or CPU route record differs from frozen v8",
    )
    routes = amendment["route_process_bindings"]
    _require(
        isinstance(routes, dict) and set(routes) == {"official-pristine", "local-custom"},
        "R3 protocol v9 needs exactly two isolated process routes",
    )
    for route, binding in routes.items():
        _require(
            isinstance(binding, dict)
            and set(binding) == {"route", "wan_root", "source_files", "adapter", "checkout"}
            and binding["route"] == route
            and isinstance(binding["wan_root"], str)
            and Path(binding["wan_root"]).is_absolute()
            and isinstance(binding["source_files"], dict)
            and bool(binding["source_files"])
            and all(
                isinstance(name, str) and name and not Path(name).is_absolute()
                and ".." not in Path(name).parts
                and isinstance(digest, str) and _SHA256.fullmatch(digest)
                for name, digest in binding["source_files"].items()
            ),
            f"R3 protocol v9 {route} process binding is malformed",
        )
        _require(
            binding["checkout"] == {
                key: frozen[
                    "official_source" if route == "official-pristine" else "local_source"
                ][key]
                for key in ("commit", "tree")
            }
            and (route != "local-custom" or binding["checkout"]["commit"]
                 == amendment["source_binding"]["wan_revision"]),
            f"R3 protocol v9 {route} checkout differs from provenance",
        )
        adapter = binding["adapter"]
        _require(
            (route == "local-custom" and adapter is None)
            or (route == "official-pristine" and isinstance(adapter, dict)
                and set(adapter) == {"path", "sha256"}
                and isinstance(adapter["path"], str) and Path(adapter["path"]).is_absolute()
                and isinstance(adapter["sha256"], str)
                and _SHA256.fullmatch(adapter["sha256"])),
            f"R3 protocol v9 {route} observation adapter is malformed",
        )
    _require(
        routes["official-pristine"]["wan_root"] != routes["local-custom"]["wan_root"],
        "R3 protocol v9 routes cannot share a Wan checkout",
    )
    components = amendment["production_component_hashes"]
    _require(
        isinstance(components, dict)
        and set(components) == {
            "experiment_pipeline.py", "fsdp_worker.py", "r3_checkout_binding.py",
            "r3_route_isolation.py", "r3_pristine_adapter.py", "r3_contracts.py",
        }
        and all(isinstance(digest, str) and _SHA256.fullmatch(digest) for digest in components.values()),
        "R3 protocol v9 production component hashes are incomplete",
    )
    _require(
        routes["official-pristine"]["adapter"]["sha256"]
        == components["r3_pristine_adapter.py"],
        "R3 protocol v9 official adapter differs from its production component",
    )


def _validate_v10_execution_amendment(protocol):
    """Permit only the approved, single-GPU Bootes prerequisite delta from frozen v9."""
    frozen_path = Path(__file__).resolve().parents[1] / "docs/r3_protocol_v9.json"
    _require(
        frozen_path.is_file() and sha256_file(frozen_path) == V9_PROTOCOL_SHA256,
        "R3 protocol v10 frozen v9 lineage is unavailable or changed",
    )
    expected = _read_json(frozen_path)
    amendment = protocol.get("execution_amendment")
    _require(isinstance(amendment, dict), "R3 protocol v10 amendment is missing")
    source = amendment.get("source_binding")
    components = amendment.get("production_component_hashes")
    _require(isinstance(source, dict) and isinstance(components, dict),
             "R3 protocol v10 source binding is missing")
    revision = source.get("parent_revision_at_freeze")
    digest = source.get("parent_production_content_sha256")
    contract_digest = components.get("r3_contracts.py")
    _require_git_revision(revision, "R3 protocol v10 source revision")
    _require_sha256(digest, "R3 protocol v10 production source")
    _require_sha256(contract_digest, "R3 protocol v10 contract component")
    old = expected["execution_amendment"]
    _require(
        revision != old["source_binding"]["parent_revision_at_freeze"]
        and digest != old["source_binding"]["parent_production_content_sha256"]
        and contract_digest != old["production_component_hashes"]["r3_contracts.py"],
        "R3 protocol v10 must bind its new validated production source",
    )
    expected["schema_version"] = 10
    expected["protocol_id"] = "r3-gpu-contracts-v10"
    expected["lineage"] = {"protocol_id": "r3-gpu-contracts-v9", "sha256": V9_PROTOCOL_SHA256}
    expected["frozen_at"] = protocol.get("frozen_at")
    expected["claim_boundary"] = (
        "Only the Bootes single-GPU backend prerequisite canary is approved; "
        "no model/checkpoint load, generation, parity, distributed execution, "
        "later stage, or scientific claim is authorized."
    )
    expected["amendment_rule"] = (
        "Stop after one backend-kernel-canary attempt. Any subsequent stage "
        "requires fresh approval and a versioned host-bound amendment."
    )
    expected["approvals"] = {"gpu_execution": "approved", "hardware_environment": "approved"}
    old["amendment_id"] = "r3-gpu-execution-amendment-v10"
    old["authorization_state"] = "approved-bounded-stage"
    old["authorization_record"] = {
        "authorized_stage": "backend-kernel-canary",
        "authorization_source": "explicit-current-session-user-approval",
        "authorized_operations": [
            "verify-bootes-host-source-environment-and-checkpoint-identity",
            "fa2-flex-sam2-single-gpu-cuda-canaries",
        ],
        "prohibited_operations": [
            "checkpoint-or-model-load", "generation", "distributed-execution",
            "later-stage-execution",
        ],
        "command": (
            "CUDA_VISIBLE_DEVICES=GPU-c247e0e3-654a-7387-8ec6-46791821a52d "
            "uv run --no-sync --locked python -m multi_sample_inference.r3_backend_canary "
            "--protocol docs/r3_protocol_v10.json --matrix docs/r3_test_matrix_v3.json "
            "--output /local_scratch2/gzappavi/r3_stage1/bootes-backend-kernel-canary.json"
        ),
        "checkpoint_inventory": "docs/u1_checkpoint_inventory.json",
        "output": "/local_scratch2/gzappavi/r3_stage1/bootes-backend-kernel-canary.json",
        "gpu_uuid": "GPU-c247e0e3-654a-7387-8ec6-46791821a52d",
        "stop_after_stage": True,
    }
    old["source_binding"]["parent_revision_at_freeze"] = revision
    old["source_binding"]["parent_production_content_sha256"] = digest
    old["production_component_hashes"]["r3_contracts.py"] = contract_digest
    old["stage_gates"][0]["authorization"] = "approved"
    _require(protocol == expected, "R3 protocol v10 exceeds its frozen bounded authorization delta")


def _validate_v11_execution_amendment(protocol):
    """Bound the one approved Bootes layer-0 checkpoint/hook attempt to frozen v10."""
    repo = Path(__file__).resolve().parents[1]
    frozen = repo / "docs/r3_protocol_v10.json"
    historical = repo / "docs/r3_protocol_v7.json"
    _require(frozen.is_file() and sha256_file(frozen) == V10_PROTOCOL_SHA256,
             "R3 v11 frozen v10 lineage is unavailable or changed")
    _require(historical.is_file() and sha256_file(historical) == V7_PROTOCOL_SHA256,
             "R3 v11 frozen hook contract is unavailable or changed")
    expected = _read_json(frozen)
    amendment = protocol.get("execution_amendment")
    _require(isinstance(amendment, dict), "R3 v11 amendment is missing")
    source = amendment.get("source_binding")
    components = amendment.get("production_component_hashes")
    _require(isinstance(source, dict) and isinstance(components, dict),
             "R3 v11 source binding is missing")
    revision = source.get("parent_revision_at_freeze")
    digest = source.get("parent_production_content_sha256")
    _require_git_revision(revision, "R3 v11 parent revision")
    _require_sha256(digest, "R3 v11 production source")
    for name in ("r3_contracts.py", "r3_checkpoint_hook_canary.py"):
        _require_sha256(components.get(name), f"R3 v11 {name} component")
    old = expected["execution_amendment"]
    _require(revision != old["source_binding"]["parent_revision_at_freeze"]
             and digest != old["source_binding"]["parent_production_content_sha256"],
             "R3 v11 must bind its new production source")
    expected["schema_version"] = 11
    expected["protocol_id"] = "r3-gpu-contracts-v11"
    expected["lineage"] = {"protocol_id": "r3-gpu-contracts-v10", "sha256": V10_PROTOCOL_SHA256}
    expected["frozen_at"] = protocol.get("frozen_at")
    expected["claim_boundary"] = (
        "Only the Bootes checkpoint-load and synthetic layer-0 hook-neutrality canary is approved; "
        "no T5/CLIP/VAE, custom model, full-model forward, generation, parity, "
        "distributed execution, later stage, or scientific claim is authorized."
    )
    expected["amendment_rule"] = (
        "Stop after one checkpoint-load-hook-canary attempt. Any subsequent stage "
        "requires fresh approval and a versioned host-bound amendment."
    )
    old["amendment_id"] = "r3-gpu-execution-amendment-v11"
    old["source_binding"]["parent_revision_at_freeze"] = revision
    old["source_binding"]["parent_production_content_sha256"] = digest
    for name in ("r3_contracts.py", "r3_checkpoint_hook_canary.py"):
        old["production_component_hashes"][name] = components[name]
    old["stage_gates"][1]["authorization"] = "approved"
    old["authorization_record"] = {
        "authorized_stage": "checkpoint-load-hook-canary",
        "authorization_source": "explicit-current-session-user-approval",
        "authorized_operations": [
            "verify-bootes-host-source-environment-and-checkpoint-identity",
            "load-official-checkpoint-local-wan-dit-and-run-one-synthetic-layer-0-hook-test",
        ],
        "prohibited_operations": [
            "text-encoder-clip-vae-load", "custom-model-load",
            "full-model-forward-or-generation", "scheduler-or-decoding",
            "distributed-execution", "later-stage-execution",
        ],
        "command": (
            "CUDA_VISIBLE_DEVICES=GPU-c247e0e3-654a-7387-8ec6-46791821a52d "
            "uv run --no-sync --locked python -m multi_sample_inference.r3_checkpoint_hook_canary "
            "--protocol docs/r3_protocol_v11.json --matrix docs/r3_test_matrix_v3.json "
            "--output /local_scratch2/gzappavi/r3_stage2/bootes-checkpoint-hook-canary.json"
        ),
        "checkpoint_inventory": "docs/u1_checkpoint_inventory.json",
        "output": "/local_scratch2/gzappavi/r3_stage2/bootes-checkpoint-hook-canary.json",
        "gpu_uuid": "GPU-c247e0e3-654a-7387-8ec6-46791821a52d",
        "stop_after_stage": True,
    }
    old["prerequisite_evidence"] = {
        "stage_id": "backend-kernel-canary",
        "path": "docs/r3_evidence/bootes-backend-kernel-canary.json",
        "sha256": "8f90996b5bb7fc5ba9f55ba4a5e20160aeab8a51009151d8691de60a18a54793",
        "protocol_sha256": V10_PROTOCOL_SHA256,
        "status": "passed",
    }
    old["hook_canary_contract"] = _read_json(historical)["execution_amendment"]["hook_canary_contract"]
    _require(protocol == expected, "R3 v11 exceeds its frozen checkpoint/hook authorization delta")


def _validate_v12_execution_amendment(protocol):
    """Only the single Bootes generator pair may advance frozen v11."""
    repo = Path(__file__).resolve().parents[1]
    frozen = repo / "docs/r3_protocol_v11.json"
    _require(frozen.is_file() and sha256_file(frozen) == V11_PROTOCOL_SHA256,
             "R3 v12 frozen v11 lineage is unavailable or changed")
    expected = _read_json(frozen)
    amendment = protocol.get("execution_amendment")
    _require(isinstance(amendment, dict), "R3 v12 amendment is missing")
    source = amendment.get("source_binding")
    components = amendment.get("production_component_hashes")
    _require(isinstance(source, dict) and isinstance(components, dict),
             "R3 v12 source binding is missing")
    revision = source.get("parent_revision_at_freeze")
    digest = source.get("parent_production_content_sha256")
    _require_git_revision(revision, "R3 v12 parent revision")
    _require_sha256(digest, "R3 v12 production source")
    _require(set(components) == set(expected["execution_amendment"]["production_component_hashes"])
             | {"r3_generator_pair.py"}, "R3 v12 production components differ")
    for name in ("r3_contracts.py", "experiment_pipeline.py", "r3_generator_pair.py"):
        _require_sha256(components.get(name), f"R3 v12 {name} component")
    old = expected["execution_amendment"]
    _require(revision != old["source_binding"]["parent_revision_at_freeze"]
             and digest != old["source_binding"]["parent_production_content_sha256"],
             "R3 v12 must bind its new production source")
    expected["schema_version"] = 12
    expected["protocol_id"] = "r3-gpu-contracts-v12"
    expected["lineage"] = {"protocol_id": "r3-gpu-contracts-v11", "sha256": V11_PROTOCOL_SHA256}
    expected["frozen_at"] = protocol.get("frozen_at")
    expected["claim_boundary"] = (
        "Only one Bootes single-rank official-pristine/custom-none generator pair is approved; "
        "stop after the pair attempt, with no repeat, other jobs, distributed execution, "
        "scientific claim, or later stage."
    )
    expected["amendment_rule"] = (
        "Stop after one single-rank-generator-canary pair attempt, including partial failure. "
        "Any subsequent stage or retry requires fresh approval and a versioned amendment."
    )
    old["amendment_id"] = "r3-gpu-execution-amendment-v12"
    old["source_binding"]["parent_revision_at_freeze"] = revision
    old["source_binding"]["parent_production_content_sha256"] = digest
    old["production_component_hashes"] = components
    old["stage_gates"][2]["authorization"] = "approved"
    authorization = amendment.get("authorization_record")
    _require(isinstance(authorization, dict), "R3 v12 authorization is missing")
    source_path = authorization.get("source")
    output = authorization.get("output")
    gpu = old["environment_binding"]["gpu_uuids"]
    _require(isinstance(source_path, dict) and set(source_path) == {"path", "sha256"}
             and isinstance(source_path["path"], str)
             and re.fullmatch(r"/[A-Za-z0-9_./-]+", source_path["path"])
             and ".." not in Path(source_path["path"]).parts,
             "R3 v12 source path is invalid")
    _require_sha256(source_path["sha256"], "R3 v12 source input")
    _require(isinstance(output, str) and re.fullmatch(r"/[A-Za-z0-9_./-]+", output)
             and ".." not in Path(output).parts
             and Path(output).is_relative_to("/local_scratch2/gzappavi")
             and not Path(output).is_relative_to(repo), "R3 v12 external output is invalid")
    _require(isinstance(gpu, list) and len(gpu) == 2 and authorization.get("gpu_uuid") in gpu,
             "R3 v12 selected GPU differs from Bootes binding")
    _require(type(authorization.get("min_free_gpu_bytes")) is int
             and authorization["min_free_gpu_bytes"] >= 70 * 1024**3,
             "R3 v12 GPU free-memory floor must be at least 70 GiB")
    timeout = authorization.get("job_timeout_seconds")
    _require(type(timeout) is int and 600 <= timeout <= 3600,
             "R3 v12 per-job timeout must be 600-3600 seconds")
    assets = authorization.get("assets")
    expected_assets = {
        "reference.png": "3f24239d8b0eef18cb022f46ef81b9a0bbcbe76cab9bffe3bf018b94be6ca17b",
        "mask-left.png": "ec534110d2c7c022847e2e9edcd1bb2c3bdeffa807cd43e336382dfce572b7bd",
        "mask-right.png": "f7c53ec340a68e501182bb2987750bfaecb7dfc9d57536edbf36a6c629820b56",
    }
    _require(isinstance(assets, dict) and set(assets) == set(expected_assets),
             "R3 v12 three synthetic asset bindings are missing or unexpected")
    for name, digest in expected_assets.items():
        asset = assets[name]
        _require(isinstance(asset, dict) and set(asset) == {"path", "sha256"}
                 and isinstance(asset["path"], str)
                 and re.fullmatch(r"/[A-Za-z0-9_./-]+", asset["path"])
                 and ".." not in Path(asset["path"]).parts
                 and Path(asset["path"]).name == name
                 and asset["sha256"] == digest,
                 f"R3 v12 {name} asset binding is invalid")
    _require(len({value["path"] for value in assets.values()}) == 3,
             "R3 v12 assets must have distinct paths")
    old["authorization_record"] = {
        "authorized_stage": "single-rank-generator-canary",
        "authorization_source": "explicit-current-session-user-approval",
        "authorized_operations": ["verify-bootes-host-source-environment-checkpoint-and-prerequisites",
                                  "two-sequential-isolated-single-rank-generator-jobs-and-offline-parity"],
        "prohibited_operations": ["repeat-attempt", "additional-jobs", "distributed-execution",
                                  "later-stage-execution", "scientific-claim"],
        "command": (
            f"CUDA_VISIBLE_DEVICES={authorization['gpu_uuid']} uv run --no-sync --locked "
            "python -m multi_sample_inference.r3_generator_pair "
            "--protocol docs/r3_protocol_v12.json --matrix docs/r3_test_matrix_v3.json "
            f"--source {source_path['path']} --output {output}"
        ),
        "checkpoint_inventory": "docs/u1_checkpoint_inventory.json",
        "source": source_path,
        "output": output,
        "gpu_uuid": authorization["gpu_uuid"],
        "min_free_gpu_bytes": authorization["min_free_gpu_bytes"],
        "job_timeout_seconds": timeout,
        "assets": assets,
        "stop_after_stage": True,
    }
    old["prerequisite_evidence"] = [
        expected["execution_amendment"]["prerequisite_evidence"],
        {
            "stage_id": "checkpoint-load-hook-canary",
            "path": "docs/r3_evidence/bootes-checkpoint-hook-canary.json",
            "sha256": "84c0a95647d0923af7d8eb5572a2a0b5143a986d06be295a7cee13aff9c0adc7",
            "protocol_sha256": V11_PROTOCOL_SHA256,
            "status": "passed",
        },
    ]
    _require(protocol == expected, "R3 v12 exceeds its single-pair authorization delta")


def _validate_protocol_lineage(protocol, schema_version):
    if schema_version == 1:
        return
    lineage = protocol.get("lineage")
    expected_lineage = {
        2: ("r3-gpu-contracts-v1", V1_PROTOCOL_SHA256),
        3: ("r3-gpu-contracts-v2", V2_PROTOCOL_SHA256),
        4: ("r3-gpu-contracts-v3", V3_PROTOCOL_SHA256),
        5: ("r3-gpu-contracts-v4", V4_PROTOCOL_SHA256),
        6: ("r3-gpu-contracts-v5", V5_PROTOCOL_SHA256),
        7: ("r3-gpu-contracts-v6", V6_PROTOCOL_SHA256),
        8: ("r3-gpu-contracts-v7", V7_PROTOCOL_SHA256),
        9: ("r3-gpu-contracts-v8", V8_PROTOCOL_SHA256),
        10: ("r3-gpu-contracts-v9", V9_PROTOCOL_SHA256),
        11: ("r3-gpu-contracts-v10", V10_PROTOCOL_SHA256),
        12: ("r3-gpu-contracts-v11", V11_PROTOCOL_SHA256),
    }[schema_version]
    _require(
        isinstance(lineage, dict)
        and lineage.get("protocol_id") == expected_lineage[0],
        f"R3 protocol v{schema_version} lineage is missing",
    )
    _require_sha256(
        lineage.get("sha256"), f"R3 protocol v{schema_version} lineage.sha256"
    )
    _require(
        lineage["sha256"] == expected_lineage[1],
        f"R3 protocol v{schema_version} lineage differs from frozen v{schema_version - 1}",
    )


def validate_protocol(protocol, matrix, *, matrix_sha256):
    _require(isinstance(protocol, dict), "protocol must be a JSON object")
    schema_version = protocol.get("schema_version")
    _require(type(schema_version) is int and schema_version in range(1, 13), "unsupported R3 protocol schema_version")
    _require(
        protocol.get("protocol_id") == f"r3-gpu-contracts-v{schema_version}",
        "unexpected R3 protocol_id",
    )
    _validate_protocol_lineage(protocol, schema_version)
    _require(protocol.get("status") == "in-progress", "R3 protocol must remain in-progress")
    matrix_ref = protocol.get("matrix")
    _require(isinstance(matrix_ref, dict), "protocol matrix reference is missing")
    _require(matrix_ref.get("matrix_id") == matrix.get("matrix_id"), "protocol matrix_id does not match matrix")
    _require_sha256(matrix_ref.get("sha256"), "protocol matrix.sha256")
    _require(matrix_ref["sha256"] == matrix_sha256, "protocol matrix hash does not match matrix file")

    approvals = protocol.get("approvals")
    _require(isinstance(approvals, dict), "protocol approvals are missing")
    for key in ("gpu_execution", "hardware_environment"):
        _require(approvals.get(key) in {"required-not-approved", "approved"}, f"invalid {key} approval state")

    hooks = protocol.get("runtime_hooks")
    required_hooks = {
        "initial_latent_all_rank",
        "all_rank_masks_and_self_cross",
        "cfg_branch_outputs",
        "actual_attention_backend",
        "fixed_tracker_call_count",
        "full_generator_parity",
    }
    _require(isinstance(hooks, dict) and set(hooks) == required_hooks, "runtime hook declarations are incomplete or unexpected")
    allowed_hook_states = {"required-not-implemented", "implemented"}
    if schema_version >= 2:
        allowed_hook_states.add("implemented-source-hook-gpu-unvalidated")
    _require(
        all(value in allowed_hook_states for value in hooks.values()),
        "runtime hook declarations have invalid states",
    )

    declarations = protocol.get("runtime_declarations")
    required_declarations = {
        "checkpoint_content_sha256",
        "hardware_identifier",
        "torch_version",
        "cuda_version",
        "flash_attention_version",
        "flex_attention_version",
        "intended_fsdp_rank_count",
    }
    _require(
        isinstance(declarations, dict) and set(declarations) == required_declarations,
        "runtime declarations are incomplete or unexpected",
    )
    for key in required_declarations - {"intended_fsdp_rank_count"}:
        value = declarations[key]
        _require(
            value is None or (isinstance(value, str) and value),
            f"runtime declaration {key} must be null or a nonempty string",
        )
    checkpoint_sha256 = declarations["checkpoint_content_sha256"]
    if checkpoint_sha256 is not None:
        _require_sha256(
            checkpoint_sha256, "runtime declaration checkpoint_content_sha256"
        )
    rank_count = declarations["intended_fsdp_rank_count"]
    _require(rank_count is None or (type(rank_count) is int and rank_count > 1), "intended FSDP rank count must be null or an integer greater than one")
    tolerance = protocol.get("numerical_tolerances", {}).get("full_generator_parity")
    _require(
        isinstance(tolerance, dict)
        and set(tolerance) == {"atol", "rtol", "amendment_required"}
        and type(tolerance["amendment_required"]) is bool
        and (schema_version >= 2 or tolerance["amendment_required"] is True),
        "full-generator parity tolerance declaration is incomplete",
    )
    for key in ("atol", "rtol"):
        value = tolerance[key]
        _require(value is None or (not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0), f"full-generator {key} must be null or finite and nonnegative")

    exact = protocol.get("exact_contracts")
    _require(
        isinstance(exact, dict)
        and exact.get("rank_initial_latent_hashes") == "exact-equality"
        and exact.get("mask_identity_hashes") == "exact-equality"
        and exact.get("finiteness") == "all-values-finite"
        and exact.get("cardinality") == "exact",
        "CPU-independent exact contracts are not frozen",
    )
    if schema_version >= 3:
        _require(
            protocol.get("backend_contract")
            == {
                "request_semantics": "expected-self-attention-coverage",
                "flash_attention_3": "unsupported-fail-before-model-load",
                "flash_attention_2": "required-for-all-routes",
                "flex_attention": "declared-and-available-dispatched-only-when-derived",
                "dispatch_derivation": "trusted-route-method-self-routing-and-schedules",
            },
            "R3 protocol v3+ backend contract is missing or unexpected",
        )
    if schema_version >= 4:
        {
            4: _validate_v4_execution_amendment,
            5: _validate_v5_execution_amendment,
            6: _validate_v6_execution_amendment,
            7: _validate_v7_execution_amendment,
            8: _validate_v8_execution_amendment,
            9: _validate_v9_execution_amendment,
            10: _validate_v10_execution_amendment,
            11: _validate_v11_execution_amendment,
            12: _validate_v12_execution_amendment,
        }[schema_version](protocol)
    return validate_matrix(matrix)


def load_protocol_bundle(protocol_path, matrix_path):
    protocol_path, matrix_path = Path(protocol_path).resolve(), Path(matrix_path).resolve()
    protocol, matrix = _read_json(protocol_path), _read_json(matrix_path)
    matrix_sha256 = sha256_file(matrix_path)
    summary = validate_protocol(protocol, matrix, matrix_sha256=matrix_sha256)
    return {
        "protocol": protocol,
        "matrix": matrix,
        "protocol_path": protocol_path,
        "matrix_path": matrix_path,
        "protocol_sha256": sha256_file(protocol_path),
        "matrix_sha256": matrix_sha256,
        "summary": summary,
    }


def _staged_authorization_blockers(protocol):
    if protocol["schema_version"] < 4:
        return []
    amendment = protocol["execution_amendment"]
    blockers = []
    if amendment["authorization_state"] not in {"approved", "approved-bounded-stage"}:
        blockers.append("explicit-user-authorization")
    if any(
        stage["authorization"] != "approved"
        for stage in amendment["stage_gates"]
    ):
        blockers.append("staged-execution-gates")
    return blockers


def _base_execution_blockers(protocol):
    blockers = []
    for key, value in protocol["approvals"].items():
        if value != "approved":
            blockers.append(f"approval:{key}")
    for key, value in protocol["runtime_declarations"].items():
        if value is None:
            blockers.append(f"runtime-declaration:{key}")
    executable_hook_states = {"implemented"}
    if protocol["schema_version"] >= 2:
        executable_hook_states.add("implemented-source-hook-gpu-unvalidated")
    for key, value in protocol["runtime_hooks"].items():
        if value not in executable_hook_states:
            blockers.append(f"runtime-hook:{key}")
    tolerance = protocol["numerical_tolerances"]["full_generator_parity"]
    for key in ("atol", "rtol"):
        if tolerance[key] is None:
            blockers.append(f"tolerance:full_generator_parity.{key}")
    if tolerance["amendment_required"]:
        blockers.append("versioned-pre-execution-amendment")
    return blockers


def _pristine_route_blockers(protocol):
    if protocol["schema_version"] < 8:
        return []
    state = protocol["execution_amendment"]["upstream_provenance_binding"][
        "route_contract"
    ]["implementation_state"]
    if state != "implemented-and-cpu-validated":
        return [f"pristine-upstream-route:{state}"]
    return []


def stage_execution_blockers(protocol, stage_id):
    """Return blockers for one explicitly authorized bounded stage."""
    blockers = _base_execution_blockers(protocol)
    if protocol["schema_version"] not in {5, 6, 7, 8, 10, 11, 12}:
        return [*blockers, "stage-scoped-authorization-unavailable"]
    amendment = protocol["execution_amendment"]
    if amendment["authorization_state"] != "approved-bounded-stage":
        blockers.append("explicit-user-authorization")
    stages = {stage["id"]: stage for stage in amendment["stage_gates"]}
    if stage_id not in stages:
        blockers.append(f"unknown-stage:{stage_id}")
        return blockers
    if stages[stage_id]["authorization"] != "approved":
        blockers.append(f"stage-not-authorized:{stage_id}")
    for prerequisite in stages[stage_id]["prerequisites"]:
        if stages[prerequisite]["authorization"] != "approved":
            blockers.append(f"stage-prerequisite-not-authorized:{prerequisite}")
    if list(R3_STAGE_IDS).index(stage_id) >= 2:
        blockers.extend(_pristine_route_blockers(protocol))
    return blockers


def execution_blockers(protocol):
    """Return blockers for full approved contract-validation execution."""
    return (
        _base_execution_blockers(protocol)
        + _staged_authorization_blockers(protocol)
        + _pristine_route_blockers(protocol)
    )


def evidence_acceptance_blockers(protocol):
    """Return blockers that prevent accepting collected GPU contract evidence."""
    blockers = execution_blockers(protocol)
    for key, value in protocol["runtime_hooks"].items():
        if value != "implemented":
            blocker = f"gpu-validation:runtime-hook:{key}"
            if blocker not in blockers:
                blockers.append(blocker)
    return blockers


def matrix_case_for_request(matrix, family, selection):
    matches = [
        case
        for case in enumerate_matrix_cases(matrix)
        if case["family"] == family and case["selection"] == selection
    ]
    _require(len(matches) == 1, "requested R3 settings do not identify exactly one matrix case")
    case = matches[0]
    if matrix["schema_version"] == 3:
        disposition = next(
            item for item in matrix["lineage_cases"] if item["case_id"] == case["case_id"]
        )
        case = case | {"lineage_disposition": disposition}
    return case


def _tensor_finite(value):
    current = value.detach() if hasattr(value, "detach") else value
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        torch = None
    if torch is not None and isinstance(current, torch.Tensor):
        return bool(torch.isfinite(current).all().item())
    import numpy as np  # noqa: PLC0415

    return bool(np.isfinite(np.asarray(current)).all())


def _validate_identity(identity, label):
    _require(isinstance(identity, dict) and set(identity) == {"shape", "dtype", "sha256"}, f"{label} is not a complete tensor identity")
    _require(isinstance(identity["shape"], list) and all(type(value) is int and value >= 0 for value in identity["shape"]), f"{label}.shape is invalid")
    _require(isinstance(identity["dtype"], str) and identity["dtype"], f"{label}.dtype is invalid")
    _require_sha256(identity["sha256"], f"{label}.sha256")


def _validate_returned_mask_identity(identity, expected, label):
    _validate_identity(identity, label)
    shape = identity["shape"]
    _require(
        len(shape) == 7 and all(value > 0 for value in shape),
        f"{label} must have positive [batch, step, block, entity, time, height, width] dimensions",
    )
    _require(
        shape[:3] == [1, expected["sampling_steps"], expected["num_layers"]],
        f"{label} cardinality does not match expected batch/step/block counts",
    )


def build_worker_observation(binding, extra_data, *, observed_rank):
    """Build the honest partial observation available after custom generation.

    The current generator returns this data only on rank zero. Requested values
    remain under ``requested`` and are never promoted to observed facts.
    """
    _require(type(observed_rank) is int and observed_rank == 0, "current worker observation is rank-zero only")
    _require(isinstance(binding, dict), "R3 worker binding must be a mapping")
    required_extra = {"simil_masks", "used_simil_masks", "generated_simil_masks", "diffusion_seed"}
    _require(required_extra <= set(extra_data), f"generator extra_data is missing R3 observations: {sorted(required_extra - set(extra_data))}")
    used = tensor_identity(extra_data["used_simil_masks"])
    alias = tensor_identity(extra_data["simil_masks"])
    generated = tensor_identity(extra_data["generated_simil_masks"])
    _require(alias == used, "simil_masks compatibility alias differs from used_simil_masks")
    expected = binding["expected"]
    _validate_returned_mask_identity(used, expected, "used mask")
    _validate_returned_mask_identity(generated, expected, "generated mask")
    _require(used["shape"] == generated["shape"], "used and generated mask shapes differ")
    seed = extra_data["diffusion_seed"]
    _require(type(seed) is int and seed == binding["requested"]["diffusion_seed"], "observed diffusion seed differs from requested seed")
    return {
        "schema_version": SCHEMA_VERSION,
        "record_kind": WORKER_OBSERVATION_KIND,
        "evidence_class": GENUINE_RUNTIME_EVIDENCE,
        "claim_scope": "partial-rank-zero-post-generator-return",
        "bindings": binding["bindings"],
        "case_id": binding["case_id"],
        "expected": expected,
        "requested": binding["requested"],
        "observed": {
            "rank": observed_rank,
            "diffusion_seed": seed,
            "used_masks": used,
            "generated_masks": generated,
            "all_returned_mask_values_finite": _tensor_finite(extra_data["used_simil_masks"])
            and _tensor_finite(extra_data["generated_simil_masks"]),
        },
        "unobserved_execution_blockers": [
            "initial_latent_identity_by_rank",
            "all_rank_mask_identity",
            "conditional_and_negative_cfg_branch_outputs",
            "actual_attention_backend",
            "self_cross_consumer_mask_identity",
            "fixed_mask_tracker_call_count",
            "full_generator_upstream_custom_none_parity",
        ],
        "r3_acceptance": False,
    }


def validate_worker_observation(record, binding):
    _require(isinstance(record, dict), "R3 worker observation must be a JSON object")
    _require(record.get("schema_version") == SCHEMA_VERSION, "unsupported worker observation schema_version")
    _require(record.get("record_kind") == WORKER_OBSERVATION_KIND, "unexpected worker observation kind")
    _require(record.get("evidence_class") == GENUINE_RUNTIME_EVIDENCE, "worker observation is not genuine runtime evidence")
    for key in ("bindings", "case_id", "expected", "requested"):
        _require(record.get(key) == binding[key], f"worker observation {key} does not match manifest binding")
    observed = record.get("observed")
    _require(isinstance(observed, dict), "worker observed values are missing")
    _require(observed.get("rank") == 0, "worker observation must be rank zero")
    _require(observed.get("diffusion_seed") == binding["requested"]["diffusion_seed"], "worker observed seed does not match")
    for key in ("used_masks", "generated_masks"):
        _validate_returned_mask_identity(
            observed.get(key), binding["expected"], f"observed.{key}"
        )
    _require(observed["used_masks"]["shape"] == observed["generated_masks"]["shape"], "worker mask shapes differ")
    _require(observed.get("all_returned_mask_values_finite") is True, "worker returned nonfinite masks")
    _require(record.get("r3_acceptance") is False, "partial worker observation cannot claim R3 acceptance")
    blockers = record.get("unobserved_execution_blockers")
    _require(isinstance(blockers, list) and blockers, "partial worker observation must name unobserved blockers")
    return True


def _validate_rank_and_mask_cardinality(record, checks, case):
    expected = record.get("expected")
    _require(
        isinstance(expected, dict)
        and set(expected) == {"rank_count", "sampling_steps", "num_layers"}
        and all(type(expected[key]) is int and expected[key] > 0 for key in expected),
        "runtime expected rank/step/block cardinalities are incomplete",
    )
    rank_count = expected["rank_count"]
    rank_evidence = checks["initial_latent_rank_identity"].get("evidence")
    if checks["initial_latent_rank_identity"]["status"] == "passed":
        identities = rank_evidence.get("rank_identities") if isinstance(rank_evidence, dict) else None
        _require(isinstance(identities, list) and len(identities) == rank_count, "initial-latent rank cardinality mismatch")
        ranks = []
        hashes = []
        for item in identities:
            _require(isinstance(item, dict) and set(item) == {"rank", "identity"}, "initial-latent rank identity is malformed")
            ranks.append(item["rank"])
            _validate_identity(item["identity"], "initial_latent_rank_identity")
            hashes.append(item["identity"]["sha256"])
        _require(sorted(ranks) == list(range(rank_count)), "initial-latent ranks are duplicated or incomplete")
        _require(len(set(hashes)) == 1, "initial-latent hashes differ across ranks")

    if "used_generated_mask_identity" in checks and checks["used_generated_mask_identity"]["status"] == "passed":
        evidence = checks["used_generated_mask_identity"]["evidence"]
        records = evidence.get("records") if isinstance(evidence, dict) else None
        expected_count = rank_count * expected["sampling_steps"] * expected["num_layers"]
        _require(isinstance(records, list) and len(records) == expected_count, "mask rank/step/block cardinality mismatch")
        coordinates = set()
        per_coordinate = {}
        for item in records:
            _require(
                isinstance(item, dict)
                and set(item) == {"rank", "step", "block", "used", "generated", "self_used_sha256", "cross_used_sha256", "finite"},
                "mask observation is malformed",
            )
            coordinate = (item["rank"], item["step"], item["block"])
            _require(
                all(type(value) is int for value in coordinate)
                and 0 <= coordinate[0] < rank_count
                and 0 <= coordinate[1] < expected["sampling_steps"]
                and 0 <= coordinate[2] < expected["num_layers"],
                "mask observation coordinate is out of range",
            )
            _require(coordinate not in coordinates, "mask rank/step/block observation is duplicated")
            coordinates.add(coordinate)
            _validate_identity(item["used"], "mask.used")
            _validate_identity(item["generated"], "mask.generated")
            _require(item["self_used_sha256"] == item["used"]["sha256"] == item["cross_used_sha256"], "self/cross used-mask identity differs")
            _require(item["finite"] is True, "mask observation is nonfinite")
            per_coordinate.setdefault(coordinate[1:], []).append(item["used"]["sha256"])
        _require(all(len(set(hashes)) == 1 for hashes in per_coordinate.values()), "used-mask identity differs across ranks")

    if "fixed_mask_no_tracker" in checks and checks["fixed_mask_no_tracker"]["status"] == "passed":
        evidence = checks["fixed_mask_no_tracker"]["evidence"]
        _require(
            isinstance(evidence, dict)
            and evidence.get("tracker_call_count") == 0
            and case["selection"].get("mask_configuration") == "fixed:fixed",
            "fixed-mask tracker evidence is invalid",
        )


def _validate_passed_checks(record, checks, case):
    expected = record["expected"]
    selection = case["selection"]

    environment = checks["environment_binding"]
    if environment["status"] == "passed":
        evidence = environment["evidence"]
        required = {"hardware_identifier", "torch_version", "cuda_version", "attention_backend", "backend_version"}
        _require(isinstance(evidence, dict) and set(evidence) == required, "environment binding evidence is incomplete")
        _require(all(isinstance(evidence[key], str) and evidence[key] for key in required), "environment binding values must be observed nonempty strings")
        _require(evidence["attention_backend"] == selection["attention_backend"], "observed environment backend differs from matrix case")
        digest = hashlib.sha256(canonical_json_bytes(evidence)).hexdigest()
        _require(record["bindings"]["environment_sha256"] == digest, "environment evidence hash differs from binding")

    checkpoint = checks["checkpoint_binding"]
    if checkpoint["status"] == "passed":
        evidence = checkpoint["evidence"]
        _require(isinstance(evidence, dict) and evidence.get("content_sha256") == record["bindings"]["checkpoint_content_sha256"], "checkpoint evidence differs from binding")

    blocks = checks["full_generator_all_blocks"]
    if blocks["status"] == "passed":
        evidence = blocks["evidence"]
        _require(isinstance(evidence, dict) and evidence.get("observed_block_indices") == list(range(expected["num_layers"])), "full-generator block evidence is incomplete")

    for check_id, branch in (("conditional_cfg_branch", "conditional"), ("negative_cfg_branch", "negative")):
        check = checks[check_id]
        if check["status"] == "passed":
            evidence = check["evidence"]
            _require(isinstance(evidence, dict) and evidence.get("branch") == branch and evidence.get("executed") is True and evidence.get("finite") is True, f"{branch} CFG evidence is incomplete")
            _validate_identity(evidence.get("output"), f"{branch}_cfg.output")

    solver = checks["solver_execution"]
    if solver["status"] == "passed":
        _require(solver["evidence"] == {"observed_solver": selection["solver"]}, "observed solver differs from matrix case")
    backend = checks["attention_backend_execution"]
    if backend["status"] == "passed":
        evidence = backend["evidence"]
        _require(
            isinstance(evidence, dict)
            and evidence.get("observed_backend") == selection["attention_backend"]
            and isinstance(evidence.get("dispatch_source"), str)
            and evidence["dispatch_source"],
            "observed attention backend evidence is incomplete or mismatched",
        )
    finite = checks["finiteness"]
    if finite["status"] == "passed":
        _require(finite["evidence"] == {"all_values_finite": True}, "finiteness evidence is incomplete")
    rejection = checks["early_rejection"]
    if rejection["status"] == "passed":
        evidence = rejection["evidence"]
        _require(
            isinstance(evidence, dict)
            and set(evidence.get("passed_check_ids", [])) >= REQUIRED_INVALID_CHECKS,
            "early-rejection evidence lacks required invalid-input checks",
        )
    for check_id in ("self_cross_used_mask_identity", "rank_mask_identity"):
        if check_id in checks and checks[check_id]["status"] == "passed":
            _require(checks[check_id]["evidence"] == {"verified": True}, f"{check_id} evidence is incomplete")
    if "full_generator_parity" in checks and checks["full_generator_parity"]["status"] == "passed":
        raise ValueError(
            "full-generator parity acceptance is unavailable until paired input/output identities and numerical comparisons are implemented"
        )


def _reload_validated_bundle(bundle):
    _require(isinstance(bundle, dict), "acceptance requires a validated protocol bundle")
    for key in (
        "protocol",
        "matrix",
        "protocol_path",
        "matrix_path",
        "protocol_sha256",
        "matrix_sha256",
    ):
        _require(key in bundle, "acceptance requires a complete validated protocol bundle")
    reloaded = load_protocol_bundle(bundle["protocol_path"], bundle["matrix_path"])
    for key in ("protocol", "matrix", "protocol_sha256", "matrix_sha256"):
        _require(bundle[key] == reloaded[key], "protocol bundle is not bound to its validated files")
    return reloaded


def _trusted_runtime_case(bundle, expected_job):
    _require(
        isinstance(expected_job, dict)
        and set(expected_job) == {"bindings", "case_id", "expected"},
        "trusted expected job binding is incomplete or unexpected",
    )
    trusted_bindings = expected_job["bindings"]
    required_bindings = {
        "manifest_sha256",
        "checkpoint_content_sha256",
        "source_sha256",
        "config_sha256",
    }
    _require(
        isinstance(trusted_bindings, dict) and set(trusted_bindings) == required_bindings,
        "trusted job identity bindings are incomplete or unexpected",
    )
    for key, value in trusted_bindings.items():
        _require_sha256(value, f"trusted job bindings.{key}")
    expected = expected_job["expected"]
    _require(
        isinstance(expected, dict)
        and set(expected) == {"rank_count", "sampling_steps", "num_layers"}
        and all(type(expected[key]) is int and expected[key] > 0 for key in expected),
        "trusted job cardinalities are incomplete",
    )
    matches = [
        case
        for case in enumerate_matrix_cases(bundle["matrix"])
        if case["case_id"] == expected_job["case_id"]
    ]
    _require(len(matches) == 1, "trusted job case is not uniquely bound to the validated matrix")
    case = matches[0]
    rank_mode = case["selection"]["rank_mode"]
    if rank_mode == "single":
        _require(expected["rank_count"] == 1, "single rank mode requires exactly one expected rank")
    else:
        intended = bundle["protocol"]["runtime_declarations"]["intended_fsdp_rank_count"]
        _require(
            intended is not None and expected["rank_count"] == intended,
            "intended FSDP rank mode differs from the protocol rank declaration",
        )
    return case


def _validate_declared_runtime_bindings(record, bundle, expected_job, checks, case):
    bindings = record.get("bindings", {})
    expected_bindings = {
        "protocol_sha256": bundle["protocol_sha256"],
        "matrix_sha256": bundle["matrix_sha256"],
        **expected_job["bindings"],
    }
    _require(
        isinstance(bindings, dict)
        and set(bindings) == {*expected_bindings, "environment_sha256"},
        "runtime evidence bindings are incomplete or unexpected",
    )
    for key, value in expected_bindings.items():
        _require(bindings.get(key) == value, f"runtime evidence {key} differs from trusted job binding")
    _require_sha256(bindings.get("environment_sha256"), "runtime evidence bindings.environment_sha256")

    declarations = bundle["protocol"]["runtime_declarations"]
    checkpoint = declarations["checkpoint_content_sha256"]
    if checkpoint is not None:
        _require(
            checkpoint == expected_job["bindings"]["checkpoint_content_sha256"],
            "trusted job checkpoint differs from the protocol declaration",
        )
    environment_check = checks["environment_binding"]
    if environment_check["status"] == "passed":
        evidence = environment_check["evidence"]
        backend_key = (
            "flash_attention_version"
            if case["selection"]["attention_backend"] == "flash_attention_2"
            else "flex_attention_version"
        )
        declared_environment = {
            "hardware_identifier": declarations["hardware_identifier"],
            "torch_version": declarations["torch_version"],
            "cuda_version": declarations["cuda_version"],
            "attention_backend": case["selection"]["attention_backend"],
            "backend_version": declarations[backend_key],
        }
        if all(value is not None for value in declared_environment.values()):
            _require(
                evidence == declared_environment,
                "observed environment differs from the protocol declaration",
            )


def validate_runtime_evidence(record, *, bundle, expected_job, accept=False):
    """Validate a future full record against trusted protocol and job inputs."""
    bundle = _reload_validated_bundle(bundle)
    case = _trusted_runtime_case(bundle, expected_job)
    _require(isinstance(record, dict), "R3 runtime evidence must be a JSON object")
    _require(record.get("schema_version") == SCHEMA_VERSION, "unsupported runtime evidence schema_version")
    _require(record.get("record_kind") == RUNTIME_RECORD_KIND, "unexpected runtime evidence kind")
    _require(record.get("expected") == expected_job["expected"], "runtime evidence cardinalities differ from trusted job binding")
    _require(record.get("case_id") == case["case_id"], "runtime evidence matrix case differs")
    _require(record.get("requested") == case["selection"], "runtime evidence requested settings differ from matrix case")
    checks = record.get("checks")
    required = set(case["required_observations"])
    _require(isinstance(checks, dict) and set(checks) == required, "runtime evidence checks are incomplete or unexpected")
    for check_id, check in checks.items():
        _require(isinstance(check, dict) and set(check) == {"status", "evidence"}, f"runtime check {check_id} is malformed")
        _require(check["status"] in {"passed", "failed", "not-run"}, f"runtime check {check_id} has invalid status")
        if check["status"] == "passed":
            _require(isinstance(check["evidence"], dict) and check["evidence"], f"passed runtime check {check_id} lacks evidence")
        else:
            _require(check["evidence"] is None or isinstance(check["evidence"], dict), f"runtime check {check_id} evidence is malformed")
    _validate_declared_runtime_bindings(record, bundle, expected_job, checks, case)
    _validate_rank_and_mask_cardinality(record, checks, case)
    _validate_passed_checks(record, checks, case)
    if accept:
        _require(record.get("evidence_class") == GENUINE_RUNTIME_EVIDENCE, "CPU fixtures cannot satisfy R3 acceptance")
        _require(all(check["status"] == "passed" for check in checks.values()), "not-run or failed checks cannot satisfy R3 acceptance")
        _require(
            not evidence_acceptance_blockers(bundle["protocol"]),
            "protocol still has evidence-acceptance blockers",
        )
        _require(record.get("r3_acceptance") is True, "runtime record does not declare R3 acceptance")
    else:
        _require(record.get("evidence_class") in {GENUINE_RUNTIME_EVIDENCE, CPU_FIXTURE_EVIDENCE}, "unknown runtime evidence class")
    return True


def write_immutable_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_json_bytes(record)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        if path.read_bytes() != content:
            raise FileExistsError(f"immutable R3 evidence conflict: {path}") from error
        return False
    with os.fdopen(fd, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    return True
