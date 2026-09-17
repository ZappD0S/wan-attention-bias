"""Safe no-generation preflight for the frozen R3 protocol and matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .r3_contracts import (
    R3_STAGE_IDS,
    evidence_acceptance_blockers,
    execution_blockers,
    load_protocol_bundle,
    stage_execution_blockers,
)


def validate_backend_runtime(dispatch_contract, observations):
    """Validate real helper observations for the opt-in v3+ worker guard."""
    required = {
        "flash_attention_2_available",
        "flash_attention_3_available",
        "flash_attention_version",
        "flex_attention_available",
        "flex_attention_version",
    }
    if not isinstance(observations, dict) or set(observations) != required:
        raise ValueError("R3 backend runtime observations are incomplete")
    if observations["flash_attention_3_available"] is not False:
        raise RuntimeError("R3 v3+ forbids FlashAttention 3 auto-selection")
    if observations["flash_attention_2_available"] is not True:
        raise RuntimeError("R3 v3+ requires available FlashAttention 2")
    versions = dispatch_contract["backend_versions"]
    if any(not isinstance(value, str) or not value for value in versions.values()):
        raise RuntimeError("R3 v3+ backend versions are undeclared")
    if observations["flash_attention_version"] != versions["flash_attention_2"]:
        raise RuntimeError("observed FlashAttention 2 version differs from the protocol")
    if observations["flex_attention_available"] is not True:
        raise RuntimeError("R3 v3 declared flex attention helper is unavailable")
    if observations["flex_attention_version"] != versions["flex_attention"]:
        raise RuntimeError("observed flex attention version differs from the protocol")
    return True


def validate_v4_runtime_environment(protocol, observations):
    """Require exact v4+ host/package/hardware observations before model load."""
    if protocol.get("schema_version", 0) < 4:
        return True
    expected = protocol["execution_amendment"]["environment_binding"]
    if not isinstance(observations, dict) or set(observations) != set(expected):
        raise ValueError("R3 v4+ runtime environment observations are incomplete")
    mismatches = [
        key for key, value in expected.items() if observations.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            "R3 v4+ runtime environment differs from the amendment: "
            + ", ".join(sorted(mismatches))
        )
    return True


def preflight(protocol_path, matrix_path, *, execution=False, stage=None):
    bundle = load_protocol_bundle(protocol_path, matrix_path)
    blockers = (
        stage_execution_blockers(bundle["protocol"], stage)
        if stage is not None
        else execution_blockers(bundle["protocol"])
    )
    acceptance_blockers = evidence_acceptance_blockers(bundle["protocol"])
    result = {
        "schema_version": bundle["protocol"]["schema_version"],
        "preflight_kind": "r3-no-generation-preflight",
        "mode": "execution" if execution else "preparation",
        "stage": stage,
        "protocol_id": bundle["protocol"]["protocol_id"],
        "protocol_sha256": bundle["protocol_sha256"],
        "matrix_id": bundle["matrix"]["matrix_id"],
        "matrix_sha256": bundle["matrix_sha256"],
        **bundle["summary"],
        "execution_ready": not blockers,
        "contract_validation_execution_ready": not blockers,
        "evidence_acceptance_ready": not acceptance_blockers,
        "blockers": blockers,
        "evidence_acceptance_blockers": acceptance_blockers,
        "gpu_or_checkpoint_access": "not-performed",
    }
    if execution and blockers:
        raise RuntimeError("R3 execution preflight blocked: " + ", ".join(blockers))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--execution", action="store_true")
    parser.add_argument("--stage", choices=R3_STAGE_IDS)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            preflight(
                args.protocol,
                args.matrix,
                execution=args.execution,
                stage=args.stage,
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
