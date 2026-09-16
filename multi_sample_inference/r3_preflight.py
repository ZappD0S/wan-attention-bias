"""Safe no-generation preflight for the frozen R3 protocol and matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .r3_contracts import (
    evidence_acceptance_blockers,
    execution_blockers,
    load_protocol_bundle,
)


def preflight(protocol_path, matrix_path, *, execution=False):
    bundle = load_protocol_bundle(protocol_path, matrix_path)
    blockers = execution_blockers(bundle["protocol"])
    acceptance_blockers = evidence_acceptance_blockers(bundle["protocol"])
    result = {
        "schema_version": bundle["protocol"]["schema_version"],
        "preflight_kind": "r3-no-generation-preflight",
        "mode": "execution" if execution else "preparation",
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
    args = parser.parse_args(argv)
    print(
        json.dumps(
            preflight(args.protocol, args.matrix, execution=args.execution),
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
