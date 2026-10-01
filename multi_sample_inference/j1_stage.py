"""Run exactly one approved J1 stage with the paths bound in its jz authorization record.

``tools/j1_slurm_stage.sh`` calls this inside one SLURM job. ``--print-output``
only resolves the bound one-attempt output path, so the wrapper can refuse an
existing attempt before any GPU work.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .r3_contracts import load_protocol_bundle
from .r3_jz import (
    J1_STAGE_IDS,
    JZ_MATRIX_PATH,
    STEP0_STAGE,
    jz_authorization,
    require_jz_protocol_path,
)


def run_stage(protocol_path, stage_id):
    repo = Path(__file__).resolve().parents[1]
    protocol_path = Path(protocol_path).resolve()
    matrix_path = repo / JZ_MATRIX_PATH
    protocol = load_protocol_bundle(protocol_path, matrix_path)["protocol"]
    require_jz_protocol_path(repo, protocol_path, matrix_path, protocol)
    authorization = jz_authorization(protocol, stage_id)
    output = Path(authorization["output"])
    if output.exists():
        raise FileExistsError(f"J1 {stage_id} attempt already exists: {output}")
    if stage_id == "backend-kernel-canary":
        from .r3_backend_canary import run_backend_canary  # noqa: PLC0415

        return run_backend_canary(protocol_path, matrix_path, output)
    if stage_id == STEP0_STAGE:
        from .j1_step0_probe import run_step0_probe  # noqa: PLC0415

        return run_step0_probe(protocol_path, matrix_path, output)
    from .r3_generator_pair import run_generator_pair  # noqa: PLC0415

    return run_generator_pair(protocol_path, matrix_path, authorization["source"]["path"], output)


def bound_output(protocol_path, stage_id):
    repo = Path(__file__).resolve().parents[1]
    protocol_path = Path(protocol_path).resolve()
    matrix_path = repo / JZ_MATRIX_PATH
    protocol = load_protocol_bundle(protocol_path, matrix_path)["protocol"]
    require_jz_protocol_path(repo, protocol_path, matrix_path, protocol)
    return jz_authorization(protocol, stage_id)["output"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=J1_STAGE_IDS)
    parser.add_argument("--print-output", action="store_true")
    args = parser.parse_args(argv)
    if args.print_output:
        print(bound_output(args.protocol, args.stage))
        return
    print(json.dumps(run_stage(args.protocol, args.stage), sort_keys=True, default=str))


if __name__ == "__main__":
    main()
