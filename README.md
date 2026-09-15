# Wan action-binding experiments

This repository is an experiment workspace for testing action binding in Wan video generation. The authoritative roadmap and claim boundaries are in [`docs/experiment_plan.md`](docs/experiment_plan.md). CPU contract checks, immutable manifest handling, and the bounded U1 one-layer parity result do **not** establish full-generator, CUDA/FSDP, video-quality, or scientific efficacy readiness.

## Environment

Use the checked-in `pyproject.toml` and `uv.lock` through `uv` only. Do not use system Python, system `pip`, or an ad hoc environment.

```bash
git submodule update --init --recursive
uv sync --locked
```

Both `wan2.1` and `lama` are required workspace submodules. A full locked sync can fail while building the SAM2 CUDA extension on an incompatible CUDA toolchain. The bounded CPU-only K1 fallback is documented in `docs/experiment_plan.md` and `docs/k1_validation.md`; it omits only `sam2` and is not a full installation or GPU-readiness result.

## Canonical M1 interface

The supported orchestration path is:

```text
experiment_pipeline -> manifest_adapter -> fsdp_worker -> upstream/custom Wan
```

The smoke fixture is **NON-SCIENTIFIC**. Validation and dry-run do not generate video:

```bash
uv run --locked python -m multi_sample_inference.experiment_pipeline validate \
  --source tests/fixtures/smoke_experiment.json
uv run --locked python -m multi_sample_inference.experiment_pipeline dry-run \
  --source tests/fixtures/smoke_experiment.json --output-dir /tmp/wan-smoke-run
```

A production source must replace every fixture asset and placeholder checkpoint with approved local, content-inventoried material; set `smoke_only` to false; preserve actor order across AB/BA; use target-sized masks; provide complete schedules and inference settings; and choose an output directory outside this repository. The CLI does not download checkpoints.

```bash
uv run --locked python -m multi_sample_inference.experiment_pipeline validate --source /path/protocol.json
uv run --locked python -m multi_sample_inference.experiment_pipeline expand \
  --source /path/protocol.json --output-dir /scratch/action-binding-run
uv run --locked python -m multi_sample_inference.experiment_pipeline status \
  --jobs /scratch/action-binding-run/jobs
uv run --locked python -m multi_sample_inference.experiment_pipeline run \
  --job /scratch/action-binding-run/jobs/JOB_ID.json --devices 0,1
```

`run` launches only the selected job and requires the declared number of visible devices. Do not run production generation until the R3 environment, checkpoint, and compute gates are satisfied.

The `annotations-export` and `annotations-import` commands provide operator interchange only. Their CSV exposes condition metadata and source paths; it is **not** a blinded H0 rater interface.

## Active and compatibility paths

The canonical active surface is `multi_sample_inference/experiment_pipeline.py`, its adapter/worker/contracts, the `wan2.1` submodule, focused tests, and the protocol/evidence documents. Existing direct dispatcher, shell/OAR/Slurm, evaluation, gallery, visualization, prompt-production, examples, notebooks, and paper paths are retained in place as historical or compatibility material. They are noncanonical and are not validated production instructions. See [`docs/k1_inventory.md`](docs/k1_inventory.md) and [`docs/k1_deletion_manifest.md`](docs/k1_deletion_manifest.md) before changing or deleting them.
