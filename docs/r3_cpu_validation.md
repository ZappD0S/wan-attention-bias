# R3 CPU-only preparation validation

Date: 2026-09-15. Scope: local CPU validation only, with
`CUDA_VISIBLE_DEVICES=''`. No SSH, checkpoint loading/download, generation,
CUDA kernel execution, GPU query, torchrun/FSDP job, or compute submission was
performed.

## Environment boundary

The earlier preparation attempt ran the required `uv sync --locked`, which
failed only while building `sam2==1.1.0` because `CUDA_HOME` was unset at the
known SAM2 CUDA-extension boundary, then successfully used the documented
`uv sync --locked --no-install-package sam2` fallback. Its temporary `/tmp`
logs were not retained across this fallback recovery session. Because the
uv-created environment remained functional, recovery did not reinstall it.
This is not a complete locked environment and does not establish SAM2, tracker,
CUDA, flash/flex, generator, or distributed readiness.

Versions observed through `uv run --no-sync`: Python 3.11.15, Torch
2.10.0+cu128 (packaged CUDA runtime 12.8; CUDA availability false with devices
hidden), NumPy 1.26.4, SciPy 1.17.1, pytest 9.0.3, and Ruff 0.15.12.

## Results

- Frozen v1 protocol/matrix preparation preflight: passed schema and identity
  checks; 592 cases and 13 named invalid-input checks enumerated; execution
  readiness false. Protocol SHA-256:
  `1b45951b65fc757615dfb91c9e07859084457a2dba34c18a4f9c4e3f83f3488b`.
  Matrix SHA-256:
  `381fd9c5829fbe405e17ec77da57a5c3b14e60c325e653571b4c4d3a4043b1a5`.
- Lineage-bound v2 preparation preflight: passed the same 592-case/13-check
  coverage with all source-hook states GPU-unvalidated and execution readiness
  false. Protocol SHA-256:
  `1adb1508be962d140070283a45dc092713e56b3b00029c406947d4989ee058b6`;
  matrix SHA-256:
  `ecad41461cd43db148af70d91a798f2342263ceee073ce92d1ae8d12b90bc643`.
  No runtime declaration, approval, rank count, or tolerance was filled from the
  local machine.
- Execution-mode preflight: expected rejection (exit 1). Frozen v1 reports 18
  contract-execution blockers: two approvals, seven runtime declarations, six
  unimplemented hooks, two tolerance values and an amendment. The checked-in v2
  reports 12 contract-validation blockers (the approvals, declarations,
  tolerances and amendment) and 18 evidence-acceptance blockers because its six
  implemented source hooks are still GPU-unvalidated. A fully populated,
  synthetic amended-v2 fixture passes contract-validation preflight while
  remaining evidence-acceptance blocked; it performs no execution.
- Focused CPU regression suite after direct restart review and final fixes: 89
  passed in 3.09 seconds. Regressions cover present-null declarations, including
  present `parity_pair: null` during source expansion; protocol/matrix/trusted
  job, checkpoint, environment, rank-mode and cardinality binding; default-no-op
  observer behavior, lifecycle/errors/coordinate overlap; exact CFG/mask/tracker/
  dispatch-site/version coordinates, including rejection of cross-attention as a
  substitute for the requested conditional self-attention backend; composed
  two-rank collector/gather validation, rank-local ordering and trusted seed
  binding; cross-event mutation detection;
  parity final finiteness; composed PyTorch float16/float32/float64 final
  collector-to-artifact identity and source validation; fail-closed PyTorch
  bfloat16; Torch-produced pair comparison; immutable artifacts, tampering,
  mismatched inputs and upstream-reference measured tolerance failures;
  malformed or mismatched checkpoint declarations; exact requested-backend
  version binding; malformed returned-mask dimensions; and missing, mutated or
  substituted observation/result/status evidence. Every created CPU artifact uses
  `non-scientific-cpu-fixture` evidence.
- Syntax compilation: passed with no output.
- Focused Ruff on the new contracts/runtime/parity/preflight/tests and Wan hook
  files passed with no findings. The integration-file run also passed after
  explicitly limiting ignores to the established `experiment_pipeline.py`
  whole-file debt listed in the command below.
- M1 validation: `valid: NON-SCIENTIFIC-smoke-only`.
- M1 no-generation dry-run: 12 jobs enumerated; no output directory created.
- Reference indexing: `qmd embed` returned “All content hashes already have
  embeddings.” This does not establish GPU/runtime evidence.

Commands:

```bash
# Earlier preparation attempt; not rerun during recovery because the uv environment worked:
CUDA_VISIBLE_DEVICES='' uv sync --locked
CUDA_VISIBLE_DEVICES='' uv sync --locked --no-install-package sam2
CUDA_VISIBLE_DEVICES='' uv run --no-sync python -m multi_sample_inference.r3_preflight \
  --protocol docs/r3_protocol.json --matrix docs/r3_test_matrix.json
CUDA_VISIBLE_DEVICES='' uv run --no-sync python -m multi_sample_inference.r3_preflight \
  --protocol docs/r3_protocol.json --matrix docs/r3_test_matrix.json --execution
CUDA_VISIBLE_DEVICES='' uv run --no-sync python -m multi_sample_inference.r3_preflight \
  --protocol docs/r3_protocol_v2.json --matrix docs/r3_test_matrix_v2.json
CUDA_VISIBLE_DEVICES='' uv run --no-sync python -m multi_sample_inference.r3_preflight \
  --protocol docs/r3_protocol_v2.json --matrix docs/r3_test_matrix_v2.json --execution
CUDA_VISIBLE_DEVICES='' uv run --no-sync --with pytest==9.0.3 python -m pytest -q \
  tests/test_repair_contracts.py tests/test_experiment_pipeline.py \
  tests/test_mask_contracts.py tests/test_generation_routes.py \
  tests/test_parity_contracts.py tests/test_r3_contracts.py \
  tests/test_r3_runtime.py
CUDA_VISIBLE_DEVICES='' uv run --no-sync python -m compileall -q \
  multi_sample_inference/experiment_pipeline.py \
  multi_sample_inference/r3_runtime.py multi_sample_inference/r3_parity.py \
  tests/test_r3_contracts.py tests/test_r3_runtime.py
CUDA_VISIBLE_DEVICES='' uv run --no-sync ruff check \
  multi_sample_inference/r3_contracts.py \
  multi_sample_inference/r3_preflight.py \
  multi_sample_inference/r3_runtime.py \
  multi_sample_inference/r3_parity.py \
  tests/test_r3_contracts.py tests/test_r3_runtime.py \
  wan2.1/wan/utils/runtime_evidence.py wan2.1/wan/modules/attention.py \
  wan2.1/wan/modules/custom_model.py wan2.1/wan/modules/model.py \
  wan2.1/wan/image2video.py wan2.1/wan/regional_prompt/image2video.py
CUDA_VISIBLE_DEVICES='' uv run --no-sync ruff check \
  multi_sample_inference/experiment_pipeline.py \
  multi_sample_inference/fsdp_worker.py \
  --ignore PLR0912,PLR0915,B904,E731,UP017,C420,E702,E701
CUDA_VISIBLE_DEVICES='' uv run --no-sync python -m \
  multi_sample_inference.experiment_pipeline validate \
  --source tests/fixtures/smoke_experiment.json
CUDA_VISIBLE_DEVICES='' uv run --no-sync python -m \
  multi_sample_inference.experiment_pipeline dry-run \
  --source tests/fixtures/smoke_experiment.json \
  --output-dir /tmp/wan-r3-smoke-dry-run
CUDA_VISIBLE_DEVICES='' qmd embed
```

## Astra restart lead check — 2026-09-16

After the user switched to the recommended GPT-6 Astra lead, the current outer
and Wan HEADs/statuses matched the saved handoff and both indexes were empty.
All existing changes were preserved. The same seven-file focused CPU suite was
rerun through the documented `uv run --no-sync --with pytest==9.0.3` command with
`CUDA_VISIBLE_DEVICES=''`: **89 passed in 3.09 seconds**. Both preparation
preflights passed with unchanged frozen hashes and 592 cases; both execution
preflights failed with the expected approval/declaration/amendment blockers.
Outer and nested whitespace checks passed. These are new CPU observations only;
the earlier compilation, Ruff and M1 checks above were not rerun in this lead pass.

Source review confirmed backend/matrix incompatibilities and FA3 auto-selection
risk; `docs/r3_protocol.md` records the lead direction for a further versioned
amendment. No production code or frozen JSON was changed, no execution amendment
was frozen, and no stage/commit/push, SSH, GPU query/job, real checkpoint access,
or generation occurred. CPU-preparation checkpoint-commit confirmation and
explicit host/checkpoint authorization remain pending.

## Authorized source checkpoint follow-up — 2026-09-16

The user approved committing the preserved CPU-preparation work and delegating
backend/matrix repair to GPT-5.6 Sol. Before the source checkpoint, syntax
compilation and the same focused Ruff checks were rerun successfully; M1 again
validated and enumerated 12 dry-run jobs without creating the output directory.
The seven-file focused suite then passed **89 tests in 3.12 seconds**; v1/v2
preparation passed, execution remained blocked as expected, frozen hashes were
unchanged, and both whitespace checks passed. `qmd embed` again reported all
content hashes already embedded. The source checkpoint is not R3 completion
or GPU acceptance.

Separately authorized read-only SSH/hardware inspection of `bootes.alias` is
recorded in `docs/r3_bootes_preflight.md`. No checkpoint or CUDA workload was
accessed/run; checkpoint selection still requires confirmation. This later
hardware inspection does not change the CPU-only scope of the checks above.

## Explicitly not run

Checkpoint-bound generator and scheduler execution; positive/negative CFG;
all-block runtime; fixed/hard/soft runtime masks; sharing modes; UniPC/DPM++;
tracker bypass; upstream/custom-`none` full-generator parity; initial latent
identity; rank mask identity; FlashAttention 2; flex attention; single-rank GPU;
and intended-rank FSDP are all **not run**. CPU fixtures and matrix enumeration
do not count as those observations.

Checked-in v1 remains inherently non-executable. The lead-approved v2 loader,
matrix, source hooks, all-rank collector and paired numerical comparator now
have CPU-only coverage. They are not execution-ready: all hooks remain marked
`implemented-source-hook-gpu-unvalidated`; GPU/hardware approvals, compatible
checkpoint and environment identities, exact backend versions, intended FSDP
rank count, parity tolerances and a further pre-execution amendment are still
required for contract-validation execution. That amended state would remain
separately blocked from evidence acceptance until real GPU observations permit
all hook states to become `implemented`. The observer is absent by default. When enabled, R3 canonicalizes equivalent
PyTorch/NumPy floating storage names (for example, `torch.float32` to `float32`)
while preserving shape and raw bytes. Final parity artifacts support PyTorch
float16, float32, and float64 without casting. PyTorch bfloat16 is explicitly
unsupported by the NumPy 1.26 artifact path and fails before file creation rather
than converting lossily. The collector retains bounded digest metadata plus at
most one declared rank-zero final-latent tensor; hash transfers are chunked, but
the artifact writer may materialize/copy one whole contiguous CPU final tensor,
so its working-memory bound is one final tensor rather than O(chunk). Tensor
aliases cache only within one event and are rehashed thereafter. Gather preserves
rank-local ordinals and runs on all ranks after successful generation. Rank-local
failures prevent valid completed evidence and may terminate peers at collectives;
partial immutable artifacts are preserved.

R3 stays **IN PROGRESS**, and actual difficulty remains pending.
