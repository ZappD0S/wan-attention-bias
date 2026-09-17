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
  multi_sample_inference/fsdp_worker.py \
  multi_sample_inference/r3_contracts.py \
  multi_sample_inference/r3_preflight.py \
  multi_sample_inference/r3_environment.py \
  multi_sample_inference/r3_backend_canary.py \
  multi_sample_inference/r3_checkpoint_hook_canary.py \
  multi_sample_inference/r3_runtime.py multi_sample_inference/r3_parity.py \
  tests/test_r3_contracts.py tests/test_r3_runtime.py
CUDA_VISIBLE_DEVICES='' uv run --no-sync ruff check \
  multi_sample_inference/r3_contracts.py \
  multi_sample_inference/r3_preflight.py \
  multi_sample_inference/r3_environment.py \
  multi_sample_inference/r3_backend_canary.py \
  multi_sample_inference/r3_checkpoint_hook_canary.py \
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

## V3 backend-compatibility preparation — 2026-09-16

The Astra-approved, Sol-implemented CPU-only slice added lineage-bound v3
preparation contracts without changing the frozen v1/v2 files. Their SHA-256
values remain exactly those recorded above. V3 protocol SHA-256 is
`91a34240c8a9d4d13caa0f6579de523f1d2ca3cb9c387cca17734c18fd2030f5`;
v3 matrix SHA-256 is
`e1776af152b3b0594c3af7497ffe95ec31db263fc9e3974bc9ea3c1d838fbd86`.
The v3 matrix explicitly dispositions all 592 inherited cases: 404 are runnable
only when their concrete route and schedules are compatible, and 188 are
source-infeasible and rejected. All 13 invalid-input checks remain.

CPU fixtures cover source-expansion rejection of impossible and concrete
incompatible requests; mixed flex/FA2 schedules; negative-CFG FA2 routing;
exact self-site counts, coordinates and versions; wrong sites, versions and
all-flash masquerading as flex; schedule/config substitution; and the pre-model
FA3/unavailable/version-mismatch guard. The unchanged seven-file suite passed
**94 tests in 3.13 seconds**. V1, v2 and v3 preparation preflights passed; all
three execution preflights rejected as expected. Syntax compilation and focused
Ruff passed, including only the previously documented integration-file ignores.
M1 validated the non-scientific smoke source and dry-ran 12 jobs without
creating the output directory. Both repository whitespace checks passed.

No GPU, SSH, checkpoint access/loading/download, generation, kernel execution,
or environment/lock change occurred. The v3 template leaves approvals,
checkpoint/hardware/runtime/backend identities, intended ranks and tolerances
unpopulated. FA3 remains unsupported rather than being worked around, and no
attention call/default was rewritten. Full-record/comparator acceptance remains
out of scope and passed full-generator parity claims still fail closed. R3 stays
**IN PROGRESS**, with its frozen estimate/models unchanged and actual difficulty
pending.

### Bounded Astra review and Sol correction

Under the operator's updated routing policy, a read-only GPT-6 Astra subagent
reviewed only the v3 trust boundary with an 18-tool, 28,000-token and 30-minute
hard bound. It returned `BLOCK` with two P1 findings: a removable task schema
marker could skip the pre-model v3 guard, and manifest inference could diverge
from the hash-bound source while retaining the source configuration claim.

The Sol main-agent fix derives v3 from the validated manifest independently of
task markers; requires the v3 dispatch contract; reconstructs inference and the
complete intervention from the hash-bound source; and binds exact case selection
and expected rank/step/layer cardinalities. Added regressions reject discriminator
removal, missing dispatch contracts, solver/case substitution and cardinality
substitution. After the fix, the focused R3 suite passed **52 tests in 2.60
seconds** and the seven-file suite passed **94 tests in 3.40 seconds**. Syntax,
focused Ruff, all three preparation preflights, all three expected-blocked
execution preflights, M1 validation/12-job no-output dry-run and whitespace
checks passed. The v1/v2/v3 hashes above remained unchanged. Initial review:
`/home/gzappavi/.pi/agent/sessions/--home-gzappavi-Documents-wan_experiments--/subagent-artifacts/outputs/e9410c24-c944-4861-af30-ffbab575eed2/r3-astra/v3-critical-review.md`.
Because that reviewer was not retained, a fresh same-role Astra fallback rechecked
only the two fixes and their immediate blast radius under a smaller 9-tool,
11,000-token and 15-minute hard bound. It found both P1s resolved, no adjacent
defect and returned `Merge verdict: OK`:
`/home/gzappavi/.pi/agent/sessions/--home-gzappavi-Documents-wan_experiments--/subagent-artifacts/outputs/806e7684-05f1-4df8-8a74-f8aba318727a/r3-astra/v3-fix-followup.md`.
No GPU, checkpoint or generation activity occurred.

## Authorized checkpoint identity revalidation — 2026-09-16

After the v3 commit, the user selected the retained Bootes U1 snapshot for R3
identity revalidation. Read-only SSH checks found 33 resolved files, zero broken
symlinks and the approved 82,272,036,045-byte inventory. The inventory SHA-256
remained `e6b7adbd6f6e5dfcb7fa06e09d5d2edfcb179e80ff25f1c743567d1948701dd4`.
A full content rehash through `checkpoint_identity(..., verify_contents=True)`
completed in 70.048 seconds and reproduced
`80c954fbb46c39139a300a7dd41ed5a8b1c6b11c5f952c1bfb3c763d6e73ebb7`.

This check ran under `uv run --no-sync` with `CUDA_VISIBLE_DEVICES` empty. It did
not load checkpoint tensors, run a CUDA kernel, change the remote environment,
or generate output. A same-boundary package probe confirmed matching local/remote
`pyproject.toml` and `uv.lock` hashes and imported Python 3.11.15, PyTorch
2.10.0+cu128, CUDA runtime metadata 12.8, flash-attn 2.8.3 and PyTorch flex
attention. `sam2` remains absent, and no alternate compiler was found under
`/usr/local` or `/opt`; only the incompatible system nvcc 12.0 is observed.

Snapshot identity is now confirmed for preparation of a further execution
amendment. At this point a fresh remote source copy and approved
Blackwell-capable toolchain provisioning remained unresolved.

## Authorized Bootes environment provisioning — 2026-09-16

The user authorized a fresh remote R3 source and complete locked environment,
with an explicit stop before GPU kernels, model loading or generation. The
source at `/local_scratch2/gzappavi/wan_experiments_r3` is clean at outer commit
`b07c64aa6b2aaea90627c30e27a4b22f194c5b41`, Wan commit
`00bde1e719ccb56c66a01a1f18a70c49b278c202` and LaMa commit
`469acc7358a1c6828b647b4ee20c93474a2f36b4`. LFS smudging was skipped, leaving
example LFS objects as pointers; the retained U1 evidence tree was not mutated.

A toolkit-only CUDA 12.8.1 installation at
`/local_scratch2/gzappavi/toolchains/cuda-12.8.1` reports nvcc V12.8.93. The
5,382,238,770-byte runfile SHA-256 is
`228f6bcaf5b7618d032939f431914fc92d0e5ed39ebe37098a24502f26a19797`.
The initial locked sync failed before package resolution because the LaMa
submodule was absent from the bundle checkout. After initializing its exact
recorded commit, GPU-hidden `uv sync --locked` resolved 287 packages and
succeeded; a second invocation resolved in 3 ms and checked 266 packages.

With `CUDA_VISIBLE_DEVICES` empty, imports report Python 3.11.13, PyTorch
2.10.0+cu128, CUDA runtime metadata 12.8, flash-attn
2.8.3+cu128torch2.10, SAM2 1.1.0 and PyTorch flex attention. Static `cuobjdump`
inspection found `sm_120` cubins in both SAM2 and FA2. The approved
`pyproject.toml` and `uv.lock` hashes are unchanged. No checkpoint/model was
loaded and no GPU kernel or generation ran.

Provisioning is complete, but at this point full-generator compatibility,
intended ranks, numerical tolerances, GPU kernel/source-hook validation,
distributed behavior and evidence acceptance remained unresolved.

## GPU-free v4 execution-amendment validation — 2026-09-16

The authorization-pending `docs/r3_protocol_v4.json` now predeclares the rehashed
checkpoint, exact Wan/LaMa revisions, parent production-content digest
`56e4cc52…`, complete Bootes environment/hardware binding, two intended FSDP
ranks and full-generator parity at `atol=1e-5`, `rtol=0.016`. The tolerance is
explicitly re-derived as a near-zero absolute floor and rounded
two-BF16-epsilon relative allowance rather than inherited from U1. Its protocol
SHA-256 is `004f2f9e…`; it reuses the frozen v3 matrix hash `e1776af1…`.

CPU guards now require clean parent/Wan/LaMa worktrees, exact nested revisions
and the parent production digest during v4 materialization. Before model load,
the worker requires the manifest-bound repository identity, exact FA2/flex
helper state, and exact host, Python, Torch/CUDA, SAM2, driver, GPU
UUID/model/capability and project/lock observations. Synthetic tests exercise
exact/missing/mismatched environment declarations and source-binding failures;
no real host observation was made in this phase.

The five cumulative stages cover bounded backend kernels, checkpoint/hook
canary, one single-rank generator pair, remaining single-rank contracts and
finally two-rank FSDP. Every stage remains `not-approved`. V4 preparation
preflight enumerates 592 cases (404 runnable, 188 rejected) and reports exactly
four blockers: the two approvals, explicit user authorization and staged gates.
Execution preflight rejects those blockers as intended; evidence acceptance is
also blocked by all six GPU-unvalidated source hooks.

The seven-file focused suite passed **97 tests in 3.30 seconds**. Focused syntax
compilation and both Ruff commands passed; M1 smoke validation and the 12-job
no-output dry-run passed; v1/v2/v3 preparation preflights still pass and their
execution preflights remain blocked; outer and nested whitespace checks pass.
All project commands used `uv` with `CUDA_VISIBLE_DEVICES=''` where runtime
imports were possible.

No SSH, GPU query/kernel, checkpoint/model load, generation, torchrun/FSDP job
or scientific measurement occurred. V4 itself is not authorized for execution:
a later immutable authorization amendment is required. Real backend kernels,
hook neutrality, full generator, distributed behavior and full-record acceptance
remain unvalidated. R3 stays **IN PROGRESS**.

## Bounded v5 authorization validation — 2026-09-17

The immutable v5 protocol (SHA-256 `d3271493…`) is lineage-bound to v4 and
records explicit authorization for only `backend-kernel-canary`. Both top-level
approvals are `approved`, but generic execution remains blocked by
`staged-execution-gates`; stage-scoped execution preflight passes only for the
backend canary and rejects the checkpoint/hook stage. The source binding covers
the dedicated no-model canary runner and shared environment observer under
production digest `7b8e027a…`. Tests also confirm exact v4/v5 environment checks
and clean exact v5 parent/Wan/LaMa binding.

With CUDA hidden, the seven-file suite passed **99 tests in 3.43 seconds**; the
R3 subset passed **57 tests in 2.58 seconds**. Focused compilation and both
Ruff commands passed. V5 preparation reports only the later-stage blocker, and
its backend-stage execution preflight reports no blockers while evidence
acceptance remains blocked. At this checkpoint the source had not yet been
transferred and no GPU kernel had run. Checkpoint/model loading, generation,
distributed execution and all later stages remain prohibited; R3 stays **IN
PROGRESS**.

The three-commit bundle from `b07c64a` through `db99ecf` was later verified and
fast-forwarded into the clean Bootes worktree. Remote exact source checks
reproduced v5 SHA-256 `d3271493…` and production digest `7b8e027a…`; the
stage-scoped execution preflight passed, while generic execution rejected
`staged-execution-gates` as intended. These were no-generation checks.

The canary was not started because both GPUs were heavily occupied by another
user. The last query reported 91,130/45,116 MiB used and 95/93% utilization.
No CUDA tensor or kernel, checkpoint/model load, generation or distributed job
ran. The temporary two-minute availability poll was cancelled and there are no
active scheduled tasks. This is an availability stop, not a failed backend
result; R3 remains **IN PROGRESS**.

## Pollux v6 relocation validation — 2026-09-17

The user selected idle `pollux.alias` for the same backend-only stage. The
lineage-bound v6 protocol (SHA-256 `f4de93f2…`) preserves every prohibition and
changes only the exact host/source/environment binding. It records Pollux
hardware identifier `f13f743d…`, production digest `69b14888…`, two new GPU
UUIDs and `NODE-no-NVLink` topology. Generic execution remains blocked; the v6
backend-stage execution preflight passes.

The clean source, 9.4-GiB locked environment, 8.7-GiB CUDA toolkit and verified
installer were copied directly from Bootes to writable Pollux local scratch.
Two GPU-hidden `uv sync --locked` passes rebound editable workspace paths. A
no-model probe reproduced Python 3.11.13, Torch 2.10.0+cu128, CUDA 12.8, FA2
2.8.3, no FA3, flex attention, SAM2 1.1.0, driver 580.173.02 and exact
project/lock hashes. The attention environment probe was changed to inspect
installed FA2/FA3 directly rather than import Wan's eager model package before
the environment gate.

Focused R3 CPU tests passed 60 tests and the seven-file suite passed 102;
focused compilation, both documented Ruff commands and whitespace checks passed. No checkpoint was copied or loaded and no Pollux CUDA tensor or
kernel had run at this freeze point.

After commit `3dbf8e0` was installed cleanly, remote stage preflight and exact
source/environment validation passed. With only idle GPU 0 visible, the bounded
runner exited 0 and wrote immutable record SHA-256 `6d5c5cae…` plus log SHA-256
`cf7d78dc…`. FA2, compiled flex attention and SAM2 all produced finite outputs;
the SAM2 semantic check returned component areas 1 and 4. The local evidence
regression binds the protocol/matrix hashes, exact environment, clean source,
CUDA outputs and all-false model/generation/distributed scope. After harvest,
the focused R3 suite passed 61 tests and the seven-file suite passed 103;
compile and both documented Ruff commands also passed.

This is a passed backend-kernel canary only. No checkpoint/model load,
generation, source-hook validation, FSDP or acceptance test occurred. The
mandatory stage stop was observed, all later stages remain unapproved, and R3
remains **IN PROGRESS**.

## Bounded v7 checkpoint/hook validation — 2026-09-17

The user explicitly authorized the next cumulative stage after the v6 stop.
Immutable v7 (SHA-256 `63e2781b…`) binds the passed backend record, exact Pollux
checkpoint/source/environment declarations, upstream Wan loader, layer-0 state
identity and a fixed small hook-neutrality input/event contract. Only the first
two cumulative stage gates are approved; generator and later stage preflights
remain blocked.

CUDA-hidden protocol and runner regressions passed in the focused R3 suite: **63
tests in 2.70 seconds**; the seven-file suite passed **105 tests in 3.49
seconds**. Focused compile, both Ruff commands, v1–v7 preparation preflights,
M1 validation and the 12-job no-output dry-run also passed. The v7
checkpoint/hook execution preflight reports no
blockers without touching a GPU, while generic execution and the generator stage
fail closed. At this freeze point no checkpoint had been copied to Pollux, no
checkpoint/model had been loaded and no v7 CUDA operation had run.

After verified commit transfer and complete checkpoint rehashing, the bounded
runner exited 0 on one visible idle Pollux GPU. Evidence SHA-256 `23a4a8df…`
and log SHA-256 `af6f919b…` bind v7, the frozen matrix, prior backend evidence,
exact source/environment/checkpoint, concrete loader and scope. Layer 0 retained
state `eb1b247b…`; observer/no-observer outputs were finite and bitwise identical
at `236f8556…`, with exact input/state preservation and expected self/cross/cross
FA2 events. No full model forward, generation or distributed operation occurred.
The mandatory stage stop was observed. Post-harvest evidence regressions passed
**64 focused R3 tests in 2.87 seconds** and **106 seven-file tests in 3.68
seconds**; compile and both Ruff commands passed. R3 remains **IN PROGRESS**.
