# R3 full-generator GPU contract protocol

Protocol v1 was frozen on 2026-09-15 for CPU-only preparation; its normative
files remain byte-for-byte `docs/r3_protocol.json` and
`docs/r3_test_matrix.json`. The explicitly lineage-bound v2 preparation files
add the lead-approved source-hook and paired-artifact semantics without changing
v1. Validating either version does not load a checkpoint, import a generator,
reserve a GPU, or establish runtime readiness. R3 remains **IN PROGRESS**.

## Claim and evidence boundary

R3 will test full checkpoint-bound generation contracts. It will not test action-
binding efficacy or video quality. A protocol or matrix validation is only CPU
preparation. A worker observation is genuine runtime evidence only when produced
by a real declared job, but the currently available rank-zero observation is
partial and always records `r3_acceptance=false`. CPU fixtures can test rejection
behavior but can never satisfy acceptance.

All completed R3 runtime records must be immutable and checked against a
validated protocol/matrix bundle whose in-memory content and SHA-256 identities
still match its files. The selected case must exist in that validated matrix.
Manifest, source, configuration, checkpoint, rank mode, and rank/step/block
cardinalities must come from the trusted job binding rather than values echoed
by the record; environment observations must match the protocol declarations.
`passed` requires actual observations; GPU-only or unimplemented checks remain
`not-run`. Missing, duplicate, partial, nonfinite, or mismatched evidence blocks
acceptance.

## Frozen matrix and stop gates

The matrix expands to 592 required cases: 576 custom-generator combinations,
eight upstream-generator cases, and eight aggregate upstream/custom-`none`
parity cases. It covers:

- custom `none`, regional prompting, Concept-Weaver, and eDiff-I;
- fixed/fixed, hard/dynamic, and soft/dynamic masks;
- `current`, `first`, and `previous_generated` sharing;
- self-attention routing off and on;
- UniPC and DPM++;
- requested FlashAttention 2 and flex attention;
- one rank and the intended FSDP rank count.

Each generator case requires real positive and negative CFG branch execution,
all configured blocks, finiteness, initial-latent identity across ranks, the
actual solver and kernel, and early-rejection evidence. Custom cases additionally
require separately identifiable generated masks, exact used-mask identity by
rank/step/block, and exact self/cross consumer identity. Fixed cases require an
observed tracker call count of zero. Full upstream/custom-`none` parity requires an explicit tolerance declared
before execution. Protocol v1 deliberately rejects every `passed` parity claim:
a paired input/job identity, paired upstream/custom-`none` output identities,
and producer/consumer numerical-comparison schema do not yet exist. A bare
finite/passed boolean or tolerance echo is never parity evidence.

Any parity, finiteness, rank-initial-latent hash, mask identity, cardinality, or
fixed-mask tracker failure stops generation experiments. The invalid-input
matrix requires rejection before output or completed status for invalid batches,
overlap, empty resolved masks, unsupported entity layouts, token overflow,
schedule errors, missing negative prompts, rank mismatch, and incomplete or
conflicting evidence.

## Frozen exact semantics and unresolved declarations

The following CPU-independent rules are frozen now:

- initial-latent hashes must be exactly equal across all expected ranks;
- used-mask hashes must be exactly equal across self/cross consumers and ranks
  for each unique step/block coordinate;
- generated and used masks remain separate identities;
- expected rank × step × block cardinality is exact, with no duplicate coordinate;
- every tested tensor path must be finite.

The v1 hardware/environment approval, checkpoint identity, exact Torch/CUDA/
FlashAttention/flex versions, intended FSDP rank count, and full-generator parity
`atol`/`rtol` are deliberately null or unapproved. They must be declared in a
new versioned pre-execution amendment. The one-layer U1 tolerance must not be
silently inherited for full-generator parity. Flex is matrix coverage, not a
claim that the active path currently selects or reports flex.

## Available top-level observation seam

An M1 source may optionally declare `r3_evidence` with protocol, matrix,
requested backend, and rank mode. Existing M1/U1 artifacts that omit the field
remain valid; an explicitly present null or malformed declaration is rejected.
Expansion then declares an immutable rank-zero worker observation path. For a
custom generator, the top-level worker can persist only facts already returned
in `extra_data`: resolved diffusion seed and shape/dtype/SHA-256 identity plus
finiteness for used and generated mask stacks. Both stacks must have the actual
returned layout `[batch, step, block, entity, time, height, width]`, with batch
one and every dimension positive. Result creation and completed status fail if
this declared record is missing, changed, substituted, malformed, nonfinite, or
bound to another job/protocol/matrix. Completed status and result identities
must also agree on the job, manifest, attempt, and video.

That v1 seam does **not** observe initial noise, non-output ranks, separate CFG
branch outputs, actual kernel dispatch, self/cross consumer object identity, or
tracker calls. Upstream returns no equivalent v1 payload. The M1 `run` command
rejects every declared R3 job while protocol approvals, declarations, hook
validation, tolerances, or amendment remain blocked; it does not use partial
evidence to bypass execution preflight. Requested backend/solver/rank values
remain labeled as requested and are never promoted to observed facts.

## Lead-approved v2 source hooks and paired comparator

Protocol/matrix v2 explicitly inherit the frozen v1 hashes. Narrow opt-in hooks
now record, without changing attention, solver, or seed arithmetic:

1. the initial latent immediately after true noise creation and before denoising
   on every rank;
2. used and generated mask tensor identities at each rank/step/block and the
   exact used-mask objects passed to self- and cross-attention;
3. actual conditional and negative-CFG output identities and finiteness;
4. every actually dispatched attention backend, its self/cross site, and its
   module/Torch version;
5. tracker calls under observer lifecycle, so completed fixed-mask runs prove an
   observed count of zero; and
6. final latent identities, with a rank-zero immutable retained tensor artifact
   for declared upstream/custom-`none` parity pairs.

The observer is absent by default and every emission/scope is then a no-op. A
non-null protocol checkpoint declaration must be a lowercase SHA-256 digest and
must match the source checkpoint before manifests are materialized or executed.
Completed source-hook evidence also requires one exact self-attention dispatch
per rank/step/CFG-branch/block. The requested backend and exact predeclared
version must occur at the conditional self-attention site for every
rank/step/block; a cross-attention dispatch cannot satisfy that contract. With
the observer installed, tensor
callbacks are immediately reduced to identity and finiteness metadata whose
hashing and device-to-host transfers are chunked. R3 normalizes a
PyTorch dtype name such as `torch.float32` to the equivalent storage name
`float32`; shape and raw bytes are unchanged, so the source observation and its
lossless NumPy artifact have one identity without weakening the frozen U1 hash
contract. Final-latent artifacts support PyTorch float16, float32, and float64.
PyTorch bfloat16 is rejected before a file is created because NumPy 1.26 cannot
serialize it losslessly through this path; it must not be cast and presented as
supported evidence. Aliases are hashed once only within one synchronous event;
the same live object is rehashed on every later event so in-place mutation cannot
be hidden. The sole retained full tensor is a declared rank-zero final-latent
parity artifact. Its writer may allocate/materialize one complete contiguous CPU
final tensor (and the resulting immutable file), rather than having O(chunk)
working memory. The offline comparator memory-maps both immutable artifacts,
verifies trusted pair/job/input/config/checkpoint/source/environment bindings,
keeps route source identities distinct, and measures finiteness and maximum
absolute and relative error in bounded chunks against predeclared tolerances,
always using upstream as the relative-tolerance/reference denominator. Only real
floating-point tensors are accepted; nonfinite comparisons fail with JSON-safe
null error measurements. A Boolean claim is not accepted as comparison evidence.
CPU synthetic artifacts are explicitly non-scientific and always record
`r3_acceptance=false`.

Every intended rank enters the post-generation `gather_object`; only rank zero
receives and persists the flattened observations. Ordinals remain rank-local,
and gathering validates rank ownership/order before flattening. Initial-latent
seed observations on every rank must equal the trusted M1 requested seed. The
collective is outside all rank-zero output paths. A rank-local generation/observer failure emits a failure
lifecycle event when possible and prevents successful evidence validation;
torchrun may terminate peers waiting at a collective, and any immutable partial
artifacts are preserved rather than overwritten. Collector metadata grows with
event count, while parity retention and offline comparison add O(final-latent
tensor bytes) storage and bounded comparison memory. No callback is placed
inside the compiled flex kernel body.

CPU tests cover observer absence/lifecycle/errors, coordinates, within-event
alias caching plus cross-event mutation, dispatch versions, tracker zero/dynamic
counts, composed rank gathering/seed binding, PyTorch float16/float32/float64
final collector-to-artifact identity and source validation, fail-closed bfloat16,
artifact immutability/tampering, Torch-produced trusted pair identities, and
comparator tolerance failures. They test mechanics only. No hook has been
exercised with a checkpoint,
CUDA attention kernel, real generator, scheduler, or FSDP, so v2 marks every
hook `implemented-source-hook-gpu-unvalidated` and execution remains blocked.
V1 remains inherently blocked; v2 loader/matrix integration exists. A synthetic
CPU fixture verifies the non-circular gate: a further v2 amendment with approvals,
all runtime identities, rank count and tolerances populated and
`amendment_required=false` may authorize **contract-validation execution** while
hooks are still `implemented-source-hook-gpu-unvalidated`. That state can never
make evidence acceptance ready or satisfy R3. Only successful GPU observations
and a subsequent version that marks every validated hook `implemented` can clear
the separate evidence-acceptance gate. The checked-in v2 template populates none
of these declarations or approvals and remains blocked.

## Astra pre-execution lead review — 2026-09-16

The user switched to `openai-codex/gpt-6-astra`, matching R3's frozen lead
recommendation. This review is local CPU/source work only, not an execution
amendment or hardware/checkpoint authorization. Frozen v1/v2 JSON files remain
unchanged. Merely filling their runtime declarations is not sufficient to resolve
the following source/matrix incompatibilities:

- `attention_backend` declares an expected observation; it does not select a
  kernel. `wan/modules/model.py` uses FlashAttention for upstream self-attention.
  In `wan/modules/custom_model.py`, flex is used only when
  `self_attention_bias_enabled` is true: non-`none` method, self routing enabled,
  and both timestep/block bias guards active. Custom `none`, disabled self
  routing, inactive schedule coordinates, and negative CFG use FlashAttention.
- Consequently upstream/flex and custom-`none`/flex cannot satisfy the current
  contract. Routed methods with mixed schedules legitimately use both kernels,
  whereas `r3_runtime.py::_validate_dispatches` requires the requested kernel at
  every conditional step/block. Cross-attention cannot repair this mismatch.
- `wan/modules/attention.py::flash_attention` defaults to FA3 when available;
  current callers do not force version 2. FA2 declarations alone cannot ensure
  FA2 execution. This is a source-level risk, not an observed host fact.

**Lead direction for the next versioned amendment:** preserve the baseline and
intervention arithmetic rather than force flex to satisfy a Cartesian label.
Give every frozen case an explicit, reasoned disposition; infeasible cases must
be rejected before launching, never counted as successful GPU coverage or silently
dropped. Define route/schedule/CFG-derived expected self-attention dispatches at
every rank/step/block, bind each observed backend to its declared version, and
require actual flex observations in flex coverage. Keep upstream/custom-`none`
parity on their common unmodified FlashAttention route. Resolve the FA2/FA3
selection explicitly before execution. The new version must retain the frozen
592-case inventory as lineage and test incompatible requests, mixed schedules,
negative CFG, wrong sites/versions, and missing/duplicate coordinates on CPU.
These directions are not implemented by the current v2 loader/validator.

Host/access and checkpoint authorization, observed complete locked environment,
checkpoint inventory, intended ranks, and predeclared generator tolerances remain
pending; no prior U1 values are adopted. Full-record/comparator integration and
independent acceptance review remain required even after real GPU hook validation.
No R3 completion or execution-readiness claim follows from this lead review.

## Lineage-preserving v3 backend preparation — 2026-09-16

The approved CPU-only compatibility slice is implemented in
`r3_protocol_v3.json` and `r3_test_matrix_v3.json`. It preserves the byte-exact
v1/v2 files and explicitly dispositions all 592 inherited cases: **404 runnable
subject to concrete configuration validation** and **188 rejected** because the
unmodified route can never provide the requested backend coverage. The 13
invalid-input checks remain present. These dispositions are preparation
inventory, not GPU passes or execution approval.

V3 treats the backend request as expected self-attention coverage, not a kernel
selector. Upstream, custom `none`, disabled self routing, inactive schedule
coordinates, and every negative-CFG self call derive FlashAttention 2. A custom
non-`none` conditional coordinate derives flex only when self routing and both
schedule guards are active. A flex request therefore requires at least one real
flex coordinate; an FA2 request requires no derived flex coordinate. Mixed flex
and FA2 schedules are valid flex-coverage configurations. Expansion and the
worker both bind the route, method, self-routing flag, full schedules, case and
backend versions, and reject incompatible or substituted configurations before
model loading. The worker derives the v3 requirement from the validated manifest
rather than a removable task marker. Before producing or accepting the task, v3
also reconstructs inference, intervention, case selection and expected
cardinalities from the hash-bound source and requires exact equality.

The opt-in v3 worker reads actual helper availability and versions. It fails
closed before model construction if FA3 is available (because the unchanged
helper would auto-select it), or if declared FA2/flex availability or versions
do not match the actual helpers. Flex is dispatched only on derived flex routes. Runtime
validation requires exactly one self dispatch per rank/step/CFG branch/block,
including negative CFG, with the derived backend and declared version; wrong
sites, duplicates, missing coordinates, or all-flash evidence for flex coverage
fail. No call/default was rewritten to force a backend.

The checked-in v3 protocol remains a preparation template: approvals,
checkpoint identity, observed hardware/runtime/backend versions, intended rank
count, and numerical tolerances are null or unapproved. Full-record/comparator
acceptance is deliberately unchanged and passed `full_generator_parity` claims
remain rejected pending separate integration and review. No R3 completion or
execution-readiness claim follows.

## Authorization-pending v4 execution amendment — 2026-09-16

`r3_protocol_v4.json` is a GPU-free pre-execution amendment over the unchanged
v3 matrix. It binds the rehashed checkpoint, exact Wan/LaMa revisions, a digest
of the tracked parent production paths, the complete Bootes package/hardware
observations, two intended FSDP ranks and staged stop gates. Manifest expansion
requires clean parent/Wan/LaMa worktrees, exact nested revisions and the frozen
parent production-content digest. The v4 pre-model guard additionally requires
exact host, Python, Torch/CUDA, FA2, flex, SAM2, driver, GPU UUID/model/capability
and project/lock-hash observations. These are declarations and guards, not GPU
observations made by this amendment.

Full-generator upstream/custom-`none` parity is predeclared at `atol=1e-5` and
`rtol=0.016`. Those values are explicitly re-derived for this test as a
near-zero absolute floor and a rounded two-BF16-epsilon relative allowance; they
are not inherited from U1, even though the independently selected values
coincide. Shape, dtype, finiteness, immutable identities and every exact contract
remain separate hard gates.

The intended FSDP rank count is two: one process for each observed 97,887-MiB
GPU. The inter-GPU path is `SYS` without NVLink, so the declaration does not
assert performance or collective success. Stages are backend-kernel canaries,
checkpoint/source-hook canary, a single-rank generator pair, remaining
single-rank contract cases and finally two-rank FSDP. Every stage is
`not-approved` and has cumulative prerequisites plus explicit stop conditions.
V4 therefore reports only the two approval blockers plus explicit user/stage
authorization blockers. An immutable later authorization amendment is required
before any GPU query/kernel, model load, generation or distributed job.

V4 does not mark any source hook GPU-validated, integrate full-record acceptance
or change generator/attention arithmetic. R3 remains **IN PROGRESS**.

## Bounded v5 backend-canary authorization — 2026-09-17

`r3_protocol_v5.json` is an immutable authorization amendment over v4 (v4
SHA-256 `004f2f9e…`; v5 SHA-256 `d3271493…`). The user's current-session
approval authorizes only transfer of the frozen source to Bootes, exact
source/environment preflight, and one FA2, compiled flex-attention and SAM2
CUDA canary on one visible GPU. It explicitly prohibits checkpoint/model
loading, generation, distributed execution and every later stage, then requires
a stop for review.

The v5 source binding includes the new backend-only canary runner and shared
runtime observer under parent production-content digest `7b8e027a…`. Generic
contract-validation preflight remains blocked by the four later stage gates;
only `--stage backend-kernel-canary` can pass. The dedicated runner revalidates
clean parent/Wan/LaMa source identities, the exact v4 environment binding, FA3
absence and one visible CUDA device before allocating its small test tensors.
It writes one immutable record outside the source tree only after all three
outputs are finite and the SAM2 component result is structurally correct.

The authorized command is bounded to:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --locked python -m multi_sample_inference.r3_backend_canary \
  --protocol docs/r3_protocol_v5.json --matrix docs/r3_test_matrix_v3.json \
  --output /local_scratch2/gzappavi/r3_stage1/backend-kernel-canary.json
```

At amendment freeze time this command had not run. A canary pass can establish
only bounded backend-kernel execution under the exact environment; it cannot
validate the checkpoint, model, source hooks, generator parity, FSDP, evidence
acceptance or any scientific result. R3 remains **IN PROGRESS**.

The source was subsequently transferred to Bootes at outer commit `db99ecf`.
Clean exact parent/Wan/LaMa checks, protocol/production hashes and the
stage-scoped execution preflight passed; generic execution remained blocked.
Both GPUs were occupied by another user's high-utilization jobs, so the canary
runner was not launched and no CUDA tensor or kernel was created. The temporary
availability poll was cancelled at the user's request, leaving no scheduled
work. Resume the same v5 stage only after an explicit continuation request and
an idle-GPU check; every later stage remains unapproved.

## Safe CPU commands now

```bash
CUDA_VISIBLE_DEVICES='' uv run --no-sync python -m multi_sample_inference.r3_preflight \
  --protocol docs/r3_protocol.json --matrix docs/r3_test_matrix.json
CUDA_VISIBLE_DEVICES='' uv run --no-sync python -m multi_sample_inference.r3_preflight \
  --protocol docs/r3_protocol_v2.json --matrix docs/r3_test_matrix_v2.json
CUDA_VISIBLE_DEVICES='' uv run --no-sync python -m multi_sample_inference.r3_preflight \
  --protocol docs/r3_protocol_v3.json --matrix docs/r3_test_matrix_v3.json
CUDA_VISIBLE_DEVICES='' uv run --no-sync python -m multi_sample_inference.r3_preflight \
  --protocol docs/r3_protocol_v4.json --matrix docs/r3_test_matrix_v3.json
CUDA_VISIBLE_DEVICES='' uv run --no-sync python -m multi_sample_inference.r3_preflight \
  --protocol docs/r3_protocol_v5.json --matrix docs/r3_test_matrix_v3.json
CUDA_VISIBLE_DEVICES='' uv run --no-sync python -m multi_sample_inference.r3_preflight \
  --protocol docs/r3_protocol_v5.json --matrix docs/r3_test_matrix_v3.json \
  --execution --stage backend-kernel-canary
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. uv run --no-sync --with pytest==9.0.3 python -m pytest -q \
  tests/test_r3_contracts.py tests/test_r3_runtime.py
```

Generic preflight reports `execution_ready=false` while later stage gates
remain blocked. The v5 command scoped to `backend-kernel-canary` is the sole
approved exception and reports ready without accessing a GPU; generic
`--execution` and every later stage must fail.

## Later GPU handoff commands (not authorized or run)

After approval, complete environment and GPU hook validation plus the further
versioned protocol/acceptance amendment. Only then use a real non-smoke source
outside this repository's output tree:

```bash
CUDA_VISIBLE_DEVICES='' uv run --locked python -m multi_sample_inference.r3_preflight \
  --protocol /path/r3_protocol_v2.json --matrix /path/r3_test_matrix_v2.json --execution
uv run --locked python -m multi_sample_inference.experiment_pipeline validate \
  --source /path/r3_source.json
uv run --locked python -m multi_sample_inference.experiment_pipeline expand \
  --source /path/r3_source.json --output-dir /scratch/r3-run
uv run --locked python -m multi_sample_inference.experiment_pipeline run \
  --job /scratch/r3-run/jobs/JOB_ID.json --devices 0
uv run --locked python -m multi_sample_inference.experiment_pipeline run \
  --job /scratch/r3-run/jobs/FSDP_JOB_ID.json --devices 0,1
uv run --locked python -m multi_sample_inference.experiment_pipeline status \
  --jobs /scratch/r3-run/jobs
```

The example `0,1` is illustrative only and must be replaced by the amended
intended rank count. No checkpoint path, host, rank count, backend version, or
tolerance is approved by this v1 document.
