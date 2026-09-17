# R3-P upstream baseline provenance gate

## Status and purpose

**Status: PLANNED.** This is a CPU/source-only prerequisite to any R3
full-generator or upstream-vs-custom parity authorization. It does not authorize
checkpoint loading, model execution, GPU work or generation.

The current “upstream” route means `wan.modules.model.WanModel` from the pinned
local `wan2.1` checkout at revision
`00bde1e719ccb56c66a01a1f18a70c49b278c202`. A clean worktree proves only that
this revision has no uncommitted edits; it does not prove that its tree is
identical to an official Wan2.1 release. Existing U1 and v7 evidence remains
valid for that exact local implementation, but must not be relabeled as a
pristine-vendor result without this gate.

Pre-work estimate and routing are frozen in `docs/experiment_plan.md`: **6/10**,
with GPT-5.6 Sol for implementation and review.

## Required source identities

Before comparison, record and freeze:

1. the official Wan2.1 repository URL, resolved without relying on a mutable
   branch name alone;
2. the exact official commit corresponding to the model/configuration in use,
   with annotated tag or model-card provenance when available;
3. the current local Wan revision and clean/dirty identity;
4. a separate read-only pristine checkout path and its exact commit/tree hash;
5. the comparison tool versions and canonical hashing procedure.

Do not modify the active `wan2.1` checkout to manufacture equivalence. The
pristine checkout must remain separate and reproducible.

## Required comparisons

Produce immutable, machine-readable evidence for both:

- a whole tracked-tree comparison, including added, removed, renamed and changed
  paths; and
- the transitive execution surface of the standard Wan I2V generator route,
  including package initializers, configuration, model, attention, T5, CLIP,
  VAE, schedulers, distributed helpers and every imported local module reached
  by that route.

Record canonical path/content hashes, an exact binary-safe diff or equivalent
manifest, and a digest over the complete comparison artifact. Generated files,
vendored dependencies and model weights must be dispositioned explicitly rather
than silently excluded.

## Difference classification

Every difference must receive exactly one disposition with path and hunk/content
hashes, rationale and reviewer:

- `identical`;
- `observation-only-instrumentation`;
- `substantive-baseline-change`;
- `outside-upstream-execution-surface`; or
- `unresolved`.

An instrumentation allowlist is acceptable only when the change is limited to
observer imports, no-op scopes or evidence emission; does not alter tensor
values, seeds, state, model/config loading, attention/backend selection,
scheduler behavior, branching or exception behavior when observation is off;
and remains separately subject to R3 runtime-neutrality checks. File-level
allowlisting is insufficient: allowed hunks/content identities must be frozen.

## Baseline decision

The completion record must choose exactly one outcome:

1. **Pristine current route:** the relevant local execution surface is
   byte-identical to the pinned official source.
2. **Instrumented vendor-derived route:** every relevant difference is
   allowlisted observation-only instrumentation. Future claims must use that
   label, not “unmodified upstream.” A truly pristine claim still requires the
   separate checkout.
3. **Separate pristine route required:** any substantive relevant difference
   exists. The next protocol must bind and execute the pristine checkout as the
   upstream reference while keeping the custom checkout distinct.

Any `unresolved` difference blocks the decision and all full-generator/parity
execution.

## Planned artifacts and acceptance

Planned durable artifacts are:

- `docs/r3_upstream_provenance.json` for source identities, classifications and
  the baseline decision;
- `docs/r3_evidence/upstream-whole-tree.diff` or a binary-safe equivalent;
- `docs/r3_evidence/upstream-execution-surface.json` for the dependency closure
  and per-path hashes; and
- focused tests that reject a moved official revision, changed local tree,
  incomplete classification, broad/unhashed allowlist or inconsistent decision.

R3-P is complete only when every tracked-tree difference is represented, every
execution-surface difference is resolved, artifact hashes reproduce, and the
roadmap records the chosen baseline outcome. The subsequent generator protocol
must bind the official commit/tree, comparison digest and chosen baseline route.
Completion proves provenance only; it does not prove numerical parity or
scientific efficacy.
