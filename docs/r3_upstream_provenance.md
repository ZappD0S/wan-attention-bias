# R3-P upstream baseline provenance gate

## Status and purpose

**Status: COMPLETED.** This CPU/source-only provenance gate is satisfied for the
bounded claim below. It does not authorize checkpoint loading, model execution,
GPU work or generation.

The current “upstream” route means `wan.modules.model.WanModel` from the pinned
local `wan2.1` checkout at revision
`00bde1e719ccb56c66a01a1f18a70c49b278c202`. A clean worktree proves only that
this revision has no uncommitted edits; it does not prove that its tree is
identical to an official Wan2.1 release. Existing U1 and v7 evidence remains
valid for that exact local implementation, but must not be relabeled as a
pristine-vendor result without this gate.

Pre-work estimate and routing are frozen in `docs/experiment_plan.md`: **6/10**,
with GPT-5.6 Sol for implementation and review.

## Recorded result

The official source is `https://github.com/Wan-Video/Wan2.1.git` at detached
commit `7c81b2f27defa56c7e627a4b6717c8f2292eee58` and tree
`91b74dfa24e32dc86350fe2ef2b5f2f2e06f6ee9`, the latest commit shared by the
official default-branch history observed at `9737cba9…` and the local custom
history. Neither the repository nor model card supplied an exact code tag. The
separate checkout at
`/home/gzappavi/.pi/agent/r3-gpu-handoff-2026-09-15/pristine-Wan2.1` is clean,
detached and recursively write-protected. The compared local source remains
commit `00bde1e719ccb56c66a01a1f18a70c49b278c202`, tree
`ee7dddb233e6acc14a89cf96951cca6536587fee`, with no active-checkout mutation.

The binary-safe whole-tree comparison covers 59 official and 70 local tracked
files: 49 identical, 10 changed, 11 added, zero removed and zero renamed. All 21
differences have exact content/patch hashes, rationale and reviewer. The AST
local-import closure contains 26 official and 27 local files; nine surface paths
differ. Two exact patches are allowlisted as observation-only instrumentation,
while seven are conservatively substantive. The required decision is therefore
**separate-pristine-route-required**: future upstream parity must bind and
execute the pristine checkout, not relabel the current local route as pristine.

The record, surface and whole-tree diff SHA-256 values are respectively
`7db6e5cf983c181aed11e4631f98b625a640bd159fed485fbb7c4f7d9266ad33`,
`ace68c00feda4785f12a2c967e447f90fe0773e2c1a60248c93715d138f64418` and
`252075384646fed6727d0239d4c4fe3624041bb67c5d245e08a2a1fd6954651e`; the
comparison digest is
`e54a8ca9757a2ade239c38a848a83a8681327d1ff44c58d343b22410abad422f`.
The reproducer, seven fail-closed provenance tests, 71-test R3 suite and
113-test expanded CPU suite passed, as did compilation, both documented Ruff
checks, M1 validation/dry-run and whitespace checks. Full locked sync failed
only at the previously documented SAM2/CUDA build boundary, so validation used
the approved `--no-install-package sam2`/`--no-sync` CPU fallback. No checkpoint
or model was loaded, no GPU was used and no generation ran.

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

## Artifacts and acceptance

Durable artifacts are:

- `docs/r3_upstream_provenance.json` for source identities, classifications and
  the baseline decision;
- `docs/r3_evidence/upstream-whole-tree.diff` or a binary-safe equivalent;
- `docs/r3_evidence/upstream-execution-surface.json` for the dependency closure
  and per-path hashes; and
- focused tests that reject a moved official revision, changed local tree,
  incomplete classification, broad/unhashed allowlist or inconsistent decision.

Every tracked-tree difference is represented, every execution-surface
difference is resolved, artifact hashes reproduce, and the roadmap records the
chosen baseline outcome. The subsequent generator protocol must bind the
official commit/tree, comparison digest and separate pristine route. Completion
proves provenance only; it does not prove numerical parity or scientific
efficacy.
