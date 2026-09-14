# Action-binding experiment plan

## Status vocabulary and amendment rule

Task status is one of **PLANNED**, **IN PROGRESS**, or **COMPLETED**. A blocker is recorded separately and never converted into a conclusion. Evidence paths and results are added only after measurement. Amendments append to the log; they do not rewrite predeclared hypotheses, success criteria, or failed runs. Historical outputs are read-only.

## Task register

| ID | Status | Depends on | Blocker | Evidence / result | Conclusion |
|---|---|---|---|---|---|
| P0 checkpoint | COMPLETED | — | Off-machine recovery still incomplete | `docs/checkpoint_manifest.md`; parent checkpoint `f092679`; Wan original `f7472d354e0cb46b2853f15bf97b8b283d4780f8` | Existing source was preserved before repair; historical efficacy remains unresolved. |
| R1 correctness repair | COMPLETED | P0 | GPU/scientific integration is tracked separately as R3 | Repair diff; `tests/test_repair_contracts.py`; `docs/attention_method_audit.md` | Blocking source contracts repaired without intervention retuning; no efficacy conclusion. |
| R2 CPU contracts | COMPLETED | R1 | None | 14 attention/helper tests; final focused suite including M1: 22 passed, no warnings (parent rerun: 1.86 s); syntax, focused lint and diff checks passed | Narrow CPU math/contracts pass; GPU behavior remains unmeasured. |
| U1 upstream-Wan baseline adapter/parity | PLANNED | R2 | Full project environment and compatible local checkpoint | Baseline adapter plus identical-layer/custom-`none` parity records | Not implemented or run; required before cleanup can distinguish the retained baseline path from superseded variants. |
| M1 immutable manifest infrastructure | COMPLETED | R2 | GPU task materialization remains blocked on the full project environment, checkpoint and approved compute | `multi_sample_inference/experiment_pipeline.py`; `manifest_adapter.py`; 8 infrastructure/mask CPU tests | Deterministic manifests, content-inventoried checkpoints, hash-verified task descriptors, selected-job launch wiring, status/results separation and annotation interchange are executable; no model run or scientific result occurred. |
| K1 evidence-preserving repository cleanup | PLANNED | R2, U1, M1 | Complete path/dependency classification; uncertain provenance is retained or archived, not deleted | Reviewed retain/archive/delete inventory; deletion manifest; updated entry points/dependencies; clean-checkout CPU validation | Not started; must finish before R3 so GPU validation covers the reduced repository rather than obsolete paths. |
| R3 GPU contracts | PLANNED | R2, U1, K1 | Compatible checkpoint, CUDA/flash/flex environment and approved compute | Per-job manifests and logs, including tensor hashes | Not run; helper tests do not exercise full attention blocks or the generator. |
| H0 blinded annotation preparation | PLANNED | M1 | Historical clip provenance and a frozen labeling rubric | Opaque randomized clip IDs, private condition mapping, rater/adjudication records and endpoint consistency checks | Current metadata-bearing CSV is operator interchange, not a blinded rater interface. |
| H1 human calibration | PLANNED | H0 | Raters and 80–120 existing eligible clips | Annotation template/import infrastructure exists; no labels collected | Not run. |
| C1 action competence | PLANNED | H1, R3 | Approved GPU compute | Single-action and both-same matched outputs | Not run. |
| P1 128-video pilot | PLANNED | C1, U1 | Approved scenes/actions and compute | 8 scenes × 2 seeds × AB/BA × 4 conditions | Not run. |
| X1 held-out confirmation | PLANNED | P1 | Requires diagnostic signal and power analysis | Frozen analysis plan and held-out manifests | Not run. |
| D1 uncertainty-aware routing | PLANNED | P1 | Only if fixed succeeds and dynamic fails | Separate amended protocol | Not run. |
| D2 cross-model benchmark | PLANNED | X1 | Current literature/model suitability review | Versioned benchmark protocol | Not run. |
| D3 gaze project | PLANNED | P1 | Separate geometry/observability validation | Separate plan; never folded into repair | Not run. |
| D4 reproducible blog | PLANNED | H1 | Provenance and permissions | Full grids, failures and caveats | Not run. |

## R1–R3: repair and executable contracts

**Hypothesis.** Restoring schedule guards, consistent effective masks, fixed-mask execution, weighted regional gates, persistent entity mapping, early seed synchronization and validation removes known contract violations without retuning the intervention.

**Procedure.** Run syntax compilation and the isolated CPU helper tests. On the approved GPU environment, compare upstream and custom `none` at one layer with identical weights, latent/timestep, text/image contexts, lengths and settings; predeclare numerical tolerances appropriate to the same kernel. Exercise conditional and negative-CFG calls; fixed/hard/soft masks; every sharing mode; both solvers; one rank and the intended FSDP rank count. Record initial-latent hashes and used/generated mask hashes by rank.

**Success criterion.** CPU contracts pass. GPU calls are finite; all rank initial-latent hashes match for each job; disabled custom `none` agrees with upstream within the predeclared tolerance; each block's self/cross consumers and exported used-mask hash agree; generated masks remain separately identifiable. Fixed masks do not call tracker code. Invalid batch, overlaps, empty masks after resizing, unsupported mapping, over-limit tokens and invalid schedules fail before output writing.

**Stop criterion.** Any parity, finiteness, rank-hash or identity failure blocks generation experiments. Do not infer scientific efficacy from contract success. Current CPU tests exercise extracted helpers, not full attention blocks or the generation path; those layer/generator checks remain PLANNED under R3/U1.

The repair did not change `_generate_noise` relative to Wan checkpoint `f7472d354e0cb46b2853f15bf97b8b283d4780f8`: it still makes one channel-major `[16, T, H, W]` `torch.randn` draw. Resolving/broadcasting the integer seed now happens before that same draw and preserves the post-draw generator for the scheduler. No new frame-major/channel-major same-seed incompatibility is introduced by this diff; checkpoint-backed confirmation remains part of R3.

## M1: manifest and task interface

The small stdlib CLI in `multi_sample_inference.experiment_pipeline` now expands a declarative source JSON into one immutable, canonical JSON per job using exclusive creation, or verifies byte-equivalent canonical content on resume. It delegates selected jobs to the existing torchrun `fsdp_worker` through `manifest_adapter`; it does not duplicate video inference or implement a scheduler. A job includes:

- stable `job_id`, schema version and source record/scene/assignment/condition/seed IDs;
- parent and Wan commit IDs plus dirty-worktree fingerprints;
- explicit local checkpoint path, label, and approved JSON inventory containing every checkpoint file's relative path, byte size, and SHA-256; no credentials or remote resolution;
- content hashes for the source JSON, prompt payload, reference image, segmentation/bbox mask specification, isolated images and resolved configuration;
- resolved global actor/entity order, AB/BA target mapping, prompt representation, full sentences and character segments;
- explicit `diffusion_seed` (not the image-generation `seed`) and rank count;
- `inference_settings`: steps, `frame_num` (`4n+1`), width/height, shift, solver and CFG;
- intervention settings: method, fixed-vs-dynamic mask source, sharing, self routing, image-context isolation, beta/strength, and complete timestep/block schedules;
- declared video, artifact, task, status and result paths.

The repaired worker accepts required task field `diffusion_seed`, explicit inference settings and explicit per-step/per-block schedules. The CLI records parent/Wan revisions plus dirty fingerprints, checkpoint inventory/content identity, source/assets/masks/prompts/config hashes, global actor order and assignment, dimensions/frames/steps/CFG/solver/shift/ranks, method/isolation settings and declared output paths. Validation and each selected run re-hash the actual checkpoint files against the approved inventory; this intentionally favors content verification over an unchecked cache and can add substantial startup I/O for a large checkpoint. Output roots must be outside the source repository so expansion cannot invalidate its own dirty fingerprint. Runtime events are append-only JSONL; immutable results are separate. Resume skips only a completed result whose manifest and video hashes verify. Task metadata records and verifies the materialized pickle SHA-256 before reuse. An MP4 from a failed/incomplete job causes a hard refusal rather than a skip or overwrite. Failed task descriptors remain available for diagnosis.

`tests/fixtures/smoke_experiment.json` and its tiny PPM/PGM assets are explicitly **NON-SCIENTIFIC** CPU fixtures, not a curated dataset or usable checkpoint; `smoke_only` jobs are rejected by `run`. Production source JSON must replace every fixture asset and checkpoint inventory, keep actor order stable across AB/BA, and explicitly set `image_context_isolation=false` for this pilot. Concept-Weaver always requires split singleton prompts even when image isolation is off. Unsupported methods, mask/source combinations, layouts, dimensions, schedules and implicit mask resizing fail explicitly.

Dynamic-mask preprocessing is frozen as `legacy_boolean_gaussian_v1`: threshold to Boolean, remove overlap, run SciPy Gaussian filtering with Boolean input and explicit Boolean output, interpret nonzero as foreground, then remove overlap again. This records the historical ambiguity instead of silently choosing a new float threshold. Task metadata records the postprocessed combined/per-actor SHA-256 and per-actor true-pixel area. Any float-smoothing threshold variant requires an amendment; empty/area behavior in the full NumPy/SciPy worker environment remains a pending integration check.

## K1: evidence-preserving repository cleanup

**Priority and scope.** Treat cleanup as a high-priority engineering gate after U1 defines the retained upstream/custom-`none` baseline route and before R3 spends GPU compute validating the repository. Start from `docs/repo_inventory/`, but verify the current tree rather than treating the inventory as authoritative. Classify every tracked top-level path, submodule, executable entry point, configuration family and direct dependency against R3, M1, H0–H1, C1, P1, X1 and the conditional D1–D4 work. Classify ignored/generated local material separately so source cleanup cannot be confused with deletion of research evidence.

**Procedure.** Produce a reviewed inventory with one of four dispositions for each item: retain in the active path; move to a clearly labeled historical/reference archive; replace and migrate its callers; or delete as duplicated, unreachable, superseded or unrelated to the plan. Record the call sites, configs, documentation links and unique provenance supporting each decision before mutation. Remove stale parallel launchers, configs, implementations, dependencies and generated/debug residue only after retained callers and historical consumers have been accounted for. Apply removals in reviewable batches; update imports, CLIs, manifests, container/build files, tests and operator documentation in the same batch. Preserve P0 checkpoints, immutable manifests, audit documents, unique historical evidence and compatibility fields required to interpret old results. Cleanup must not retune an intervention or silently change an experimental contract.

**Success criterion.** Every remaining active path and direct dependency has a documented role in the plan, and every removal has a path-level rationale and provenance disposition. A fresh checkout with required submodules has no references to removed paths and passes manifest validation/dry-run, the focused CPU suite, syntax compilation and focused lint. R3 is then run from the cleaned revision so full-layer, generator, CUDA and distributed contracts validate the code that will actually be used.

**Stop criterion.** If an item's provenance, historical consumer or relevance to a retained/conditional task is uncertain, retain or archive it pending a specific decision rather than deleting it. Do not rewrite history, delete historical outputs in place, or describe a clean source tree as evidence of model correctness or efficacy.

## H1: observation calibration

**Prerequisite (H0).** The current CSV exposes `condition_id` and source paths: do not send it directly to blinded raters. Create randomized opaque clip IDs with a private mapping; preserve actor identity and requested action while concealing method. Add historical-clip ingestion, rater IDs, adjudication and consistency checks between actor labels and the joint endpoint before collecting primary labels. The importer currently validates labels and repeated clip fields, not these scientific requirements.

**Hypothesis.** Blinded humans can reliably distinguish requested and wrong actions per actor, while known VLM shortcuts can be quantified.

**Procedure.** Select approximately 80–120 existing clips without cherry-picking, balanced where provenance allows among both-correct, swapped, both-A, both-B, mixed and neither. Multiple raters label each actor's absolute target-action occurrence, wrong-action occurrence, identity, visibility, quality, uncertainty and failures; adjudicate disagreement. Include frozen-first-frame, correct/wrong verb, AB/BA option order, identical-description, neither/both, full-frame/crop and suitable temporal reversal/shuffle controls.

**Primary calibration output.** Agreement and error by category. Frozen videos scoring as motion are evaluator failures. Existing VLM relative comparisons and coin-null p-values are secondary diagnostics, not primary efficacy evidence. The historical evaluator remains available only for historical/secondary comparison because it measures relative description preference rather than absolute actor-level target and wrong-action presence; it is not the primary endpoint.

**Stop criterion.** If actor/action labels cannot reach acceptable predeclared agreement, revise the annotation unit and action set before generation.

## C1: model competence gates

**Hypothesis.** Each retained action is renderable alone and in a both-subjects-same-action setting.

**Procedure.** With matched seeds/settings, test each action alone and both-same before paired binding. Record absolute requested/wrong action by actor, JOINT both-correct, quality, uncertainty and all failures.

**Success criterion.** Predeclare a minimum action-level competence rate after calibration. Incompetent actions form a reported stratum; do not discard them post hoc.

**Stop criterion.** If the model lacks adequate competence on most actions, revise the action set rather than attribute paired failure to binding.

## P1: matched falsification pilot

**Design.** Freeze eight stationary, counterfactually feasible scenes before generation. Use exactly two explicit video diffusion seeds, both AB and BA assignments, and four conditions: `8 × 2 × 2 × 4 = 128` videos.

1. Fair vanilla Wan with an explicit joint assignment prompt.
2. Custom `none` using the same split representation as interventions (mechanism ablation).
3. Fixed, disjoint regional text control with self-attention routing and image isolation off.
4. Dynamic regional text control with the same settings and self-attention routing/image isolation off.

The fair vanilla prompt and split custom-none are deliberately distinct controls: one tests a fair task input; the other isolates the hook under the same representation. Do not describe them as numerically equivalent. Add image isolation and self-routing one at a time only after the four-condition diagnostic. Do not cherry-pick scenes/seeds, tune strengths/temperature, or sweep until significance.

**Primary outcomes.** Human absolute requested-action and wrong-action occurrence for each actor; **JOINT both actors correct**; visual quality; uncertainty; visibility/tracking/runtime failure denominators. Secondary outcomes may include independently calibrated automatic scores, never the old relative comparison or method-versus-coin-null p-value as primary efficacy.

**Success criterion.** A predeclared, uncertainty-qualified paired improvement in joint success without an unacceptable quality/failure penalty, plus interpretable fixed-versus-dynamic behavior. This pilot detects a diagnostic signal; it is not a powered publication claim.

**Stop/interpretation rules.** Isolated-action failure triggers action revision. A prompt-only win is prompting. Fixed success plus dynamic failure motivates localization work. Competent actions and functioning fixed control with no convincing human gain stop tracker refinement and strength sweeps. Automatic-only gain is evaluator failure.

## X1 and conditional successors

Only after a diagnostic signal, freeze controller and metric, select the smallest worthwhile joint-success effect (project example: +10 percentage points), and power a held-out comparison using independent scenes and scene-clustered paired analysis. A confidence interval crossing zero is inconclusive; an upper bound below the threshold rules out that gain on the tested domain. Equivalence needs a predeclared margin. No tuning on held-out data.

Uncertainty-aware routing (D1) is conditional on fixed rescue/dynamic failure. Cross-model benchmarking (D2) is conditional on a frozen validated task and current model/literature review. Gaze (D3) remains separate and must represent unobservable/offscreen states rather than false. A blog (D4) remains valid without efficacy if it shows full seed grids, failures and provenance honestly.

## Exact next commands

The isolated test environment used Python 3.12.3, CPU torch 2.14.0, pytest 9.1.1, NumPy 2.5.3, SciPy 1.18.1 and ruff 0.16.7. It is **not** the project's Python/CUDA generation environment and does not establish production compatibility. To recreate it separately:

```bash
uv venv --python 3.12 .venv-tests
uv pip install --python .venv-tests/bin/python \
  pytest==9.1.1 numpy==2.5.3 scipy==1.18.1 ruff==0.16.7
uv pip install --python .venv-tests/bin/python \
  torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
```

These commands exercise only CPU bookkeeping and tests; they do not start inference or claim indexing:

```bash
python -m multi_sample_inference.experiment_pipeline validate \
  --source tests/fixtures/smoke_experiment.json
python -m multi_sample_inference.experiment_pipeline dry-run \
  --source tests/fixtures/smoke_experiment.json --output-dir /tmp/wan-smoke-run
.venv-tests/bin/python -m pytest -q \
  tests/test_repair_contracts.py tests/test_experiment_pipeline.py tests/test_mask_contracts.py
python -m compileall -q wan2.1/wan multi_sample_inference tests
```

For a real protocol, first copy the smoke JSON structure to a new research source, replace all assets and the placeholder with an existing local checkpoint, set `smoke_only` false, use target-sized dynamic masks, and choose a new output directory outside this repository. Then, in the existing full project CUDA environment:

```bash
python -m multi_sample_inference.experiment_pipeline validate --source /path/protocol.json
python -m multi_sample_inference.experiment_pipeline expand \
  --source /path/protocol.json --output-dir /scratch/new-action-binding-run
python -m multi_sample_inference.experiment_pipeline status \
  --jobs /scratch/new-action-binding-run/jobs
python -m multi_sample_inference.experiment_pipeline run \
  --job /scratch/new-action-binding-run/jobs/JOB_ID.json --devices 0,1
python -m multi_sample_inference.experiment_pipeline annotations-export \
  --jobs /scratch/new-action-binding-run/jobs --output /scratch/new-action-binding-run/annotations.csv
# Operator interchange only: H0 must provide blinding before primary human labels.
# After operator reconciliation fills every actor row:
python -m multi_sample_inference.experiment_pipeline annotations-import \
  --template /scratch/new-action-binding-run/annotations.csv \
  --jobs /scratch/new-action-binding-run/jobs \
  --output /scratch/new-action-binding-run/annotations.json
```

`run` launches exactly the selected job and requires the declared number of visible devices. It does not download a checkpoint. The adapter additionally requires the project's existing torch/NumPy/SciPy/diffusers stack; CUDA/flash/flex, checkpoint-backed execution and multi-rank validation remain pending. The current worker route records custom `none`, not an upstream-Wan numerical-equivalence claim; an unknown `upstream` method is rejected. The CLI intentionally does not compute inferential statistics.

After reference-document changes, run `qmd embed` as required by the workspace instructions. This repository currently is not a qmd collection; success for existing collections is not evidence that these documents were indexed.

## Amendment log

- 2026-09-12 — Initial living plan created from the checkpoint assessment and repair contract. No efficacy conclusion entered.
- 2026-09-12 — R1 and R2 marked COMPLETED after 14 narrow CPU tests, syntax compilation, new-module lint and whitespace checks passed. One torch warning records that NumPy was absent from the isolated minimal environment. R3 and every scientific measurement remain PLANNED.
- 2026-09-12 — M1 marked COMPLETED after four infrastructure tests covered deterministic paired expansion and hashes, immutable resume, real worker-task schema compatibility, subprocess failure/resume, and annotation validation. No checkpoint-backed generation or human annotation was performed. Upstream-Wan execution remains outside the current worker route.
- 2026-09-12 — Review repairs reject joint Concept-Weaver at source and worker contracts, tie actual checkpoint files to an approved inventory, hash-check reused task pickles, and freeze/record legacy dynamic-mask preprocessing evidence. The focused suite now has 22 passing CPU tests. U1 was added as an explicit PLANNED pilot dependency; full layer/generator, adapter materialization, CUDA, and scientific generations remain PLANNED. Standalone NumPy/SciPy preprocessing tests now pass.
- 2026-09-12 — Independent final review found no remaining blocker in this bounded source/infrastructure slice; parent reran the focused suite (22 passed). Repository-wide pytest collection remains blocked by missing `diffusers` and `sync_batchnorm` in the isolated CPU environment. H0 was added after parent inspection confirmed that annotation interchange exposes condition metadata and does not yet enforce actor-to-joint scientific consistency. No human result or ready-to-run blinded study is claimed.
- 2026-09-13 — K1 was added as an evidence-preserving repository-cleanup gate after U1 and before R3. It requires explicit retain/archive/replace/delete decisions, protects provenance and historical outputs, and makes the cleaned revision—not obsolete parallel paths—the target of GPU contract validation.

Final review evidence: `/home/gzappavi/.pi/agent/sessions/--home-gzappavi-Documents-wan_experiments--/subagent-artifacts/outputs/ba6ee532-e9cd-4518-a833-920a1239b263/repair-sol/final-review.md`. Sibling `contracts.md`, `infrastructure.md`, `review.md` and `resolution.md` record implementation, findings and dispositions. All five children in this retry used Sol; no GPU jobs were launched.
