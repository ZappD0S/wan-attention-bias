# K1 candidate validation record

## Candidate and scope

- Base revision: `962c2b13a4db784ec0262ca64ff014e136c9dd7f`
- Candidate identity before commit: base revision plus the current K1 working-tree diff
- Branch at preparation: `research/action-binding-repair`
- Validation status: **accepted** after bounded fresh-candidate checks and independent final review; two P2 wording corrections from review are incorporated

Until a task-scoped commit exists, this document does not assign a commit identity or claim clean-checkout acceptance.

## Preparation checks actually run

| Check | Result | Evidence |
|---|---|---|
| Base/top-level inventory | Pass | `git rev-parse HEAD` returned the base above; `git ls-tree --name-only HEAD` returned 33 top-level families. |
| Sole-deletion provenance | Pass | `HEAD:main.py` is blob `ace69f63f1bd494e84360742db334671bb966861`; pre-deletion SHA-256 was `33dff1ce146d3b1f9809b5c3ec97ea2531c9f36fa970efd0f71f0c5f7d2633a9`. |
| Dependency metadata count | Pass | A `uv run --no-sync` stdlib `tomllib` check found 116 project dependencies and three development-group declarations. This parsed metadata only; it did not import project packages. |
| Historical malformed config | Confirmed/retained | A `uv run --no-sync` stdlib JSON check reported `Extra data: line 5 column 4` for `multi_sample_inference/param_configs/all.json`. No legacy bytes were changed. |
| Submodule state | Pass with recovery limitation | `lama` was `469acc7358a1c6828b647b4ee20c93474a2f36b4`; `wan2.1` was `9f52e9abceb49c5cbf4a7f435a84ef0621c0029f`. The fresh worktree verified both exact gitlinks after the Wan commit was reconstructed from the clean local submodule; configured-remote retrieval failed as detailed below. |
| Locked metadata fingerprints | Recorded | Pre-batch SHA-256: `pyproject.toml` `37ccb3859ca658181198569d9d2ae5d4398d1a17d4a07721e5d340a1d02d63a6`; `uv.lock` `d31fb115afdeb800de705fd080316598f1ba7e9de98c42b67cd0fdcea5d8a361`; `.gitmodules` `8f17a4eee186d9aa194c134765b1d905d8ec9a94f4c602d0ca0897b16e5a2534`. |
| Ignored evidence-family aggregates | Pass for inventory only | Present ignored files/bytes were independently recounted as 210/97,679,120; 841/484,499,043; 3/793,962; 4/485,284,020; and 3,730/2,599,615,584 for the five families listed in `k1_inventory.md`. No content identity or recovery claim follows. |
| Git LFS local check | Pass with limitation | `git lfs ls-files` returned 985 and `git lfs fsck` returned OK. All 985 tracked image paths are pointers in the working tree. Remote recovery and rights were not checked. |

No project CPU test, lint, compilation, full sync, GPU operation or scientific measurement is reported by the preparation checks above.

## Bounded fresh-checkout validation

The parent created a detached worktree from the recorded base, applied the complete K1 candidate diff plus new files, and initialized LaMa at its expected gitlink. A standard recursive submodule update could not retrieve Wan commit `9f52e9ab…` because the configured remote did not advertise that custom commit. The parent then reconstructed Wan from the original clean local submodule repository and checked out the exact recorded gitlink. The validated parent source is fresh, but this procedure is not independent remote-recovery evidence; the existing P0 off-machine recovery limitation remains.

`uv sync --locked` failed solely while building `sam2==1.1.0`, where its build requested a CUDA extension and reported that `CUDA_HOME` was unset. The approved fallback then succeeded:

```bash
uv sync --locked --no-install-package sam2
```

All subsequent checks used `uv run --no-sync`; pytest was supplied with the plan-pinned `--with pytest==9.0.3`. Recorded versions were Python 3.11.15, torch 2.10.0+cu128, NumPy 1.26.4, SciPy 1.17.1 and Ruff 0.15.12. `sam2` and project-local pytest were not installed; pytest 9.0.3 ran in uv's transient `--with` environment.

Required bounded commands executed:

```bash
uv run --no-sync python -m multi_sample_inference.experiment_pipeline validate \
  --source tests/fixtures/smoke_experiment.json
uv run --no-sync python -m multi_sample_inference.experiment_pipeline dry-run \
  --source tests/fixtures/smoke_experiment.json --output-dir /tmp/wan-k1-fresh-smoke
uv run --no-sync --with pytest==9.0.3 python -m pytest -q \
  tests/test_repair_contracts.py tests/test_experiment_pipeline.py tests/test_mask_contracts.py \
  tests/test_generation_routes.py tests/test_parity_contracts.py
uv run --no-sync python -m compileall -q wan2.1/wan multi_sample_inference tests
uv run --no-sync ruff check multi_sample_inference tests
```

Results:

| Check | Result | Evidence |
|---|---|---|
| Full locked sync | Expected bounded failure | `sam2==1.1.0` build failed because `CUDA_HOME` was unset; no other package failure was reported. |
| Bounded locked sync | Pass | `uv sync --locked --no-install-package sam2` exited 0. |
| M1 fixture validation | Pass | Reported `valid: NON-SCIENTIFIC-smoke-only`. |
| M1 fixture dry-run | Pass | Reported 12 job IDs, created no output directory or MP4, and launched no generation. |
| Five focused CPU suites | Pass | **42 passed** under pytest 9.0.3: 2.81 s initially and 1.65 s in the final candidate rerun. |
| Syntax compilation | Pass | `compileall` exited 0 for `wan2.1/wan`, `multi_sample_inference` and `tests`. |
| Broad Ruff probe | Known pre-existing findings | `ruff check multi_sample_inference tests` reported 34 findings confined to `experiment_pipeline.py`, legacy `multi_sample_inference.py`, and `test_experiment_pipeline.py`; no surviving Python source was modified, and only the greeting-only root `main.py` was deleted in K1. |
| Established focused Ruff scope | Pass | Ruff exited 0 for the M1 worker/adapter/routes/contracts/U1 modules and their generation, mask, parity and repair tests. |

The passing focused Ruff scope was `multi_sample_inference/{__init__,fsdp_worker,generation_routes,manifest_adapter,mask_contracts,parity_contracts,task_contracts,u1_parity_runner,upstream_parity,utils}.py` and `tests/test_{generation_routes,mask_contracts,parity_contracts,repair_contracts}.py`. The broad probe is recorded rather than hidden; K1 did not refactor unchanged M1 or historical code to repair existing style debt.

This bounded result does not establish a complete project install, independent submodule recovery, SAM2 availability, CUDA compatibility, generator/FSDP readiness, runtime parity, video quality, blinding or efficacy. Any future non-SAM2 sync failure remains a blocker for the task using that environment; full R3 work still requires the complete locked environment.

## Source and policy checks actually run

| Check | Result | Evidence |
|---|---|---|
| VS Code parse and command audit | Pass | A `uv run --no-sync` stdlib check parsed both files as JSON, found three process tasks, and verified each invokes `uv run --locked ... multi_sample_inference.experiment_pipeline` with only `validate` or `dry-run`. The module and smoke fixture paths exist. |
| Sole-deletion/reference audit | Pass | Root `main.py` is absent; `evaluation/main.py` and `video_gallery/main.py` remain. No tracked Python/shell/OAR/Slurm/VS Code command invokes root `main.py`; historical inventory mentions were classified and retained. |
| Inventory/dependency coverage | Pass | A `uv run --no-sync` stdlib/TOML check matched all 33 base top-level names in order and matched 116 unique documented requirements exactly to `[project].dependencies`; all three development declarations are covered. |
| Change/deletion allowlist | Pass | `git status --short` listed only the README, plan, VS Code label/tasks, three K1 records and root deletion. A scoped diff found no changes in legacy source/config families; the malformed `all.json` blob still matches `HEAD`. |
| Lock/submodule integrity | Pass | `pyproject.toml`, `uv.lock` and `.gitmodules` retain the preparation SHA-256 values; both submodule status lines retain the expected revisions. |
| Whitespace/staging | Pass | `git diff --check` produced no output; `git diff --cached --name-only` was empty. |
| Independent final review | Pass with two P2 notes corrected | GPT-5.6 Sol found no blocker and returned `Merge verdict: OK with notes`; review artifact: `/home/gzappavi/.pi/agent/sessions/--home-gzappavi-Documents-wan_experiments--/subagent-artifacts/outputs/fd45cb05-2134-4c8b-b864-fea5f74eb478/k1-final-review-sol.md`. |
| Reference indexing | Pass with repository-scope limitation | `qmd embed` reported “All content hashes already have embeddings.” This repository is not configured as a qmd collection, so the result is not evidence that these documents entered an index. |

The fresh-candidate source/policy audit was rerun after provisioning: both VS Code files parsed, all three tasks invoked `uv` and only `validate`/`dry-run`, all 116 direct requirements and three development declarations remained covered, the candidate allowlist and whitespace check passed, and the index remained unstaged. Independent review accepted K1 with only the two corrected wording notes. Roadmap reconciliation and the task-scoped commit complete the workflow.

## Full environment boundary

Production M1 source/checkpoint materialization, full SAM2/CUDA installation, generator/CFG/FSDP/distributed calls and GPU validation remain R3 or later work. They are not bypassed or established by K1's bounded CPU protocol.
