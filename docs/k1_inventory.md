# K1 repository disposition inventory

## Scope and decision rule

This inventory was regenerated from parent revision `962c2b13a4db784ec0262ca64ff014e136c9dd7f` on 2026-09-14, before the K1 candidate mutations. That revision has 33 tracked top-level families and 116 direct project requirements. `docs/repo_inventory/` was used only as historical context; current paths, imports, launchers, metadata, submodule revisions and local ignored-family aggregates were checked independently.

K1 reduces supported operational ambiguity without destroying uncertain provenance. “Retain—historical/compatibility” means the bytes remain at their existing path but the path is noncanonical. It does not mean the path was moved to an archive or validated for current execution. A later move or deletion requires caller migration, historical-output attribution and recovery review first.

## Tracked top-level families (33/33)

Counts are parent-index entries at the base revision; each submodule gitlink counts as one.

| Family (count) | Disposition | Current caller or role | Status and migration condition |
|---|---|---|---|
| `.gitattributes` (1) | Retain—active | Applies Git LFS to the 985 tracked prompt-image PNG paths. | Keep while the asset bundle remains; changing it requires fresh-clone/LFS recovery validation. |
| `.gitignore` (1) | Retain—active | Separates credentials, checkpoints, environments, containers and generated media/tensors from tracked source. | Ignored does not mean disposable; update only with an evidence audit. |
| `.gitmodules` (1) | Retain—active | Declares `wan2.1` and `lama`; `pyproject.toml` uses both as workspace sources. | No submodule metadata changed in K1. |
| `.python-version` (1) | Retain—active | Selects Python 3.11, matching project metadata. | Canonical environment metadata. |
| `.vscode/` (1) | Replace/migrate callers | The existing direct legacy dispatcher launch remains behaviorally unchanged but is labeled historical/noncanonical; new tasks call M1 through `uv`. | Remove the legacy launch only after scheduler/operator and output-consumer cutover. |
| `AGENTS.md` (1) | Retain—active | Repository workflow, validation and commit policy. | Authoritative contributor guidance. |
| `README.md` (1) | Replace/migrate (this batch) | `pyproject.toml` packaging metadata points here. | Empty placeholder replaced by canonical M1 operator and claim-boundary guidance. |
| `baseline_comparison/` (7) | Retain—historical/compatibility | Notebook, materially distinct backup and unique input images; no canonical M1 caller. | Preserve embedded run evidence; move only after checkpoint/run provenance and publication rights are resolved. |
| `containers/` (2) | Retain—compatibility | Builder consumes the definition; root `run.sh` consumes the generated SIF. The image copies both submodules and installs the root lock. | Site-sensitive, not a validated canonical R3 environment. Migrate only with all five generation/evaluation launcher callers. |
| `debug_utils.py` (1) | Retain—active | Imported by `fsdp_worker.py`, `regional_prompting.py` and mask visualization. | Active worker dependency; refactor only with tests and caller migration. |
| `docs/` (33) | Retain—active/evidence | Roadmap, audits, checkpoint record, U1 protocol/evidence and historical inventory. | P0/U1 evidence is immutable; historical path mentions are intentional provenance. |
| `download_weights.sh` (1) | Retain—compatibility | Manual helper for the legacy local checkpoint layout; M1 itself never downloads checkpoints. | Unpinned and unverified, therefore noncanonical. Replace only with manifest-aware provisioning and caller migration. |
| `evaluation/` (16) | Retain—historical/compatibility | OAR/shell launch `evaluation.main`; H1 permits the relative VLM evaluator only as historical/secondary comparison. | Not the H1 primary endpoint. Attribute ignored videos/logs and migrate launchers before moving source. |
| `examples/` (10) | Retain—historical/compatibility | `regional_prompting.py` hard-codes `dogs_no_interaction`; other examples are manual research inputs. | Preserve uncertain demo provenance and permissions; not M1 production sources. |
| `image_prompt_generation/` (995) | Retain—compatibility/evidence | Producer chain and `schema.py` create the JSON/LFS asset bundle consumed by the legacy dispatcher and launch config. | Preserve the 197-record bundle atomically. Production M1 migration requires an approved content-addressed source and asset/recovery review. |
| `inversion/` (1) | Retain—historical/compatibility | Standalone Diffusers inversion experiment with no canonical caller. | Move only after external input/output provenance is inventoried. |
| `lama` (1) | Retain—workspace submodule | `big_lama.py` imports it for historical image preparation; container and root workspace include it. | Pinned at `469acc7358a1c6828b647b4ee20c93474a2f36b4`; remove only with producer, lock, workspace and container migration. |
| `main.py` (1) | Delete (this batch) | Six-line greeting scaffold; no package entry, caller, experiment behavior or unique evidence. | Sole deletion; recovery details are in `k1_deletion_manifest.md`. Nested `main.py` files are unrelated and retained. |
| `mask_visualization/` (3) | Retain—historical/compatibility | Manual diagnostics import legacy inference/Wan helpers; ignored tensor artifacts are cited by the checkpoint record. | Not a focused test family. Content-level recovery review is required before any move. |
| `multi_sample_inference/` (20) | Retain—mixed active/compatibility | Canonical M1/U1 modules coexist with legacy dispatcher, configs and shell/OAR/Slurm launchers. | Canonical: `experiment_pipeline -> manifest_adapter -> fsdp_worker -> generation_routes`. Legacy files stay noncanonical until operator/output consumers migrate. |
| `paper/` (7) | Retain—historical/evidence | Incomplete manuscript, notes, bibliography and references; possible D4 provenance. | Not validated results; move only after citation/license and evidence-link review. |
| `plans/` (1) | Retain—historical/evidence | Unapproved tracker-improvement note potentially relevant only to conditional D1. | Must not be mistaken for a frozen protocol. |
| `prompts/` (4) | Retain—historical/evidence | Human-authored evaluation/leakage/scene/resume notes. | No canonical caller; uncertain D4 and protocol provenance prevents deletion. |
| `pyproject.toml` (1) | Retain—active | Root package, direct requirements, tool configuration and both workspace members. | Unchanged in K1; `uv` and this file are authoritative with `uv.lock`. |
| `regional_prompting.py` (1) | Retain—historical/compatibility | Manual hard-coded Wan experiment using examples, root helpers and local weights. | Noncanonical; migrate evidence/callers before archival or deletion. |
| `run.sh` (1) | Retain—compatibility | Container gateway called by three legacy generation and two evaluation launcher surfaces. | Site-specific and not canonical M1 guidance; migrate all callers together. |
| `schema.py` (1) | Retain—compatibility | Typed legacy prompt/asset contract imported by image preparation and the legacy dispatcher. | Required to interpret the preserved bundle; move only with both callers and old-data compatibility. |
| `scratch/` (1) | Retain—historical/evidence | Interactive tokenizer/subsequence investigation; no canonical test collection. | Preserve research provenance; do not advertise as a test. |
| `tests/` (14) | Retain—active | Five focused CPU suites plus explicitly non-scientific smoke fixtures. | K1 acceptance surface; smoke data cannot be used for generation/scientific claims. |
| `utils.py` (1) | Retain—active | Imported by active `fsdp_worker.py` and several compatibility paths. | Keep until active helper migration has focused coverage. |
| `uv.lock` (1) | Retain—active | Exact project resolution used by `uv`. | SHA-256 is recorded in validation; unchanged in K1. |
| `video_gallery/` (4) | Retain—historical/compatibility | Old NiceGUI scalar-score interface reads the legacy output layout. | It is unblinded and not H0-ready; replace only with private mappings, actor labels, adjudication and consistency checks. |
| `wan2.1` (1) | Retain—critical workspace submodule | M1 worker, U1 runner and repair tests consume upstream/custom Wan. | Pinned at `9f52e9abceb49c5cbf4a7f435a84ef0621c0029f`; required for R3/C1/P1. |

After deleting root `main.py`, 32 of these top-level families remain. No family was moved.

## Executable entry-point families

| Entry family | Disposition and boundary |
|---|---|
| `python -m multi_sample_inference.experiment_pipeline` (`validate`, `dry-run`, `expand`, `status`, `run`, annotation import/export) | Canonical M1 surface. `run` requires an explicit selected job; smoke fixtures are rejected for generation. Annotation interchange is unblinded and is not H0. |
| `python -m multi_sample_inference.manifest_adapter` and `multi_sample_inference.fsdp_worker` | Retained internal M1/R3 chain; normally invoked by the pipeline, not used as an alternative scheduler. |
| `python -m multi_sample_inference.u1_parity_runner` | Retained evidence-reproduction tool for bounded U1; not full-generator parity. |
| `python -m multi_sample_inference.multi_sample_inference` plus `run_inference.sh`, `submit_job.sh`, `job.oar`, `job.slurm` | Retained historical/compatibility family. It bypasses immutable M1 source/manifests and remains noncanonical until scheduler/operator and output consumers migrate. |
| `.vscode/tasks.json` | Canonical `uv`-backed validate/dry-run tasks only; none launches generation. |
| `.vscode/launch.json` | Existing direct dispatcher launch retained and labeled historical/noncanonical; attach configuration retained. |
| `run.sh`, `containers/build_image.sh`, `download_weights.sh` | Retained site/checkpoint compatibility operations; not validated canonical GPU setup. |
| `evaluation.main` plus `evaluation/run.sh`/`job.oar` | Historical secondary evaluator, never the H1 primary endpoint. |
| Image-generation modules | Retained source-asset provenance; remote/model-dependent and not an immutable production-source builder. |
| `regional_prompting.py`, `video_gallery/main.py`, inversion, mask-visualization and scratch scripts | Retained manual historical tools, not active M1 entry points. |
| Submodule CLIs | Retained inside pinned gitlinks; most are not parent entry points and were not independently classified as dead code. |
| Root `main.py` | Deleted greeting-only scaffold; no caller migration required. |

Executable-bit inspection found tracked executables at `containers/build_image.sh`, `download_weights.sh`, both evaluation launchers, `multi_sample_inference/job.oar`, `multi_sample_inference/run_inference.sh`, and root `run.sh`. Other launcher files can still be invoked explicitly by their interpreter; executable mode is not evidence that a path is canonical.

## Configuration families

| Family | Disposition |
|---|---|
| `tests/fixtures/smoke_experiment.json` plus fixture assets/inventory | Canonical NON-SCIENTIFIC validation fixture; retain. |
| Production M1 source | Intentionally external and explicit. No tracked scientific production source exists; it must meet the README/plan contract. |
| `multi_sample_inference/param_configs/*.json` | Historical legacy dispatcher inputs. `all.json` is currently malformed (`JSONDecodeError: Extra data: line 5 column 4`) and is preserved byte-for-byte as non-runnable evidence, not silently repaired. |
| `image_prompt_generation/{final_dataset,prompts_modified}.json` and backups | Retained source/processed/alternate research data. The processed JSON and 985 referenced LFS paths are an indivisible compatibility bundle. |
| `examples/*/config.json` | Historical regional-prompting inputs; not M1 manifests. |
| Evaluation/prompt text families | Historical evaluator and research-method inputs; no primary-endpoint status. |
| `containers/cuda_ubuntu.def`, `pyproject.toml`, `uv.lock` | Retained environment families; only the root project/lock through `uv` is authoritative. Container success is unverified under K1. |
| Git/VS Code/Python metadata | Retained or migrated as described above; VS Code canonical tasks invoke `uv` directly. |
| U1 checkpoint inventory/evidence JSON | Immutable completed-task evidence; retained and not rewritten. |
| Submodule configuration | Retained within pinned gitlinks; parent cleanup does not selectively alter it. |

## Direct project requirements (116/116)

All 116 root `[project].dependencies` entries are retained. Absence of a parent AST import is not proof of removability: the two workspace packages, dynamic model loading, CLI/backend selection, notebooks, and historical compatibility routes also consume the environment. These categorized lists cover each declaration individually; versions remain as locked. None is described as unused.

### Canonical/model runtime and retained root-helper closure (24)

Directly used by M1/U1/root helpers, declared by retained Wan, or part of checkpoint/model execution and serialization:

- `accelerate`, `dashscope`, `diffusers`, `easydict`, `einops`, `filelock`
- `flash-attn`, `ftfy`, `imageio`, `imageio-ffmpeg`, `numpy`, `opencv-python-headless`
- `packaging`, `pillow`, `safetensors`, `sentencepiece`, `tokenizers`, `torch`
- `torchvision`, `tqdm`, `transformers`, `triton`, `typing-extensions`, `wan`

This category does not claim that CPU validation imports every package. Wan explicitly declares several of them and full use remains an R3 concern.

### Retained submodule, source-production, evaluation, gallery or notebook compatibility (26)

These have a tracked compatibility consumer, support a retained submodule/tool family, or remain an unresolved named backend for those preserved workflows:

- `bitsandbytes`, `datasets`, `eva-decord`, `gradio`, `gradio-client`, `great-tables`
- `huggingface-hub`, `ipdb`, `ipykernel`, `ipywidgets`, `lama-inpainting`, `maestro`
- `msgspec`, `nicegui`, `omegaconf`, `openai`, `pandas`, `peft`
- `pyyaml`, `qwen-vl-utils`, `rich`, `sam2`, `scikit-image`, `scikit-learn`
- `supervision`, `torchcodec`

Examples of concrete seams are LaMa image preparation, schema decoding, SAM2/Qwen historical evaluation, the NiceGUI/Great Tables gallery, scene authoring, model/checkpoint acquisition and video decoding. A package may serve only a preserved compatibility route; that is a retention rationale, not a readiness claim.

### Retained locked support closure pending package-by-package pruning proof (66)

These direct declarations primarily support the retained networking/UI/model stack or are promoted transitive/build/runtime utilities. Their exact directness is unresolved. They remain to avoid silently changing the locked environment before a recursive workspace, dynamic-import, CLI and backend audit:

- `aiofiles`, `aiohappyeyeballs`, `aiohttp`, `aiosignal`, `annotated-types`, `anyio`
- `attrs`, `brotli`, `certifi`, `cffi`, `charset-normalizer`, `click`, `cryptography`
- `fastapi`, `ffmpy`, `frozenlist`, `fsspec`, `groovy`, `h11`, `hf-xet`
- `httpcore`, `httpx`, `idna`, `importlib-metadata`, `jinja2`, `markdown-it-py`
- `markupsafe`, `mdurl`, `more-itertools`, `mpmath`, `multidict`, `networkx`
- `orjson`, `propcache`, `protobuf`, `psutil`, `pycparser`, `pydantic`, `pydantic-core`
- `pydub`, `pygments`, `python-dateutil`, `python-multipart`, `pytz`, `regex`
- `requests`, `safehttpx`, `semantic-version`, `setuptools`, `shellingham`, `six`
- `sniffio`, `starlette`, `sympy`, `tomlkit`, `typer`, `typing-inspection`, `tzdata`
- `urllib3`, `uvicorn`, `wcwidth`, `websocket-client`, `websockets`, `widgetsnbextension`
- `yarl`, `zipp`

The active `mask_contracts.py` also imports SciPy, which is presently available only through the locked transitive graph rather than as a root direct declaration. K1 does not alter that graph; dependency-normalization requires a separately reviewed lock change.

### Development group (3)

- `ipykernel` and `jupyterlab`: retained notebook/development compatibility.
- `ruff`: retained focused lint gate.

`ipykernel` is intentionally present in both the runtime direct list and the development group; these are two metadata declarations, not two distinct package names.

## Submodules

| Path | Parent gitlink | Role and disposition |
|---|---|---|
| `wan2.1` | `9f52e9abceb49c5cbf4a7f435a84ef0621c0029f` | Critical M1/U1/R2/R3 implementation; retain unchanged. |
| `lama` | `469acc7358a1c6828b647b4ee20c93474a2f36b4` | Retained compatibility source-image dependency; remove only with producer/workspace/container migration. |

A fresh checkout must initialize both recursively. Their internal entry points and dependencies prevent parent-only import scans from authorizing requirement removal.

## Ignored and generated local material

The five evidence-bearing families were counted on 2026-09-14 at base revision `962c2b13a4db` by enumerating present regular files below each family, selecting files matched by `git check-ignore`, and summing logical sizes from `stat`. Counts include caches; they do not classify every file as research evidence.

| Family | Ignored files | Logical bytes | Conservative classification |
|---|---:|---:|---|
| `image_prompt_generation/` | 210 | 97,679,120 | Generated diagnostics and caches; provenance differs across files. Preserve pending content-level review. |
| `evaluation/` | 841 | 484,499,043 | Generated videos, logs and caches associated with the historical evaluator. Preserve pending run attribution. |
| `inversion/` | 3 | 793,962 | Generated media with unresolved provenance. Preserve. |
| `mask_visualization/` | 4 | 485,284,020 | Media and large tensor artifacts cited by checkpoint documentation. Preserve. |
| `multi_sample_inference/` | 3,730 | 2,599,615,584 | Outputs, configs/status material, media and caches; map to runs/manifests before mutation. |
| **Total** | **4,788** | **3,667,871,729** | **Zero ignored files moved or deleted in K1.** |

Other ignored categories are classified without exposing sensitive names or contents:

- project/test environments and tool caches: locally reproducible in principle, but untouched by this source batch;
- credentials and local service configuration: sensitive, excluded from documentation and never treated as cleanup evidence;
- model checkpoints/weights: external large inputs requiring approved inventory and recovery records;
- generated container images and build caches: machine-specific runtime material, not source;
- generated videos, masks, tensors, logs, status/result files and ad hoc media: possible research evidence until privately attributed and inventoried.

All 985 tracked prompt-image paths are Git LFS pointers in this working tree. `git lfs fsck` succeeded against locally present objects, which shows local object availability only; it does not prove remote recovery or redistribution rights. Aggregate preservation is likewise **not recovery**: no content hashes, off-machine backup, LFS remote availability, permissions or run attribution are established by this inventory. Before any future destructive operation, create a private content-level manifest and verify recovery without publishing credentials or sensitive paths.

## Intentional historical references

Historical inventories and evidence may continue to mention root `main.py` and other noncanonical paths. Those references explain prior tree state and are not active callers. They must not be rewritten merely to make a text search empty. The deletion audit distinguishes exact root-path execution references from same-basename files such as `evaluation/main.py` and `video_gallery/main.py`.
