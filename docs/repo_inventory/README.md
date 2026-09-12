# Repository inventory

This directory documents the repository as it exists in the inspected working tree. It is an inventory and architecture map, not a cleanup or repair plan. The detailed documents were produced by separate scoped reviews; each includes a first-pass review of obvious bugs, stale state, reproducibility gaps, or missing tests in its assigned area.

## At-a-glance map

| Area | What it contains | Detailed document |
|---|---|---|
| Root workflow and examples | Root Python helpers, regional prompting script, schemas, launchers, dependency metadata, examples, and repository-level configuration | [`root-core.md`](root-core.md) |
| Image/prompt dataset production | Remote scene generation, Flux reference images, Grounding DINO detection, SAM2 masks, LaMa inpainting, processed JSON, and generated image assets | [`image-prompt-generation.md`](image-prompt-generation.md) |
| Multi-sample generation | Parameter-grid expansion, task scheduling, distributed/FSDP workers, mask construction, cluster launchers, and dated generation outputs | [`multi-sample-inference.md`](multi-sample-inference.md) |
| Evaluation | SAM2 video cropping, Qwen-VL auditors, score aggregation, VideoBench adapter/reference code, batch/runtime scripts, and evaluation artifacts | [`evaluation.md`](evaluation.md) |
| Inversion, mask inspection, and gallery | Wan latent inversion, regional mask visualization, partial noising, the empty soft-mask area, and the NiceGUI video gallery | [`media-experiments-and-gallery.md`](media-experiments-and-gallery.md) |
| Containers and baselines | Root Apptainer definition, nested Apptainer-image repository, build scripts/CI, and the interactive baseline-comparison notebook | [`infrastructure-and-baselines.md`](infrastructure-and-baselines.md) |
| LaMa dependency | Nested LaMa repository, `saicinpainting` layers, inference/training/evaluation pathways, configs, Docker routes, and parent integration | [`lama-submodule.md`](lama-submodule.md) |
| Wan2.1 dependency | Pinned and initialized source checkout, expected package role, author-scoped implementation inventory, and reproducibility impact | [`wan2.1/README.md`](wan2.1/README.md); [`wan2.1-submodule.md`](wan2.1-submodule.md) |
| Paper and loose research directories | LaTeX draft, bibliography and paper notes, plus the current state of empty `old/`, `plans/`, `preprocessing/`, `prompts/`, `scratch/`, and `weights/` directories | [`paper-and-experiments.md`](paper-and-experiments.md) |

## Detailed table of contents

### 1. Root contracts and the main regional-prompting path

Start with [`root-core.md`](root-core.md) for the repository's top-level shape and the effective root experiment:

- `regional_prompting.py` is the main root-level experimental workflow. It loads the custom Wan I2V implementation, prepares example-image bounding boxes and masks, constructs looking/interaction schedules, generates video, and writes debug media.
- `main.py` is only a smoke/demo placeholder; it does not launch generation.
- `schema.py` defines typed prompt/video dataset contracts used by downstream generation, while the root regional-prompting examples use a separate `base_prompt`/`characters`/`wlw` JSON shape.
- `utils.py` and `debug_utils.py` provide mask, box, tensor-layout, resizing, and video-visualization helpers.
- `run.sh` is the container-oriented operational wrapper; `download_weights.sh` is the Hugging Face checkpoint helper.
- `examples/` contains four hand-authored regional-prompting cases, but the script currently hardcodes one of them rather than exposing an example-selection CLI.
- The root dependency and workspace configuration is GPU- and platform-sensitive, with editable `wan2.1` and `lama` members.

The root review identifies likely mask/debug shape errors, import-time 14B model loading, hard-coded paths and sampling parameters, no seed/CLI, helper edge cases, schema drift, and downloader/runtime validation gaps.

### 2. Prompt and reference-image dataset production

See [`image-prompt-generation.md`](image-prompt-generation.md) for the full production chain:

1. `generate_scenes.py` requests scene/prompt data from the Zhipu API using `metaprompt.md`, incrementally updates `final_dataset.json`, and performs remote audit passes.
2. `generate_prompt_image.py` reads the current top-level JSON dataset, generates Flux reference images, detects subjects with Grounding DINO, retries failed detections, and writes debug boxes.
3. It runs SAM2 and LaMa to create single-character images, writes rectangular segmentation-mask files and enlarged boxes, then serializes `prompts_modified.json`.
4. `big_lama.py` is the local adapter for a checkpoint under `weights/big-lama`.
5. `images/` currently contains 197 numbered asset directories; backups contain alternate ten- and 200-scene datasets.

The detailed review records the current 197-item versus 200-target discrepancy, unvalidated remote JSON, potentially unbounded refill behavior, unsafe audit IDs, stale backup schemas/paths, hard-coded model and output settings, ambiguous serialized mask semantics, missing LaMa weights, and the absence of failure manifests or tests.

### 3. Multi-sample inference and experiment output

See [`multi-sample-inference.md`](multi-sample-inference.md):

- `multi_sample_inference.py` decodes processed prompt records, expands parameter grids and prompt variants, prepares masks and task pickles, creates output/config folders, and schedules local GPU teams.
- `fsdp_worker.py` launches the custom Wan I2V model under `torchrun`, initializes distributed/FSDP execution, builds regional prompt metadata, generates 81-frame 480x832 videos, and lets rank 0 export videos and attention-mask visualizations.
- Shell and batch launchers cover local container execution, Slurm, and OAR/Abaca environments.
- The historical parameter files are currently deleted in the working tree, although their committed contents were inspected. The current local `param_configs/` directory therefore has no usable JSON grid.
- Date-named output roots contain substantial ignored experiment artifacts and configuration snapshots; they are not tracked source or complete reproducibility fixtures.

Important current risks include invalid/deleted parameter inputs, launcher/CLI drift, ignored checkpoint-path settings, non-pinned model downloads, questionable boolean conversion of smoothed “soft” masks, lost failed tasks, signal/requeue mismatch, short output-folder hashes, site-specific launch assumptions, and no scheduler/mask/distributed failure tests.

### 4. Evaluation and scoring

See [`evaluation.md`](evaluation.md):

- `evaluation.main` discovers generated-video directories, reads action/bounding-box configuration, runs SAM2 to make fixed object crops, and aggregates auditor scores.
- `evaluation.engine` supplies Qwen-VL loading, video/text prompting, generation, and soft-logit scoring.
- `evaluation.auditor` contains direct, blind, pairwise, and discrete scoring variants.
- `evaluation.videobench` is a partially integrated multi-turn evaluator: the local Qwen adapter is present, while an OpenAI-oriented `original.py` and an alternate engine remain stale/reference paths.
- `run.sh`, `job.oar`, logs, crop media, and input/output directories document a GPU/remote-storage batch workflow.

The review highlights exact-two-object assumptions, uncaught malformed/empty-input failures, repeated SAM2 work and overwrites, frame-rate/geometry assumptions, hard-coded CUDA/bfloat16 settings, score/bootstrap edge cases, brittle parsing, commented-out VideoBench registration, stale reference code, and the absence of fixtures, persisted result files, and tests.

### 5. Inversion, mask visualization, and gallery UI

See [`media-experiments-and-gallery.md`](media-experiments-and-gallery.md):

- `inversion/inversion_test.py` is a monolithic Wan latent ODE inversion/reconstruction experiment with T2V/I2V branches and MP4 outputs.
- `mask_visualization/mask_visualization_tests.py` runs regional I2V generation and saves a large NPZ of video/similarity-mask data before writing outlined-mask videos.
- `partially_noised_video.py` encodes a video, adds scheduler noise, and saves a latent tensor artifact.
- `soft_mask_test/` is currently empty.
- `video_gallery/` is a NiceGUI application that discovers `config.json` plus `video_<n>.mp4` files under `../multi_sample_inference/output/`, stores scores, renders a gallery, and builds a score table.

The configured gallery output tree is empty in this checkout, and the existing inversion/mask artifacts are not automatically connected to it. The detailed review also records likely inversion-direction and tensor-layout issues, absent provenance/manifests, CWD-relative paths, unstable gallery IDs, folder/URL collisions, nested prompt rendering mismatch, and missing media/gallery tests.

### 6. Containers and baseline comparisons

See [`infrastructure-and-baselines.md`](infrastructure-and-baselines.md):

- `containers/cuda_ubuntu.def` builds the root CUDA runtime with UV, the editable workspace packages, compiler/tool dependencies, and frozen Python dependencies.
- `containers/build_image.sh` is a host-specific Apptainer wrapper with a hard-coded cache path.
- `apptainer_images/` is a separate untracked nested repository with an `arch_ml` build helper and GitLab CI, but its referenced definitions/SIFs are missing or staged-as-deleted in the current checkout.
- `baseline_comparison/baselines_comparison.ipynb` interactively generates nine qualitative Wan baseline cases from five local images; the `.bak` notebook is a materially different historical run/configuration.

The infrastructure review identifies floating image/package bases, user- and site-specific paths, missing nested build inputs, absent checkpoints and result persistence, lack of build/notebook smoke tests, and active-versus-backup sampling-step drift.

### 7. LaMa image-inpainting dependency

See [`lama-submodule.md`](lama-submodule.md):

- The nested repository is checked out cleanly at the parent gitlink commit, detached from its local branch, and contains the full `saicinpainting` package plus configs, scripts, models, Docker files, and notebook.
- Its inference entry point is `bin/predict.py`; training and evaluation are Hydra/Lightning-based, with historical virtualenv/Conda/Docker pathways.
- The parent only uses the `big_lama.py` adapter and the `weights/big-lama` checkpoint path from image-prompt generation.
- No parent LaMa checkpoint or benchmark data is present, and the nested project retains legacy dependency/configuration assumptions.

The detailed review calls out the missing default training config, stale Docker and location documentation, legacy dependency drift against the parent UV environment, narrow package discovery, a broken secondary evaluation path, possible non-modulo parent inputs, and no parent-level integration smoke test.

### 8. Wan2.1 model submodule

See the [author-scoped Wan2.1 inventory](wan2.1/README.md) and the [submodule metadata report](wan2.1-submodule.md):

- The parent has a gitlink pinned to `f7472d3`, and the submodule worktree is now initialized at that exact detached commit.
- The author-scoped swarm reviewed 16 active files touched by `gzappavi <gianluca.zappavigna@inria.fr>`; the owner explicitly excluded the abandoned `wan/finetune/image2video.py` path. It produced nine focused reports covering package exports, sampling paths, regional prompting, models, encoders, VAE, and mask utilities.
- The source tree is now locally available for imports/build attempts, but weights remain external and the static review found internal API, shape, packaging, and CUDA/reproducibility risks.
- The old `wan2.1-submodule.md` report records the pre-initialization state; it should be read as historical evidence rather than the current worktree status.

### 9. Paper and loose experimental directories

See [`paper-and-experiments.md`](paper-and-experiments.md):

- `paper/main.tex` is an incomplete draft describing inference-time compositional control for Wan2.1, with LaMa/SAM2/self-attention/cross-attention ideas and planned experiments but no completed results tables.
- `old_notes.tex` preserves an earlier attention-masking formulation; local Markdown paper notes cover Concept Weaver and MultiTalk.
- `references.bib` contains both used and unused entries; there are no local PDFs or figures for the paper notes.
- `old/`, `plans/`, `preprocessing/`, `prompts/`, `scratch/`, and `weights/` are empty at inspection time. `old/` and `weights/` are ignored; the other empty directories have no durable Git representation.

The paper review emphasizes the empty root README, incomplete manuscript/TODOs, absent datasets/checkpoints/results/build instructions, remote/non-self-contained figures, and the difference between ignored local experiment state and tracked provenance.

## End-to-end conceptual flow

The repository's pieces suggest the following intended lifecycle, although the complete lifecycle is not currently runnable from a clean checkout:

```text
metaprompt + remote scene generation
        |
        v
base prompt dataset (`final_dataset.json`)
        |
        v
Flux reference images + Grounding DINO boxes
        |
        v
SAM2 masks + LaMa single-character images
        |
        v
processed prompt dataset (`prompts_modified.json`)
        |
        v
multi-sample parameter-grid scheduling -> Wan2.1 regional I2V videos
        |
        +--------------------+
        |                    |
        v                    v
video gallery          SAM2/Qwen-VL evaluation
        |                    |
        +-----------> paper/experiment analysis
```

Separate but related branches are the root `regional_prompting.py` demo, baseline notebook, latent inversion experiment, and mask-visualization scripts. They reuse the same Wan checkpoints and custom Wan implementation but are not unified behind one CLI or manifest format.

## Cross-cutting current-state findings

These findings recur across the detailed inventories:

1. **The principal model source is now initialized but not yet runtime-validated.** `wan2.1/` is checked out at its pinned gitlink commit, so the earlier missing-worktree blocker is removed. The author-scoped static reports still identify likely import, API, shape, CUDA, dependency, and checkpoint-provenance blockers that require focused runtime validation.
2. **Model artifacts are external and absent.** `weights/` is empty/ignored, while root and subprojects expect Wan checkpoints and `weights/big-lama`; SAM2, Qwen, Flux, DINO, and LaMa assets are also downloaded or provisioned externally.
3. **The root README provides no operational guidance.** The detailed inventories therefore rely on source/config inspection and identify many CWD-relative, hard-coded, host-specific, or cluster-specific invocation assumptions.
4. **Version-control state is not a clean release snapshot.** Four tracked multi-sample parameter files are deleted locally; several directories/files are untracked, including paper material, mask artifacts, evaluation additions, and the nested Apptainer repository. Generated media and weights are intentionally or incidentally ignored.
5. **Configuration contracts are fragmented.** Root `schema.py`, regional example JSON, processed prompt JSON, multi-sample parameter grids, evaluation config JSON, and gallery discovery each define related but non-identical structures. There is no single manifest or adapter layer covering generation, evaluation, and gallery consumption.
6. **Reproducibility is weak.** Important paths, model IDs, devices, resolutions, frame counts, prompts, scheduler settings, and cluster assumptions are often hard-coded; model revisions and generated-run metadata are not consistently pinned or persisted.
7. **Validation and testing are sparse.** The scoped reviews found syntax/static checks and experiment artifacts, but no comprehensive unit/integration suite, clean-checkout smoke path, model-free fixture suite, or durable evaluation-result schema.
8. **There are stale or incomplete parallel paths.** The VideoBench reference implementation, alternate evaluation engine, notebook backups, LaMa legacy docs, empty soft-mask area, paper TODOs, and historical parameter grids all preserve useful context but should not be assumed to be current runnable interfaces.

## What was and was not changed

- Added documentation files under `docs/repo_inventory/`, including the author-scoped `docs/repo_inventory/wan2.1/` inventory.
- Initialized the existing `wan2.1` submodule at the parent-pinned commit; no source files inside the submodule were modified.
- No model weights, datasets, generated media, or pre-existing working-tree changes were modified. No container was built; no model inference, notebook execution, or web server was run.
- Checks performed by the scoped agents were non-mutating and mainly included syntax/static inspection, directory and artifact inventories, JSON/notebook parsing, shell syntax checks, and Git/status inspection.

## Decision on merging the scoped documents

The original nine parent-repository reports remain separate references. The nine new Wan2.1 author-scoped reports likewise retain non-overlapping primary ownership and are more useful as focused references than as a merged monolith. The cross-links and end-to-end sections provide navigation without duplicating long architecture and bug-review sections.
