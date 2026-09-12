# `multi_sample_inference/` repository inventory

## Scope and status

This directory contains a dispatcher, a distributed generation worker, local and cluster launchers, parameter-grid snapshots, and a large set of ignored generated runs. At the time of inspection, the four tracked parameter files are deleted in the working tree (`git status` reports `D` for `all.json`, `massive_run.json`, `massive_run2.json`, and `only_concept_weaver.json`); their committed contents were inspected with `git show HEAD`. The working-tree `param_configs/` therefore has no JSON files. Date-named runs and `output/`/`abaca_output/` are ignored by the repository rules and are not tracked source.

Tracked source files:

- `__init__.py` is empty and makes the directory a package.
- `multi_sample_inference.py` is the dispatcher and CLI.
- `fsdp_worker.py` is the `torchrun` child process and output writer.
- `utils.py` provides the six-character URL-safe MD5-derived output-folder name.
- `run_inference.sh` is the local/container convenience launcher.
- `submit_job.sh` selects an architecture-specific Slurm allocation.
- `job.slurm` is the Slurm batch script.
- `job.oar` is the OAR/Abaca batch script.
- `param_configs/*.json` are tracked historical inputs, currently deleted locally (described below).

The root `pyproject.toml` is the applicable package metadata. It requires Python `>=3.11,<3.12`, uses a workspace editable `wan` dependency, pins `torch==2.10` and CUDA-128 PyTorch-family packages through a PyTorch index, and declares the inference stack including `diffusers`, `huggingface-hub`, `numpy`, `scipy` transitively through the environment, `scikit-learn`, `msgspec`, `tqdm`, OpenCV/Pillow-related packages, and `torchcodec`. The root `run.sh` requires Apptainer or Singularity, binds the repository as `/workspace`, uses a CUDA image by default (`containers/cuda_ubuntu.sif`), enables `--nv`, and starts commands with `/workspace` as the working directory. `wan2.1` is a git submodule (`.gitmodules`, URL `ZappD0S/Wan2.1.git`, branch `dev`); its checkout in this worktree is empty, so the `wan.regional_prompt` implementation and its package metadata are not locally inspectable here even though the code imports them.

## Architecture and process flow

The primary entry point is:

```text
python -m multi_sample_inference.multi_sample_inference \
  --prompts-file <processed dataset JSON> \
  --param-grid-file <JSON grid> \
  --output-path <directory>
```

`multi_sample_inference.py` decodes the prompt file as `VideoGenerationDataset[ProcessedVideoSpecification]` using the root `schema.py`. Each processed item is expected to contain action prompt variants, original/reference image and per-character image/mask paths, bounding boxes, enlarged bounding boxes, and a seed. The dispatcher loads the reference and single-character images with Diffusers, expands the JSON parameter grid with scikit-learn `ParameterGrid`, and creates one task for every parameter combination, permitted action-prompt type, and repeat index.

For each action variant, it joins each segment list into a prompt sentence, extracts character segments using the action mask, chooses `general_prompt` (or the hard-coded fallback `high quality video, background scenery`), and writes an action-folder `config.json`. The folder name is the first six characters of a URL-safe base64 MD5 of the complete configuration (sorted JSON keys). The actual task is serialized as a pickle under `<output>/.tasks/task_<UUID>.pkl`; the pickle carries images, masks, prompt data, parameter data, model path, output paths, and repeat number.

Mask preparation is done once per prompt/parameter combination. `simil_masks_type == "fixed"` rasterizes `enlarged_bboxes` at `(480, 832)` via root `utils.create_mask_from_bbox`; other values load the segmentation masks and threshold them at 128. Overlapping pixels are removed, a Gaussian filter with sigma 5 is applied, the result is converted to boolean Torch masks, and `invert` reverses character-mask order. The worker passes these masks as `face_masks`.

The dispatcher detects CUDA hardware and starts one Python thread per GPU team. Each thread sets `CUDA_VISIBLE_DEVICES`, drains the shared queue, and invokes:

```text
python -m torch.distributed.run --nproc_per_node=<team size> \
  --master_port=<29500 + team id> -m multi_sample_inference.fsdp_worker \
  --task-file <pickle> --mode solo|fsdp [--t5-cpu]
```

A team is a local multi-process distributed job, not a cross-node job. The worker requires `LOCAL_RANK`, selects that CUDA device, initializes an NCCL process group, unpickles the task, constructs `wan.regional_prompt.WanI2V` with the Wan I2V 14B 480P config, and enables DIT FSDP in `fsdp` mode. T5 is sharded only for FSDP jobs without `--t5-cpu`; solo mode still uses a one-process distributed group. Generation is fixed at 40 sampling steps, 81 frames, and 480x832 maximum area. It builds regional prompt metadata, a full timestep/block bias schedule, per-character face masks, and a frame-repeated identity `wlw_matrix`, then calls `WanI2V.generate`.

Only distributed rank 0 exports the generated tensor to `video_<repeat>.mp4` at 16 FPS and writes soft attention visualizations when `extra_data["simil_masks"]` exists. For each character it writes `video_<repeat>_soft_masks_char_<character>_step_0.mp4` and the corresponding `step_-1` file. Saving retries `FileNotFoundError` up to five times. Root `utils.normalize_video_tensor` converts channel-first `[-1, 1]` output to RGB `[0, 1]` frame order; root `debug_utils.py` upsamples attention masks and overlays them with OpenCV.

Hardware policy in `auto_configure_hardware()` is local-GPU VRAM based: at least 65 GB uses solo/T5 GPU; 35--64 GB uses two-GPU FSDP/T5 GPU teams; below 35 GB uses four-GPU FSDP teams where available, otherwise two-GPU FSDP/T5 CPU or solo/T5 CPU. A leftover-GPU fallback always creates solo workers and chooses CPU T5 below 65 GB. No GPU is a hard runtime error. `team_thread()` uses a separate master port per team, but there is no node-level coordination for simultaneous dispatcher processes.

## Configuration schema and precedence

The dispatcher has exactly three required CLI arguments; there are no CLI overrides for model, resolution, sampling steps, hardware mode, or checkpoint location. Effective precedence is:

1. Constants in the Python sources control resolution, frame count, sampling steps, model config, Gaussian sigma, and default general prompt.
2. The selected parameter-grid JSON controls combinations. Each grid object maps a parameter name to a list of values; `ParameterGrid` takes the Cartesian product. `repeat` is consumed by the dispatcher and defaults to 1. `prompt_types` filters action-suite keys and otherwise defaults to every decoded action key. The remaining parameter mapping is passed unchanged to `WanI2V.generate` through `bias_kwargs`.
3. Prompt data supplies image paths, masks/bounding boxes, action text, and general prompt. The selected action variant and generated `params` are persisted in each output folder's `config.json`.
4. Scheduler environment variables select only launcher behavior and paths. `NODE_ARCH` and `PARAM_CONFIG_NAME` are exported by `submit_job.sh`; the batch scripts derive the parameter-file path and output root. The Python process does not consume `CHECKPOINT_PATH`; it calls `snapshot_download("Wan-AI/Wan2.1-I2V-14B-480P", local_files_only=False)` every run and stores that resolved path in tasks.

The committed grids are historical and not all mutually compatible with current code:

- `all.json` is malformed as committed: it contains multiple top-level objects separated by commas without an enclosing array, so JSON decoding fails. Its entries also omit `simil_masks_type`, which the dispatcher indexes as a required key.
- `massive_run.json` contains two grid objects: concept-weaver/soft masks with `mask_sharing: null`, `self_attention_masking: true`, and split sentences; and no-bias/soft masks with default prompts.
- `massive_run2.json` contains concept-weaver/soft masks, `mask_sharing: first`, both values of `self_attention_masking`, and split sentences.
- `only_concept_weaver.json` contains concept-weaver/soft masks for `mask_sharing` values `first`, `prev`, and `null`, both self-attention values, and split sentences, plus a no-bias/soft/default-prompt baseline.

Generated `config.json` files show later experiments using `fixed`, `hard`, and `soft` mask modes and, in later runs, `mask_sharing` values. They are useful provenance snapshots, not an alternative source schema. JSON parameter files are copied to `<output-path>/<basename>` by `sync_param_grid`; a same-named destination with different bytes aborts the run, while an identical copy is accepted. Action-folder configurations are rewritten on each scheduling pass. Existing `video_<repeat>.mp4` files are skipped, providing coarse idempotent resume behavior; task pickles are temporary and are deleted after a child process returns, including after a failed child.

## Launchers and runtime assumptions

- `run_inference.sh` assumes execution from the repository root, uses `run.sh`, points at `image_prompt_generation/prompts_modified.json`, the deleted `param_configs/only_concept_weaver.json`, and `multi_sample_inference/debug_output/`.
- `submit_job.sh -a h100|a100 -c <name>` maps H100 to `gpu_p6`/`xvh@h100` and A100 to `gpu_p5`/`xvh@a100`, then submits `job.slurm` with exported environment.
- `job.slurm` requests one node, one task, 24 CPUs, one GPU, and a 20-hour limit, asks Slurm to send `USR1` 60 seconds before timeout, loads an architecture module plus Singularity, checks `nvidia-smi`, binds scratch/work, sets Hugging Face and Singularity cache paths, and writes output below `$SCRATCH/attn_bias/output_videos/<config>`. It expects the submission current directory to be the repository root. The one-GPU allocation does not exercise the dispatcher’s intended multi-GPU team modes.
- `job.oar` targets the Abaca queue/MUSA resource with a 48-hour walltime, loads Apptainer, uses `/storage/attn_bias/output_videos/<config>`, and attempts idempotent resubmission after a checkpoint signal. It also assumes repository-root execution and the same relative prompt/config paths.
- The cluster scripts depend on site-specific `module`, `sbatch`/`scontrol` or OAR behavior, NVIDIA drivers, CUDA-compatible container runtime, shared storage, and a cached or downloadable Hugging Face model. The local launcher depends on the checked-out Wan submodule, the CUDA image, and the image/prompt assets produced by `image_prompt_generation`.

## Output and dated artifact inventory

The Python code does not create date-named roots; it creates the caller-supplied output root, `.tasks`, a copied parameter JSON, and six-character action folders. The inspected working tree contains empty `multi_sample_inference/output/` and `multi_sample_inference/abaca_output/` directories, plus ignored date-named experiment roots. These are generated artifacts, not source or guaranteed reproducible fixtures.

There are 15 date roots (including the separately named `2026-02-18_2`), containing 266 JSON configuration snapshots, 1,134 MP4 files, and 200 PNGs in total (about 1.0 GiB by file-stat summation). Useful run-level summary:

| Root | Folder/config scale | Artifact counts (JSON / MP4 / PNG) | Notes |
|---|---:|---:|---|
| `2026-02-05` | 10 | 10 / 20 / 20 | Early concept-weaver outputs; debug prompt composites also present. |
| `2026-02-14`, `2026-02-17`, `2026-02-18` | 10--11 | 11/22/20, 10/20/20, 10/20/20 | Repeated fixed-mask-era runs; `2026-02-14` has `CKBt-I.bak`. |
| `2026-02-18_2`, `2026-02-20`, `2026-02-23` | 9--11 | 10/20/20, 9/18/0, 11/22/20 | Continued fixed-mask runs; `2026-02-20` has no PNG composites. |
| `2026-03-04`, `2026-03-06` | 20 each | 20 / 40 / 20 each | Larger fixed-mask/baseline batches. |
| `2026-03-17`, `2026-03-18` | 13--22 | 22/66/10, 13/39/10 | Soft-mask experiments and explicitly named `old`, `old2`, and `1st_block` folders; `.bak` artifacts occur. |
| `2026-03-30`, `2026-04-01` | 20 each | 20/107/10, 20/140/10 | Hard-mask experiment families; MP4 counts include attention-mask videos. |
| `2026-04-03` | 50 | 50 / 350 / 0 | Largest hard/fixed-mask comparison batch. |
| `2026-04-09` | 30 | 30 / 210 / 0 | Latest soft-mask batch; folders contain base videos plus per-character step 0/-1 overlays. |

A typical action folder has `config.json`, one or more `video_<n>.mp4` files, and, in the newer attention runs, two soft-mask videos per character and repeat. Early folders also contain `video_with_masks_<n>.mp4`. The `debug/prompt_0` through `prompt_9` trees contain source-adjacent PNG box/mask visualizations rather than generated model videos. No binary/media files were opened exhaustively; the inventory uses names, extensions, sizes, and selected text metadata.

## First cursory bug/gap review (no fixes made)

- **Deleted and stale inputs:** all four tracked parameter configs are deleted in the current worktree. Both `run_inference.sh` and `job.slurm` default/construct paths to `only_concept_weaver.json`, so those launchers fail before inference unless the deleted file is restored externally. `.vscode/launch.json` references that same deleted file and also passes obsolete `--checkpoint-path` and `--t5-cpu` arguments that the dispatcher CLI does not define.
- **Committed config defect:** `param_configs/all.json` is invalid JSON (extra top-level data); even if wrapped in an array, its entries omit the dispatcher-required `simil_masks_type` key. This is directly inconsistent with `process_parameter_grid()`.
- **Checkpoint reproducibility:** the scheduler sets `CHECKPOINT_PATH`, but Python ignores it and always performs a non-local-only Hugging Face snapshot download. The model revision is not pinned in code or task metadata. Runs therefore depend on network/cache state and the current remote model contents.
- **Mask semantic mismatch:** after Gaussian smoothing, `multi_sample_inference.py` converts the floating mask to boolean without a threshold. Thus any positive Gaussian tail becomes `True`; in addition, the `soft` branch ultimately sends boolean `face_masks` rather than preserving soft values. This is inconsistent with the `soft` name and should be checked against the expected `WanI2V` mask contract.
- **Failure accounting:** a failed `torchrun` is caught in `team_thread()`, but its task pickle is then deleted and `task_done()` is called, so the task is lost rather than retried. If all team threads stop, the dispatcher breaks its progress loop and still reaches normal process exit; the batch wrappers can report success despite incomplete work.
- **Timeout/requeue mismatch:** the Python dispatcher handles `USR1`, `SIGTERM`, and `SIGINT`, but `job.oar` forwards `SIGUSR2`, which Python does not handle and which can terminate it rather than allowing a clean save. The Python signal path also stops scheduling but has no explicit incomplete-work exit status, while the Slurm wrapper’s requeue logic expects a special exit code.
- **Portability and launch assumptions:** all launchers assume repository-root working directory, site-specific module names/partitions/accounts, Apptainer/Singularity, NVIDIA drivers, and a particular container image. The current checkout’s empty `wan2.1` submodule makes imports such as `wan.regional_prompt` unavailable without submodule population.
- **Collision/provenance risk:** action directories use only six characters of an MD5-derived name. The full configuration is stored inside the folder, but there is no check that a pre-existing hash folder’s configuration matches before files are reused; short-hash collision or manually mixed artifact directories could silently combine results.
- **Dependency and testing gaps:** `multi_sample_inference.py` directly imports `scipy.ndimage.gaussian_filter`, but root `pyproject.toml` does not declare `scipy` directly (it may arrive transitively); that weakens environment reproducibility. There are also no tests under `multi_sample_inference/` for grid decoding, task expansion, signal/requeue behavior, mask construction, distributed failure handling, or output naming. The repository inspection used non-mutating file/status and metadata checks only; no GPU inference was run.
