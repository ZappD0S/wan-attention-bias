# Media experiments and video gallery inventory

Scope: `inversion/`, `mask_visualization/`, `soft_mask_test/`, and `video_gallery/`. This is a current-state inventory, not a reproduction guide. Large media/model files were inspected structurally (names, sizes, container/header metadata where available), not as text.

## Current-state snapshot

| Directory | Current contents | Status |
| --- | --- | --- |
| `inversion/` | `inversion_test.py`; three MP4 artifacts | Runnable-looking, top-level script; requires unavailable local model/input assumptions |
| `mask_visualization/` | `__init__.py` (empty), two top-level notebook-style scripts, `output.npz`, one `.pt`, three MP4 artifacts | Experiment outputs are present; scripts execute at import/top level and are not packaged tests |
| `soft_mask_test/` | Empty | No implementation, script, or artifact present |
| `video_gallery/` | `data_manager.py`, `gallery_view.py`, `main.py`, `table_view.py` | UI code present; its configured generated-data directory is currently empty |

The repository working tree identifies the `video_gallery` Python files and `inversion/inversion_test.py` as tracked. The `mask_visualization` files other than ignored MP4s are currently untracked. All five MP4s in scope are ignored by the repository's `*.mp4` rule. This matters for provenance and for whether a fresh checkout contains the displayed results.

## `inversion/`

### Purpose and entry point

`inversion/inversion_test.py` is a monolithic Wan 2.1 latent ODE experiment. It encodes an input video to VAE latents, runs a scheduler/transformer ODE in a nominal inversion direction, runs the solver again for reconstruction, decodes the result, and writes an MP4. The file has no `main()` or argument parser: importing it or running it executes the full GPU job.

The active constants are:

- `MODEL_PATH = "../weights/Wan2.1-T2V-14B-Diffusers/"` (the I2V Diffusers path is a commented alternative).
- `VIDEO_IN = "./video.mp4"`.
- Prompt: `A video of two golden retrievers.`
- Optional I2V image: `../image_prompt_generation/images/0.png`.
- Scheduler shift `3.0`, default ODE step count `50`, default output FPS `15`, and at most `81` input frames.

The relative paths are relative to the process working directory, not to the script file; the apparent intended invocation directory is therefore `inversion/`.

### Data flow and parameters

1. `build_wan_pipeline` loads the Diffusers transformer and VAE from the selected model directory in `bfloat16` on CUDA. It selects `WanImageToVideoPipeline` when `transformer.config.in_channels == 36`, otherwise `WanPipeline`, and replaces the scheduler with `FlowMatchEulerDiscreteScheduler` configured for 1,000 training timesteps and the selected shift. VAE slicing is enabled and tiling initially disabled.
2. `get_prompt_embeddings` loads the tokenizer and UMT5 text encoder separately (4-bit, CPU device map), uses a temporary `WanPipeline` to encode the prompt without classifier-free guidance, then moves the embeddings to CUDA/bfloat16 and releases the temporary encoder.
3. `encode_video` loads `VIDEO_IN`, takes up to 81 frames, preprocesses them, encodes them with the VAE, and normalizes using `latents_mean`/`latents_std` from the VAE config. The resulting `orig_latents` is the ODE state.
4. For I2V only, the optional first-frame image is resized from the latent spatial dimensions (`* 8`), encoded by CLIP for image context, and VAE-encoded into a zero-padded 16-channel conditioning tensor. A four-channel mask marks only the first latent frame as conditioned. T2V bypasses these conditionals.
5. `run_ode` sets scheduler timesteps, optionally reverses their order, concatenates state/conditioning/mask for I2V (36 channels total), calls the transformer with text and optional CLIP context, and applies `z += noise_pred * dt` for all but the final timestep.
6. The top-level code calls `run_ode(..., reverse=True)` to create `inverted_noise`, then calls it again with `reverse=True` to create `recon_latents`. It offloads and deletes the transformer, denormalizes and decodes through the VAE, and exports `reconstruction_result_{t2v|i2v}.mp4`.

### Outputs and artifacts

- `video.mp4` — 293,586 bytes; the input/source artifact for this experiment. Its SHA-256 is identical to `mask_visualization/video.mp4`.
- `reconstruction_result.mp4` — 257,210 bytes; an unsuffixed older/result artifact not named by the current top-level code.
- `reconstruction_result_t2v.mp4` — 243,166 bytes; matches the currently selected T2V mode's output naming.
- No model files are stored in `inversion/`; model paths resolve into the repository's `weights/`, which is currently empty.

### Cursory review

- **Likely directionality gap:** both the inversion and reconstruction calls pass `reverse=True` (lines 268–274). Given the function's explicit direction switch and the names `inverted_noise`/`recon_latents`, the second call appears to lack the opposing integration direction; this is unverified without a model run.
- **Device API inconsistency:** `build_wan_pipeline` accepts `device`, but `get_vae_norm_params`, `encode_video`, and `run_ode` hard-code CUDA. The helper arguments and pipeline device can therefore disagree, and CPU/non-CUDA execution is not supported despite the nominal parameterization.
- **Path and provenance brittleness:** all inputs/outputs are CWD-relative, there is no CLI/config serialization, and the active T2V model differs from the commented I2V alternative. Existing output names include both an unsuffixed result and the current suffixed result, with no recorded run parameters.
- **Validation/test gap:** there are no input existence checks, frame/shape checks, scheduler compatibility checks, or tests. The code assumes a CUDA-capable environment, bfloat16 support, 14B model components, 4-bit loading support, and sufficient VRAM.

## `mask_visualization/`

### Purpose and entry points

This directory contains a regional-prompt/mask inspection experiment for Wan I2V and a separate partial-noising utility. Both are notebook-style scripts with `# %%` cell markers but no callable entry point or CLI; executing the file performs model loading and inference.

### `mask_visualization_tests.py`

The script fixes `TARGET_SIZE = (480, 832)`, selects `PROMPT_TYPE = "default"`, and uses the custom `WanI2V` checkpoint `../weights/Wan2.1-I2V-14B-480P/` on CUDA device 0 with `t5_cpu=True`. Its hard-coded prompt data describes two golden retrievers, two bounding boxes, and three available prompt variants (`default`, `no_locative`, and `split_sentences`). The chosen default has one action segment with location words and a character mask `[0, 1, 0, 1]`; a safeguard sentence is prepended with mask `[0]`.

Data flow:

1. Prompt segment lists become complete prompt sentences; character-only segments are selected using the segment masks.
2. `images/0.png` is joined under `../image_prompt_generation/`, loaded, and paired with masks generated from the two hard-coded bounding boxes.
3. `run_inference` (outside this scoped directory) receives the Wan object, prompt sentences, image, character segments, spatial masks, and `config = {"bias_method": "none"}`. It returns video plus `extra_data["simil_masks"]`.
4. The video is moved to CPU and normalized by `normalize_video_tensor`; normalized frames and similarity masks are saved as `output.npz`.
5. The archive is reopened, normalized frames are exported as `video.mp4`, and masks are transformed by removing the batch dimension, selecting the first transformer block, unscaling to video resolution, and rearranging to `T, char, diff_step, H, W`.
6. Only the first diffusion-step mask for each of two characters is outlined with OpenCV contours and two Matplotlib colormaps. The resulting frames are exported as `video_with_masks.mp4` at 16 FPS.

Structural metadata from the existing `output.npz` (read from the NumPy archive headers) is:

- `video_norm`: little-endian float32, shape `(81, 464, 832, 3)`, 375,238,784-byte payload.
- `simil_masks`: boolean, shape `(1, 40, 40, 2, 21, 29, 52)`, 101,337,728-byte payload. This agrees with the in-code comment describing batch, denoising step, block, character, temporal, height, and width dimensions.
- Archive size: 476,576,784 bytes (about 455 MiB).

Existing outputs are `video.mp4` (293,586 bytes, same bytes as `inversion/video.mp4`) and `video_with_masks.mp4` (304,818 bytes). The directory also has an empty `__init__.py` and no tests despite the script name containing `tests`.

Cursory review:

- **External input currently does not match the checked-out tree:** the script asks for `image_prompt_generation/images/0.png`. The repository has `image_prompt_generation/images/0/` (a directory containing LFS-pointer PNG entries such as `original.png`), not that file. The selected checkpoint is also absent because `weights/` is empty.
- **CWD/import assumptions:** `IMG_DIR`, checkpoint, input/output names, and `Path.cwd().parent` depend on where the notebook is launched. The repository-root path is added to `sys.path` only after imports from `wan`, so that adjustment cannot make those earlier imports work.
- **Output reproducibility gap:** prompt type, boxes, scheduler/inference settings, source image, and run metadata are not written alongside the NPZ. The saved mask archive is large but has no manifest/version.
- **Shape/selection assumptions:** there is no validation that `simil_masks` has two characters, that video and masks agree, or that `extra_data` contains the expected key. The visualization intentionally discards all but the first block and first diffusion step, while the commented averaging alternative is not active.

### `partially_noised_video.py`

This utility loads the same custom Wan I2V model/checkpoint and the local `./video.mp4`, using CUDA, bfloat16, `832 x 480`, UniPC with 40 sampling steps and shift `5.0`. It crops frames to Wan's `4n + 1` alignment, preprocesses them, VAE-encodes them, samples matching random noise, chooses the scheduler timestep nearest `0.3 * num_train_timesteps`, adds noise, and saves `video_latents_{normalized_t:.2f}_noise.pt`.

The current artifact `video_latents_0.29_noise.pt` is 8,108,832 bytes (PyTorch zip serialization; the tensor storage is 8,107,008 bytes). The filename records only a rounded normalized timestep; the script itself has a TODO noting that it does not save the source image or actual timestep.

Cursory review:

- **Processing inconsistency:** the script preprocesses once and rearranges `[B,T,C,H,W]` to `[B,C,T,H,W]`, then immediately preprocesses the frames a second time and overwrites `video_tensor` without that rearrangement. The comments also disagree about the return layout. This makes the tensor layout dependent on the library's actual API and is a likely execution/shape risk.
- Frame alignment is performed twice (`target_len` and `target_num_frames`) and the first preprocessed tensor is discarded. There are no checks for empty/too-short video, VAE output shape, or timestep range.
- The saved tensor is not linked to the source prompt/image or exact scheduler configuration, so later consumers cannot reliably reproduce its provenance.

## `soft_mask_test/`

The directory is currently empty (4 KiB directory entry only). There is no script, data, model artifact, output, or documented entry point. Any planned soft-mask experiment is therefore absent from this checkout rather than merely failing at runtime.

## `video_gallery/`

### Data contract and manager

`data_manager.py` configures:

- `VIDEOS_PATH = Path("../multi_sample_inference/output/")`.
- `SCORES_FILE = VIDEOS_PATH / "video_scores.json"`.
- Expected generated-data layout: recursively discover `config.json`; for each config's parent, discover files matching exactly `video_<integer>.mp4`.

`discover_groups` parses each config, reads `prompt_data` and `prompt_type` (defaulting to `{}` and `"unknown"`), assigns a group ID by exact equality of `prompt_data`, and stores the parent directory basename plus synthetic `_gid` and `_videos` fields. Each video is represented as `(URL, filename, integer index)`, with URL `/videos/<parent-basename>/<filename>`. `generate_dataset_json` emits one record per video containing group/type/folder/file/index/score and then merges `params` into that record.

Scores are global in-memory `video_scores`, loaded from `video_scores.json` if present and written immediately by `save_score`. The score key is `<folder-basename>/<video-index>`.

Current state: the repository's `multi_sample_inference/output/` exists but is empty, so no `config.json`, generated video, or score file is available for this gallery. The gallery therefore has no local generated data to display in this checkout, despite the separate experiment outputs described above not being under the configured path.

### Views and connection to generated data

`main.py` is the application entry point. It mounts `VIDEOS_PATH` as NiceGUI static files at `/videos`, loads scores, discovers groups once at startup, and renders two tabs before calling `ui.run(title="Video Gallery", dark=True, show=False)`.

`gallery_view.py` renders one sidebar link and one content section per numeric group. Within each group it creates prompt-type tabs; each config becomes a card with a carousel of discovered MP4 URLs, score toggle (1–5, clearable), and sorted `params` badges. The score callback writes through `data_manager.save_score`. The download action serializes the same discovered records and current score map as `scores.json`. Thus the gallery does not consume `inversion/` or `mask_visualization/` outputs directly: those files would need to be copied or regenerated into the external `multi_sample_inference/output/<config-folder>/` contract, together with matching configs, before they can appear.

`table_view.py` builds a score matrix for a selected `prompt_type`: one row per group, one column per formatted `params` configuration, and mean score across that configuration's discovered videos. Its prompt-type sidebar is derived from discovered configs; it does not scan media independently.

### Cursory review and integration gaps

- **Currently empty integration point:** no generated configs/videos exist under the configured output directory, so the app's expected data path is disconnected from all in-scope experiment artifacts.
- **CWD-relative path:** `../multi_sample_inference/output/` works only when the process is launched from a directory whose parent contains that path as intended (apparently `video_gallery/`). Launching `python video_gallery/main.py` from the repository root resolves a different parent-relative location. The static mount and score writes inherit this brittleness.
- **Nested-path/collision risk:** discovery searches recursively but retains only `folder.name` for static URLs and score keys. Two nested folders with the same basename collide; parent directories are discarded, and the generated URL cannot represent a deeper relative path.
- **Unstable grouping:** group IDs are assigned in filesystem traversal order, so numeric labels can change between runs. Exact prompt-data equality is the only grouping criterion; there is no explicit stable ID.
- **Missing validation/error coverage:** malformed JSON is skipped for a narrow set of exceptions, but config shape and required fields are not validated. There is no check that a discovered MP4 is playable or that a config's params are JSON-safe. No automated tests are present in `video_gallery/`.
- **Schema/API edge cases:** `save_score` accepts the clearable toggle's `None` despite the `int` annotation; it assumes the output parent is writable. `record.update(params)` can overwrite the standard fields if a generated `params` contains a colliding key. Folder/index-only score keys can apply a score to the wrong file when basenames collide.
- **Prompt rendering mismatch:** gallery prompt rendering joins `action_prompts[*]["segments"]` directly. The in-scope mask experiment stores segments as nested lists, so that form would raise a `TypeError` and be replaced by the broad `Prompt data unavailable` fallback if such a config is placed in the gallery.
- **Stale/artifact gap:** gallery startup scans only its configured output tree and has no import, symlink, or manifest mechanism for the existing experiment MP4s. The MP4s are ignored, and large NPZ/PT outputs are local/untracked in `mask_visualization`, so a clean checkout cannot reproduce the displayed media state.

## External assumptions summary

Across these directories, successful execution assumes: downloaded Wan 2.1 I2V/T2V weights in the exact names and layouts referenced by the scripts; the repository's custom `wan` implementation and other project modules importable on `PYTHONPATH`; CUDA hardware with bfloat16 and sufficient memory for a 14B model; compatible Diffusers/Transformers/PyTorch, bitsandbytes (for 4-bit text loading), OpenCV, Matplotlib, NumPy, einops, NiceGUI, pandas, and great-tables installations; valid video/image assets and the expected frame/layout conventions; and, for the gallery, generated `config.json` plus `video_<n>.mp4` records under `multi_sample_inference/output/`. The checked-out `weights/` and gallery output directories are empty, and the referenced `images/0.png` path is absent, so these assumptions are not satisfied by the current repository state.

## Cheap checks performed

- Parsed all eight Python files in scope with `ast.parse`; all are syntactically valid.
- Enumerated files, sizes, archive headers, MP4 container headers, hashes, and directory contents without decoding large media as text.
- No model inference, web-server launch, media decode, file generation, or mutation was performed.
