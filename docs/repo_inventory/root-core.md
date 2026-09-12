# Root/core inventory

## Scope and current shape

This inventory covers the requested root/core files and `examples/`. The repository is an experiment-oriented Wan video-generation workspace rather than a packaged application with a single working CLI. The only conventional Python entry point, `main.py`, is a smoke/demo placeholder; the substantive root-level workflow is the notebook-style `regional_prompting.py` script.

Observed from the working tree: the requested source, scripts, configuration files, and example assets are tracked. `.env` exists locally but is not listed by `git ls-files` and is ignored by `.gitignore`; it contains a Hugging Face token and should be treated as sensitive local configuration. No deleted status was observed for the requested paths. `docs/repo_inventory/root-core.md` is the inventory artifact being created.

## Root modules and entry points

- **`main.py`** — Defines `main()` and prints `Hello from wan-experiments!` when run directly. It is a syntactically valid, but functionally empty, package-style entry point. It does not invoke Wan, consume an example, or expose a generation CLI.
- **`regional_prompting.py`** — The effective root experiment. It is written as an interactive/Jupyter-style sequence (`# %%`) with substantial import-time work. It constructs `WanI2V` from the editable `wan2.1` workspace package using the `i2v_14B` configuration, hardcodes GPU device 0, enables CPU T5, and points at `./weights/Wan2.1-I2V-14B-480P/`. It then uses the fixed example directory `examples/dogs_no_interaction`, so changing examples currently requires editing the file rather than passing an argument.
- **`utils.py`** — Small tensor/image helpers used by the regional workflow. It creates a filled boolean mask from `(left, top, right, bottom)`, finds the largest connected contour and returns its bounding box, converts a `C T H W` video array to normalized `T H W C`, and supplies a context manager that temporarily disables `tqdm` progress bars.
- **`debug_utils.py`** — Visualization/debug output helpers. It draws boxes or colored masks on PIL images, resizes trailing `(T, H, W)` tensor dimensions with trilinear interpolation, and writes hard-mask, soft-mask, looking/“wlw”, or attention overlays as MP4 files through OpenCV.
- **`schema.py`** — A `msgspec` data-contract layer for a different/more structured dataset representation. It defines masked text prompts, joint action prompts, an action-prompt suite, image/mask paths, minimal and processed video specifications, and a generic dataset wrapper. There is no visible connection from this schema to the JSON consumed by `regional_prompting.py`.
- **`run.sh`** — The operational launcher, not a Python entry point. It loads `.env` if present, selects Apptainer or Singularity, binds the repository at `/workspace`, binds a Hugging Face cache, optionally binds `$STORAGE_DIR`, and executes either an interactive container shell (no arguments) or the supplied command. With `DEBUG=1`, it prepends `python -m debugpy --listen ... --wait-for-client`; a target argument is then mandatory. The script is executable.
- **`download_weights.sh`** — A thin Hugging Face CLI helper. It creates the directory supplied as `$1` and downloads `Wan-AI/Wan2.1-I2V-14B-480P` under `$1/Wan2.1-I2V-14B-480P/`. It assumes the `hf` command and authentication are already available; it has no argument validation or strict-error setting.

A typical intended execution shape is therefore `./run.sh python regional_prompting.py` (inside the configured GPU container), after weights are downloaded. That is an inference from the launcher and script; there is no documented root CLI and the script currently has runtime issues noted below.

## Regional prompting data flow and contracts

`regional_prompting.py` reads an example directory containing `original.png` and `config.json`:

1. It loads the RGB source image and records `(width, height)`.
2. It loads `characters.bbox_format`, sorts `characters.list` by `id`, removes each character's `id`, and extracts bounding boxes and descriptions. The code uses the resulting list positions as the indices referenced by `wlw[*].pair`.
3. `rescale_img_and_bboxes` wraps the boxes in torchvision `BoundingBoxes` with the original `(height, width)` canvas, converts them to `XYXY`, resizes the image and center-crops it to `(480, 832)`, and returns transformed boxes. A mask is then made per box with `create_mask_from_bbox`; the helper requires height no greater than width, which holds for this target.
4. It builds `wlw_matrix` with shape `(characters, characters, frames)` for 81 frames. Each `wlw` pair and normalized time interval is expanded into a boolean observer/observed matrix. A singleton pair such as `[0]` is duplicated to address the diagonal. The corresponding formatted prompt is stored in `control_prompts` as `descr_list` plus `prompt`.
5. It creates all-true timestep and transformer-block bias schedules, then passes `control_prompts`, schedules, face masks, the wlw matrix, and `beta=1.0` as `bias_kwargs` to `wan_i2v.generate`. The base and negative prompts, 480×832 area, 40 sampling steps, and 81 frames are hardcoded in the script (apart from the example's `base_prompt`).
6. The returned video is normalized from `[-1, 1]` and rearranged to `T H W C`. Extra similarity masks and attention maps are upscaled for inspection. The script attempts to write a timestamped `people_masks.mp4` under the example's `output/` directory and separately caches `example.mp4` in the current working directory.

The effective example JSON contract is therefore:

- `base_prompt: string`;
- `characters.bbox_format: string` accepted by torchvision's bounding-box converter (the examples use `XYWH`);
- `characters.list: [{id: integer, descr: string, bbox: [x, y, width, height]}]`;
- `wlw: [{prompt_template: string, pair: [character index, ...], time_intervals: [[start, end], ...]}]`.

`prompt_template.format(*pair descriptions)` supplies the control prompt. Intervals are expected to be normalized to `[0, 1]`; the implementation checks bounds with `assert`, requires a nonempty interval list, and maps endpoints to frame indices by rounding.

`schema.py` describes a separate contract: `MaskedTextPrompt` and `JointActionPrompt` pair nested text segments with nested integer masks; `VideoSpecification` contains appearance and action prompts; `ProcessedVideoSpecification` additionally requires boxes, asset paths, enlarged boxes, and a seed; `VideoGenerationDataset[T]` wraps a safeguard suffix and a list of either specification type. These types are definitions only in the inspected root surface—no loader, serializer call, or example JSON uses them.

## Examples and their relation to code

Each example directory is a hand-authored input case, not a standalone runnable example. The code currently selects only `examples/dogs_no_interaction`; the other directories document the same expected shape and can be selected only by editing `PROMPT_CONFIG`.

| Directory | Input content and intended regional behavior |
| --- | --- |
| `dogs_no_interaction/` | A 1536×1024 source image and two dogs. Dog 0 jumps for the full interval; dog 1 lays down and sleeps for the full interval. This is the current hardcoded case. |
| `2animals/` | A 1536×1024 source image, an older 1024×1024 source, and `sample1.png`, plus rabbit/turtle configuration. Rabbit looks at turtle for the full interval. |
| `3animals/` | A 1536×1024 source image and chicken/cat/dog configuration. Cat looks at chicken during the first half, cat looks at dog during the second half, and dog looks at cat throughout. |
| `women_looking_at_each_other/` | A 3840×2160 source image and two detailed woman descriptions. The left woman looks at the right during the first half, then the reverse during the second half. |

The examples exercise both singleton `pair` entries (`dogs_no_interaction`) and two-character observer/observed entries. Their boxes are authored in source-image coordinates and are transformed by the script; no per-example masks or generated outputs are tracked. Generated `output` paths and video files are excluded by `.gitignore`.

## Dependencies and configuration

`pyproject.toml` declares Python `>=3.11,<3.12`, project version `0.1.0`, and a broad GPU/ML, image/video, web/UI, and notebook dependency set. The regional path directly relies on PyTorch, torchvision, NumPy, OpenCV, Pillow, matplotlib, einops, tqdm, `msgspec` (for the schema), and the editable `wan` package. The wider declaration also includes Diffusers/Transformers, Hugging Face tooling, flash-attention, bitsandbytes, SAM2, Lama, and several application/UI libraries.

The UV workspace includes editable `wan2.1` (`wan`) and `lama` (`lama-inpainting`) members. `torch`, `torchvision`, `torchaudio`, and `torchcodec` resolve from the explicit CUDA 12.8 PyTorch index; torch is pinned to `2.10` and torchcodec to `0.10`. Architecture-specific flash-attention wheels are supplied by custom GitHub URLs. `uv.lock` records Python 3.11 resolution and the three workspace packages (`lama-inpainting`, `wan`, and `wan-experiments`), with the root package dependencies and dev group resolved. Installation is consequently GPU/platform-sensitive rather than a lightweight CPU setup.

`.env` is loaded by `run.sh` and exports Hugging Face-related values. The launcher itself uses `HF_TOKEN`, maps the host Hugging Face cache to `/.cache/huggingface`, and chooses the container image from `CONTAINER_IMAGE` or `containers/cuda_ubuntu.sif`. `STORAGE_DIR` and `STORAGE_BIND_PATH` control an optional storage bind. `.gitignore` excludes `.env`, weights, containers (`*.sif`), temporary files, caches, outputs, debug artifacts, and videos. `.gitattributes` applies Git LFS treatment to PNGs under `image_prompt_generation/images/`, not to the root examples. `.gitmodules` declares `wan2.1` on the `dev` branch and `lama` as external submodules; the editable workspace dependencies therefore require those checkouts.

The VS Code launch configuration is not a regional-prompting launcher. It launches `multi_sample_inference.multi_sample_inference` with prompt/parameter files and a weights path, and separately provides an attach configuration for port 5678 with `/workspace` path mapping. This is consistent with the broader experiment repository but does not remove the need to invoke `regional_prompting.py` manually.

## Cursory bug/gap review

The following are static observations from the inspected files; implications marked **inference** were not confirmed by running a model generation.

- **Likely immediate runtime error (observed code path):** `rescale_img_and_bboxes` converts each transformed box to a Python list (`bbox.tolist()`). The later drawing loop calls `bbox.tolist()` again, which lists do not provide. The script is therefore likely to fail before generation unless that loop is changed or the boxes remain tensors.
- **Likely mask/video shape mismatch (observed code):** the post-generation code averages `simil_masks[0, -1]` over a dimension, producing a single spatial mask before `unscale`, then passes `face_masks.transpose(0, 1)` to `write_video_wlw_masks`. That writer expects one collection of per-character masks for each frame and uses strict zips; the supplied tensor appears to be a transposed single-mask `(H, T, W)`-style value, not `(T, characters, H, W)`. **Inference:** the debug MP4 path is likely to raise a strict-zip error or draw incorrectly even if generation succeeds.
- **Import-time and environment coupling (observed):** importing `regional_prompting` instantiates a 14B model and assumes CUDA device 0 plus a relative checkpoint directory. It is not safe as a reusable import or CPU smoke test. The script also has no argument/configuration interface for examples, weights, device, resolution, or seed.
- **Output and reproducibility gaps (observed):** sampling steps, frame count, bias schedules, beta, negative prompt, and device are hardcoded; no seed is set. `example.mp4` is written to the process working directory while the debug output is placed below the selected example. **Inference:** repeated runs are difficult to compare and may leave outputs in unexpected locations.
- **Helper edge cases (observed):** `create_bbox_from_mask` calls `.numpy()` directly on a torch tensor, so a CUDA tensor is unsupported despite the type acceptance; it raises for empty masks. `create_mask_from_bbox` asserts an aspect condition and does not clip coordinates. `write_debug_video_attn` divides by the maximum attention value without guarding an all-zero map. These may be acceptable for current debug inputs but are not defensive APIs.
- **Contract drift (observed):** the `msgspec` schema's appearance/action/asset structure does not match the example `base_prompt`/`characters`/`wlw` structure, and no adapter is present in the inspected surface. **Inference:** the schema may belong to an unfinished or separate pipeline rather than the currently runnable regional experiment.
- **Operational gaps (observed):** `download_weights.sh` does not validate `$1` and does not enable `set -e`; `run.sh` requires Apptainer/Singularity and a suitable image, GPU passthrough, dependencies, and weights. A missing or empty first argument to the downloader will not produce the intended destination. The local `.env` includes a credential-like token even though it is ignored, so it must not be committed or exposed.

A cheap AST parse of the five Python files passed; no model, container, download, or mutating runtime test was performed.
