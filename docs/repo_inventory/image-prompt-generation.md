# Image prompt generation inventory

## Scope and snapshot

This inventory covers only `image_prompt_generation/` and its direct relationship to the repository-level `schema.py`, `utils.py`, and model weights. The directory contains three Python programs, one metaprompt, current JSON datasets, archived JSON datasets, generated image artifacts, and Python bytecode caches. A selective JSON/static check found:

- `final_dataset.json`: 197 unprocessed scene specifications.
- `prompts_modified.json`: 197 processed specifications; every item has two bounding boxes, two segmentation-mask paths, two single-character paths, and a seed (42 through 238).
- `images/`: 197 numeric directories (`0` through `196`), each with `original.png`, `segmentation_mask_0.png`, `segmentation_mask_1.png`, `single_char_0.png`, `single_char_1.png`, and a `debug/original_with_boxes.png`.
- `backup/`: two ten-scene snapshots and two alternate 200-scene datasets.
- `__pycache__/`: compiled bytecode, including a stale `lama_inpainter` bytecode name for which no corresponding source file is present. It is a cache, not an input or output contract.

Image files and bytecode were not opened or decoded for this inventory; only names, counts, and referenced-path existence were checked.

## End-to-end workflow

### 1. Scene/prompt dataset generation

`generate_scenes.py` is the text-data producer. It reads `image_prompt_generation/metaprompt.md`, resumes from `image_prompt_generation/final_dataset.json` if present, and targets 200 items (`generate_scenes.py:20-24`, `191-205`). It requires `ZHIPU_API_KEY` and sends requests through the OpenAI client to `https://api.z.ai/api/coding/paas/v4`, using model `glm-5.1` (`17-26`).

Each iteration asks the service for simple `Theme | Two [Subjects]` seed lines, excluding previously collected subject segment text (`29-82`). Seeds are sent in batches (normally ten) to the metaprompt as a system message, with JSON-object response mode; the code reads the response's `dataset` member (`88-119`). Progress is written immediately, with a fixed safeguard suffix (`162-188`). Up to five audit passes request another remote JSON response containing `remove_ids`; selected items are removed and the dataset is refilled (`122-177`, `209-231`). The service is therefore responsible for both initial language generation and the redundancy audit; there is no local generation or deterministic seed for this stage.

The metaprompt defines the intended contract: a root `safeguard_suffix` plus ten scenes per response, two similar subjects, an appearance prompt split into five segments, three action-prompt variants, and masks that mark exactly two subject/action segments. Its dynamic batch substitution only replaces the literal phrase `10 distinct scenes` (`generate_scenes.py:89-91`).

### 2. Reference image generation and detection

`generate_prompt_image.py` loads every top-level `*.json` in `image_prompt_generation/`, decoding each as the base `VideoGenerationDataset[VideoSpecification]`; it skips only the existing `prompts_modified.json` by `samefile` comparison (`56-78`). Thus the current normal input is `final_dataset.json`; files under `backup/` are not included by this glob. Decode failures are printed and skipped rather than terminating the run.

The script loads Flux (`black-forest-labs/FLUX.1-dev`) and Grounding DINO (`IDEA-Research/grounding-dino-base`) (`81-97`). For each item it takes the first appearance segment list, selects entries whose mask is `1`, and joins all appearance strings into the Flux prompt (`347-360`). Flux requests 832 by 480 pixels, guidance scale 3.5, 50 inference steps, and sequence length 512. A CPU torch generator supplies seed `42 + item index + attempt * 1000` (`115-130`, `302-322`).

Grounding DINO receives the two selected character descriptions after trailing punctuation is removed. It uses detection threshold 0.29 and text threshold 0.1, then NMS at IoU 0.5 and retains the highest-scoring requested number of boxes (`133-170`). Existing `original.png` files are reused unless regeneration is forced; `main()` does not expose or pass a force option, so its normal call uses `False` (`285-300`, `496-503`). A failed detection is retried ten times, and an item still failing is skipped (`273-326`, `366-368`). Successful images are saved as `images/<input-index>/original.png`; debug box visualizations are saved as `images/<input-index>/debug/original_with_boxes.png` (`329-336`).

### 3. Segmentation, inpainting, and processed JSON

After generation/detection, generation models are released and the script loads SAM2 (`facebook/sam2.1-hiera-large`) and a local LaMa checkpoint (`generate_prompt_image.py:100-112`). SAM2 is run with detected boxes. Its masks are clipped to each integer box, convex-hulled, and dilated with a 10-pixel elliptical kernel (`390-437`). For each character, a removal mask is the union of all *other* character masks, dilated by 100 pixels; LaMa inpaints that mask to produce a single-character image (`440-475`).

The saved `segmentation_mask_*.png` files are not the SAM2 masks: they are rectangular masks made directly from each detected box by `create_mask_from_bbox` (`513-526`). The SAM2-derived masks are used internally for inpainting but are not serialized as paths. Single-character images are saved as `single_char_*.png` (`528-532`). Bounding boxes are enlarged by up to 50% where possible without positive-area overlap, clipped to the image dimensions, and stored as `enlarged_bboxes` (`182-227`, `534-535`).

The final processed dataset preserves each base prompt and adds metadata, then writes msgspec-formatted JSON to `prompts_modified.json` (`478-493`, `544-555`). The output list contains only successful items, so failed detections can make it shorter than its source.

`big_lama.py` is the small local inference adapter. It reads `<checkpoint>/config.yaml`, enables predict-only/no-op visualization, loads `<checkpoint>/models/best.ckpt`, freezes and moves the model, then converts PIL RGB/L masks to normalized tensors before inpainting (`10-30`, `33-64`).

## Data formats and conventions

The repository-level `schema.py` supplies the typed shape. `VideoGenerationDataset` has `safeguard_suffix` and `dataset`. A base `VideoSpecification` has `appearance_prompt` and `action_prompts`; all prompt objects use `segments: list[list[str]]` paired with same-shaped integer `mask` arrays (`schema.py:7-27`, `66-77`).

The observed base item shape is:

- `appearance_prompt`: one nested five-string list, conventionally intro/context, subject A, connector/locative, subject B, punctuation; mask `[0, 1, 0, 1, 0]`.
- `action_prompts.default`: one nested four-string list with locative, action A, locative, action B; mask `[0, 1, 0, 1]`.
- `action_prompts.no_locative`: one nested three-string list with action A, conjunction, action B; mask `[1, 0, 1]`.
- `action_prompts.split_sentences`: a `general_prompt` string, two one-string sentence lists, and masks `[[1], [1]]`.

The processed `ProcessedVideoSpecification` adds `bboxes` and `enlarged_bboxes` as lists of four-float coordinates, `img_paths` (original, segmentation-mask list, single-character list), and integer `seed` (`schema.py:29-50`). Current processed records consistently have two entries in each box/image list. Paths are stored as repository-relative strings such as `image_prompt_generation/images/0/original.png`, while the scripts construct them from relative `Path` values.

Generated directory names are input enumeration indices, not stable semantic IDs. Within each directory, the current producer's names are `original.png`, `segmentation_mask_0.png`/`_1.png`, `single_char_0.png`/`_1.png`, and `debug/original_with_boxes.png`. Nine early debug directories also contain `debug/all_boxes_detected.png`, an artifact not written by the current source (the current writer uses `original_with_boxes.png`).

## Code, source data, outputs, and caches

**Code:** `generate_scenes.py` orchestrates remote scene creation; `generate_prompt_image.py` orchestrates Flux, DINO, SAM2, LaMa, masking, and serialization; `big_lama.py` adapts LaMa. The shared typed definitions are in root `schema.py`, and box-mask construction is imported from root `utils.py`.

**Source data:** `metaprompt.md` is the generation instruction; `final_dataset.json` is the current base prompt dataset. `backup/prompts.json` is an older ten-scene base snapshot. `backup/final_dataset1.json.bak` and `backup/final_dataset2.json.bak` are separate alternate 200-scene base datasets, not processed outputs.

**Generated outputs:** `prompts_modified.json` is the current processed dataset; `images/` contains the corresponding generated/reference, rectangular mask, inpainted single-character, and debug PNGs. The current processed JSON references existing files for all 197 records.

**Caches/archive:** `__pycache__/` is ignored compiled Python output. `backup/` is an archive area rather than an input directory for the normal top-level JSON glob. The `all_boxes_detected.png` files appear to be leftover debug artifacts from an earlier implementation.

## Model and service dependencies

The Python project requires Python 3.11 and declares OpenAI, msgspec, PyTorch/torchvision, Accelerate, Diffusers, Transformers, SAM2, OpenCV, scikit-image, Pillow, YAML/OmegaConf, and the editable `lama-inpainting` workspace dependency (`pyproject.toml:1-3`, `8-123`, `167-176`). Runtime access is needed to download the three Hugging Face models named above. GPU-oriented dtype/device settings are used: Flux is loaded with bfloat16 and balanced device mapping, SAM2 with float16, and DINO/LaMa use the Accelerate-selected device.

LaMa additionally requires local `weights/big-lama/config.yaml` and `weights/big-lama/models/best.ckpt` by the call at `generate_prompt_image.py:110` and the path construction in `big_lama.py:14-25`. At inventory time `weights/` exists but is empty, so a fresh image-processing invocation cannot load this checkpoint without provisioning it. The scene stage requires the Zhipu API key and network access; likely Hugging Face authentication/network access is also required for model downloads, although the scripts do not document or validate those prerequisites.

## Invocation assumptions

Run from the repository root so the hard-coded relative paths resolve. The scene producer assumes `image_prompt_generation/metaprompt.md` and writes `image_prompt_generation/final_dataset.json` (`generate_scenes.py:20`, `193-205`). The image producer similarly assumes `./image_prompt_generation/` and `./weights/big-lama` (`generate_prompt_image.py:32-35`, `110`). Because it imports `.big_lama`, it is intended to be launched as a package module (for example, `python -m image_prompt_generation.generate_prompt_image` from the root), rather than as a standalone file whose relative import may fail. No project entry points or task-specific usage documentation were found.

## Cursory bug and gap review (no fixes made)

- **Target/count gap:** the generator targets 200 (`generate_scenes.py:21`), but the current base and processed datasets each contain 197 records, and the image directory range stops at 196. The image stage explicitly skips failures (`generate_prompt_image.py:366-368`), which can explain divergence but does not record a failure manifest.
- **Weak generation validation:** batch responses are accepted via `json.loads(...).get("dataset", [])` without validating the prompt schema (`generate_scenes.py:103-119`). Seed parsing accepts any line containing `|` (`79-82`). Repeated empty/invalid seed responses can loop indefinitely because `ensure_dataset_full` has no retry limit (`162-175`).
- **Audit index safety:** audit IDs are not type/range validated; only `idx < len(dataset)` is checked before `pop` (`generate_scenes.py:222-225`). Negative IDs therefore address items from the end, and non-integer IDs could raise.
- **Stale/inconsistent backups:** `backup/final_dataset1.json.bak` and `final_dataset2.json.bak` are distinct 200-scene prompt sets and share no exact subject segment with the current set in a static comparison. The ten-scene `backup/prompts_modified.json` has no `seed` field and references misspelled `segmentaion_mask_0.png`/`_1.png`, which do not exist; its paired `backup/prompts.json` is an older base schema. None is marked with provenance or a generation date.
- **Hard-coded paths and hidden inputs:** paths, model IDs, thresholds, dimensions, and output names are constants (`generate_scenes.py:17-24`; `generate_prompt_image.py:32-37`, `115-127`). The image loader aggregates all top-level JSON files rather than accepting an explicit input, and filesystem glob order is not made explicit (`56-78`).
- **Reproducibility:** the Flux seed is recorded for newly generated images, but an existing image reused at `generate_prompt_image.py:286-300` is assigned `42 + index` even if its original successful attempt used an alternate retry seed. Remote scene generation/auditing has no request seed or captured response metadata, and model/library revisions are not pinned by these scripts.
- **Segmentation diagnostics:** the empty-mask warning uses `i` left over from the preceding box loop rather than the current mask index (`generate_prompt_image.py:413-427`), so its reported index can be wrong (or be undefined for an empty box list). The serialized segmentation paths describe box rectangles while the internal inpainting masks are SAM2-derived (`513-526` versus `405-435`), a potentially confusing schema/semantics mismatch.
- **Absent checks and documentation:** there are no tests in this directory, no validation/report for asset dimensions or prompt semantics, no failure manifest, and no task-specific README or invocation document. Current JSON shape checks pass for the inspected datasets, but that is not enforced by `generate_scenes.py` before data is written.

**Status:** inventory written to `docs/repo_inventory/image-prompt-generation.md`; no source, dataset, image, backup, or cache files were modified.