# `wan2.1/wan/custom_image2video.py` — static inventory

## Scope and attribution

- Repository: `/home/zapp/Documents/wan_experiments/wan2.1`, detached at `f7472d354e0cb46b2853f15bf97b8b283d4780f8`.
- Attribution filter: exact author `gzappavi <gianluca.zappavigna@inria.fr>`.
- The assigned path is **not present at HEAD**. It was deleted by gzappavi commit `2d1918018afb26a53800d6407199c961b8cdb322` (`wip`, 2025-10-30). The latest extant snapshot is the 550-line blob at `e7020106f6c36be3113300b5920c7d5f69c17d6f` (2025-10-28), used for line references below.
- Five gzappavi commits directly affect the named path: `44e09dab` (add/rename, 2025-10-09), `60c6729f` (2025-10-21), `209e9d97` (2025-10-25), `e7020106` (2025-10-28), and `2d191801` (delete). `git log --follow` additionally attributes eight earlier predecessor-path revisions to the same author; no other file contents were inspected.

## Pipeline behavior and data flow

`CustomWanI2V.__init__` (lines 50–156) creates a CUDA device, loads T5 text encoding, VAE, CLIP, and `CustomWanModel` checkpoints from `checkpoint_dir`, then optionally installs USP attention/forward hooks and/or FSDP sharding. The model is put in evaluation/no-grad mode.

`generate` (399–550) performs the following flow:

1. Converts the input image to a tensor and normalizes it to approximately `[-1, 1]` (450).
2. Chooses an explicit seed or a random seed, and creates a device-local Torch generator (452–454).
3. `_build_latents` (158–215) derives latent height/width from input aspect ratio and `max_area`, resizes the first image frame, pads the remaining frames with zeros, VAE-encodes that sequence, prepends a first-frame mask, and creates Gaussian latent noise.
4. `_build_context` (217–323) resolves the negative prompt, encodes the base and negative prompts, constructs ordered pairwise “gaze” prompts from `descr_list` and `link_text`, tokenizes the combined prompt, and creates GPU token masks for each gaze/description span. It also obtains CLIP image features.
5. `_get_scheduler` (325–349) selects UniPC or DPM++ and produces sampling timesteps.
6. For every timestep, `_compute_noise_pred` (351–397) runs conditional and unconditional model passes and applies classifier-free guidance: `uncond + guide_scale * (cond - uncond)` (393–395). The timestep bias is taken from `timestep_bias_schedule` (497).
7. The scheduler updates the latent (522–529). On rank zero, the VAE decodes the final latent; all ranks stack similarity masks and rank zero returns `(video, similarity_masks)` (535–550).

## gzappavi commit evolution

`git log --follow` identifies this sequence for the file’s lineage (subjects are recorded metadata, not independent behavior claims):

| Date | Commit | Subject / evolution |
|---|---|---|
| 2025-09-18 | `dda8f6f` | formatting |
| 2025-09-18 | `62af287` | improved mask creation |
| 2025-09-22 | `37ac1cd` | model-weight dtype / GPU OOM fix |
| 2025-09-23 | `8f2130b` | WanI2V refactor; face-mask arguments |
| 2025-10-02 | `c98cb08` | first working version |
| 2025-10-02 | `760b26f` | bias-kwargs refactor and bug fixes |
| 2025-10-03 | `957d700` | additional bug fixes |
| 2025-10-07 | `4ec6025` | progress |
| 2025-10-09 | `44e09da` | assigned custom class/path added while original class was restored elsewhere |
| 2025-10-21 | `60c6729` | replaced simple context inputs with generated pairwise prompt/token data; added `find_subsequence` |
| 2025-10-25 | `209e9d9` | switched to nested subsequence masks and a combined prompt; removed per-pair text embeddings |
| 2025-10-28 | `e702010` | separated base context from expanded prompt-token/mask data; removed 63 lines of older logic |
| 2025-10-30 | `2d19180` | deleted the assigned file (`0` insertions, `550` deletions) |

The last two functional revisions are particularly relevant to maintenance: `e702010` changed `full_prompt` handling, and the next author commit removed the path entirely. The checked-out later HEAD therefore cannot import this module from its assigned location.

## Static bug/gap review

Findings are cursory and based only on the latest extant snapshot (`e702010`, line numbers in that snapshot).

1. **`offload_model=True` does not offload T5 in the normal `t5_cpu=False` path.** `_build_context` moves `self.text_encoder.model` to CUDA at 224, but the cleanup at 309–317 only moves encoded tensors for `t5_cpu=True` and explicitly offloads CLIP. There is no corresponding T5 `.cpu()` operation for `offload_model=True`. This conflicts with the option’s documentation at 439–440 and can retain substantial T5 VRAM.
2. **Possible scheduler device mismatch under offload.** Timesteps are created on `self.device` (333 and 344), while the offload branch moves `latent` to CPU at 515–517. `scheduler.step` then receives CPU latent/noise tensors but the original `t` at 522–528. Whether this works depends on scheduler internals not inspected here; the file does not normalize `t` to the latent device.
3. **Prompt embeddings and masks are not obviously aligned.** `full_prompt` is the base prompt plus all gaze prompts (269), and `full_token_mask` is built from it (272–277), but `full_prompt_tokens` is encoded from only `[base_prompt]` (270). The model consequently receives base-only embeddings alongside masks describing the expanded prompt (319–321). This may be intentional for the custom model, but the local naming and code provide no explanation or assertion of the required alignment.
4. **Required input contracts are unchecked.** The method assumes an image shape that unpacks as three dimensions (159), compatible face-mask rank (174–176), valid positive dimensions/`max_area`, and the documented `4n+1` frame count (404–406), without validation. It also unconditionally pops `face_masks`, `link_text`, `descr_list`, and `timestep_bias_schedule` (235–236, 458–459, 491), and indexes the bias schedule once per timestep (496–497); malformed caller data produces late `KeyError`, shape errors, or `IndexError`.
5. **Pairwise prompt work scales quadratically.** `permutations(range(n_characters), r=2)` (242) creates two entries for every ordered pair and then tokenizes each pair’s descriptions (248–262). The expanded prompt and all GPU masks are retained (269–307). There is no bound or explicit resource warning for large `descr_list` values.
6. **Description normalization can alter user text.** `a_descr.capitalize()` (244) uppercases its first character and lowercases the remainder, while `b_descr` is not normalized. This can silently change names/acronyms and makes the two sides asymmetric.
7. **Mask construction does not assert that expected spans were found.** The only assertion (298) checks containment of a description mask within its gaze mask. There is no local check that each gaze/description mask is non-empty after tokenization/truncation, so an absent span could pass through as an all-false mask.
8. **Dead/incorrect documentation residue.** `find_subsequence` (35–47) has no executable call in this snapshot; it appears to be leftover from the earlier masking implementation. The return-shape documentation says `(C, N H, W)` at 444, omitting a separator and not clearly describing the actual tensor layout.

## Reproducibility and maintenance risks

- Explicit nonnegative seeds are propagated to the initial noise and scheduler generator (190–198, 452–454, 522–528), but `seed=-1` intentionally uses Python’s global random source (452). CUDA kernels, VAE/model operations, distributed execution, and USP/FSDP paths are not made deterministic in this file.
- Reproduction requires the exact external checkpoint directory and config object; their schemas, versions, and artifact identities are not documented here. The file dynamically imports optional USP support (129–144), and only two solver names are accepted (325–347).
- `offload_model` changes device placement and may expose the scheduler device issue above. Cleanup is mostly on the successful path; exceptions during context construction or sampling do not use a `finally` block to restore/offload modules.
- Most importantly, the file is deleted in the supplied revision. A reproducible checkout at HEAD must recover the historical blob or use a different integration path before this implementation can run.

## Validation performed

- Verified detached `HEAD` and file absence/presence with Git tree checks.
- Inspected author-filtered/path-followed metadata and file-only diffs.
- Parsed the latest 550-line snapshot with Python `ast.parse` successfully.
- Ran `git diff --check` on the latest functional revision; it reported no whitespace errors.
- No imports, model inference, builds, tests requiring runtime dependencies, web apps, or GPU execution were run.

## Limitations

This report intentionally does not inspect or evaluate imported modules, callers, configs, checkpoints, or any other source file. Consequently, the imported model/tokenizer/scheduler contracts and runtime device behavior cannot be confirmed beyond what is visible in this file. Findings marked “possible” require an in-scope runtime or dependency review to resolve.
