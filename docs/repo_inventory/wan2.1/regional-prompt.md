# wan2.1 regional-prompt inventory

## Scope and attribution

- **Repository state:** `wan2.1` submodule, detached at `f7472d354e0cb46b2853f15bf97b8b283d4780f8`.
- **File reviewed:** `wan/regional_prompt/image2video.py` (669 lines at this revision)
- **Attribution filter:** Git history was queried for this path from `HEAD`, retaining commits whose author name and email are exactly `gzappavi <gianluca.zappavigna@inria.fr>`. There are **25 matching commits**, from 2025-10-30 through 2026-05-07; no non-matching-author path commits were found in that history. No source outside the assigned file was read or analyzed.

## Regional-prompt behavior and data flow

### Construction and device placement

`WanI2V.__init__` (`image2video.py:40-149`) records configuration, rank, and memory/distribution options, then constructs a T5 text encoder, Wan VAE, CLIP encoder, and custom Wan model from the checkpoint directory. The text encoder is initially constructed on CPU (`85-92`); VAE and CLIP target the selected CUDA device (`96-106`). The transformer is put in evaluation/no-gradient mode (`108-114`), converted to `config.param_dtype` (`141`), and optionally sharded (`143-147`). USP replaces attention/model forward methods and sets a sequence-parallel size (`119-136`). Distributed barriers are used during initialization (`138-139`).

### Image, geometry, and conditioning latents

- `_img_to_tensor` (`151-152`) converts a PIL image to a tensor, maps `[0, 1]` to approximately `[-1, 1]`, and moves it to the configured CUDA device.
- `_get_lat_h_w` (`154-171`) derives latent height/width from input height/width, `max_area`, VAE stride, and patch size. It currently asserts `height / width <= 1` (`157`).
- `_generate_noise` (`173-192`) samples a 16-channel float32 latent noise tensor with temporal length `ceil(frame_num / vae_stride[0])`.
- `_build_latents` (`194-230`) rescales the image to VAE-aligned dimensions, concatenates it as the first frame with zero-valued remaining frames (`215-224`), encodes the sequence with the VAE, and prepends a temporal mask whose first latent frame is one (`225-228`). It also computes a patch/sequence-parallel-aligned `max_seq_len` (`207-213`).

The public `generate` method (`601-669`) obtains a seed, creates a device generator, generates initial noise from `img.size`, and delegates to `generate_from_latents`.

### Prompt/context preparation

`_build_context` (`232-379`) consumes an untyped `bias_kwargs` mapping. It requires `prompt_data_list` and `general_prompt` (`261-267`). For every sentence/data pair (`269-271`) it:

1. encodes the sentence with T5 (`273`);
2. optionally encodes a per-sentence character image with CLIP (`276-281`);
3. tokenizes the full sentence (`285-294`);
4. tokenizes every control prompt and finds nested subsequences in the full sentence (`296-313`);
5. finds nested character-description subsequences inside each control prompt (`315-336`), prepending zeros for prior sentence context so masks line up with concatenated context;
6. records control indices and the resulting masks in `tokens_data_list` (`338-344`).

Sentence contexts and full-token masks are concatenated (`346-350`). An optional general prompt is encoded separately (`264-267`). CLIP also encodes the main input image (`364`); T5 and CLIP modules may be moved back to CPU when offloading is enabled (`352-368`). The generated context objects, image contexts, token data, and mask are returned through `bias_kwargs` (`370-379`). The negative prompt is encoded as `context_null` (`240-252`); an empty supplied negative prompt falls back to `self.sample_neg_prompt` (`240-241`).

The implicit data contract is therefore that each `prompt_data_list` item has `control_prompts` (iterable pairs of indices and prompt data), `single_char_img`, and each control item has `prompt` and `char_descr_list`. This contract is not represented by annotations in the public methods.

### Denoising, guidance, and output

`_get_scheduler` (`381-405`) supports only `unipc` and `dpm++`; all other solver names raise `NotImplementedError`. `generate_from_latents` (`456-599`) creates the schedule, optionally finds a starting index for `t_i` (`477-490`), accepts either an integer or an existing `torch.Generator` (`492-505`), prepares image latents and context, then iterates over decreasing timesteps (`537-558`).

At each step `_compute_noise_pred` (`407-454`) runs the model once with conditional context and regional bias, then once with negative context and `bias=False` (`427-444`). Classifier-free guidance combines the two predictions (`450-452`). The sampler updates the latent (`566-576`), and similarity masks are moved to CPU and accumulated. The model can be offloaded after sampling (`579-581`). Rank zero decodes the final latent with the VAE (`583-585`); all ranks synchronize, and the result includes stacked similarity masks as `extra_data` (`587-597`). Nonzero ranks return `None`.

## gzappavi commit evolution

All entries below are commits selected because they touched the assigned path; only its diffs are summarized. Subjects are reproduced from Git; descriptions summarize the scoped diff.

| Date | Commit | Scoped evolution |
|---|---|---|
| 2025-10-30 | `2d19180` | Created the regional-prompt implementation. |
| 2025-11-03 | `e595db8` | Reworked prompt/control data from the earlier gaze/link form toward `control_prompts` and token masks. |
| 2025-11-05 | `9aa8930` | Adjusted full-prompt construction/encoding; temporarily added a full-prompt debug print. |
| 2025-11-06 | `7e77413` | Formatting and control-index representation cleanup. |
| 2025-12-05 | `d7b424f` | Removed permutation-based prompt construction, introduced the module negative-prompt default, and changed the prompt API. |
| 2025-12-08 | `9c66f45` | Added pyright-oriented annotations/ignore comments. |
| 2025-12-08 | `3e3450e` | Removed a prompt consistency assertion. |
| 2025-12-11 | `2c7611c` | Added timestep normalization passed in bias kwargs and briefly added timestep logging. |
| 2025-12-12 | `3e150c0` | Corrected normalization to divide by `num_train_timesteps`. |
| 2025-12-12 | `aa51486` | Removed attention-weight collection from model results and returned metadata. |
| 2025-12-15 | `e2a0a79` | Removed the timestep debug print. |
| 2026-01-15 | `0c85186` | Changed context preparation and the public API to handle multiple sentences. |
| 2026-01-16 | `53a2ee8` | Fixed control-prompt iteration and context tensor shape. |
| 2026-01-16 | `e696613` | Tightened tokenization/mask handling and added text-encoder offloading. |
| 2026-01-21 | `77c3e0f` | Split noise creation from conditioning and added `generate_from_latents`/`t_i` support for arbitrary starting latents. |
| 2026-01-26 | `5df2134` | Fixed accumulation of full-token masks across sentences rather than overwriting the prior value. |
| 2026-01-27 | `b47b881` | Corrected image-shape handling and added the height/width aspect assertion. |
| 2026-01-27 | `5b00890` | Follow-up shape/stride cleanup and explicit model device movement. |
| 2026-02-03 | `8d477bf` | Added PIL image typing and concept-weaver context data: optional general prompt and per-sentence character-image contexts. |
| 2026-02-24 | `c6d5608` | Removed face-mask conditioning, retained similarity-mask output, and initialized the temporary sampler output. |
| 2026-03-23 | `390de16` | Removed obsolete pyright suppressions around distributed calls. |
| 2026-04-07 | `2a7238b` | Replaced the earlier negative-prompt text with a camera-motion-aware version. |
| 2026-04-14 | `38b21b3` | Broadcast integer seeds in `generate_from_latents` and documented decreasing timesteps. |
| 2026-05-06 | `dfc103b` | Added another transformer dtype conversion before sharding/device placement. |
| 2026-05-07 | `1593f4d` | Added an extra CUDA cache clear after text-encoder offload. |

The trajectory is an iterative prototype: prompt-region masks and multi-sentence context were added first, then arbitrary-latent resumption, concept-weaver image context, similarity-mask output, distributed seeding, and memory/dtype changes. Commit subjects include multiple `wip`, `progress`, and bug-fix commits, so the history documents intent more reliably than a stable API specification.

## Static bug/gap review

Findings are cursory and based only on the two scoped files; they were not tested with model components.

1. **`generate_from_latents` mutates caller input and makes repeated calls fragile (high).** At `image2video.py:507`, `bias_kwargs.pop("timestep_bias_schedule")` occurs before the defensive copy at `509`. A caller reusing the same mapping for a second call will normally receive `KeyError`; the later copy does not undo the mutation. The other prompt keys are popped from a copy inside `_build_context`, which makes this one mutation particularly inconsistent.

2. **The public random-seed path bypasses distributed seed broadcast (medium).** `generate` resolves `-1` to a random integer and creates a generator at `650-652`, then passes that generator to `generate_from_latents` (`669`). The broadcast branch at `495-498` runs only when `generate_from_latents` receives an `int`, not a `torch.Generator`. Thus, with multiple ranks and the public default/random-seed path, ranks can begin from different noise. Explicit identical seeds avoid this particular issue; externally supplied generators also bypass the broadcast.

3. **Arbitrary-latent resume does not preserve multistep scheduler history (medium, static risk).** A fresh scheduler is created at `473-475`, and `t_i` merely selects a suffix of its timestep tensor (`477-490`). No scheduler state or prior model evaluations are accepted or restored. Because the selected solver classes are multistep schedulers, resuming at a later timestep may not be equivalent to continuing the original run. The API should either define this approximation or carry the required scheduler state.

4. **Several input constraints are undocumented or unchecked (medium).** `_get_lat_h_w` asserts a landscape/non-taller image at `157`, so portrait input fails with an assertion rather than a documented validation error. Width, positive `max_area`, and sensible `frame_num` are not checked (`154-171`, `173-191`, `194-230`). The docstring says `frame_num` should be `4n+1` (`generate` docstring around `620`), but no such check exists; values below one can produce invalid/negative frame dimensions at `185`, `215`, or `225`. `sampling_steps=0` leaves no loop iterations and later makes `torch.stack([])` fail at `593`.

5. **Non-RGB PIL images are not rejected before a hard-coded three-channel path (medium).** `_img_to_tensor` accepts whatever channel count `TF.to_tensor` produces (`151-152`), while `_build_latents` allocates `torch.zeros(3, ...)` (`215`) and concatenates it with the converted image (`221`). Grayscale or four-channel images can consequently fail at concatenation or downstream encoders. The PIL type annotation alone does not establish RGB input.

6. **Negative-prompt documentation disagrees with the actual default (low/medium).** Both generation entry points default `n_prompt` to the module constant (`469`, `610`). The fallback to `config.sample_neg_prompt` occurs only when the caller explicitly passes an empty string (`240-241`). The `generate` docstring says an omitted negative prompt uses `config.sample_neg_prompt` (around `627`), which is not what the signature implements.

7. **The prompt/bias API is implicit and weakly typed (maintenance gap).** `bias_kwargs` has no declared mapping type in `_build_context`, `generate_from_latents`, or `generate` (`237`, `458`, `608`). Required keys are popped without validation (`261-262`, `507`), and `prompt_data_list`/`control_prompts` have a non-obvious nested structure (`269-344`). Mixed optional character-image presence is only caught after processing by an assertion (`370-372`), rather than reported as a clear contract error. This makes caller compatibility and future refactors difficult to verify.

8. **Runtime checks rely on `assert` (low).** Input/invariant checks at `157`, `335`, `370-372`, `432`, and `544-545` are disabled under Python optimization. User-facing validation and safety-critical shape checks should use explicit exceptions if they must hold in production.

9. **The implementation is CUDA-bound despite CPU-related knobs (low/operational).** The device is always constructed as `cuda:<device_id>` (`75`), sampling uses `torch.autocast("cuda", ...)` (`527-530`), and offload cleanup calls CUDA synchronization (`587-588`). `t5_cpu` and `init_on_cpu` control staging/offload, not CPU-only execution. This should be explicit in the API/environment contract.

10. **There is a small source-maintenance residue.** `Any` and `TypedDict` are imported at `image2video.py:11` but unused in the current file; the `SentenceData` declaration that previously used them was removed in the 2026-02-24 commit. The generation docstring also names `input_prompt` although the parameter is `prompt_sentences` (`603-604`), and its return-shape prose contains a malformed `(C, N H, W)` expression (around `641-645`).

## Reproducibility and maintenance risks

- Randomness is split between initial noise generation in `generate` and sampler-generator use in `generate_from_latents` (`650-669`, `568-573`); the generator's consumed state is not surfaced. Distributed reproducibility therefore depends on how the entry point is invoked, especially for random defaults.
- The `timestep_bias_schedule` is indexed against the full schedule using `i + i0` (`537-540`) but its length is never validated. A short schedule fails late; an oversized schedule is silently ignored. Resume callers also need to know whether the schedule is aligned to all timesteps or only the resumed suffix.
- Model/text/CLIP movement and cache clearing are manually interleaved (`243-249`, `352-368`, `532-535`, `579-588`). An exception during sampling can leave modules on a different device than the caller expects; there is no `try/finally` restoration boundary.
- Similarity masks for every timestep are retained and then stacked (`538`, `560-564`, `593`), which can create substantial host-memory pressure for long or high-resolution runs even though it reduces GPU residency.
- The public methods lack return/argument types for the bias mapping, latent, scheduler options, and result. Together with positional arguments and many defaults (`456-471`, `601-610`), this leaves the evolving prompt-data schema as the main compatibility risk.
- Configuration/checkpoint compatibility is assumed: stride, patch, dtype, text lengths, and model outputs are consumed without local shape/schema checks (`81-112`, `154-230`, `431-454`). This report did not inspect the external definitions, so it cannot establish whether those assumptions are guaranteed.

## Validation performed

- Parsed the scoped Python file with `ast.parse`: it succeeded.
- Ran `git diff --check` for the two scoped paths: no whitespace errors reported.
- Confirmed the submodule `HEAD` is `f7472d354e0cb46b2853f15bf97b8b283d4780f8` and the scoped working-tree paths were clean.
- Inspected only Git metadata/diffs for the assigned path and its current contents.
- Did **not** run model inference, builds, web applications, imports requiring model dependencies, or generated tests/artifacts.

## Limitations

This is a static inventory of the assigned file at the specified detached revision. It does not verify external model signatures, tokenizer behavior, scheduler numerical semantics, distributed launch behavior, checkpoint/config values, or actual tensor shapes at runtime. Commit summaries and diff review establish scoped evolution and authorship, but cannot prove why every historical change was made or whether the current implementation works end-to-end.
