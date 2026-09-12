# `wan2.1/wan/image2video.py` — static repository report

## Scope and attribution

- Repository: `/home/zapp/Documents/wan_experiments`.
- Submodule: `wan2.1`, detached at `f7472d354e0cb46b2853f15bf97b8b283d4780f8`.
- Assigned scope: `wan2.1/wan/image2video.py` only. The current file is 372 lines and is unchanged between its last file-touching commit (`d68ab4f`) and the checked-out submodule HEAD.
- Author filter: exact Git author `gzappavi <gianluca.zappavigna@inria.fr>`. Ten commits in the file's path history match that identity. The checked-out HEAD is also authored by that identity, but does not touch the assigned path and is therefore excluded from file evolution below.
- No model inference, build, web app, or source/config-file inspection outside this assigned file and its Git metadata/diffs was performed.

## Current pipeline behavior and data flow

### Initialization (`WanI2V.__init__`, lines 31–132)

1. A CUDA device is selected directly from `device_id` (line 67), and configuration values such as parameter dtype, VAE stride, patch size, and negative prompt are retained.
2. T5 is constructed initially on CPU, loading its checkpoint/tokenizer paths from `checkpoint_dir` (lines 77–84). The VAE and CLIP are constructed on the selected CUDA device (lines 86–98).
3. `WanModel` is loaded from the checkpoint directory with `config.param_dtype`, then put in evaluation/no-gradient mode (lines 100–102).
4. Optional FSDP and USP paths alter model placement/wrapping. USP replaces attention and model forwards and obtains a sequence-parallel world size (lines 104–122); an initialized distributed process group is synchronized with a barrier (lines 124–130).
5. `sample_neg_prompt` is saved for use when no negative prompt is supplied.

### Generation (`WanI2V.generate`, lines 134–372)

1. The input image is passed through `torchvision.transforms.functional.to_tensor`, normalized from approximately `[0, 1]` to `[-1, 1]`, and moved to CUDA (line 184).
2. The image aspect ratio and `max_area` determine latent height/width, aligned down to VAE stride and patch-size multiples (lines 186–203). A sequence length is calculated and rounded up to a multiple of `self.sp_size` (lines 204–210).
3. A CUDA `torch.Generator` is seeded. Initial Gaussian latent noise has 16 channels and temporal size `(frame_num - 1) // 4 + 1` (lines 212–223).
4. A conditioning mask is built and reshaped to four channels by 21 latent-time groups (lines 225–231). The first source frame is repeated across four temporal positions; later positions are zero. The image is resized to the aligned spatial dimensions and concatenated with `F - 1` zero frames, then VAE-encoded. The mask and encoded image latent are concatenated into `y` (lines 254–267).
5. T5 produces positive and negative text contexts. Depending on `t5_cpu`, the text encoder runs on CPU or is moved to CUDA; encoded contexts are placed on CUDA, and the encoder is optionally returned to CPU (lines 237–247). CLIP produces an image embedding from the image with a singleton temporal dimension, then is optionally returned to CPU (lines 249–252).
6. Under CUDA autocast, no-grad, and the model's optional `no_sync` context, either UniPC or DPM++ flow scheduling is selected. Unsupported solver names raise `NotImplementedError` (lines 275–298).
7. Each timestep performs two model evaluations with the same image latent conditioning: conditional context (`arg_c`) and negative context (`arg_null`) (lines 303–335). Classifier-free guidance is computed as `unconditional + guide_scale * (conditional - unconditional)` (lines 337–339).
8. With `offload_model=True`, predictions and the evolving latent are moved to CPU between operations and CUDA's allocator cache is cleared. The scheduler advances the latent (lines 341–352). The transformer is moved to CPU only after all sampling steps (lines 357–359).
9. Rank 0 decodes the final latent with the VAE and returns `videos[0]`; nonzero ranks return `None`. Distributed ranks synchronize after cleanup (lines 361–372).

## gzappavi commit evolution

The path history identifies the following ten matching commits (oldest first):

| Date | Commit | Subject | File-level evolution |
|---|---|---|---|
| 2025-09-18 | `dda8f6f` | `fixed formatting` | Formatting/line-wrapping cleanup; no intended pipeline change. |
| 2025-09-18 | `62af287` | `improved mask creation` | Added an `einops.rearrange` helper, but called it with `frame_num=81`, establishing the fixed 81-frame mask behavior that remains in the current file. |
| 2025-09-22 | `37ac1cd` | `fixed model weights dtype that caused gpu out of memory bug` | Switched model loading to `CustomWanModel.from_pretrained(..., torch_dtype=config.param_dtype)` and modernized the autocast block. |
| 2025-09-23 | `8f2130b` | `refactored WanI2V; introduce arguments for face_masks` | Large experimental refactor: added latent/context/scheduler/noise-prediction helpers, face-mask handling, bias-related arguments, and similarity-mask output. |
| 2025-10-02 | `c98cb08` | `first working version` | Extended the experimental path with description/link text tokens, bias kwargs, boolean face masks, and a tuple return containing video and similarity masks. |
| 2025-10-02 | `760b26f` | `refactred bias kwargs passing; fixed bugs` | Propagated bias settings to unconditional prediction and changed timestep/similarity-mask handling. |
| 2025-10-03 | `957d700` | `fixed a bunch of bugs` | Renamed link-text plumbing to list-based `link_list`/`link_tokens_list`. |
| 2025-10-07 | `4ec6025` | `progress` | Reworked temporal latent sizing and VAE input/mask construction; stacked similarity masks in the returned tuple. |
| 2025-10-09 | `44e09da` | `readded original WanI2V class and renamed modified one to CustomWanI2V` | Reverted this file from the experimental custom/bias/face-mask implementation to the original `WanI2V` path, including removal of the extra outputs and restoration of the fixed mask. |
| 2025-10-21 | `d68ab4f` | `updated obsolete autocast function` | Restored dtype-aware `WanModel` loading, switched from `torch.cuda.amp.autocast` to `torch.amp.autocast("cuda", ...)`, and applied formatting changes. |

Thus, the current assigned file is the post-revert/original pipeline with later dtype and autocast updates, not the intermediate face-mask/bias/similarity-mask design.

## Static bug and gap review

Findings are cursory and based only on the assigned file and its history.

| Severity | Finding | Evidence and impact |
|---|---|---|
| **High** | `frame_num` is documented as configurable but the conditioning mask is hard-coded to 81 frames. | Noise and VAE input use `frame_num` (lines 217 and 261), while the mask always starts as `(1, 81, lat_h, lat_w)` (line 225) and is reshaped to 21 latent-time groups (lines 227–231). Any supported value other than 81 (for example another `4n+1` value) can make the mask temporal dimension disagree with the encoded image latent, causing concatenation failure or invalid conditioning. |
| **Medium** | The code relies on unvalidated shape and parameter preconditions. | The docstring says `frame_num` should be `4n+1` (line 159), but no check enforces it. There are also no explicit checks for positive `max_area`, nonzero image width/height, a 3D/three-channel image, or positive `sampling_steps`; failures occur later in arithmetic, interpolation, tensor construction, or scheduler setup. |
| **Medium** | Default-seed reproducibility is not established across distributed ranks. | For `seed < 0`, each process independently calls Python `random.randint` (line 212), then uses the resulting seed to create rank-local CUDA noise (lines 213–223). There is no broadcast or other rank synchronization of that seed. Explicit nonnegative seeds are more reproducible, but CUDA kernels, AMP, FSDP, and USP can still introduce platform/runtime-dependent variation. |
| **Medium** | `offload_model` provides only partial/coarse offloading despite its broad docstring wording. | T5 and CLIP are returned to CPU after encoding (lines 241–252), but the transformer is moved to CUDA before the entire sampling loop and returned to CPU only afterward (lines 317–320 and 357–359). Predictions and latents are copied to CPU, which reduces intermediate activation residency, but the largest model remains on GPU for all steps. An exception inside the loop also bypasses the normal final transformer offload/cleanup path. |
| **Medium** | Runtime compatibility is tied to the newer `torch.amp` API without a local fallback or version assertion. | The current call is `torch.amp.autocast("cuda", dtype=...)` (line 276), introduced by `d68ab4f` as a replacement for the old `torch.cuda.amp` form. Environments using an older PyTorch release may fail at runtime before sampling; the file does not state or check the required version. |
| **Low** | Temporal downsampling assumptions are duplicated and only partly configuration-driven. | Sequence length uses `self.vae_stride[0]` (line 205), while noise uses a literal divisor of 4 (line 217) and mask grouping uses literal repeats of 4 (lines 227–230). This is consistent with the current model's apparent four-frame temporal grouping, but changing the configured VAE stride without changing this file can produce mismatched latent shapes. |
| **Low** | The public input/output contract is ambiguous or inaccurate. | The input docstring labels `img` as `PIL.Image.Image` while describing it as a tensor with shape `[3, H, W]` (lines 154–155). The return description writes `(C, N H, W)` and hard-codes 81 frames despite `frame_num` being an argument (lines 176–182). No runtime type/shape normalization beyond `to_tensor` is documented. |

## Reproducibility and maintenance risks

- Results depend on checkpoint contents and configuration fields supplied externally (`config.*` and `checkpoint_dir`); this file does not record versions, checkpoint hashes, or a complete generation manifest.
- `seed=-1` intentionally introduces nondeterminism. Even with a fixed seed, deterministic behavior is not guaranteed by this file for CUDA/AMP/distributed execution.
- Output spatial size is derived from `max_area` and alignment floors rather than returned or logged as metadata. Small or unusual inputs can therefore produce dimensions different from a caller's intuitive target.
- The pipeline is GPU-only by construction (`cuda:{device_id}` at line 67), and memory behavior depends on optional FSDP/USP setup and the `offload_model` path. The file does not provide a CPU fallback or explicit VRAM diagnostics.
- The normal cleanup sequence is not protected by `try/finally` (lines 275–370). Errors during model calls or scheduler steps can leave large modules resident and can complicate recovery in long-lived processes.
- The file's history includes a substantial experimental custom-model/bias/face-mask branch followed by a revert. That evolution increases the value of keeping call contracts and the frame-mask invariant explicitly tested, especially because the current docstring still describes a generic configurable pipeline while the implementation retains an 81-frame assumption.

## Validation performed

- Verified the submodule HEAD is `f7472d354e0cb46b2853f15bf97b8b283d4780f8` and inspected Git path history, author identities, commit parents, subjects, and diffs restricted to `wan/image2video.py`.
- Confirmed the current file has no diff from `d68ab4f` for this path and contains 372 lines.
- Parsed the assigned Python file with `ast.parse` successfully.
- Ran `git diff --check` for the assigned path; no whitespace errors were reported.
- No inference, build, web app, or runtime generation test was run.

## Limitations

This report does not validate tensor shapes against the implementations of the imported encoders, VAE, model, schedulers, distributed wrappers, or callers, because those are outside the assigned-file scope. Consequently, shape-related findings are static consequences of this file's own formulas and hard-coded dimensions, not end-to-end runtime observations. Dependency-version compatibility, checkpoint correctness, distributed collectives, and output quality likewise remain untested.
