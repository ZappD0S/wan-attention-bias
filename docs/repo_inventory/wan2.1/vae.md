# `wan2.1/wan/modules/vae.py` review

## Scope and attribution

- **Snapshot:** submodule `wan2.1`, detached at `f7472d354e0cb46b2853f15bf97b8b283d4780f8`.
- **Reviewed file:** `wan2.1/wan/modules/vae.py` only. No other source or configuration file was read or analyzed.
- **Author filter:** exact Git author `gzappavi <gianluca.zappavigna@inria.fr>`.
- The file has two matching author commits in its followed history:
  - `d68ab4f3a60a618a0056bda446d884bab2a61663` (2025-10-21), `updated obsolete autocast function`.
  - `d7b424f09b048284759aa734d9aaf31bcaeb8d63` (2025-12-05), `wip`.
- The earlier file introduction is `65386b2e...` by `WanX-Video-1 <wanx.ai@alibabacloud.com>` and is not attributed to gzappavi; it is treated only as historical baseline, not as part of the author evolution.

## Behavior and data flow

1. **Causal temporal primitives.** `CausalConv3d` converts PyTorch 3-D padding into explicit `F.pad` padding with all temporal padding preceding the current chunk (lines 16–41). `CACHE_T = 2` (line 13) bounds the temporal history retained between streaming chunks.
2. **Feature blocks.** `RMS_norm` normalizes across channels for channel-first tensors (lines 44–61). `ResidualBlock` applies normalized SiLU/causal-convolution layers plus an identity or 1×1×1 causal shortcut (lines 205–247). `AttentionBlock` performs attention after flattening `(batch,time)` into independent 2-D frames (lines 250–292). Despite its docstring, it is spatial per-frame attention, not attention across time.
3. **Resampling.** Spatial resampling flattens `[B,C,T,H,W]` to `[B*T,C,H,W]`, applies 2-D nearest-neighbor/convolution operations, and restores the time axis (lines 157–160). The 3-D modes add a causal temporal convolution: downsampling uses stride 2 (lines 102–108, 162–179), while upsampling doubles time by producing two channel groups and reshaping them into adjacent frames (lines 91–96, 115–156). Cache entries carry prior feature frames, with `'Rep'` marking the first upsample call.
4. **Encoder.** `Encoder3d` starts from RGB (`3` input channels), builds residual/downsample stages from `dim_mult`, optionally inserts attention, and emits `2*z_dim` channels for posterior parameters (lines 295–350). In `WanVAE_.encode`, the first input frame is processed alone and the remaining frames in groups of four (`iter_ = 1 + (T-1)//4`, lines 582–602). The output is split into `mu` and `log_var`; only `mu` is shifted/scaled using the supplied two-element `scale` (lines 603–611).
5. **Decoder.** `Decoder3d` reverses the channel hierarchy, applies residual/middle blocks and configured spatial/temporal upsampling, then emits three channels (lines 409–466). `WanVAE_.decode` reverses latent scaling, applies a 1×1×1 convolution, decodes one latent time slice per iteration with streaming caches, and concatenates decoded slices (lines 613–640).
6. **Posterior and wrapper.** `reparameterize` samples `mu + exp(0.5*clamped(log_var))*epsilon` (lines 642–646). `WanVAE_.forward` returns reconstruction, mean, and log variance; `.sample` can return deterministic `mu` or a stochastic latent (lines 576–580, 648–654). `_video_vae` constructs on the meta device and assigns a checkpoint (lines 666–690). The public `WanVAE` stores 16-channel mean/std normalization constants, loads an evaluation/no-gradient model, encodes a list of `[C,T,H,W]` videos, and decodes a list of latent tensors (lines 693–773). Decode results are converted to float and clamped to `[-1,1]`.

## gzappavi commit evolution

- **`d68ab4f3`** removed `torch.cuda.amp as amp` and replaced the two wrapper contexts with `torch.amp.autocast("cuda", dtype=self.dtype)` (current lines 2, 759, 766). This is an API modernization, while retaining an explicit CUDA device type.
- **`d7b424f0`** is mostly formatting, but also changes behavior/API:
  - `WanVAE_.forward`, `encode`, `decode`, and `sample` now accept and propagate `scale`.
  - `encode` returns `(mu, log_var)` and applies latent mean normalization; the public wrapper consequently selects `[0]` (lines 603–611, 759–763).
  - `decode` reverses normalization (lines 616–621).
  - Log variance is clamped before exponentiation (lines 642–646).
  - Sampling delegates to `reparameterize` and adds deterministic mode (lines 648–654), and the public wrapper adds a `sample` method (lines 775–783).
- No matching author commit in the file history adds tests or runtime validation; the `wip` message also provides little intent documentation for the API changes.

## Static bug/gap review

Findings are static and cursory; they were not confirmed by model execution.

### Higher-confidence issues

1. **`z_dim` is presented as configurable but normalization is fixed at 16 channels.** `WanVAE.__init__` accepts `z_dim` (lines 694–700), but `mean` and `std` each contain exactly 16 values (lines 704–738). The tensor path then views those tensors as `[1, self.z_dim, 1, 1, 1]` (lines 604–607 and 616–619). Any public instance with `z_dim != 16` cannot reshape the 16-element statistics and is therefore incompatible unless the caller bypasses this wrapper and supplies matching scale tensors.
2. **The public wrapper hard-codes CUDA autocast despite exposing `device`.** `device` is configurable and used for the statistics/model (lines 694–700, 740–752), but `encode`, `decode`, and `sample` all request the CUDA autocast backend (lines 759, 766, 776). A CPU or other-device instance therefore has a portability mismatch (typically autocast is disabled or unsupported rather than honoring `dtype`).
3. **Empty temporal inputs are not guarded.** For `T == 0`, the encoder iteration count is zero (line 586), so `out` is read before assignment at line 603. Likewise, decoding a latent with zero time steps skips the loop beginning at line 624 and returns the unassigned `out` at line 640. The file documents shapes but does not explicitly reject this case.

### Medium-confidence/API and maintenance gaps

4. **Mutable list defaults include a stateful `feat_idx`.** Several `forward` methods default to `feat_idx=[0]` (lines 113, 225, 352, and 468), while cache paths mutate `feat_idx[0]` (for example lines 117–120 and 242–244). Calls that omit the argument share and advance the same list across invocations. The main encode/decode paths pass fresh lists (lines 589 and 625), but direct use of these modules is order-dependent. Constructor defaults such as `dim_mult=[1, 2, 4, 4]` are also mutable lists (for example lines 300–303), even though this file does not currently mutate them.
5. **Attention naming/documentation does not match its computation.** The docstring calls the block “Causal self-attention” (line 252), but reshaping `(b,c,t,h,w)` to `(b*t,c,h,w)` (line 270) makes each time slice independent; the attention call at lines 273–287 has no temporal keys or causal temporal mask. This is a documentation/design clarity gap if temporal attention was intended.
6. **The wrapper's `sample` contract is ambiguous and likely surprising.** `WanVAE_.sample` encodes its `imgs` argument and returns a latent sample (lines 648–654), while the public method names its inputs `zs` but passes them into that encoder (lines 775–778). It then clamps the latent sample to `[-1,1]` (lines 779–780), unlike public `encode` and unlike the model's sampling method. The method therefore does not decode `zs`, and the clamp can alter the sampled latent distribution; callers need an explicit contract.
7. **Decoder channel adjustment is hard-coded for four stages.** The decoder accepts arbitrary `dim_mult` (lines 410–418), but adjusts `in_dim` only for literal stage indices 1, 2, and 3 (lines 444–452). The channel halving needed after each preceding upsample is architecture-length dependent; configurations with more than four stages can reach a channel mismatch, and the assumption is undocumented.
8. **Checkpoint loading has weak reproducibility/compatibility controls.** `_video_vae` uses a relative default checkpoint path through `WanVAE` (line 697), loads with `torch.load(..., map_location=device)` without a checkpoint hash/version or shape/config validation (lines 686–688), and does not request a restricted weights-only load. This makes provenance, accidental checkpoint mismatch, and serialized-checkpoint safety harder to control.

## Reproducibility and maintenance risks

- Results depend on an external checkpoint path and the 16 hard-coded latent statistics; neither the checkpoint identity nor the statistics' provenance is recorded in this file.
- Stochastic `reparameterize` uses an unseeded `torch.randn_like`, so `.sample` is nondeterministic unless the caller controls global RNG state; the deterministic option returns `mu` but exists only on the internal model method, not the public wrapper.
- Streaming behavior depends on mutable instance cache fields (`_feat_map`, `_enc_feat_map`) reset by `clear_cache` (lines 656–663). Concurrent/interleaved calls on one model instance would share state and are not protected.
- The spelling `temperal_*` is repeated in constructor arguments and attributes (lines 303, 417, 542, 552), creating an API compatibility burden. Defaults are positional/list-based and checkpoint architecture assumptions are implicit.
- The public wrapper assumes inputs are already on the configured device and that CUDA autocast is available; its short shape docstring does not state those requirements.

## Validation performed

- Parsed the assigned file with Python's `ast.parse`: **OK**.
- Ran `git diff --check` for the file from its initial upload through the checked-out snapshot: **OK**.
- Inspected only the assigned file, its Git history, and diffs limited to that path. No model inference, build, web-app execution, import test, or checkpoint load was performed.

## Limitations

This report cannot establish runtime shape behavior, device-specific autocast behavior, checkpoint compatibility, numerical reconstruction quality, or whether any wrapper method is called with a different contract elsewhere, because callers, tests, dependencies, configurations, and checkpoints were outside the permitted scope. Line references refer to the reviewed snapshot at `f7472d354e0cb46b2853f15bf97b8b283d4780f8`.
