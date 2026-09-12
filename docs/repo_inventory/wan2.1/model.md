# `wan2.1/wan/modules/model.py` report

## Scope and attribution

- **Repository state reviewed:** detached `f7472d354e0cb46b2853f15bf97b8b283d4780f8`.
- **Scope:** only `wan2.1/wan/modules/model.py` was read or analyzed. Git metadata and diffs were inspected for attribution; no other source/config file was reviewed.
- **Author filter:** exact author identity `gzappavi <gianluca.zappavigna@inria.fr>`.
- **Matching history:** one commit touches this file:
  - `d68ab4f3a60a618a0056bda446d884bab2a61663` (2025-10-21), *updated obsolete autocast function*.
  - Parent: `60c6729f332679d1bb2a51118736204d1741c6c3`.
- The matching commit is an ancestor of the reviewed HEAD. No earlier matching-author commit for this path was found in the available history.

## Architecture and control flow

`WanModel` is a diffusion-transformer backbone. It accepts a batch represented as lists of variable-size video tensors and returns a list of reconstructed tensors.

1. **Position utilities (lines 17–69):** `sinusoidal_embedding_1d` creates float64 sinusoidal timestep features. `rope_params` builds complex rotary frequencies, and `rope_apply` splits those frequencies across temporal, height, and width axes, applies them to valid (unpadded) tokens, and leaves padding unchanged.
2. **Normalization and attention (lines 72–228):** `WanRMSNorm` and `WanLayerNorm` force normalization calculations through float32 before restoring the input type. `WanSelfAttention` projects Q/K/V, optionally RMS-normalizes Q/K, applies RoPE to Q/K, delegates attention to `flash_attention`, and projects the result. Text-to-video cross-attention attends to text; image-to-video cross-attention separately attends to image and text portions of the context, then sums both outputs.
3. **Transformer block (lines 237–316):** each `WanAttentionBlock` uses pre-normalized self-attention, cross-attention, and a GELU feed-forward network. Six timestep-conditioned modulation vectors provide additive/multiplicative modulation and residual gates for the attention/FFN paths.
4. **Output and image projection (lines 319–368):** `Head` applies two-way timestep modulation and projects each hidden token to all channels within a 3D patch. `MLPProj` maps 1,280-dimensional image features into the transformer width; first/last-frame mode adds a learned 514-token positional tensor.
5. **Construction (lines 371–490):** a strided `Conv3d` patchifies the input; text and timestep MLPs produce conditioning; `num_layers` blocks are instantiated; a fixed 1,024-position RoPE table is created. `t2v` selects text cross-attention; every other accepted model type selects image-style cross-attention. Image projection is instantiated only for `i2v` and `flf2v`.
6. **Forward pass (lines 492–581):** image modes require `clip_fea` and `y`; optional `y` channels are concatenated to each input. Inputs are patchified, flattened, and zero-padded to `seq_len`. Timestep features are projected to six modulation vectors. Text is padded to `text_len` and embedded; image features, when supplied, are prepended. All blocks run, the head projects patches, and `unpatchify` (lines 583–606) restores channel/temporal/spatial order using an einsum.
7. **Initialization (lines 608–630):** linear layers receive Xavier weights and zero biases, text/time linear layers are then normal-initialized, patch convolution is Xavier-initialized, and the output head weight is zeroed.

## gzappavi commit evolution

The sole matching commit is a focused PyTorch autocast API migration. It:

- removes `import torch.cuda.amp as amp`;
- changes the two decorators at lines 30 and 41 from `@amp.autocast(enabled=False)` to `@torch.amp.autocast("cuda", enabled=False)`;
- changes five context managers (current lines 296, 304, 311, 343, and 545) to `torch.amp.autocast("cuda", dtype=torch.float32)`.

The diff is 7 insertions and 8 deletions, with no architectural or tensor-flow change evident in the file. The migration is present in the reviewed HEAD.

## Static bug/gap review

Findings are cursory and based only on this file.

| Severity | Evidence | Finding / consequence |
|---|---|---|
| Medium | Lines 552–558, 570; lines 457–459 | Every text context shorter than `text_len` is zero-padded before a biased text MLP, but `context_lens` is always `None`. If the attention dependency interprets `None` as no key-length mask (the apparent call-site intent), padded positions become nonzero after projection and can participate in cross-attention. Exact-length context inputs avoid this gap. |
| Medium | Lines 479–484, 538; lines 57–59 | RoPE is hard-coded to 1,024 positions per axis, while the only forward check compares total token count to caller-supplied `seq_len`. There is no explicit check that each `(F, H, W)` grid axis is at most 1,024; an oversized axis can make frequency expansion/attention fail rather than produce a controlled validation error. |
| Medium | Lines 436, 466, 486–487, 560–562 | `vace` is accepted, but is routed to image-style cross-attention and does not create `self.img_emb`. If a `vace` forward receives non-`None` `clip_fea`, line 561 attempts to use a missing attribute. The combination is not rejected with a clear error. |
| Low/maintenance | Lines 476–484, 525–527 | `self.freqs` is a plain tensor rather than a registered buffer. Forward manually moves it to the patch layer's device, but it is not part of normal module buffer/state-dict handling. This makes device movement, serialization, altered RoPE settings, and distributed/checkpoint maintenance less robust. |
| Low/maintenance | Lines 30, 41, 296, 304, 311, 343, 545 | The migrated API hard-codes the `cuda` device type, and several regions request float32 autocast. On CPU-only execution this may be disabled or warned about, and older PyTorch versions may not provide the new API/signature. The file has no compatibility guard; deployment therefore depends on the PyTorch version/device behavior. |
| Low/maintenance | Lines 529–530 | `zip(x, y)` silently truncates on mismatched list lengths. The docstring says the conditional list has the same shape as `x`, but no batch-cardinality check gives a clear failure for malformed input. |
| Low/maintenance | Lines 359–366 | First/last-frame positional embedding is fixed at `[1, 514, 1280]`, and the input is reshaped as paired samples. The required 514-token and paired-batch invariant is implicit; incompatible feature shapes fail at reshape/addition rather than being validated. |

## Reproducibility and maintenance risks

- The model depends on implicit shape contracts: even `dim`, compatible head dimensions, fixed text length, paired first/last-frame image features, and a 1,024-position RoPE table. Some are asserted (lines 19, 32, 112, 436, 477, 538), but several mode- and feature-shape contracts are not.
- The plain `freqs` tensor is deterministically regenerated during construction but is absent from registered module state. A checkpoint cannot independently preserve a changed/custom frequency table.
- Attention masking behavior is delegated to `flash_attention`, while this file always supplies `None` for context lengths. Correctness for padded text consequently depends on an external contract not documented here.
- The autocast migration improves API currency but increases sensitivity to PyTorch version and execution device. The explicit CUDA device type appears in decorators as well as forward-time contexts.
- Initialization uses random draws (`modulation` parameters and normal/Xavier initialization), so reproducible construction still requires caller-controlled random seeds and a stable PyTorch/runtime stack; seed handling is not provided by this file.

## Validation performed

- Parsed the current file with Python's AST parser: **OK**.
- Ran `git diff --check` on the gzappavi commit for this path: **no whitespace errors reported**.
- Confirmed the exact-author path history and that the matching commit is an ancestor of the reviewed HEAD.
- No model inference, build, web-app execution, or dependency-source inspection was performed.

## Limitations

This is a static, single-file review. The behavior and API contract of `flash_attention`, Diffusers mixins, callers, checkpoint loaders, and runtime PyTorch versions were not inspected. Therefore the masking, device, and autocast consequences above are identified from call sites and documented shapes; runtime confirmation would require tests outside the permitted scope.
