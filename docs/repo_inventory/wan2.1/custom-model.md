# Static report: `wan2.1/wan/modules/custom_model.py`

## Scope and attribution

- Repository: `/home/zapp/Documents/wan_experiments/wan2.1`.
- Reviewed revision: detached `f7472d354e0cb46b2853f15bf97b8b283d4780f` (`f7472d3`, “further changes to reduce VRAM usage”).
- Reviewed file only: `wan/modules/custom_model.py` (945 lines at this revision).
- Git history was filtered to author name `gzappavi` and author email `gianluca.zappavigna@inria.fr`; no committer-only attribution was used. The assigned file has 37 matching commits in the reachable history, from 2025-09-17 through 2026-05-07. The worktree was clean and detached at the requested commit.
- No other source or configuration file was read or analyzed.

## Behavior and data flow

`CustomWanModel` is a diffusion-transformer backbone. `forward`:

1. Optionally concatenates conditional video `y` to each input sample, applies 3-D patch embedding, records per-sample grid sizes and sequence lengths, and pads every token sequence to `seq_len` (lines 723–739).
2. Resizes face masks and the word-level/temporal (`wlw`) matrix to the latent grid (lines 741–766), computes timestep embeddings and six-way block modulation (lines 768–772), and pads/embeds text (lines 774–789).
3. Builds per-sentence image/text contexts and a general prompt context from `bias_kwargs` (lines 791–837).
4. Runs each attention block, optionally sharing similarity masks according to `mask_sharing` (lines 839–873). Each block applies modulated self-attention, cross-attention, and an FFN (lines 450–492).
5. Projects tokens with `Head`, unpatchifies them back to video tensors, and returns **a tuple**: the list of output tensors and the collected similarity masks (lines 875–883).

Self-attention projects and RoPE-transforms Q/K, then generates a background-plus-face similarity mask. Static face regions are blended with hard or soft Q/K-derived masks using a timestep-dependent interpolation (`tau_start=0.15`, `tau_end=0.4`; lines 74–112). Masked attention precomputes `AᵀM` and applies `log(AᵀMB)` as a flex-attention score bias (lines 166–210); unmasked paths use `flash_attention` (lines 140–153).

The only registered cross-attention class is `CustomWanI2VCrossAttention` (lines 399–401). It splits the context into image tokens and 512 text tokens (lines 294–299), always adds an image-attention result, and has three conditioning paths: ordinary attention; Concept Weaver, which combines general and per-character contexts using similarity weights (lines 306–329); and regional prompting or EDiff-I, using token masks and scaled dot-product attention (lines 331–396). `MLPProj` maps 1280-channel image features and optionally adds a fixed first/last-frame positional parameter (lines 526–548).

## Evolution of gzappavi changes

The history is iterative and contains several commits titled `wip` or `progress`; the major phases are:

| Period / representative commits | Evolution in this file |
|---|---|
| 2025-09-17, `4734c3e` | Created the custom model implementation. |
| 2025-09-23–10-03, `8f2130b`, `c98cb08`, `760b26f`, `957d700` | Introduced face-mask-aware conditioning, a first working bias implementation, and refactored `bias_kwargs` handling/fixes. |
| 2025-11–12, `3c22dca`, `aa51486`, `5124164`, `8180035` | Iterated on latent mixing and image cross-attention, removed attention-weight collection, fixed image-output flattening, and added `bias_method == "none"`. |
| 2026-01, `0c85186`, `53a2ee8`, `2c5c951`, `3c28540` | Added multiple-sentence/token-mask handling, generalized padding, and fixed a shallow-copy bug where padded token masks were not installed in the new list. |
| 2026-02, `8d477bf`, `c6d5608` | Added Concept Weaver context mixing and then flex-based self-attention masking. |
| 2026-03, `ae071fa`, `390de16` | Made self-attention masking conditional, replaced block-mask logic with score-modulated flex attention, and added hard/soft mask options and timestep blending. |
| 2026-04, `2a7238b`, `38b21b3` | Added propagated/first-mask sharing and adjusted returned mask collection. |
| 2026-05-07, `1593f4d`, `f7472d3` | Reduced intermediate VRAM use by deleting QKV inputs earlier and casting modulated inputs back to the activation dtype before FFN/self-attention. |

## Static bug and gap review

### High: advertised model variants are internally inconsistent

- `WAN_CROSSATTENTION_CLASSES` contains only `"i2v_cross_attn"` (lines 399–401), but the constructor selects `"t2v_cross_attn"` for the default `model_type="t2v"` (lines 624, 656–669). Instantiating the documented/default T2V variant therefore reaches a missing dictionary key.
- `self.img_emb` is created only for `i2v` and `flf2v` (lines 688–689), while `forward` calls `self.img_emb(clip_fea)` unconditionally (line 788). Thus T2V lacks the attribute even aside from the constructor issue, and VACE (accepted at line 624 but not given `img_emb`) also lacks it. The class docstring and constructor docs claim support for all four variants, but this file does not provide a complete path for them.

### High: masked flex attention does not account for the sequence padding performed by `forward`

`forward` pads token sequences to the caller’s `seq_len` (lines 732–739), whereas similarity masks are built for `T*H*W` positions (lines 87–91 and 138). The flex score modifier indexes `AM_all[q_idx]` and `B_all[kv_idx]` (lines 37–49), but `AM_all`/`B_all` have only the unpadded mask length (lines 166–180). The intended alignment padding is commented out (lines 158–163, 187–197). When `seq_len` exceeds the actual grid token count and self-attention masking is active, padded indices can exceed the mask arrays; at minimum, padded tokens have no defined mask semantics.

### High: batch handling contradicts the documented `[B, ...]` inputs

Both similarity-mask generation and cross-attention mask construction destructure `grid_sizes` as `[[T, H, W]]` (lines 75 and 232), which only accepts a one-row tensor. `CustomWanModel.forward` explicitly stacks one grid row per input sample (lines 728–730), and the attention docstrings describe batch-shaped inputs. A batch larger than one consequently fails these destructuring operations; the code also uses the first grid’s `T,H,W` for mask preparation (line 742).

### Medium: self-attention mask work is performed even on bypass paths

`new_simil_masks = self._generate_simil_masks(...)` executes before the `bias_method == "none"` / `self_attention_masking == False` branch (lines 138–145). Therefore an unmasked run still requires `face_masks`, `simil_masks_type`, and `normalized_timestep`, and may perform expensive dynamic mask computation. This is inconsistent with the stated purpose of the bypass and makes the no-mask input contract unnecessarily strict.

The comment says masks are computed only in the first block (lines 135–138), but `_generate_simil_masks` is called on every block. `mask_sharing` only supplies a prior mask to self-attention (lines 155–156 and 864–872); the block then passes the newly generated mask to cross-attention (lines 468–485). Consequently the sharing option does not actually avoid per-block generation and does not share the same mask through cross-attention.

### Medium: EDiff-I mask semantics may not exclude masked tokens

The EDiff-I branch converts a boolean attention mask to a float tensor and multiplies it by `strength * normalized_timestep` (lines 370–380). In the resulting additive SDPA mask, `False` entries become zero rather than negative infinity, while `True` entries receive a positive bias. If the intended behavior is hard token exclusion—as the preceding boolean mask construction suggests—this implementation instead leaves disallowed tokens eligible and boosts allowed tokens. The intended soft-vs-hard semantics should be made explicit and tested.

### Medium: input contracts are implicit and largely unchecked

The forward path unconditionally accesses `face_masks`, `wlw_matrix`, `full_token_mask`, `sentence_contexts`, `single_char_img_contexts`, `general_prompt_context`, and `tokens_data_list` (lines 744–837), including an unconditional `pop` at line 796. There is no local schema validation or method-specific guard for ordinary/no-bias calls. The code does validate some lengths with `strict=True`, but does not validate `blocks_bias_schedule` length before indexing it at line 849, nor validate that mask dimensions match the selected grid and token indices.

### Low: static maintenance/documentation issues

- `gc`, `create_block_mask`, and `compute_soft_simil_masks_iterative` are imported but unused (lines 2, 11, and 16).
- The model forward docstring says it returns a list (lines 712–715), while the implementation returns `(list, simil_masks)` (line 883).
- `unpatchify` documents fixed `/8` spatial dimensions (line 898), although the actual output dimensions are determined by configurable `patch_size` (lines 901–918).
- `tau_start` and `tau_end` are hard-coded TODO values (lines 79–82), and several attention paths rely on CUDA-specific autocast contexts (lines 462, 478, 489, 520, and 769).

## Reproducibility and maintenance risks

- `@torch.compile` wraps the flex-attention routine at import time (lines 33–52), and the implementation depends on `torch.nn.attention.flex_attention`, CUDA autocast, and a custom flash-attention path. PyTorch version, GPU capability, compiler backend, and sequence shape can materially affect whether it imports or runs.
- The method-specific `bias_kwargs` protocol is a mutable, untyped dictionary whose keys are consumed and removed (`pop` at lines 741, 796, 839–840). Callers must reproduce an undocumented schema and padding convention exactly.
- Similarity masks mix binary spatial masks with dynamic Q/K-derived values and assert that every token’s mask sum is approximately one (lines 182–185). Overlapping/interpolated face masks and numerical changes can violate this assertion; the file itself notes interpolation/overlap concerns at lines 747–752.
- The final implementation reflects many short incremental commits and retains stale comments/imports. The absence of local shape checks around the flex score modifier makes failures likely to surface deep inside compiled attention rather than at a clear API boundary.

## Validation performed

- Confirmed the requested detached HEAD and exact author/author-email attribution from Git metadata.
- Enumerated matching file history and reviewed the assigned-file diffs at the major evolution points above.
- Ran Python `ast.parse` on the assigned file: **passed**.
- Ran `git show --check` for the reviewed revision: **passed**; no whitespace errors were reported.
- Did not run inference, model execution, builds, web applications, or external tests.

## Limitations

This is a cursory static review of one file and its Git history only. Imported implementations, caller contracts, dependency versions, checkpoint compatibility, runtime tensor behavior, and GPU performance were not inspected. The findings concerning compiled attention, batch/padding behavior, and EDiff-I semantics should be confirmed with focused unit tests in an environment that can execute the relevant PyTorch attention operators.
