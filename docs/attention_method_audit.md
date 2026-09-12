# Attention method audit

## Scope and terminology

This document describes the implemented mechanisms, not claimed paper reproductions. The repaired path distinguishes **generated masks** (a block's localization output) from **used masks** (the effective masks consumed by that block's self- and cross-attention). Existing `simil_masks` exports remain a compatibility alias for used masks; `used_simil_masks` and `generated_simil_masks` are also exported explicitly.

Mask-sharing names are normalized as follows:

- `none` / `current` / `null`: each block uses its own generated mask.
- `first` / `block0`: block 0's generated mask is used by all later blocks.
- `prev` / `previous-generated`: block 0 uses its own generated mask; each later block uses the preceding block's generated mask.

## Implemented mechanisms

### Localization

`fixed` / `static` repeats the disjoint reference subject masks, plus their complement as background, over latent time. It bypasses attention tracking and is the primary stationary diagnostic.

`hard` and `soft` are legacy attention-derived localization modes. Both calculate head-averaged first-frame attention and aggregate each region by **sum**. Hard localization discretizes the classes. Soft localization applies a class softmax with legacy temperature 0.1. Between normalized denoising progress 0.15 and 0.4, these modes interpolate from fixed masks to their dynamic result. Sum aggregation and temperature are retained to avoid silently retuning historical settings. MultiTalk's local excerpt says **average similarity**; this implementation is therefore an adaptation, not a reproduction. No optional area-mean mode was added in this repair.

### Self-attention routing

For subject labels A and B plus background, routing compatibility permits same-subject communication and communication involving background, while suppressing direct A–B communication. It is active only when all three conditions hold: the current timestep/block/CFG `bias` flag is true, `bias_method != "none"`, and `self_attention_masking` is true. In particular the negative CFG pass uses baseline self-attention.

### Regional cross-attention

Token eligibility is a weight in `[0, 1]` formed from valid-token padding, temporal gates, and spatial subject gates. Regional attention adds `log(eligibility)` to attention logits; exact zeros are excluded. Thus exact 0/1 gates equal binary SDPA, while fractional values continuously attenuate attention. An all-masked query produces zero text-attention output rather than NaN. This fractional rule is an explicit continuous extension for this project, **not a claim of paper reproduction**. The existing beta mix between baseline and regional text attention remains unchanged.

The eDiff-I-named path remains a finite positive bias:

`strength * normalized_timestep * eligibility`.

Zero eligibility receives no added preference; it is not hard exclusion. This describes the code and does not assert equation-level fidelity to the eDiff-I paper.

### CW-inspired fusion

The `concept_weaver` path independently computes a general/background conditioned cross-attention result and one result per subject sentence/image context, then fuses those outputs spatially with the effective masks. This borrows the masked feature-fusion primitive described in the local Concept Weaver text. It does not implement that method's personalized concept-bank training, template inversion/source-feature injection, or base-feature suppression. Calling it **CW-inspired masked context fusion** is deliberate. No suppression or training was added as a repair.

### Identity and token mapping

Supported mappings are (1) one joint sentence with one character segment per entity, globally enumerated in mask/image order, and (2) one singleton sentence per entity, where sentence `i` remains bound to entity `i`. This fixes repeated entity-zero indexing for split prompts while preserving joint-prompt mapping. Other layouts fail explicitly.

Independently encoded sentence contexts are concatenated. Token masks for later sentences use the sum of all preceding encoded context lengths, including each sentence's special tokens. The prior global-offset design was correct and is retained. Padding tokens now begin ineligible, and over-limit context/mask lengths fail instead of truncating control silently.

## Boundaries and validation status

CPU contract tests cover schedules/CFG guards, exact shared-mask object selection, fixed mask partition/dtype, regional binary equivalence, fractional behavior, all-masked finiteness, finite eDiff-I behavior, identity layouts, global offsets, seed broadcast helper, and inference defaults. These tests do not exercise flash/flex CUDA kernels, distributed FSDP generation, the full conditioned model, tokenizer-specific corner cases, or tracker quality. Upstream/custom-none full-layer numerical parity remains a required GPU-capable contract test; no generator-equivalence claim is made from source or helper tests.
