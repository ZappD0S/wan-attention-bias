# Wan 2.1 text and CLIP encoders

## Scope and attribution

This report covers only the checked-out files `wan2.1/wan/modules/clip.py` and `wan2.1/wan/modules/t5.py` at submodule commit `f7472d354e0cb46b2853f15bf97b8b283d4780f8`. Git history was path-filtered to those two files. The author identity was matched exactly as `gzappavi <gianluca.zappavigna@inria.fr>`.

The path history contains the initial upload by `WanX-Video-1`, followed by two matching-author commits:

| Commit | Date | File(s) | Observed change |
|---|---|---|---|
| `d68ab4f3a60a618a0056bda446d884bab2a61663` | 2025-10-21 | `clip.py` | Replaced deprecated `torch.cuda.amp.autocast` with `torch.amp.autocast("cuda", ...)`. |
| `c6d56089a1e4aaaf73c9688c8bef69b57affe619` | 2026-02-24 | `t5.py` | Large formatting/reflow change and a tokenizer special-token docstring. |

Although the latter commit message says “added self-attention masking and other stuff,” its path diff shows the masking logic already existed in the parent; the relevant change is formatting, not a newly introduced masking behavior. Later matching-author commits in the repository history, including the checked-out `f7472d3`, did not modify either assigned file.

## Encoder behavior and data flow

### CLIP (`clip.py`)

- `VisionTransformer` converts `[B, 3, H, W]` images into patch tokens with a strided convolution (`clip.py:248-253`), optionally prepends a learned class token (`254-258`), adds learned positional embeddings, optionally interpolating them, applies dropout and optional pre-normalization (`279-293`), and runs the configured residual attention blocks (`294-300`).
- Each attention block uses multi-head `flash_attention`, followed by a projection and an MLP (`53-153`). The block supports pre-norm and post-norm residual arrangements, QuickGELU/GELU/SwiGLU, and dropout.
- `XLMRobertaWithHead` delegates token encoding to the inherited XLM-R implementation, averages non-padding positions using `pad_id`, then projects to `out_dim` (`303-325`). `XLMRobertaCLIP.forward` returns the visual result and text result as a tuple (`406-416`).
- The default factory configures a ViT-H/14-like visual branch and 24-layer XLM-R text branch with a shared output dimension of 1024 (`471-498`).
- `_clip` constructs and casts the model and, when requested, creates a square bicubic resize plus tensor conversion and normalization transform (`434-468`).
- `CLIPModel` loads a checkpoint and a Hugging Face tokenizer (`501-525`). Its `visual` method transposes/interpolates each video tensor to the configured square image size, converts values from an assumed `[-1, 1]` range before normalization, then runs the visual transformer under CUDA autocast with the last transformer block omitted (`527-542`). It returns the intermediate token sequence, not a pooled CLIP embedding.

### T5 (`t5.py`)

- `T5Attention` projects queries, keys, and values, reshapes them into heads, adds optional relative-position bias and a 2-D/3-D mask, computes unscaled dot-product attention, performs softmax in float, and applies the output projection/dropout (`75-124`).
- `T5SelfAttention` is a pre-normalized self-attention plus gated feed-forward residual block (`147-181`). `T5CrossAttention` adds decoder self-attention, encoder-decoder attention, and the feed-forward residual (`184-226`). Residual additions pass through `fp16_clamp`.
- `T5RelativeEmbedding` creates relative position matrices and maps distances into logarithmic buckets, with bidirectional buckets for encoder self-attention and unidirectional buckets for decoder self-attention (`229-277`).
- `T5Encoder` embeds IDs, applies dropout, reuses one bidirectional position-bias tensor when `shared_pos=True`, runs its blocks, and applies final norm/dropout (`280-333`).
- `T5Decoder` embeds IDs, creates a lower-triangular causal mask when needed, runs decoder self-attention/cross-attention/FFN blocks, and applies final norm/dropout (`336-398`).
- `T5Model` shares one token embedding between encoder and decoder and applies a vocabulary projection to decoder states (`401-458`). `_t5` supports full, encoder-only, and decoder-only construction (`461-502`). The `umt5_xxl` preset uses vocabulary size 256,384, model dimension 4096, 64 heads, 24 encoder and decoder layers, 32 buckets, and `shared_pos=False` (`505-519`).
- `T5EncoderModel` builds the frozen encoder-only preset, loads a checkpoint, optionally applies a sharding function, tokenizes with special tokens, moves IDs/masks to the call-site device, and trims each encoded sequence to the count of positive mask entries (`522-565`).

## gzappavi evolution

The matching-author history is narrowly focused on maintenance rather than changing the encoder architecture:

1. **Autocast modernization (`d68ab4f`, `clip.py`)**: one-line migration at current `clip.py:540`, preserving CUDA autocast and the requested dtype while using the newer API.
2. **T5 reformat/documentation (`c6d5608`, `t5.py`)**: 187 insertions and 135 deletions are predominantly line wrapping, spacing, and parenthesization. The current masking path at `t5.py:110-113` is materially the same as its parent. The only user-facing documentation addition is the explicit note that `T5EncoderModel.__call__` adds special tokens (`559`).

## Static bug/gap review

Findings are cursory and based only on the two files; they were not runtime-tested.

| Area | Evidence | Risk or gap |
|---|---|---|
| Visual output contract | `VisionTransformer` defines `post_norm` and a `head` (`268`, `271-277`), but `forward` returns immediately after the transformer (`294-300`) and never invokes either. `XLMRobertaCLIP` therefore returns visual tokens while its text branch applies a projection. | The class/factory naming and `out_dim` suggest a projected CLIP image embedding, but this implementation exposes unnormalized, unprojected visual tokens. The intermediate output may be intentional for `CLIPModel.visual`, so this is best treated as a contract ambiguity/maintenance gap unless callers require CLIP embeddings. |
| Unused CLIP factory controls | `_clip` accepts `pretrained`, `return_tokenizer`, and `tokenizer_padding` (`434-442`), but does not load pretrained weights, create/return a tokenizer, or use the padding option. | Callers can reasonably expect these options to have effects; silently ignored arguments make configuration and reproducibility error-prone. |
| `pretrained_name` nullability | Transform setup calls `pretrained_name.lower()` (`454`) without guarding `None`, although `_clip` defaults `pretrained_name=None` (`435`). | Direct use of the private helper with transforms enabled and no name raises an attribute error. Public factory use supplies a default name, but the helper’s own signature is inconsistent. |
| Positional interpolation assumptions | `pos_interpolate` derives source and target square grids from the square root of total sequence length (`22-38`). | Rectangular patch grids, or lengths that are not a square plus the inferred prefix, can produce incorrect grids or a reshape failure. Interpolation also converts positions to float (`32`) without explicitly restoring the model dtype, potentially increasing activation dtype/memory. |
| Declared CLIP scale is unused | `XLMRobertaCLIP` creates `log_scale` (`404`), while `forward` returns only `xi, xt` (`414-416`). | No contrastive logit scale is applied or returned in this module. This may be deliberate because downstream code computes logits, but it is a discoverability/contract gap. |
| Encoder-only/decoder-only initialization | `init_weights` initializes the embedding only when visiting a `T5Model` (`27-35`), while `_t5` constructs `T5Encoder` or `T5Decoder` directly in the one-sided modes (`475-484`). Their `nn.Embedding` is not covered by a matching initialization branch. | Fresh encoder-only or decoder-only models use the framework’s default embedding initialization rather than the intended T5Model embedding initialization. Loaded checkpoints mask this issue in `T5EncoderModel`, but random-model use is inconsistent. |
| CUDA-dependent default argument | `T5EncoderModel.__init__` evaluates `device=torch.cuda.current_device()` in the function signature (`527`). | Import/class definition or use in a CPU-only environment can require CUDA before the caller has supplied a device. A concrete default such as `None`, resolved at runtime, would be more portable. |
| Relative bucket edge case | `_relative_position_bucket` computes `log(rel_pos / max_exact)` for all positions, including zero (`263-276`), before selecting the small-distance branch. | Zero distances transiently produce `-inf` before `where` selects the exact bucket. It is normally discarded, but can trigger numerical diagnostics and the implementation has no explicit guards for very small/invalid bucket counts or `max_exact == 0`. |
| Operational defaults and loading | `T5EncoderModel` defaults checkpoint/tokenizer paths to `None` (`528-530`) but unconditionally loads and constructs from them (`546-555`). `CLIPModel` likewise requires paths positionally and calls `torch.load` (`503`, `517-519`). | The defaults are not standalone runnable configurations. Checkpoint serialization compatibility and trust/`torch.load` policy are also not documented in these modules. |
| Mutable configuration default | `_t5` uses `tokenizer_kwargs={}` (`466`). | It is not mutated in the current code, but a mutable default is a maintenance hazard if future changes add defaults or edits. |

## Reproducibility and maintenance risks

- Both wrappers depend on external checkpoint and tokenizer assets, but paths, expected checkpoint key layout, tokenizer behavior, and preprocessing value range are implicit in code (`clip.py:408-412`, `517-525`, `537`; `t5.py:546-560`).
- CLIP preprocessing assumes videos can be concatenated after transposition and that their values are suitable for `.mul_(0.5).add_(0.5)` (`527-537`). There is no explicit shape/range validation.
- CLIP visual execution hard-codes the CUDA autocast device type (`540`), so CPU-only execution is not an equivalent path even when the model was constructed on CPU.
- T5 attention allocates an explicit `[B, heads, query_length, key_length]` bias tensor (`107`), and the attention uses explicit einsums (`116-121`); memory use grows quadratically with sequence length. This is particularly relevant to the large UMT5-XXL defaults.
- Dropout is present throughout the T5 blocks and embeddings (`89`, `123`, `137-143`, `311`, `367`), so deterministic outputs require evaluation mode. The provided encoder wrapper sets evaluation mode (`539-545`), while direct constructors do not.
- The 2026 T5 commit substantially reformatted the file without corresponding semantic tests in the file history. Formatting-only changes can obscure whether behavior was intended to change, especially given the commit message’s masking claim.

## Validation performed

- Verified the detached submodule HEAD and path-filtered author history/commit diffs with Git.
- Confirmed both assigned files pass Python AST parsing.
- Ran Git whitespace/error checking for both assigned paths; no issues were reported.
- Did **not** run model inference, builds, web applications, checkpoint loading, or dependency tests, in accordance with the requested static-only scope.

## Limitations

The review does not inspect or assess any other source/configuration file, including the imported attention, tokenizer, XLM-R, or transformer implementations. Consequently, it cannot establish downstream shape contracts, tokenizer mask semantics, checkpoint compatibility, CUDA availability, or whether the unprojected visual-token behavior is intentional. Findings are static risks and review prompts, not confirmed runtime failures.
