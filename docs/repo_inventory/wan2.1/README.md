# Wan2.1 author-scoped inventory

## Scope and attribution

This inventory documents only files touched by commits authored by the exact identity:

`gzappavi <gianluca.zappavigna@inria.fr>`

The initialized submodule is reviewed at detached commit `f7472d354e0cb46b2853f15bf97b8b283d4780f` (`f7472d3`, “further changes to reduce VRAM usage”). Git metadata and author-filtered diffs were used for attribution. Although the author touched 17 files historically, the owner has identified `wan/finetune/image2video.py` as definitively abandoned; it is excluded. Source analysis is restricted to these 16 active files:

- `.gitignore`
- `pyproject.toml`
- `wan/__init__.py`
- `wan/modules/__init__.py`
- `wan/image2video.py`
- `wan/custom_image2video.py`
- `wan/regional_prompt/__init__.py`
- `wan/regional_prompt/image2video.py`
- `wan/modules/model.py`
- `wan/modules/custom_model.py`
- `wan/modules/clip.py`
- `wan/modules/t5.py`
- `wan/modules/vae.py`
- `wan/utils/simil_mask.py`
- `wan/utils/smooth_mask.py`
- `wan/utils/subsequence.py`

The reports intentionally remain separated by implementation seam. They are static inventories and cursory bug/gap reviews, not fixes.

## Reports

| Area | Files | Report |
|---|---|---|
| Package metadata and exports | `.gitignore`, `pyproject.toml`, `wan/__init__.py`, `wan/modules/__init__.py`, `wan/regional_prompt/__init__.py` | [package-and-exports.md](package-and-exports.md) |
| Standard image-to-video | `wan/image2video.py` | [image2video.md](image2video.md) |
| Historical custom image-to-video | `wan/custom_image2video.py` | [custom-image2video.md](custom-image2video.md) |
| Regional prompting | `wan/regional_prompt/image2video.py` | [regional-prompt.md](regional-prompt.md) |
| Standard transformer model | `wan/modules/model.py` | [model.md](model.md) |
| Custom transformer and masking | `wan/modules/custom_model.py` | [custom-model.md](custom-model.md) |
| CLIP and T5 | `wan/modules/clip.py`, `wan/modules/t5.py` | [text-encoders.md](text-encoders.md) |
| VAE | `wan/modules/vae.py` | [vae.md](vae.md) |
| Similarity, smoothing, and subsequence utilities | `wan/utils/simil_mask.py`, `wan/utils/smooth_mask.py`, `wan/utils/subsequence.py` | [mask-utilities.md](mask-utilities.md) |

## End-to-end picture

The scoped code provides several CUDA-oriented image-to-video paths around shared text/image encoders, VAE encoding/decoding, and diffusion-transformer sampling:

1. Package initializers expose standard Wan classes and selected experimental exports.
2. Standard `WanI2V` converts an input image to a masked latent condition, encodes T5/CLIP context, runs classifier-free-guided UniPC or DPM++ sampling, and decodes on rank zero.
3. The historical custom path adds prompt-pair/token masks, bias schedules, similarity-mask collection, and custom attention; that file is deleted at the reviewed HEAD.
5. The regional path builds multi-sentence contexts and token masks, optionally mixes per-sentence image contexts, applies regional bias during custom-model sampling, and returns video plus similarity-mask metadata.
6. The model layer patchifies latent video, applies timestep-modulated attention blocks, and unpatchifies predictions. The custom model adds face/similarity masking, Concept Weaver/EDiff-I/regional branches, and returns similarity masks.
7. The VAE supplies causal temporal encoding/decoding with streaming caches; CLIP and T5 provide image/text context; utility modules construct and smooth masks and locate nested token subsequences.

## Highest-priority static findings

- **Custom model variants are inconsistent:** `CustomWanModel` advertises multiple model types, but its cross-attention registry only defines `i2v_cross_attn`; `t2v` selection can fail, and `img_emb` is called unconditionally despite not being created for all variants.
- **Custom attention has shape-contract risks:** masking assumes a single grid row/batch, while `forward` constructs one grid row per input; padded token sequences can exceed the mask arrays when active masking is enabled.
- **Standard I2V hard-codes an 81-frame conditioning mask:** `frame_num` drives noise and VAE tensors but the mask remains shaped for 81 frames, so other documented `4n+1` values can misalign.
- **Package export is malformed:** `wan/modules/__init__.py` lists `CustomWanModelVaceWanModel` in `__all__` without importing or defining it.
- **Regional generation mutates caller state:** `generate_from_latents` pops `timestep_bias_schedule` before copying `bias_kwargs`, making reuse of the same mapping fail; its public random-generator path also bypasses distributed seed broadcast.
- **VAE configuration is only partly generic:** `WanVAE` accepts `z_dim` while its normalization statistics are fixed to 16 channels, and its public encode/decode/sample wrappers hard-code CUDA autocast.
- **Mask utilities need explicit edge contracts:** zero/invalid temperatures, empty attention inputs, batch/layout assumptions, NumPy truth-value handling, and smoothing validation are not guarded.
- **Packaging and reproducibility remain weak:** dependencies are mostly unpinned, package discovery is explicit only for `wan` despite subpackages, model/checkpoint provenance is external, and several paths depend on CUDA, optional distributed setup, compiled/flex attention, and undocumented mutable dictionaries.

These are source-level findings. They were not confirmed through model execution or end-to-end imports.

## Cross-cutting status and limitations

The submodule is initialized but remains detached at the parent gitlink revision. The historical `custom_image2video.py` implementation is absent from the checked-out tree, so its report analyzes the latest same-author snapshot before deletion and labels it historical.

Every lane performed bounded static checks such as AST parsing, Git history/diff inspection, and `git diff --check` for its assigned files. No model inference, GPU run, dependency installation, wheel build, notebook execution, web-app run, or end-to-end test was performed. Imported callers, checkpoints, dependency implementations, and files outside the author-scoped list were not analyzed; findings depending on those contracts remain risks rather than runtime confirmations.
