# U1 upstream/custom-`none` one-layer parity protocol

Frozen on 2026-09-14 before loading either Wan model or executing a checkpoint-backed layer. Checkpoint hashing and dependency/CUDA import checks preceded the freeze; no model weights or numerical outputs had been loaded.

## Claim boundary

This protocol tests numerical agreement between one concrete upstream `WanAttentionBlock` and the corresponding custom `CustomWanAttentionBlock` with intervention routing disabled. It does not establish full-generator, scheduler, VAE, text/CLIP encoder, FSDP, multi-rank, video-quality, or scientific efficacy parity. Those remain R3 work.

## Frozen checkpoint and source

- Hugging Face repository: `Wan-AI/Wan2.1-I2V-14B-480P`
- Snapshot revision: `6b73f84e66371cdfe870c72acd6826e1d61cf279`
- Snapshot file count and bytes: 33 files; 82,272,036,045 bytes
- Approved inventory: `docs/u1_checkpoint_inventory.json`
- Inventory-file SHA-256: `e6b7adbd6f6e5dfcb7fa06e09d5d2edfcb179e80ff25f1c743567d1948701dd4`
- Canonical checkpoint-content SHA-256: `80c954fbb46c39139a300a7dd41ed5a8b1c6b11c5f952c1bfb3c763d6e73ebb7`
- Parent source revision: `e073adaa742c4e3528a1e488e6413f12d02ae220`, plus the recorded dirty fingerprint at execution
- Wan source revision: `9f52e9abceb49c5cbf4a7f435a84ef0621c0029f`
- Selected layer: index 0, I2V cross-attention block; no layer may be substituted after observing results

The runner must verify every inventory path, size, and SHA-256 before model loading. It must load both concrete model classes directly from that verified directory and reject any architecture or selected-layer state mismatch before executing either block.

## Frozen environment

The execution target is GPU 0 on `bootes.alias`, an NVIDIA RTX PRO 6000 Blackwell Max-Q (compute capability 12.0), using the checked-in `uv.lock`: Python 3.11, PyTorch `2.10.0+cu128`, CUDA runtime 12.8, bfloat16 model/input execution, and flash-attn `2.8.3` FA2. FA3 is unavailable and is not an allowed substitution.

A full `uv sync --locked` currently fails while compiling the unrelated `sam2` extension because Bootes exposes CUDA toolkit/nvcc 12.0, which cannot compile `sm_120`. The bounded U1 layer environment was therefore synced with `uv sync --locked --no-install-package sam2`. The parity runner must import the actual upstream/custom Wan blocks and FA2 before execution; it must not import or invoke tracker code. This exception permits only the bounded one-layer diagnostic and prevents claiming full-project or generator readiness.

## Frozen inputs and cases

Run exactly two cases, `conditional` and `negative`, in one process with the same loaded models. Each uses a separately seeded, deterministic synthetic post-embedding input so both blocks receive exact clones:

- seeds: 2026091401 (`conditional`) and 2026091402 (`negative`)
- generation: for each case, one CPU `torch.Generator` makes `x`, then `e`, then `context`; each starts as a float32 standard-normal draw, `x` and `context` are cast to bfloat16, and `e` is multiplied by 0.1
- batch: 1
- `x`: bfloat16 `[1, 32760, 5120]`
- `e`: float32 `[1, 6, 5120]`
- `seq_lens`: integer `[32760]`
- `grid_sizes`: integer `[[21, 30, 52]]`, the 81-frame 480p latent grid
- `freqs`: the selected checkpoint-loaded model rotary frequencies, copied identically to CUDA
- `context`: bfloat16 `[1, 769, 5120]` (257 image tokens followed by 512 text tokens)
- `context_lens`: `None`
- one fixed subject mask on spatial rows `[8:22)` and columns `[13:39)`, producing a background/subject partition over all 32,760 latent tokens
- custom controls: `bias_method="none"`, `bias=false`, `self_attention_masking=false`, `simil_masks_type="fixed"`, normalized timestep 0.5

Synthetic inputs isolate block equivalence and are not represented as encoded prompts, images, or generated latents. The two labels test separate shared-context tensors corresponding to the positive and negative-CFG call sites; they do not execute classifier-free guidance or a diffusion step.

## Frozen tolerance and acceptance

For each case, require matching output shape and dtype, finite values, and elementwise

`abs(custom - upstream) <= 1e-5 + 0.016 * abs(upstream)`.

The fixed `atol=1e-5` and `rtol=0.016` are PyTorch 2.10's default bfloat16 `torch.testing.assert_close` tolerances, selected before outputs exist. Output hashes and maximum absolute/relative errors are diagnostics; bitwise identity is not required.

The run stops without a parity claim on any checkpoint inventory mismatch, source/runtime conflict, concrete-type mismatch, architecture mismatch, state key/shape/dtype/value mismatch, missing mask evidence, nonfinite output, or tolerance failure. Both cases must pass. Records are immutable and must include checkpoint/source/environment, state, exact input/output identities, diagnostic generated/used masks, and the predeclared tolerance.
