# R3 Pollux backend-canary preflight

## Scope and authorization

On 2026-09-17 the user asked to use `pollux.alias` instead of occupied Bootes
for the already bounded first R3 stage. This authorizes relocation of the frozen
source/environment and one FA2, compiled flex-attention and SAM2 CUDA canary on
one visible Pollux GPU, followed by a mandatory stop. It does not authorize a
checkpoint/model load, generation, distributed execution or any later stage.

## Hardware and storage observation

A no-allocation `nvidia-smi` query found two idle NVIDIA RTX PRO 6000 Blackwell
Max-Q Workstation Edition GPUs with 97,887 MiB each, driver 580.173.02 and
compute capability 12.0:

- GPU 0: `GPU-4bace78e-bb7b-8593-36a8-a40f81258a10`
- GPU 1: `GPU-f0253119-3c25-53a0-40e8-17bd437d4e83`

The peer topology is `NODE` without NVLink, not Bootes's observed `SYS`
topology. `/local_scratch2` had 3.3 TiB available but no writable user directory;
the first copy attempt failed before transfer. The writable
`/local_scratch/gzappavi` instead had 560 GiB available and is the bound Pollux
root.

## Source and environment relocation

Pollux pulled the clean Bootes source and environment directly over SSH with
`rsync`: 10,054,558,002 source/environment bytes, 9,309,912,132 toolkit bytes,
and the 5,382,238,770-byte official CUDA installer. The copied source was clean
at outer revision `e9f08e6`, Wan `00bde1e…` and LaMa `469acc73…`. The installer
again hashed to `228f6bca…`; copied `nvcc` reports CUDA 12.8, V12.8.93.

Because editable workspace links retained the old absolute Bootes path, two
GPU-hidden `uv sync --locked` passes rebound the copied environment to the
Pollux source. A direct no-model probe then observed Python 3.11.13, PyTorch
2.10.0+cu128, CUDA runtime 12.8, flash-attn 2.8.3, no FA3, PyTorch flex
attention and SAM2 1.1.0. Project and lock hashes remained
`37ccb385…` and `d31fb115…`. No checkpoint was copied or loaded, and no CUDA
tensor or kernel ran during provisioning.

## Immutable v6 relocation amendment

`docs/r3_protocol_v6.json` is lineage-bound to immutable v5 SHA-256
`d3271493…`. It changes only the exact host/source/environment authorization for
the same bounded backend stage. It binds Pollux hardware identifier
`f13f743d…`, parent production digest `69b14888…`, the copied toolkit path and
`NODE-no-NVLink` topology. Its own SHA-256 is `f4de93f2…`.

Generic execution remains blocked by `staged-execution-gates`; the v6
`backend-kernel-canary` execution preflight alone reports no blockers. At this
pre-execution checkpoint the canary has not run and its output does not exist.

## Bounded backend-canary result

Commit `3dbf8e0` was transferred by verified bundle and fast-forwarded into the
clean Pollux source. Exact parent/Wan/LaMa, production, protocol and environment
checks passed. Immediately before execution both GPUs reported 2 MiB used and
0% utilization. Only GPU 0 was exposed to the runner.

The authorized command ran from 2026-09-17T12:54:49Z through 12:55:09Z and
exited 0. Immutable evidence
`docs/r3_evidence/pollux-backend-kernel-canary.json` hashes to
`6d5c5cae…`; the captured log hashes to `cf7d78dc…`. FA2 produced a finite BF16
`[1,128,4,64]` output, compiled flex attention produced a finite BF16
`[1,4,128,64]` output, and SAM2 returned finite integer label/count tensors with
exact foreground areas 1 and 4. The record binds v6 `f4de93f2…`, matrix
`e1776af1…`, clean source revisions and the exact Pollux environment.

The evidence explicitly records no checkpoint/model load, generation or
distributed execution. A local evidence-binding regression passed with the
focused 61-test R3 suite and 103-test seven-file suite; compile and both Ruff
checks passed. This completes only the first backend-kernel stage. The
mandatory stop was observed; every later stage remains unapproved and R3
remains **IN PROGRESS**.

## Immutable v7 checkpoint/hook amendment

After that stop, the user explicitly said to go ahead with the previously
identified next stage. V7 (SHA-256 `63e2781b…`) authorizes only checkpoint
transfer/exact revalidation, one upstream Wan DiT load, and a small layer-0
source-hook neutrality comparison on one visible Pollux GPU. It binds v6
backend evidence `6d5c5cae…`, production digest `5e563044…`, checkpoint content
`80c954fb…`, inventory `e6b7adbd…` and expected selected-layer state
`eb1b247b…`.

The runner compares the same frozen 128-token input with no observer and with
the R3 observer installed. Outputs must be finite and bitwise identical, inputs
and layer state must remain exact, and observations must be exactly lifecycle,
self FA2 and the two I2V cross FA2 dispatches. T5/CLIP/VAE, custom model,
full-model forward, generation, scheduler/decoding, distributed and later-stage
execution are prohibited. At amendment freeze, no checkpoint was present on
Pollux and no v7 model load or CUDA operation had run.
