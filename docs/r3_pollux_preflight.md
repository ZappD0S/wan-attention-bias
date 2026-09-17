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
R3 remains **IN PROGRESS**.
