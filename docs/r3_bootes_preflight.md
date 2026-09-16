# R3 Bootes hardware inspection — 2026-09-16

## Authorization and boundary

The user approved a source-preparation commit, selected `bootes.alias`, authorized
SSH/hardware inspection, and approved delegation of the CPU implementation to
GPT-5.6 Sol under GPT-6 Astra lead. Bootes was already documented in the
2026-09-14 entries of `docs/experiment_plan.md`.

The initial inspection used read-only shell/device queries. No checkpoint was
accessed, no environment was installed or changed, and no CUDA computation or
generation was run. The checkpoint remained a candidate at that stage. Frozen
v1/v2/v3 protocol approvals and runtime declarations remained unchanged.

## Observed hardware

Observed over SSH at `2026-09-16T09:09:22Z`; reported hostname `bootes.alias`:

- Two NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition GPUs, each
  97,887 MiB total, 2 MiB used and 0% utilization; no compute processes reported.
  Driver: `580.173.02`.
- GPU 0 UUID: `GPU-c247e0e3-654a-7387-8ec6-46791821a52d`.
  GPU 1 UUID: `GPU-b7ce3d3e-fcc2-d2d5-91dc-c11e024f4cd5`.
- Inter-GPU topology: `SYS`, across NUMA nodes 0 and 1, not NVLink.
- Two Intel Xeon Gold 5416S sockets, 16 cores/socket, two threads/core:
  32 physical cores and 64 logical CPUs.
- Host RAM: 251 GiB total, 221 GiB available. Swap: 19 GiB total.
- `/local_scratch2/gzappavi`: filesystem 3.7 TiB total, 820 GiB used,
  2.7 TiB available (`df -h` rounded values).
- Default compiler: `/usr/bin/nvcc`, CUDA `12.0.140`.
- `uv`: `/home/gzappavi/.local/bin/uv`, version `0.11.21`.

Queries: `hostname`, `date -u`, `nvidia-smi --query-gpu`,
`nvidia-smi --query-compute-apps`, `nvidia-smi topo -m`, `free -h`, selected
`lscpu` fields, `df -h /local_scratch2/gzappavi`, `nvcc --version`, and
`uv --version`. These were metadata queries, not peer-access or kernel tests.

## Capacity assessment and remaining gates

This is a credible hardware candidate for single-GPU and two-rank 14B inference:
BF16 diffusion weights alone are approximately 28 GB, below either GPU's
capacity, with substantial host offload and scratch capacity. That estimate
excludes activations, text/image encoders, VAE, workspaces and artifact growth.
Full-shape canaries and peak-memory measurements are required to prove fit;
GPU memory is not assumed to be pooled. `SYS` communication may slow FSDP.
The intended rank count is still undeclared, not silently set to two.

Hardware capacity is **not runtime readiness**. The observed default CUDA 12.0
compiler is the same known blocker for building SAM2 for Blackwell `sm_120`.
A complete locked `uv` environment, a compatible selected compiler, validated
Torch/FA2/flex imports and kernels, and measured distributed behavior remain
required. The earlier U1 SAM2 omission does not authorize an incomplete R3 GPU
environment. No alternate compiler was searched for or selected in this probe.

## Authorized checkpoint identity revalidation

The user subsequently selected the existing U1 snapshot for R3 identity
revalidation. At `2026-09-16T09:51:53Z`, a read-only SSH check confirmed the
snapshot directory named for revision
`6b73f84e66371cdfe870c72acd6826e1d61cf279`, 33 resolved files, zero broken
symlinks, and the approved 82,272,036,045-byte inventory. The inventory file
SHA-256 remained
`e6b7adbd6f6e5dfcb7fa06e09d5d2edfcb179e80ff25f1c743567d1948701dd4`.
A full 70.048-second content rehash through the project checkpoint verifier
returned content identity
`80c954fbb46c39139a300a7dd41ed5a8b1c6b11c5f952c1bfb3c763d6e73ebb7`,
matching the retained U1 evidence exactly. The selected path is
`/local_scratch2/gzappavi/hf/hub/models--Wan-AI--Wan2.1-I2V-14B-480P/snapshots/6b73f84e66371cdfe870c72acd6826e1d61cf279`.

This establishes snapshot identity and integrity only. The verifier ran with
`CUDA_VISIBLE_DEVICES` empty and did not load model weights, import/run kernels,
change the environment, or generate output.

A further read-only package probe found that the retained U1 environment is bound
to the same `pyproject.toml` and `uv.lock` hashes as the current tree. With GPUs
hidden it imports Python 3.11.15, PyTorch 2.10.0+cu128, CUDA runtime metadata 12.8,
flash-attn 2.8.3 and `torch.nn.attention.flex_attention`; `sam2` is not installed.
Module discovery reports FA2 present and no `flash_attn_interface`/FA3 package.
A bounded search found no alternate `nvcc` under `/usr/local` or `/opt`, leaving
`/usr/bin/nvcc` 12.0 as the only observed compiler. The retained U1 repository is
an intentionally dirty evidence worktree at old parent revision `e073adaa`; it
must not be overwritten or treated as the current R3 source.

The completed v3 CPU compatibility repair and checkpoint identity now permit
preparation of a further versioned execution amendment, but not execution. A
fresh remote R3 source copy and a complete locked environment still need an
explicitly approved provisioning approach for a Blackwell-capable compiler.
The amendment must bind observed environment/backend identities, approved rank
count, full-generator tolerances and staged GPU stop gates before any CUDA
execution or generation. R3 remains IN PROGRESS; no GPU contract, runtime parity
or scientific result is claimed.
