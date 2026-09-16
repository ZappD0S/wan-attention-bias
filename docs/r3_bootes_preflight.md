# R3 Bootes hardware inspection — 2026-09-16

## Authorization and boundary

The user approved a source-preparation commit, selected `bootes.alias`, authorized
SSH/hardware inspection, and approved delegation of the CPU implementation to
GPT-5.6 Sol under GPT-6 Astra lead. Bootes was already documented in the
2026-09-14 entries of `docs/experiment_plan.md`.

This inspection used read-only shell/device queries. No checkpoint was accessed,
no environment was installed or changed, and no CUDA computation or generation
was run. The existing U1 checkpoint is only a candidate until the user confirms
it for R3. Frozen v1/v2 protocol approvals and runtime declarations remain unchanged.

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

Before checkpoint work or generation: obtain explicit checkpoint confirmation,
revalidate checkpoint identity, finish CPU backend/matrix compatibility repair,
then freeze a versioned execution amendment with observed environment/backend
identities, rank count and full-generator tolerances. R3 remains IN PROGRESS;
no GPU contract, parity or scientific result is claimed.
