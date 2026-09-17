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

The completed v3 CPU compatibility repair and checkpoint identity permitted
preparation of a further versioned execution amendment, but not execution. At
this point a fresh remote R3 source copy and a complete locked environment still
needed an explicitly approved Blackwell-capable compiler.

## Authorized fresh source and locked-environment provisioning

The user then authorized fresh R3 provisioning while requiring a stop before
GPU kernels, model loading or generation. Commit `b07c64a` was transferred by
Git bundle into `/local_scratch2/gzappavi/wan_experiments_r3`; the outer tree is
at `b07c64aa6b2aaea90627c30e27a4b22f194c5b41`, Wan is at
`00bde1e719ccb56c66a01a1f18a70c49b278c202`, and LaMa is at
`469acc7358a1c6828b647b4ee20c93474a2f36b4`. All three worktrees are clean. Git
LFS smudging was deliberately skipped during the source transfer, so tracked LFS
example assets remain pointers and must not be selected as execution inputs.
The retained U1 evidence tree was not changed.

The official NVIDIA CUDA 12.8.1 toolkit-only runfile was downloaded from
`developer.download.nvidia.com` and installed without a driver at
`/local_scratch2/gzappavi/toolchains/cuda-12.8.1`. The 5,382,238,770-byte
installer SHA-256 is
`228f6bcaf5b7618d032939f431914fc92d0e5ed39ebe37098a24502f26a19797`;
`nvcc` reports CUDA 12.8, V12.8.93. The system CUDA 12.0 installation was not
modified.

The first `uv sync --locked` attempt stopped before resolution because the
fresh outer clone did not yet contain its LaMa submodule. Initializing the exact
recorded LaMa commit repaired that source-layout omission. With GPUs hidden,
`CUDA_HOME` bound to the new toolkit, and `TORCH_CUDA_ARCH_LIST=12.0`, the next
locked sync resolved 287 packages and succeeded from `2026-09-16T11:55:43Z` to
`11:58:27Z`; an immediate second locked sync resolved in 3 ms and checked 266
installed packages without changes.

A GPU-hidden import probe reports Python 3.11.13, PyTorch 2.10.0+cu128 with CUDA
runtime metadata 12.8, flash-attn 2.8.3+cu128torch2.10, SAM2 1.1.0 and importable
PyTorch flex attention. Static `cuobjdump` inspection found `sm_120` cubins in
both SAM2 `_C.so` and the prebuilt FA2 extension. Project and lock hashes still
match the approved values. These checks loaded no checkpoint or model and ran no
GPU kernel or generation.

The complete locked installation removes the prior provisioning blocker, but it
does not establish runtime readiness. At this point the execution amendment
still needed to bind runtime/backend identities, ranks, tolerances and staged
GPU stop gates.

## GPU-free v4 amendment preparation

The authorization-pending `docs/r3_protocol_v4.json` now binds the provisioning
observations without recontacting Bootes. It predeclares the two observed GPUs as
the intended two-rank FSDP configuration; exact GPU UUID/model/capability,
driver, host, Python, Torch/CUDA, FA2, flex, SAM2 and project/lock hashes must be
re-observed by the pre-model worker guard. The parent production-content digest
plus exact Wan/LaMa revisions and clean-worktree requirement bind the execution
source independently of later documentation-only commits.

Full-generator parity is predeclared at `atol=1e-5`, `rtol=0.016`, explicitly
re-derived as a near-zero floor plus rounded two-BF16-epsilon relative allowance
rather than inherited from U1. Stages proceed only from bounded backend kernels,
to checkpoint/hook canary, single-rank generator pair, remaining single-rank
contracts and finally two-rank FSDP; every stage is still `not-approved` and has
hard stop conditions.

The v4 preparation itself used no SSH, GPU query/kernel, checkpoint/model load or
generation. It does not validate FA2/flex/SAM2 kernels, source hooks,
full-generator compatibility, distributed behavior or full-record acceptance.
A later immutable authorization amendment is required before execution. R3
remains **IN PROGRESS**; no GPU contract, runtime parity or scientific result is
claimed.

## Bounded v5 backend-canary authorization

On 2026-09-17 the user explicitly approved the proposed first stage only:
transfer the frozen source to the clean Bootes worktree, execute exact
source/environment preflight, run one bounded FA2, compiled flex-attention and
SAM2 CUDA canary on one visible GPU, then stop and report. The immutable v5
protocol records that scope while checkpoint/model loading, generation,
distributed execution and the four later stages remain `not-approved`.

The dedicated canary runner is fail closed: it requires the clean exact
parent/Wan/LaMa binding, v5's package/hardware identities, FA3 absence, and one
visible CUDA device. Its output path must be outside the repository. At this
pre-execution checkpoint no GPU kernel had yet run, so this section records
authorization and controls rather than a result.
