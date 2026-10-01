#!/bin/bash
#SBATCH --job-name=j1-stage
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --account=xvh@a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fswork/projects/rech/xvh/ukl39yh/j1_runs/slurm-logs/%x-%j.out
# Run exactly one approved J1 stage on one SLURM-allocated A100, offline.
# Usage: sbatch [--qos=... --time=...] tools/j1_slurm_stage.sh <stage> <docs/r3_protocol_jz_vN.json>
# The protocol's authorization record binds the exact sbatch command per stage.
# Contract-case chunk stages (jz-v5 on) are chained by tools/j1_submit_contract_chunks.sh.
# Logs go outside the checkout: an untracked log would fail the clean-worktree gate.
set -eo pipefail

STAGE=${1:?stage id required}
PROTOCOL=${2:?jz protocol path required}
REPO=${J1_REPO:-$WORK/wan_experiments_j1}
TOOLS=${J1_TOOLS:-$WORK/j1_tools}

module purge
# Compute nodes have no system git (login nodes do); the source gates need it.
module load arch/a100 cuda/12.8.0 git/2.53.0
set -u

# Compute nodes are offline: never reach the Hub or download interpreters.
export HF_HOME=$WORK/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export UV_PYTHON_PREFERENCE=only-managed UV_PYTHON_DOWNLOADS=never UV_OFFLINE=1
export UV_CACHE_DIR=$TOOLS/uv-cache UV_PYTHON_INSTALL_DIR=$TOOLS/python
export PATH=$TOOLS/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1
# Job-unique torchrun rendezvous port (read by torch.distributed.run via PET_*),
# so shared gpu_p5 nodes never collide on the default 29500.
export PET_MASTER_ADDR=127.0.0.1
export PET_MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

cd "$REPO"
echo "start $(date -u +%FT%TZ) job $SLURM_JOB_ID node $(hostname) stage $STAGE commit $(git rev-parse HEAD)"
OUTPUT=$(uv run --no-sync --locked python -m multi_sample_inference.j1_stage \
  --protocol "$PROTOCOL" --stage "$STAGE" --print-output)
if [[ -e "$OUTPUT" ]]; then
  echo "refusing: J1 $STAGE attempt already exists at $OUTPUT" >&2
  exit 3
fi
nvidia-smi --query-gpu=uuid,name,driver_version,compute_cap,memory.total --format=csv,noheader
uv run --no-sync --locked python -m multi_sample_inference.j1_stage \
  --protocol "$PROTOCOL" --stage "$STAGE"
echo "end $(date -u +%FT%TZ)"
