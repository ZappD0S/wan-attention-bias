#!/bin/bash
# Submit the contract-case chunks of one frozen jz protocol, exactly as bound, chained
# with afterok so a failed or cancelled chunk cancels every later one. Login node only.
# Usage: tools/j1_submit_contract_chunks.sh <docs/r3_protocol_jz_vN.json>
# Each bound command is printed by j1_stage from the validated protocol; only the
# literal $J1_PREVIOUS_CHUNK_JOB_ID placeholder is substituted, without eval.
set -euo pipefail

PROTOCOL=${1:?jz protocol path required}
REPO=${J1_REPO:-$WORK/wan_experiments_j1}
TOOLS=${J1_TOOLS:-$WORK/j1_tools}
CHUNKS=4
# shellcheck disable=SC2016  # the literal placeholder, never expanded
PLACEHOLDER='$J1_PREVIOUS_CHUNK_JOB_ID'

export UV_PYTHON_PREFERENCE=only-managed UV_PYTHON_DOWNLOADS=never
export UV_CACHE_DIR=$TOOLS/uv-cache UV_PYTHON_INSTALL_DIR=$TOOLS/python
export PATH=$TOOLS/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1

cd "$REPO"
test -z "$(git status --porcelain)" || { echo "refusing: dirty checkout" >&2; exit 3; }
previous=
for index in $(seq 1 "$CHUNKS"); do
  stage=single-rank-contract-cases-chunk-$index
  command=$(uv run --no-sync --locked python -m multi_sample_inference.j1_stage \
    --protocol "$PROTOCOL" --stage "$stage" --print-command)
  if [[ $index -eq 1 ]]; then
    [[ $command != *"$PLACEHOLDER"* ]] || { echo "chunk 1 must not depend on a job" >&2; exit 3; }
  else
    [[ $command == *"$PLACEHOLDER"* ]] || { echo "chunk $index lacks afterok" >&2; exit 3; }
    command=${command//"$PLACEHOLDER"/$previous}
  fi
  read -ra argv <<< "$command"
  [[ ${argv[0]} == sbatch ]] || { echo "bound command is not sbatch" >&2; exit 3; }
  previous=$("${argv[@]}")
  [[ $previous =~ ^[0-9]+$ ]] || { echo "sbatch returned no job id: $previous" >&2; exit 3; }
  echo "$stage $previous"
done
