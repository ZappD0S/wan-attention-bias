#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --job-name=multi_sample_inference

# NOTE: Hardware-specific directives (--partition, --account, --constraint)
# are intentionally left out of this script header.
#
# This allows the same script to be submitted to either A100 or H100 nodes
# without editing the file. These values must be passed via the command line
# or a wrapper script at submission time.
#
# Usage Example (for H100):
# sbatch --partition=gpu_p6 --account=xvh@h100 --constraint=h100 job.slurm

#SBATCH --output=logs/worker_%j.out
#SBATCH --error=logs/worker_%j.err
#SBATCH --open-mode=append

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-task=1
#SBATCH --hint=nomultithread

# Signal handler for Slurm timeout
#SBATCH --signal=B:USR1@60

# Function to run when the timeout signal is received
resubmit_handler() {
  echo "STATUS: Received SLURM timeout signal. Resubmitting job..."
  # Requeue the current job
  scontrol requeue $SLURM_JOB_ID
  exit 0
}

# Register the handler for the USR1 signal
trap 'resubmit_handler' USR1

module purge

if [ "$NODE_ARCH" == "a100" ]; then
  echo "STATUS: Configuring environment for A100"
  module load arch/a100
elif [ "$NODE_ARCH" == "h100" ]; then
  echo "STATUS: Configuring environment for H100"
  module load arch/h100
else
  echo "ERROR: NODE_ARCH is unset or unknown ('$NODE_ARCH'). Exiting." >&2
  exit 1
fi

module load cuda/12.8.0 ffmpeg/6.1.1-cuda

echo "STATUS: Checking GPU availability via nvidia-smi..."
nvidia-smi || {
  echo "ERROR: nvidia-smi failed. Driver not found." >&2
  exit 1
}

PROMPTS_FILE="../image_prompt_generation/prompts_modified.json"
OUTPUT_DIR="$SCRATCH/attn_bias/output_videos/"
mkdir -p $OUTPUT_DIR

CHECKPOINT_PATH="$SCRATCH/.weights/"
mkdir -p $CHECKPOINT_PATH

source ../.venv/bin/activate || {
  echo "Failed to activate venv"
  exit 1
}

echo "STATUS: Starting Python inference at $(date)"

# Run in background to allow trap catching
python -m multi_sample_inference \
  --prompts-file "$PROMPTS_FILE" \
  --param-grid-file "../multi_sample_inference/param_grid.json" \
  --output-path "$OUTPUT_DIR" \
  --checkpoint-path $CHECKPOINT_PATH &

# Get the Process ID of the python command
PYTHON_PID=$!

# Wait for the python process to finish
# 'wait' is an interruptible command.
wait $PYTHON_PID
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo "STATUS: Nothing left to do. Exit."
  exit 0
elif [ $EXIT_CODE -eq 2 ]; then
  echo "STATUS: There is stills some work to do. Requeueing for more work..."
  scontrol requeue $SLURM_JOB_ID
elif [ $EXIT_CODE -eq 143 ] || [ $EXIT_CODE -eq 130 ]; then
  echo "STATUS: Job terminated by signal (likely requeued by trap)."
  exit 0
else
  echo "STATUS: Python script failed with Exit Code $EXIT_CODE."
  exit $EXIT_CODE
fi
