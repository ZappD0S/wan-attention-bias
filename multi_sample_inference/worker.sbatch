#!/bin/bash

#SBATCH --partition=gpu_p5
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --account=xvh@a100
#SBATCH --constraint=a100

#SBATCH --job-name=multi_sample_inference
#SBATCH --output=logs/worker_%A_%a.out  # %A=Job ID, %a=Task ID
#SBATCH --error=logs/worker_%A_%a.err

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-task=1
#SBATCH --hint=nomultithread

# Send SIGUSR1 to the script 60 seconds before the time limit expires
#SBATCH --signal=B:USR1@60

# Function to run when the timeout signal is received
resubmit_handler() {
  echo "STATUS: Received SLURM timeout signal. Resubmitting job..."
  sbatch "$0"
  exit 0
}

# Register the handler for the USR1 signal
trap 'resubmit_handler' USR1

# Load necessary modules
module purge
module load arch/a100
module load cuda/12.8.0
module load ffmpeg/6.1.1-cuda

PROMPTS_FILE="./image_prompt_generation/prompts_modified.json"
OUTPUT_DIR="$SCRATCH/attn_bias/output_videos/"
mkdir -p $OUTPUT_DIR

CHECKPOINT_PATH="$SCRATCH/.weights/"
mkdir -p $CHECKPOINT_PATH

source .venv/bin/activate

# run the command in the background (&) so the shell can catch the trap.
python -m multi_sample_inference.multi_sample_inference \
  --prompts-file "$PROMPTS_FILE" \
  --param-grid-file "./multi_sample_inference/param_grid.json" \
  --output-path "$OUTPUT_DIR" \
  --checkpoint-path $CHECKPOINT_PATH &

# Get the Process ID of the python command
PYTHON_PID=$!

# Wait for the python process to finish
# 'wait' is an interruptible command.
wait $PYTHON_PID
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo "STATUS: Work completed successfully. Resubmitting for more..."
  sbatch "$0"
elif [ $EXIT_CODE -eq 2 ]; then
  echo "STATUS: Nothing left to do. Chain finished gracefully."
  exit 0
elif [ $EXIT_CODE -eq 143 ] || [ $EXIT_CODE -eq 130 ]; then
  # these codes often indicate the process was terminated by a signal (128+signal)
  # the trap handler usually catches this first, but this is a backup.
  echo "STATUS: Job terminated by SLURM signal."
else
  echo "STATUS: Python script failed with Exit Code $EXIT_CODE (Unhandled Exception)."
  echo "Stopping chain to prevent infinite error loops."
  exit $EXIT_CODE
fi
