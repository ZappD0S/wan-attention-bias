#!/bin/bash

#SBATCH --partition=gpu_p5
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --account=xvh@a100
#SBATCH --constraint=a100

#SBATCH --job-name=video_processing_worker
#SBATCH --output=logs/worker_%A_%a.out  # %A=Job ID, %a=Task ID
#SBATCH --error=logs/worker_%A_%a.err

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1

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

# TODO: do we need to create jobs?
uv run python -m multi_sample_inference.multi_sample_inference \
  --prompts-file "$PROMPTS_FILE" \
  --output-path "$OUTPUT_DIR" \
  --checkpoint-path $CHECKPOINT_PATH
