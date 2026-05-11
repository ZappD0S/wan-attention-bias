#!/bin/bash

PROMPTS_FILE="./image_prompt_generation/prompts_modified.json"
OUTPUT_DIR="./multi_sample_inference/debug_output/"
PARAM_GRID_FILE="./multi_sample_inference/param_configs/only_concept_weaver.json"

./run.sh python -m multi_sample_inference.multi_sample_inference \
  --prompts-file "$PROMPTS_FILE" \
  --param-grid-file $PARAM_GRID_FILE \
  --output-path "$OUTPUT_DIR"
