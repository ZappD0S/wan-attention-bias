#!/bin/bash

# Parse named arguments using getopts (single letters only)
# a: is -a [arch], c: is -c [config]
while getopts "a:c:" opt; do
  case $opt in
  a) NODE_ARCH="$OPTARG" ;;
  c) PARAM_CONFIG_NAME="$OPTARG" ;;
  *)
    echo "Usage: $0 -a [a100|h100] -c [param_name]"
    exit 1
    ;;
  esac
done

if [[ -z "$NODE_ARCH" || -z "$PARAM_CONFIG_NAME" ]]; then
  echo "Error: Both -a (arch) and -c (param-config) are mandatory."
  echo "Usage: $0 -a h100 -c my_config_file"
  exit 1
fi

if [[ "$NODE_ARCH" != "h100" && "$NODE_ARCH" != "a100" ]] || [ -z "$PARAM_CONFIG_NAME" ]; then
  echo "Usage: ./submit_job.sh [a100|h100] [param_file_name_without_extension]"
  echo "Example: ./submit_job.sh h100 all"
  exit 1
fi

if [ "$NODE_ARCH" == "h100" ]; then
  PARTITION="gpu_p6"
  ACCOUNT="xvh@h100"
  CONSTRAINT="h100"
elif [ "$NODE_ARCH" == "a100" ]; then
  PARTITION="gpu_p5"
  ACCOUNT="xvh@a100"
  CONSTRAINT="a100"
else
  echo "Usage: ./submit.sh [a100|h100]"
  exit 1
fi

export NODE_ARCH=$NODE_ARCH
export PARAM_CONFIG_NAME=$PARAM_CONFIG_NAME

# Use --export=ALL to ensure NODE_ARCH is passed to the job
sbatch --partition=$PARTITION --account=$ACCOUNT --constraint=$CONSTRAINT --export=ALL ./multi_sample_inference/job.slurm
