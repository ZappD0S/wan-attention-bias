#!/bin/bash

TYPE=$1 # Takes 'a100' or 'h100'

if [ "$TYPE" == "h100" ]; then
  PARTITION="gpu_p6"
  ACCOUNT="xvh@h100"
  CONSTRAINT="h100"
  export NODE_ARCH="h100" # Set an env var for the slurm script
elif [ "$TYPE" == "a100" ]; then
  PARTITION="gpu_p5"
  ACCOUNT="xvh@a100"
  CONSTRAINT="a100"
  export NODE_ARCH="a100" # Set an env var for the slurm script
else
  echo "Usage: ./submit.sh [a100|h100]"
  exit 1
fi

# Use --export=ALL to ensure NODE_ARCH is passed to the job
sbatch --partition=$PARTITION --account=$ACCOUNT --constraint=$CONSTRAINT --export=ALL job.slurm
