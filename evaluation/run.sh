#!/bin/bash

OUTPUT_DIR="./evaluation/logs"
VIDEOS_MNT_DIR="./evaluation/output"
mkdir -p "$VIDEOS_MNT_DIR" "$OUTPUT_DIR"

if ! mountpoint -q "$VIDEOS_MNT_DIR"; then
  sshfs sophia:/srv/storage/robotlearn@storage2.grenoble.grid5000.fr/gzappavi/attn_bias/output_videos/massive_run $VIDEOS_MNT_DIR
fi

./run.sh python -m evaluation.main \
  --videos-dir "$VIDEOS_MNT_DIR" \
  --output-dir "$OUTPUT_DIR"
