#!/bin/bash

mkdir -p "$1"
hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$1"
