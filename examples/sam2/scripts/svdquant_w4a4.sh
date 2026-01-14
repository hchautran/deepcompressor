#!/usr/bin/env bash
# set -euo pipefail

CONFIG_PATH="${1:-examples/sam2/configs/hiera_large_w4a4_svdquant.yaml}"

export DEEPCOMPRESSOR_DISABLE_EXT=1
export SAM2_ALLOW_REPO_IMPORT=1

python -m deepcompressor.app.sam2.ptq "$CONFIG_PATH"
