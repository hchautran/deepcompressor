#!/bin/bash
# SAM2 SVDQuant W4A4 Quantization Script
#
# Usage:
#   ./run_quantize.sh [model] [quant_config] [calib_path] [output_dir]
#
# Examples:
#   ./run_quantize.sh sam2.1-hiera-large w4a4
#   ./run_quantize.sh sam2-hiera-base-plus w4a4-fast /path/to/coco/val2017
#   ./run_quantize.sh sam2.1-hiera-large w4a4 /path/to/coco/val2017 /path/to/output

set -e

# Default values
MODEL=${1:-"sam2.1-hiera-large"}
QUANT_CONFIG=${2:-"w4a4"}
CALIB_PATH=${3:-"/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/deepcompressor/data/coco/val2017"}
OUTPUT_DIR=${4:-"runs/sam2"}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$(dirname "$EXAMPLES_DIR")")"

# Config paths
DEFAULT_CONFIG="$EXAMPLES_DIR/configs/__default__.yaml"
MODEL_CONFIG="$EXAMPLES_DIR/configs/model/${MODEL}.yaml"
SVDQUANT_DEFAULT="$EXAMPLES_DIR/configs/svdquant/__default__.yaml"
QUANT_CONFIG_FILE="$EXAMPLES_DIR/configs/svdquant/${QUANT_CONFIG}.yaml"

# Check if config files exist
if [ ! -f "$MODEL_CONFIG" ]; then
    echo "Error: Model config not found: $MODEL_CONFIG"
    echo "Available models:"
    ls -1 "$EXAMPLES_DIR/configs/model/" | sed 's/.yaml$//'
    exit 1
fi

if [ ! -f "$QUANT_CONFIG_FILE" ]; then
    echo "Error: Quant config not found: $QUANT_CONFIG_FILE"
    echo "Available configs:"
    ls -1 "$EXAMPLES_DIR/configs/svdquant/" | sed 's/.yaml$//'
    exit 1
fi

# Check calibration path
if [ ! -d "$CALIB_PATH" ]; then
    echo "Warning: Calibration path not found: $CALIB_PATH"
    echo "Please provide a valid path to COCO val2017 images"
fi

echo "=========================================="
echo "SAM2 SVDQuant W4A4 Quantization"
echo "=========================================="
echo "Model:         $MODEL"
echo "Quant Config:  $QUANT_CONFIG"
echo "Calib Path:    $CALIB_PATH"
echo "Output Dir:    $OUTPUT_DIR"
echo "=========================================="

# Build config argument
CONFIG_ARG="--config $DEFAULT_CONFIG"
CONFIG_ARG="$CONFIG_ARG --config $MODEL_CONFIG"
CONFIG_ARG="$CONFIG_ARG --config $SVDQUANT_DEFAULT"
CONFIG_ARG="$CONFIG_ARG --config $QUANT_CONFIG_FILE"

# Run quantization
cd "$PROJECT_ROOT"
python -m deepcompressor.app.sam2.ptq \
    $CONFIG_ARG \
    --quant.calib.path "$CALIB_PATH" \
    --output.root "$OUTPUT_DIR" \
    --cache.root "$OUTPUT_DIR" \
    "$@"

echo "=========================================="
echo "Quantization complete!"
echo "Output saved to: $OUTPUT_DIR"
echo "=========================================="
