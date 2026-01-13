# SAM2 Post-Training Quantization

This example demonstrates how to quantize SAM2 (Segment Anything Model 2) using DeepCompressor with nunchaku backend support.

## Overview

SAM2 uses a Hiera (Hierarchical Vision Transformer) backbone for image encoding. This pipeline quantizes the image encoder to reduce model size and improve inference speed.

**Supported Models:**
| Model | HuggingFace ID | Parameters |
|-------|----------------|------------|
| tiny | facebook/sam2.1-hiera-tiny | 38M |
| small | facebook/sam2.1-hiera-small | 46M |
| base-plus | facebook/sam2.1-hiera-base-plus | 80M |
| large | facebook/sam2.1-hiera-large | 224M |

**Quantization Configurations:**
| Config | Weights | Activations | Use Case |
|--------|---------|-------------|----------|
| W4A4 | 4-bit | 4-bit | Maximum compression |
| W4A8 | 4-bit | 8-bit | Balanced accuracy/size |
| W8A8 | 8-bit | 8-bit | Higher accuracy |

## Requirements

```bash
pip install deepcompressor
pip install transformers safetensors
```

For nunchaku backend support:
```bash
pip install nunchaku
```

## Quick Start

### Using a Configuration File

```bash
python quantize_sam2.py --config configs/hiera_tiny_w4a8.yaml
```

### With Command-Line Arguments

```bash
python quantize_sam2.py \
    --model.name tiny \
    --quant.calib.path /path/to/coco/val2017 \
    --output.root ./outputs/sam2
```

### Convert to Nunchaku Format

```bash
python quantize_sam2.py \
    --config configs/hiera_tiny_w4a8.yaml \
    --convert-nunchaku
```

## Configuration Files

### Basic W4A8 Configuration

```yaml
# configs/hiera_tiny_w4a8.yaml
model:
  name: tiny
  device: cuda
  dtype: float16

cache:
  root: ./cache/sam2

quant:
  calib:
    path: /path/to/coco/val2017
    num_samples: 128
    batch_size: 1
    image_size: 1024

  wgts:
    dtype: uint4
    group_shapes: [[1, 128]]
    kernel_gptq:
      damp_percentage: 0.01
      block_size: 128

  ipts:
    dtype: uint8
    group_shapes: [[1, -1]]
    static: false

  smooth:
    proj:
      granularity: Layer
      strategy: GridSearch
      num_grids: 20
      alpha: 0.5

output:
  root: ./outputs/sam2
  job: hiera_tiny_w4a8

save_model: true
seed: 42
```

### Advanced: W4A8 with SVDQuant

For better accuracy with 4-bit weights, enable SVDQuant low-rank compensation:

```yaml
# configs/hiera_large_w4a8_svdquant.yaml
quant:
  wgts:
    dtype: uint4
    group_shapes: [[1, 128]]
    kernel_gptq:
      damp_percentage: 0.01
      block_size: 128
      num_inv_tries: 250
    low_rank:
      rank: 32
      exclusive: false

  rotation:
    transforms: [hadamard]
    random: false
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--config` | Path to YAML configuration file | None |
| `--model.name` | Model variant (tiny, small, base-plus, large) | tiny |
| `--model.device` | Device (cuda or cpu) | cuda |
| `--quant.calib.path` | Path to calibration images | "" |
| `--quant.calib.num_samples` | Number of calibration samples | 128 |
| `--output.root` | Output directory | ./outputs/sam2 |
| `--save-model` | Save the quantized model | false |
| `--convert-nunchaku` | Convert to nunchaku format | false |
| `--seed` | Random seed | 42 |
| `--verbose`, `-v` | Enable verbose logging | false |

## Calibration Dataset

The quantization process requires a calibration dataset. You can use:

1. **COCO val2017**: Download from [COCO dataset](https://cocodataset.org/)
2. **SA-1B subset**: Download from [Segment Anything](https://segment-anything.com/)
3. **Custom images**: Any directory containing JPEG/PNG images

```bash
# Example with COCO val2017
python quantize_sam2.py \
    --config configs/hiera_tiny_w4a8.yaml \
    --quant.calib.path /data/coco/val2017
```

## Output Structure

After quantization, the output directory contains:

```
outputs/sam2/hiera_tiny_w4a8/
├── cache/
│   ├── model.pt          # Quantized weights
│   ├── scale.pt          # Quantization scales
│   ├── smooth.pt         # Smooth quantization scales (if enabled)
│   └── branch.pt         # SVDQuant branches (if enabled)
├── nunchaku/             # (if --convert-nunchaku)
│   ├── hiera_blocks.safetensors
│   └── unquantized_layers.safetensors
└── config.yaml           # Run configuration
```

## Quantization Techniques

### Smooth Quantization

Reduces activation outliers by migrating quantization difficulty from activations to weights:

```yaml
quant:
  smooth:
    proj:
      granularity: Layer
      strategy: GridSearch
      num_grids: 20
      alpha: 0.5  # Balance between activation and weight quantization
```

### Rotation (Hadamard Transform)

Applies Hadamard rotation to improve weight distribution:

```yaml
quant:
  rotation:
    transforms: [hadamard]
    random: false
```

### SVDQuant (Low-Rank Compensation)

Adds low-rank branches to compensate for quantization error:

```yaml
quant:
  wgts:
    low_rank:
      rank: 32        # Rank of compensation matrices
      exclusive: false
```

## Validating Quantized Models

After quantization, validate the model quality by comparing outputs with the original model.

### Quick Validation

```bash
python validate_sam2.py \
    --original facebook/sam2.1-hiera-tiny \
    --quantized ./outputs/sam2/hiera_tiny_w4a8/model \
    --images /path/to/validation/images \
    --num-samples 32
```

### Full Validation with Layer Comparison

```bash
python validate_sam2.py \
    --original facebook/sam2.1-hiera-tiny \
    --quantized ./outputs/sam2/hiera_tiny_w4a8/model \
    --images /path/to/coco/val2017 \
    --num-samples 100 \
    --compare-layers \
    --save-report validation_report.json
```

### Validation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Cosine Similarity | Similarity between original and quantized outputs | >= 0.95 |
| SNR (dB) | Signal-to-Noise Ratio | >= 20 dB |
| MSE | Mean Squared Error | Lower is better |
| MAE | Mean Absolute Error | Lower is better |

### Quality Interpretation

| Cosine Similarity | Quality | Recommendation |
|-------------------|---------|----------------|
| >= 0.99 | Excellent | Ready for production |
| 0.95 - 0.99 | Good | Suitable for most use cases |
| 0.90 - 0.95 | Acceptable | May need fine-tuning for sensitive tasks |
| 0.80 - 0.90 | Fair | Consider less aggressive quantization |
| < 0.80 | Poor | Increase calibration samples or reduce quantization |

### Programmatic Validation

```python
from deepcompressor.app.sam2 import validate_quantization
from deepcompressor.app.sam2.dataset.calib import Sam2CalibConfig

# Build dataloader
calib_config = Sam2CalibConfig(
    path="/path/to/images",
    num_samples=32,
)
dataloader = calib_config.build_dataloader()

# Validate
result = validate_quantization(
    original_model=original_model,
    quantized_model=quantized_model,
    dataloader=dataloader,
    compare_intermediate=True,
)

print(f"Cosine Similarity: {result.cosine_similarity:.4f}")
print(f"SNR: {result.snr_db:.2f} dB")
```

## Tips for Best Results

1. **Use more calibration samples** for larger models (256+ for large)
2. **Enable smooth quantization** for W4A8 configurations
3. **Use SVDQuant** for W4A4 to maintain accuracy
4. **Enable rotation** when using aggressive quantization (W4A4)
5. **Validate after quantization** to ensure quality meets requirements

## Troubleshooting

### CUDA Out of Memory

- Reduce `batch_size` in calibration config
- Use a smaller model variant
- Run on CPU with `--model.device cpu` (slower)

### Poor Accuracy

- Increase `num_samples` for calibration
- Enable smooth quantization
- Try SVDQuant with higher rank
- Use W4A8 instead of W4A4

## License

This example follows the same license as DeepCompressor.
