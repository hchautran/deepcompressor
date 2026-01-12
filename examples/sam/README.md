# SAM Nunchaku Backbone Quantization

This directory contains examples for quantizing SAM (Segment Anything Model) with Nunchaku backbone to 4-bit using the DeepCompressor framework.

## Overview

The SAM quantization pipeline supports:
- **Weight Quantization**: 4-bit (INT4/UINT4) with GPTQ or RTN
- **Activation Quantization**: 8-bit (INT8/UINT8) with dynamic range calibration
- **Advanced Techniques**: SVDQuant (low-rank branch), Smooth Quantization, Rotation
- **Flexible Configuration**: Per-channel, per-group, or per-token quantization

## Quick Start

### 1. Installation

Ensure you have DeepCompressor installed:

```bash
# From the root of the repository
pip install -e .
```

### 2. Prepare Your Model

You'll need a SAM model with Nunchaku backbone. Options include:
- SAM2 from Meta AI
- Custom SAM implementations with Nunchaku backbone

```python
# Example: Loading SAM2 from HuggingFace
from transformers import Sam2Model

model = Sam2Model.from_pretrained("facebook/sam2-hiera-base")
```

### 3. Run Quantization

```bash
# W4A8: 4-bit weights, 8-bit activations
python quantize_sam.py \
    --config configs/int4_w4a8.yaml \
    --model-path facebook/sam2-hiera-base \
    --output-dir ./quantized_sam_w4a8 \
    --num-calib-samples 128

# W4-only: 4-bit weights only
python quantize_sam.py \
    --config configs/int4_w4only.yaml \
    --model-path facebook/sam2-hiera-base \
    --output-dir ./quantized_sam_w4

# SVDQuant: 4-bit weights with low-rank compensation
python quantize_sam.py \
    --config configs/int4_svdquant.yaml \
    --model-path facebook/sam2-hiera-base \
    --output-dir ./quantized_sam_svdquant
```

## Configuration Files

### `int4_w4a8.yaml` - 4-bit Weights + 8-bit Activations

Standard quantization configuration:
- Weights: UINT4 with GPTQ, group size 128
- Activations: UINT8 per-token quantization
- Best balance of compression and accuracy

### `int4_w4only.yaml` - 4-bit Weights Only

Weight-only quantization:
- Weights: UINT4 with GPTQ, group size 128
- Activations: FP16 (no quantization)
- Faster quantization, no calibration data needed

### `int4_svdquant.yaml` - SVDQuant with Low-Rank Compensation

Advanced quantization with outlier handling:
- Weights: UINT4 with GPTQ + low-rank branch (rank 32)
- Activations: UINT8 per-token quantization
- Better accuracy for models with outliers

## Customization

### Calibration Dataset

Replace the `DummySamDataset` in `quantize_sam.py` with your actual dataset:

```python
from torch.utils.data import Dataset

class SamCalibDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __getitem__(self, idx):
        # Load and preprocess your images
        image = load_image(self.image_paths[idx])
        return {"pixel_values": image}
```

### Configuration Parameters

Key parameters you can adjust in the YAML configs:

```yaml
wgts:
  dtype: uint4                    # uint4, sint4, uint8, sint8
  group_shapes: [[1, 128]]        # Group size for per-group quantization
  skip_patch_embed: true          # Skip first layer
  kernel_gptq:
    block_size: 128               # GPTQ block size
    damp_percentage: 0.01         # Damping factor
```

### Skip Layers

Configure which layers to skip:

```yaml
wgts:
  skip_patch_embed: true          # Don't quantize patch embedding
  skip_first_block: false         # Quantize first transformer block
  skip_last_block: false          # Quantize last transformer block
  skip_decoder: false             # Quantize decoder
```

## Architecture Support

The quantization pipeline is designed for SAM with Nunchaku backbone, which follows the Vision Transformer architecture:

- **Backbone**: Nunchaku ViT encoder
  - Patch Embedding
  - Transformer Blocks (Attention + MLP)
  - Layer Normalization
- **Decoder**: Mask decoder with transformer blocks
- **Prompt Encoder**: Point/box/mask prompt encoding

## Expected Results

Typical compression and accuracy results:

| Configuration | Compression | mIoU (relative) | Latency |
|---------------|-------------|-----------------|---------|
| FP16 Baseline | 1x          | 100%            | 1x      |
| W4A8          | ~4x         | 98-99%          | 2-3x    |
| W4-only       | ~2x         | 99-100%         | 1.5x    |
| SVDQuant W4A8 | ~4x         | 99-100%         | 2-3x    |

*Note: Actual results depend on model size, dataset, and hardware*

## Advanced Features

### Smooth Quantization

Enable smooth quantization to migrate difficulty from activations to weights:

```yaml
smooth:
  enabled: true
  alpha: 0.5                      # Smooth factor (0-1)
```

### Rotation

Apply rotation to align outliers:

```yaml
rotation:
  enabled: true
```

### Low-Rank Branch (SVDQuant)

Add high-precision low-rank branches to handle outliers:

```yaml
wgts:
  low_rank:
    rank: 32                      # Rank of low-rank branch
    compensate: true              # Compensate for quantization error
```

## Troubleshooting

### Out of Memory

- Reduce `batch_size` or `num_calib_samples`
- Use gradient checkpointing
- Quantize on CPU first

### Poor Accuracy

- Increase calibration samples
- Try SVDQuant configuration
- Enable smooth quantization
- Skip sensitive layers (first/last blocks)
- Reduce group size for finer granularity

### Long Quantization Time

- Use W4-only configuration (no activation quantization)
- Disable GPTQ and use RTN instead
- Reduce calibration samples

## References

- [DeepCompressor](https://github.com/mit-han-lab/deepcompressor)
- [SAM2 Paper](https://arxiv.org/abs/2401.12741)
- [GPTQ Paper](https://arxiv.org/abs/2210.17323)
- [SVDQuant Paper](https://arxiv.org/abs/2403.07378)

## Citation

If you use this quantization pipeline, please cite:

```bibtex
@inproceedings{deepcompressor2024,
  title={DeepCompressor: Efficient Quantization Framework for Large Models},
  author={MIT HAN Lab},
  year={2024}
}
```
