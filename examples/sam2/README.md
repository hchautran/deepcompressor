# SAM2 SVDQuant W4A4 Quantization Examples

This directory contains example configurations and scripts for quantizing SAM2 models using SVDQuant with W4A4 (4-bit weights, 4-bit activations).

## Prerequisites

1. **SAM2 Installation**: Install SAM2 from the official repository:
   ```bash
   pip install sam2
   # or from source:
   # git clone https://github.com/facebookresearch/sam2.git
   # cd sam2 && pip install -e .
   ```

2. **Calibration Dataset**: Download COCO val2017 images for calibration:
   ```bash
   # Download COCO val2017
   wget http://images.cocodataset.org/zips/val2017.zip
   unzip val2017.zip -d /path/to/coco/
   ```

## Quick Start

### Using Python Script

```bash
# Quantize SAM2.1 Hiera Large (recommended)
python scripts/quantize_sam2.py \
    --model sam2.1-hiera-large \
    --calib_path /path/to/coco/val2017

# Quantize with custom settings
python scripts/quantize_sam2.py \
    --model sam2-hiera-base-plus \
    --rank 64 \
    --num_samples 256 \
    --output_dir ./outputs
```

### Using Shell Script

```bash
# Make script executable
chmod +x scripts/run_quantize.sh

# Run with defaults
./scripts/run_quantize.sh sam2.1-hiera-large w4a4

# Run with custom paths
./scripts/run_quantize.sh sam2.1-hiera-large w4a4 /path/to/coco/val2017 /path/to/output
```

### Using YAML Configs

```bash
# Run with YAML configuration
python -m deepcompressor.app.sam2.ptq \
    --config configs/__default__.yaml \
    --config configs/model/sam2.1-hiera-large.yaml \
    --config configs/svdquant/__default__.yaml \
    --config configs/svdquant/w4a4.yaml \
    --quant.calib.path /path/to/coco/val2017
```

## Configuration Files

### Model Variants

| File | Model | Description |
|------|-------|-------------|
| `configs/model/sam2.1-hiera-large.yaml` | SAM2.1 Hiera Large | Recommended, best accuracy |
| `configs/model/sam2.1-hiera-base-plus.yaml` | SAM2.1 Hiera Base+ | Good balance |
| `configs/model/sam2.1-hiera-small.yaml` | SAM2.1 Hiera Small | Faster inference |
| `configs/model/sam2.1-hiera-tiny.yaml` | SAM2.1 Hiera Tiny | Fastest, lowest accuracy |
| `configs/model/sam2-hiera-*.yaml` | SAM2 variants | Original SAM2 models |

### Quantization Configs

| File | Description |
|------|-------------|
| `configs/svdquant/w4a4.yaml` | Standard W4A4 with SVDQuant (rank=32) |
| `configs/svdquant/w4a4-fast.yaml` | Fast W4A4 (fewer iterations) |
| `configs/svdquant/w4a4-r64.yaml` | High quality W4A4 (rank=64) |
| `configs/svdquant/w4a8.yaml` | W4A8 for higher accuracy |

## SVDQuant Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rank` | 32 | Low-rank dimension. Higher = better accuracy, more memory |
| `num_iters` | 100 | Calibration iterations. More = better but slower |
| `group_size` | 128 | Quantization group size |
| `num_samples` | 128 | Number of calibration images |

## Output Structure

```
runs/sam2/
├── cache/
│   └── quant/
│       ├── smooth.pt    # Smooth quantization scales
│       ├── branch.pt    # SVDQuant low-rank branches
│       ├── wgts.pt      # Weight quantizer state
│       └── acts.pt      # Activation quantizer state
└── sam2.1-hiera-large/
    └── svdquant_w4a4_r32/
        └── model.pt     # Quantized model checkpoint
```

## Tips

1. **Start with fewer samples**: Use `--num_samples 64` for quick testing before full calibration.

2. **Adjust rank for quality**: Increase `--rank 64` for better accuracy at the cost of memory.

3. **Fast mode**: Use `w4a4-fast.yaml` config for rapid prototyping.

4. **Memory issues**: Reduce batch size with `--batch_size 1` if running out of GPU memory.

## Calibration Path

The default calibration path is set to:
```
/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/deepcompressor/data/coco/val2017
```

Override with `--quant.calib.path` or `--calib_path` as needed.
