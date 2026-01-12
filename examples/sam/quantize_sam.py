#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM Nunchaku Backbone Quantization Example

This script demonstrates how to quantize a SAM model with Nunchaku backbone
to 4-bit weights using the DeepCompressor framework.

Usage:
    python quantize_sam.py --config configs/int4_w4a8.yaml --model-path /path/to/sam --output-dir ./quantized_sam
"""

import argparse
import os
from pathlib import Path

import torch
import yaml
from omniconfig import OmniConfig
from torch.utils.data import DataLoader, Dataset

from deepcompressor.app.sam import SamModelStruct, SamQuantConfig, ptq


class DummySamDataset(Dataset):
    """Dummy dataset for SAM calibration.

    Replace this with your actual dataset loader (e.g., COCO, SA-1B).
    """

    def __init__(self, num_samples: int = 128, image_size: int = 1024):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random image tensor (replace with real data)
        image = torch.randn(3, self.image_size, self.image_size)
        return {"pixel_values": image}


def load_sam_model(model_path: str, device: str = "cuda"):
    """Load SAM model.

    Args:
        model_path: Path to SAM model checkpoint or HuggingFace model name
        device: Device to load model on

    Returns:
        Loaded SAM model
    """
    try:
        # Try loading with transformers (for SAM2)
        from transformers import Sam2Model

        print(f"Loading SAM model from {model_path}")
        model = Sam2Model.from_pretrained(model_path)
        model = model.to(device)
        model.eval()
        return model
    except ImportError:
        print("transformers not installed, trying native SAM loading...")
        # Add your custom SAM loading code here
        raise NotImplementedError("Please implement SAM model loading for your setup")


def main():
    parser = argparse.ArgumentParser(description="Quantize SAM with Nunchaku backbone to 4-bit")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/int4_w4a8.yaml",
        help="Path to quantization configuration YAML file",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to SAM model checkpoint or HuggingFace model name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./quantized_sam",
        help="Directory to save quantized model",
    )
    parser.add_argument(
        "--num-calib-samples",
        type=int,
        default=128,
        help="Number of calibration samples",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for calibration",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run quantization on",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=1024,
        help="Input image size",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache",
        help="Directory for caching intermediate results",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("SAM Nunchaku Backbone Quantization")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    print("=" * 80)

    # Load configuration
    print("\nLoading configuration...")
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    # Create SamQuantConfig from YAML
    config = OmniConfig.create(config_dict, schema=SamQuantConfig)

    # Override calibration settings from command line
    if config.quant.calib:
        config.quant.calib.num_samples = args.num_calib_samples
        config.quant.calib.batch_size = args.batch_size

    print(f"Configuration loaded:")
    print(f"  - Weight dtype: {config.quant.wgts.dtype if config.quant.wgts else 'None'}")
    print(f"  - Activation dtype: {config.quant.ipts.dtype if config.quant.ipts else 'None'}")
    print(f"  - GPTQ enabled: {config.quant.wgts.enabled_gptq if config.quant.wgts else False}")
    print(f"  - Calibration samples: {config.quant.calib.num_samples if config.quant.calib else 0}")

    # Load SAM model
    print("\nLoading SAM model...")
    model = load_sam_model(args.model_path, device=args.device)
    print(f"Model loaded successfully")

    # Create calibration dataloader
    # Replace DummySamDataset with your actual dataset
    print("\nPreparing calibration dataset...")
    calib_dataset = DummySamDataset(
        num_samples=config.quant.calib.num_samples if config.quant.calib else 0,
        image_size=args.image_size,
    )
    calib_loader = DataLoader(
        calib_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    print(f"Calibration dataset ready with {len(calib_dataset)} samples")

    # Create cache directory
    os.makedirs(args.cache_dir, exist_ok=True)

    # Run quantization
    print("\nStarting quantization...")
    quantized_model = ptq(
        model=model,
        config=config.quant,
        calib_loader=calib_loader if config.quant.enabled_ipts or config.quant.enabled_opts else None,
        cache=None,  # You can configure cache here
        load_dirpath="",
        save_dirpath=args.output_dir,
        copy_on_save=False,
        save_model=True,
    )

    print("\nQuantization complete!")
    print(f"Quantized model saved to: {args.output_dir}")

    # Optional: Test the quantized model
    print("\nTesting quantized model...")
    quantized_model.module.eval()
    with torch.no_grad():
        # Get a sample from calibration
        sample = next(iter(calib_loader))
        if isinstance(sample, dict):
            sample = {k: v.to(args.device) for k, v in sample.items()}
            output = quantized_model.module(**sample)
        else:
            sample = sample.to(args.device)
            output = quantized_model.module(sample)
    print("Test forward pass successful!")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
