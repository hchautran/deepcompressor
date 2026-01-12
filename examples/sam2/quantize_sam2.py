#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM2 (Segment Anything Model 2) Quantization from HuggingFace

This script demonstrates how to quantize a SAM2 model from HuggingFace
to 4-bit weights using the DeepCompressor framework.

Usage:
    # Quantize SAM2 Hiera-Tiny with W4A8
    python quantize_sam2.py --model tiny --config configs/hiera_tiny_w4a8.yaml

    # Quantize SAM2 Hiera-Base with W4A8
    python quantize_sam2.py --model base --config configs/hiera_base_w4a8.yaml

    # Quantize with COCO dataset
    python quantize_sam2.py --model tiny --config configs/hiera_tiny_w4a8.yaml \\
        --dataset coco --coco-root /path/to/coco

    # Weight-only quantization (no calibration needed)
    python quantize_sam2.py --model tiny --config configs/hiera_tiny_w4only.yaml
"""

import argparse
import os
from pathlib import Path

import torch
import yaml
from omniconfig import OmniConfig

from deepcompressor.app.sam2 import (
    Sam2QuantConfig,
    get_coco_calibration_loader,
    get_sam2_processor,
    load_sam2_from_huggingface,
    print_model_info,
    ptq,
)


def main():
    parser = argparse.ArgumentParser(description="Quantize SAM2 from HuggingFace to 4-bit")
    parser.add_argument(
        "--model",
        type=str,
        default="tiny",
        choices=["tiny", "small", "base", "large"],
        help="SAM2 model size (tiny, small, base, large)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to quantization configuration YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./quantized_sam2",
        help="Directory to save quantized model",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="random",
        choices=["random", "coco", "sa1b"],
        help="Calibration dataset (random, coco, sa1b)",
    )
    parser.add_argument(
        "--coco-root",
        type=str,
        default="",
        help="Root directory of COCO dataset (required if dataset=coco)",
    )
    parser.add_argument(
        "--sa1b-root",
        type=str,
        default="",
        help="Root directory of SA-1B dataset (required if dataset=sa1b)",
    )
    parser.add_argument(
        "--num-calib-samples",
        type=int,
        default=None,
        help="Number of calibration samples (overrides config)",
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
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Model data type",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache",
        help="Directory for caching intermediate results",
    )
    parser.add_argument(
        "--no-save-model",
        action="store_true",
        help="Don't save the quantized model",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("SAM2 HuggingFace Model Quantization")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Config: {args.config}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    print("=" * 80)

    # Parse dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]

    # Load configuration
    print("\nLoading configuration...")
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    config = OmniConfig.create(config_dict, schema={"quant": Sam2QuantConfig})
    config = config.quant

    # Override calibration settings from command line
    if args.num_calib_samples is not None and config.calib:
        config.calib.num_samples = args.num_calib_samples
    if config.calib:
        config.calib.batch_size = args.batch_size

    print(f"Configuration loaded:")
    print(f"  - Weight dtype: {config.wgts.dtype if config.wgts else 'None'}")
    print(f"  - Activation dtype: {config.ipts.dtype if config.ipts else 'None'}")
    print(f"  - GPTQ enabled: {config.wgts.enabled_gptq if config.wgts else False}")
    print(f"  - Calibration samples: {config.calib.num_samples if config.calib else 0}")

    # Load SAM2 model from HuggingFace
    print("\nLoading SAM2 model from HuggingFace...")
    model, model_struct = load_sam2_from_huggingface(
        model_name=args.model,
        device=args.device,
        torch_dtype=torch_dtype,
    )

    # Print model information
    print_model_info(model, model_struct)

    # Prepare calibration dataloader
    calib_loader = None
    needs_calibration = (config.enabled_ipts or config.enabled_opts) and config.calib.num_samples > 0

    if needs_calibration:
        print("\nPreparing calibration dataset...")

        if args.dataset == "coco":
            if not args.coco_root:
                raise ValueError("--coco-root must be specified when using COCO dataset")

            processor = get_sam2_processor(args.model)
            calib_loader = get_coco_calibration_loader(
                coco_root=args.coco_root,
                split="val2017",
                num_samples=config.calib.num_samples,
                batch_size=args.batch_size,
                processor=processor,
                num_workers=4,
            )
            print(f"Using COCO dataset from {args.coco_root}")

        elif args.dataset == "sa1b":
            from deepcompressor.app.sam2 import get_sa1b_calibration_loader

            if not args.sa1b_root:
                raise ValueError("--sa1b-root must be specified when using SA-1B dataset")

            processor = get_sam2_processor(args.model)
            calib_loader = get_sa1b_calibration_loader(
                sa1b_root=args.sa1b_root,
                num_samples=config.calib.num_samples,
                batch_size=args.batch_size,
                processor=processor,
                num_workers=4,
            )
            print(f"Using SA-1B dataset from {args.sa1b_root}")

        elif args.dataset == "random":
            # Create random calibration data
            from torch.utils.data import DataLoader, TensorDataset

            print("Using random calibration data")
            random_images = torch.randn(
                config.calib.num_samples,
                3,
                1024,
                1024,
                dtype=torch_dtype,
            )
            calib_dataset = TensorDataset(random_images)
            calib_loader = DataLoader(
                calib_dataset,
                batch_size=args.batch_size,
                shuffle=False,
            )

        print(f"Calibration dataset ready with {config.calib.num_samples} samples")

    # Create cache directory
    os.makedirs(args.cache_dir, exist_ok=True)

    # Run quantization
    print("\nStarting quantization...")
    quantized_model_struct = ptq(
        model=model_struct,
        config=config,
        calib_loader=calib_loader,
        cache=None,
        load_dirpath="",
        save_dirpath=args.output_dir if not args.no_save_model else "",
        copy_on_save=False,
        save_model=not args.no_save_model,
    )

    if not args.no_save_model:
        print(f"\nQuantized model saved to: {args.output_dir}")

    # Optional: Test the quantized model
    print("\nTesting quantized model...")
    quantized_model_struct.module.eval()
    with torch.no_grad():
        # Create a test input
        test_input = torch.randn(1, 3, 1024, 1024, dtype=torch_dtype, device=args.device)
        try:
            output = quantized_model_struct.module(pixel_values=test_input)
            print("Test forward pass successful!")
        except Exception as e:
            print(f"Test forward pass failed: {e}")

    print("\n" + "=" * 80)
    print("Quantization Complete!")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"1. Load quantized model from: {args.output_dir}")
    print(f"2. Evaluate on your validation dataset")
    print(f"3. Compare with FP16 baseline")


if __name__ == "__main__":
    main()
