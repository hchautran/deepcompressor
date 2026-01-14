#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SAM2 SVDQuant W4A4 Quantization Script

This script quantizes SAM2 models using SVDQuant with W4A4 (4-bit weights, 4-bit activations).

Usage:
    python quantize_sam2.py --model sam2.1-hiera-large --calib_path /path/to/coco/val2017
    python quantize_sam2.py --model sam2-hiera-base-plus --rank 64 --num_samples 256
    python quantize_sam2.py --help

Requirements:
    - COCO val2017 images for calibration
    - SAM2 installed (pip install sam2 or from source)
"""

import argparse
import os
import sys

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="SAM2 SVDQuant W4A4 Quantization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quantize SAM2.1 Hiera Large with default settings
    python quantize_sam2.py --model sam2.1-hiera-large --calib_path /path/to/coco/val2017

    # Quantize with higher rank for better accuracy
    python quantize_sam2.py --model sam2.1-hiera-large --rank 64 --calib_path /path/to/coco/val2017

    # Fast quantization with fewer samples
    python quantize_sam2.py --model sam2-hiera-small --num_samples 64 --num_iters 10

Available models:
    - sam2-hiera-tiny
    - sam2-hiera-small
    - sam2-hiera-base-plus
    - sam2-hiera-large
    - sam2.1-hiera-tiny
    - sam2.1-hiera-small
    - sam2.1-hiera-base-plus
    - sam2.1-hiera-large (recommended)
        """,
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="sam2.1-hiera-large",
        help="SAM2 model variant (default: sam2.1-hiera-large)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to model checkpoint (optional, downloads from HuggingFace if not provided)",
    )

    # Calibration configuration
    parser.add_argument(
        "--calib_path",
        type=str,
        default="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/deepcompressor/data/coco/val2017",
        help="Path to calibration images (COCO val2017)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=128,
        help="Number of calibration samples (default: 128)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for calibration (default: 1)",
    )

    # SVDQuant configuration
    parser.add_argument(
        "--rank",
        type=int,
        default=32,
        help="SVDQuant low-rank dimension (default: 32, higher = better accuracy)",
    )
    parser.add_argument(
        "--num_iters",
        type=int,
        default=100,
        help="Number of SVDQuant iterations (default: 100)",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=128,
        help="Quantization group size (default: 128)",
    )

    # Quantization precision
    parser.add_argument(
        "--weight_bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="Weight quantization bits (default: 4)",
    )
    parser.add_argument(
        "--activation_bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="Activation quantization bits (default: 4)",
    )

    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/sam2",
        help="Output directory (default: runs/sam2)",
    )
    parser.add_argument(
        "--save_model",
        type=str,
        default="",
        help="Path to save quantized model (optional)",
    )

    # Device configuration
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Model dtype (default: float16)",
    )
    parser.add_argument(
        "--sam2_repo_path",
        type=str,
        default="",
        help="Path to sam2 repository (auto-detected if not provided)",
    )

    # Other options
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed (default: 12345)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Import deepcompressor modules
    try:
        from deepcompressor.app.sam2 import SAM2Ptq, SAM2PtqRunConfig
        from deepcompressor.app.sam2.cache import SAM2PtqCacheConfig
        from deepcompressor.app.sam2.dataset.calib import SAM2CalibConfig
        from deepcompressor.app.sam2.pipeline import SAM2PipelineConfig
        from deepcompressor.app.sam2.quant import SAM2QuantConfig
        from deepcompressor.app.sam2.quant.quantizer import (
            SAM2ActivationQuantizerConfig,
            SAM2WeightQuantizerConfig,
        )
        from deepcompressor.calib.config import SkipBasedQuantLowRankCalibConfig
        from deepcompressor.data.dtype import QuantDataType
        from deepcompressor.utils.config.output import OutputConfig
    except ImportError as e:
        print(f"Error: Failed to import deepcompressor modules: {e}")
        print("Make sure deepcompressor is installed and SAM2 app is properly configured.")
        sys.exit(1)

    # Parse dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # Create weight quantization config
    weight_dtype = f"sint{args.weight_bits}"
    wgts_config = SAM2WeightQuantizerConfig(
        dtype=QuantDataType.from_str(weight_dtype),
        group_shapes=((-1, -1, args.group_size),),
        low_rank=SkipBasedQuantLowRankCalibConfig(
            rank=args.rank,
            num_iters=args.num_iters,
            compensate=True,
            early_stop=True,
        ),
    )

    # Create activation quantization config
    activation_dtype = f"sint{args.activation_bits}"
    ipts_config = SAM2ActivationQuantizerConfig(
        dtype=QuantDataType.from_str(activation_dtype),
        group_shapes=((-1, -1, args.group_size),),
        static=False,  # Dynamic quantization
        allow_unsigned=True,
    )

    # Create output activation config (disabled)
    opts_config = SAM2ActivationQuantizerConfig(dtype=None)

    # Create calibration config
    calib_config = SAM2CalibConfig(
        data="COCO",
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        path=args.calib_path,
    )

    # Create main quant config
    quant_config = SAM2QuantConfig(
        wgts=wgts_config,
        ipts=ipts_config,
        opts=opts_config,
        calib=calib_config,
    )

    # Create pipeline config
    pipeline_config = SAM2PipelineConfig(
        name=args.model,
        checkpoint=args.checkpoint,
        dtype=dtype,
        device=args.device,
        sam2_repo_path=args.sam2_repo_path,
    )

    # Create output config
    output_config = OutputConfig(
        root=args.output_dir,
        dirname=f"svdquant_w{args.weight_bits}a{args.activation_bits}_r{args.rank}",
    )

    # Create cache config
    cache_config = SAM2PtqCacheConfig(root=args.output_dir)

    # Create run config
    run_config = SAM2PtqRunConfig(
        cache=cache_config,
        output=output_config,
        pipeline=pipeline_config,
        quant=quant_config,
        seed=args.seed,
        save_model=args.save_model,
    )

    # Print configuration
    print("=" * 60)
    print("SAM2 SVDQuant Quantization")
    print("=" * 60)
    print(f"Model:            {args.model}")
    print(f"Weight bits:      {args.weight_bits}")
    print(f"Activation bits:  {args.activation_bits}")
    print(f"SVDQuant rank:    {args.rank}")
    print(f"Group size:       {args.group_size}")
    print(f"Num samples:      {args.num_samples}")
    print(f"Num iterations:   {args.num_iters}")
    print(f"Calib path:       {args.calib_path}")
    print(f"Output dir:       {args.output_dir}")
    print(f"Device:           {args.device}")
    print(f"Dtype:            {args.dtype}")
    if args.sam2_repo_path:
        print(f"SAM2 repo:        {args.sam2_repo_path}")
    print("=" * 60)

    # Check calibration path
    if not os.path.exists(args.calib_path):
        print(f"\nWarning: Calibration path does not exist: {args.calib_path}")
        print("Please provide a valid path to COCO val2017 images.")

    # Create PTQ instance and run
    print("\nStarting quantization...")
    sam2_ptq = SAM2Ptq.from_config(run_config)
    result = sam2_ptq.run()

    print("\n" + "=" * 60)
    print("Quantization complete!")
    print(f"Output saved to: {args.output_dir}")
    if args.save_model:
        print(f"Model saved to: {args.save_model}")
    print("=" * 60)


if __name__ == "__main__":
    main()
