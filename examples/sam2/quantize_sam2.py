#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SAM2 post-training quantization script.

This script performs post-training quantization on SAM2 (Segment Anything Model 2)
using the DeepCompressor framework with nunchaku backend support.

Usage:
    python quantize_sam2.py --config configs/hiera_tiny_w4a8.yaml

    # Or with command-line overrides:
    python quantize_sam2.py \
        --model.name tiny \
        --quant.wgts.dtype uint4 \
        --quant.ipts.dtype uint8 \
        --quant.calib.path ../../coco/val2017 \
        --output.root ./outputs/sam2

Supported model variants:
    - tiny: facebook/sam2.1-hiera-tiny (38M params)
    - small: facebook/sam2.1-hiera-small (46M params)
    - base-plus: facebook/sam2.1-hiera-base-plus (80M params)
    - large: facebook/sam2.1-hiera-large (224M params)

Quantization configurations:
    - W4A4: 4-bit weights, 4-bit activations
    - W4A8: 4-bit weights, 8-bit activations
    - W8A8: 8-bit weights, 8-bit activations

For best accuracy, enable:
    - smooth quantization (reduces activation outliers)
    - rotation (Hadamard transform for better weight distribution)
    - SVDQuant (low-rank compensation for outliers)
"""

import argparse
import sys
import traceback
import torch


def main():
    """Main entry point for SAM2 quantization."""
    # Import here to avoid slow imports on --help
    from deepcompressor.app.sam2 import Sam2PtqRunConfig
    from deepcompressor.utils import tools

    # Parse configuration
    parser = argparse.ArgumentParser(
        description="SAM2 Post-Training Quantization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--model.name",
        type=str,
        default="tiny",
        dest="model_name",
        help="Model variant: tiny, small, base-plus, large",
    )
    parser.add_argument(
        "--model.device",
        type=str,
        default="cuda",
        dest="model_device",
        help="Device to run on (cuda or cpu)",
    )
    parser.add_argument(
        "--quant.calib.path",
        type=str,
        default="",
        dest="calib_path",
        help="Path to calibration dataset (directory of images)",
    )
    parser.add_argument(
        "--quant.calib.num_samples",
        type=int,
        default=128,
        dest="calib_num_samples",
        help="Number of calibration samples",
    )
    parser.add_argument(
        "--output.root",
        type=str,
        default="./outputs/sam2",
        dest="output_root",
        help="Output root directory",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save the quantized model",
    )
    parser.add_argument(
        "--convert-nunchaku",
        action="store_true",
        help="Convert to nunchaku format after quantization",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args, unknown_args = parser.parse_known_args()

    # Setup logging
    log_level = tools.logging.DEBUG if args.verbose else tools.logging.INFO
    tools.logging.setup(level=log_level)
    logger = tools.logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("SAM2 Post-Training Quantization")
    logger.info("=" * 60)

    # Load configuration from YAML if provided
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        import yaml

        with open(args.config) as f:
            config_dict = yaml.safe_load(f)
    else:
        # Build configuration from command-line arguments
        config_dict = {
            "model": {
                "name": args.model_name,
                "device": args.model_device,
                "dtype": "float16",
            },
            "quant": {
                "calib": {
                    "path": args.calib_path,
                    "num_samples": args.calib_num_samples,
                },
                "wgts": {
                    "dtype": "uint4",
                    "group_shapes": [[1, 128]],
                },
                "ipts": {
                    "dtype": "uint8",
                    "group_shapes": [[1, -1]],
                },
                "opts": {
                    "dtype": None,
                },
            },
            "cache": {
                "root": "./cache/sam2",
            },
            "output": {
                "root": args.output_root,
            },
            "seed": args.seed,
            "save_model": "true" if args.save_model else "false",
        }

    # Create and validate configuration
    try:
        from deepcompressor.app.sam2.config import Sam2PtqRunConfig, Sam2ModelConfig
        from deepcompressor.app.sam2.cache.config import Sam2PtqCacheConfig
        from deepcompressor.app.sam2.quant.config import Sam2QuantConfig
        from deepcompressor.app.sam2.quant.quantizer.config import (
            Sam2WeightQuantizerConfig,
            Sam2ActivationQuantizerConfig,
        )
        from deepcompressor.app.sam2.dataset.calib import Sam2CalibConfig
        from deepcompressor.utils.config.output import OutputConfig

        # Build sub-configs
        model_config = Sam2ModelConfig(**config_dict.get("model", {}))

        calib_config = Sam2CalibConfig(**config_dict.get("quant", {}).get("calib", {}))

        wgts_dict = config_dict.get("quant", {}).get("wgts", {})
        ipts_dict = config_dict.get("quant", {}).get("ipts", {})
        opts_dict = config_dict.get("quant", {}).get("opts", {})

        # Handle dtype conversion
        from deepcompressor.data.dtype import QuantDataType

        if wgts_dict.get("dtype"):
            wgts_dict["dtype"] = QuantDataType.from_str(wgts_dict["dtype"])
        if ipts_dict.get("dtype"):
            ipts_dict["dtype"] = QuantDataType.from_str(ipts_dict["dtype"])
        if opts_dict.get("dtype"):
            opts_dict["dtype"] = QuantDataType.from_str(opts_dict["dtype"])

        wgts_config = Sam2WeightQuantizerConfig(**wgts_dict)
        ipts_config = Sam2ActivationQuantizerConfig(**ipts_dict)
        opts_config = Sam2ActivationQuantizerConfig(**opts_dict)

        quant_config = Sam2QuantConfig(
            wgts=wgts_config,
            ipts=ipts_config,
            opts=opts_config,
            calib=calib_config,
        )

        cache_config = Sam2PtqCacheConfig(**config_dict.get("cache", {}))
        output_config = OutputConfig(**config_dict.get("output", {}))

        config = Sam2PtqRunConfig(
            model=model_config,
            quant=quant_config,
            cache=cache_config,
            output=output_config,
            seed=config_dict.get("seed", 42),
            save_model=config_dict.get("save_model", ""),
        )

    except Exception as e:
        logger.error(f"Failed to create configuration: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Run quantization
    try:
        logger.info("Starting quantization pipeline...")
        config.main()
        logger.info("Quantization complete!")

        # Convert to nunchaku format if requested
        if args.convert_nunchaku:
            logger.info("Converting to nunchaku format...")
            from deepcompressor.backend.nunchaku.convert_sam2 import convert_to_nunchaku_sam2_state_dicts
            import os

            quant_path = os.path.join(config.output.running_job_dirpath, "cache")
            save_model = config.save_model
            if save_model:
                if isinstance(save_model, str):
                    save_model_value = save_model.lower()
                else:
                    save_model_value = save_model
                if save_model_value in ("false", "none", "null", "nil", False):
                    pass
                elif save_model_value in ("true", "default", True):
                    quant_path = os.path.join(config.output.running_job_dirpath, "model")
                else:
                    quant_path = save_model

            # Load state dicts
            state_dict = torch.load(os.path.join(quant_path, "model.pt"))
            scale_dict = torch.load(os.path.join(quant_path, "scale.pt"))
            smooth_dict = {}
            branch_dict = {}
            if os.path.exists(os.path.join(quant_path, "smooth.pt")):
                smooth_dict = torch.load(os.path.join(quant_path, "smooth.pt"))
            if os.path.exists(os.path.join(quant_path, "branch.pt")):
                branch_dict = torch.load(os.path.join(quant_path, "branch.pt"))

            converted, other = convert_to_nunchaku_sam2_state_dicts(
                state_dict=state_dict,
                scale_dict=scale_dict,
                smooth_dict=smooth_dict,
                branch_dict=branch_dict,
            )

            import safetensors.torch

            nunchaku_path = os.path.join(quant_path, "nunchaku")
            os.makedirs(nunchaku_path, exist_ok=True)
            safetensors.torch.save_file(converted, os.path.join(nunchaku_path, "hiera_blocks.safetensors"))
            safetensors.torch.save_file(other, os.path.join(nunchaku_path, "unquantized_layers.safetensors"))
            logger.info(f"Nunchaku format saved to {nunchaku_path}")

    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
