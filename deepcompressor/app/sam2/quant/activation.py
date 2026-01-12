# -*- coding: utf-8 -*-
"""SAM2 activation quantization utilities."""

import typing as tp

import torch
from tqdm import tqdm

from deepcompressor.calib import DynamicRangeCalibrator
from deepcompressor.utils import tools
from deepcompressor.utils.hooks import TensorsCache

from ..nn.struct import Sam2ModelStruct
from .quantizer import Sam2ActivationQuantizer

__all__ = ["calibrate_sam_activations", "quantize_sam_activations"]


def calibrate_sam_activations(
    model: Sam2ModelStruct,
    quantizers: dict[str, Sam2ActivationQuantizer],
    calib_loader: tp.Iterable,
    num_samples: int = 128,
) -> dict[str, tp.Any]:
    """Calibrate SAM2 activation quantizers.

    Args:
        model (`Sam2ModelStruct`):
            The SAM2 model structure.
        quantizers (`dict[str, Sam2ActivationQuantizer]`):
            Dictionary of activation quantizers for each module.
        calib_loader (`Iterable`):
            Calibration data loader.
        num_samples (`int`, *optional*, defaults to 128):
            Number of calibration samples to use.

    Returns:
        `dict[str, Any]`: Calibration state dictionary.
    """
    logger = tools.logging.getLogger(__name__)
    logger.info("Calibrating SAM2 activations...")

    # Create tensor cache for collecting activations
    cache = TensorsCache()
    hooks = []

    # Register forward hooks to collect activations
    for key, module_name, module, parent_struct, field_name in model.named_key_modules():
        if key not in quantizers:
            continue

        quantizer = quantizers[key]
        if not quantizer.is_enabled():
            continue

        # Register hook to capture inputs/outputs
        def make_hook(quant_key):
            def hook(mod, inp, out):
                if isinstance(inp, tuple):
                    inp = inp[0]
                cache.append(quant_key, inp.detach())
                if out is not None:
                    cache.append(f"{quant_key}.output", out.detach())

            return hook

        handle = module.register_forward_hook(make_hook(key))
        hooks.append(handle)

    # Run calibration samples through the model
    model.module.eval()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(calib_loader, desc="Calibrating", total=num_samples, leave=False)):
            if idx >= num_samples:
                break

            # Forward pass
            if isinstance(batch, dict):
                _ = model.module(**batch)
            elif isinstance(batch, (list, tuple)):
                _ = model.module(*batch)
            else:
                _ = model.module(batch)

    # Remove hooks
    for handle in hooks:
        handle.remove()

    logger.info("Computing optimal dynamic ranges...")

    # Calibrate dynamic ranges for each quantizer
    calib_dict = {}
    for key, quantizer in tqdm(quantizers.items(), desc="Computing ranges", leave=False):
        if not quantizer.is_enabled():
            continue

        if key not in cache:
            logger.warning(f"No cached activations for {key}")
            continue

        # Get cached activations
        activations = cache.get(key)

        if quantizer.config.enabled_calib_range:
            # Use dynamic range calibrator
            calibrator = DynamicRangeCalibrator(config=quantizer.config.calib_range)
            dynamic_range = calibrator.calibrate(activations, quantizer.config)
            quantizer.dynamic_range = dynamic_range

            calib_dict[key] = {
                "dynamic_range": dynamic_range,
            }

    logger.info(f"Calibrated {len(calib_dict)} activation quantizers")
    return calib_dict


def quantize_sam_activations(
    model: Sam2ModelStruct,
    quantizers: dict[str, Sam2ActivationQuantizer],
) -> None:
    """Apply activation quantizers to SAM2 model.

    Args:
        model (`Sam2ModelStruct`):
            The SAM2 model structure.
        quantizers (`dict[str, Sam2ActivationQuantizer]`):
            Dictionary of activation quantizers for each module.
    """
    logger = tools.logging.getLogger(__name__)
    logger.info("Applying activation quantizers...")

    # Register quantizers as hooks on modules
    for key, module_name, module, parent_struct, field_name in model.named_key_modules():
        if key not in quantizers:
            continue

        quantizer = quantizers[key]
        if not quantizer.is_enabled():
            continue

        # Register quantizer as forward hook
        hook = quantizer.as_hook()
        module.register_forward_hook(hook)

    logger.info("Activation quantizers applied")
