# -*- coding: utf-8 -*-
"""SAM weight quantization utilities."""

import typing as tp

import torch
from tqdm import tqdm

from deepcompressor.utils import tools

from ..nn.struct import SamModelStruct
from .quantizer import SamWeightQuantizer

__all__ = ["quantize_sam_weights"]


def quantize_sam_weights(
    model: SamModelStruct,
    quantizers: dict[str, SamWeightQuantizer],
    develop_dtype: torch.dtype = torch.float32,
) -> dict[str, tp.Any]:
    """Quantize SAM model weights.

    Args:
        model (`SamModelStruct`):
            The SAM model structure.
        quantizers (`dict[str, SamWeightQuantizer]`):
            Dictionary of weight quantizers for each module.
        develop_dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            Development dtype for quantization.

    Returns:
        `dict[str, Any]`: State dictionary of quantized weights.
    """
    logger = tools.logging.getLogger(__name__)
    logger.info("Quantizing SAM weights...")

    state_dict = {}

    # Iterate over all quantizable modules in the model
    for key, module_name, module, parent_struct, field_name in tqdm(
        list(model.named_key_modules()),
        desc="Quantizing weights",
        leave=False,
    ):
        if key not in quantizers:
            continue

        quantizer = quantizers[key]
        if not quantizer.is_enabled() or quantizer.should_skip(module_name):
            continue

        # Get weight parameter
        if isinstance(module, torch.nn.Linear):
            weight = module.weight.data
        elif isinstance(module, torch.nn.Conv2d):
            weight = module.weight.data
        else:
            continue

        # Quantize weight
        try:
            qweight = quantizer.quantize(
                weight,
                develop_dtype=develop_dtype,
                return_with_dequant=True,
                return_with_quant=False,
            )

            # Update module weight with dequantized version
            module.weight.data = qweight.data

            # Store quantization parameters in state dict
            state_dict[f"{module_name}.weight"] = {
                "scale": qweight.scale,
                "zero": qweight.zero if qweight.zero is not None else None,
            }

        except Exception as e:
            logger.error(f"Error quantizing {module_name}: {e}")
            raise

    logger.info(f"Quantized {len(state_dict)} weight tensors")
    return state_dict
