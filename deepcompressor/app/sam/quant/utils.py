# -*- coding: utf-8 -*-
"""SAM quantization utilities."""

import typing as tp

import torch

from ..nn.struct import SamModelStruct

__all__ = ["get_sam_modules_for_quantization", "load_sam_weights_state_dict"]


def get_sam_modules_for_quantization(
    model: SamModelStruct,
    skip_patterns: list[str] | None = None,
) -> dict[str, torch.nn.Module]:
    """Get SAM modules eligible for quantization.

    Args:
        model (`SamModelStruct`):
            The SAM model structure.
        skip_patterns (`list[str]` or `None`, *optional*, defaults to `None`):
            List of name patterns to skip.

    Returns:
        `dict[str, torch.nn.Module]`: Dictionary of modules to quantize.
    """
    if skip_patterns is None:
        skip_patterns = []

    modules_dict = {}

    for key, module_name, module, parent_struct, field_name in model.named_key_modules():
        # Check if module should be skipped
        should_skip = any(pattern in module_name for pattern in skip_patterns)
        if should_skip:
            continue

        modules_dict[key] = module

    return modules_dict


def load_sam_weights_state_dict(
    model: SamModelStruct,
    state_dict_path: str,
    device: torch.device | str = "cpu",
) -> None:
    """Load quantized SAM weights from state dict.

    Args:
        model (`SamModelStruct`):
            The SAM model structure.
        state_dict_path (`str`):
            Path to the state dict file.
        device (`torch.device` or `str`, *optional*, defaults to `"cpu"`):
            Device to load weights to.
    """
    state_dict = torch.load(state_dict_path, map_location=device)

    # Load quantization parameters
    for key, module_name, module, parent_struct, field_name in model.named_key_modules():
        weight_key = f"{module_name}.weight"
        if weight_key in state_dict:
            # Load scale and zero point if they exist
            param_dict = state_dict[weight_key]
            if "scale" in param_dict:
                # Store quantization parameters as buffer
                if not hasattr(module, "_quant_scale"):
                    module.register_buffer("_quant_scale", param_dict["scale"].to(device))
                else:
                    module._quant_scale = param_dict["scale"].to(device)

            if "zero" in param_dict and param_dict["zero"] is not None:
                if not hasattr(module, "_quant_zero"):
                    module.register_buffer("_quant_zero", param_dict["zero"].to(device))
                else:
                    module._quant_zero = param_dict["zero"].to(device)
