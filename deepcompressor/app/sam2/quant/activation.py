# -*- coding: utf-8 -*-
"""SAM2 activation quantization."""

import gc
import typing as tp

import torch
import torch.nn as nn

from deepcompressor.utils import tools

from ..nn.struct import Sam2HieraBlockStruct, Sam2ModelStruct
from .config import Sam2QuantConfig

__all__ = ["quantize_sam2_activations"]


def quantize_sam2_activations(
    model: Sam2ModelStruct,
    config: Sam2QuantConfig,
    quantizer_state_dict: dict | None = None,
    orig_state_dict: dict | None = None,
) -> dict | None:
    """Quantize SAM2 model activations.

    Args:
        model: SAM2 model structure.
        config: Quantization configuration.
        quantizer_state_dict: Pre-computed quantizer state dict.
        orig_state_dict: Original model state dict.

    Returns:
        Activation quantizer state dict or None.
    """
    logger = tools.logging.getLogger(__name__)

    if not config.enabled_ipts and not config.enabled_opts:
        return None

    if quantizer_state_dict is None:
        quantizer_state_dict = {}

    ipts_config = config.ipts
    opts_config = config.opts
    develop_dtype = config.develop_dtype

    for block_idx, block in enumerate(model.block_structs):
        logger.debug(f"- Quantizing activations for block {block_idx}")

        # Quantize attention activations
        if block.attn_structs:
            for attn_struct in block.attn_structs:
                _setup_activation_quantization(
                    attn_struct,
                    ipts_config=ipts_config,
                    opts_config=opts_config,
                    develop_dtype=develop_dtype,
                    quantizer_state_dict=quantizer_state_dict,
                )

        # Quantize FFN activations
        if block.ffn_struct is not None:
            _setup_ffn_activation_quantization(
                block.ffn_struct,
                ipts_config=ipts_config,
                opts_config=opts_config,
                develop_dtype=develop_dtype,
                quantizer_state_dict=quantizer_state_dict,
            )

        gc.collect()
        torch.cuda.empty_cache()

    return quantizer_state_dict if quantizer_state_dict else None


def _setup_activation_quantization(
    attn_struct,
    *,
    ipts_config,
    opts_config,
    develop_dtype: torch.dtype,
    quantizer_state_dict: dict,
) -> None:
    """Setup activation quantization for attention modules."""
    # QKV projections
    if attn_struct.q_proj is not None:
        _setup_linear_activation_quantization(
            attn_struct.q_proj,
            name=attn_struct.q_proj_name,
            key=attn_struct.qkv_proj_key,
            ipts_config=ipts_config,
            opts_config=opts_config,
            quantizer_state_dict=quantizer_state_dict,
        )

    if attn_struct.k_proj is not None:
        _setup_linear_activation_quantization(
            attn_struct.k_proj,
            name=attn_struct.k_proj_name,
            key=attn_struct.qkv_proj_key,
            ipts_config=ipts_config,
            opts_config=opts_config,
            quantizer_state_dict=quantizer_state_dict,
        )

    if attn_struct.v_proj is not None:
        _setup_linear_activation_quantization(
            attn_struct.v_proj,
            name=attn_struct.v_proj_name,
            key=attn_struct.qkv_proj_key,
            ipts_config=ipts_config,
            opts_config=opts_config,
            quantizer_state_dict=quantizer_state_dict,
        )

    # Output projection
    if attn_struct.o_proj is not None:
        _setup_linear_activation_quantization(
            attn_struct.o_proj,
            name=attn_struct.o_proj_name,
            key=attn_struct.out_proj_key,
            ipts_config=ipts_config,
            opts_config=opts_config,
            quantizer_state_dict=quantizer_state_dict,
        )


def _setup_ffn_activation_quantization(
    ffn_struct,
    *,
    ipts_config,
    opts_config,
    develop_dtype: torch.dtype,
    quantizer_state_dict: dict,
) -> None:
    """Setup activation quantization for FFN modules."""
    # Up projections
    for up_proj, up_name in zip(ffn_struct.up_projs, ffn_struct.up_proj_names):
        _setup_linear_activation_quantization(
            up_proj,
            name=up_name,
            key=ffn_struct.up_proj_key,
            ipts_config=ipts_config,
            opts_config=opts_config,
            quantizer_state_dict=quantizer_state_dict,
        )

    # Down projections
    for down_proj, down_name in zip(ffn_struct.down_projs, ffn_struct.down_proj_names):
        _setup_linear_activation_quantization(
            down_proj,
            name=down_name,
            key=ffn_struct.down_proj_key,
            ipts_config=ipts_config,
            opts_config=opts_config,
            quantizer_state_dict=quantizer_state_dict,
        )


def _setup_linear_activation_quantization(
    module: nn.Linear,
    *,
    name: str,
    key: str,
    ipts_config,
    opts_config,
    quantizer_state_dict: dict,
) -> None:
    """Setup activation quantization for a linear layer."""
    # Check skips
    if ipts_config is not None and key in ipts_config.skips:
        return
    if opts_config is not None and key in opts_config.skips:
        return

    # Register activation quantization hooks
    if ipts_config is not None and ipts_config.is_enabled():
        if not hasattr(module, "_input_quantizer_hook"):
            hook = module.register_forward_pre_hook(_input_quantize_hook)
            module._input_quantizer_hook = hook
            module._input_quant_config = ipts_config

    if opts_config is not None and opts_config.is_enabled():
        if not hasattr(module, "_output_quantizer_hook"):
            hook = module.register_forward_hook(_output_quantize_hook)
            module._output_quantizer_hook = hook
            module._output_quant_config = opts_config


def _input_quantize_hook(module: nn.Module, inputs: tuple) -> tuple:
    """Forward pre-hook for input quantization."""
    if not hasattr(module, "_input_quant_config"):
        return inputs

    config = module._input_quant_config
    if not config.is_enabled():
        return inputs

    # Simple per-token quantization
    x = inputs[0]
    if config.quant_dtype is not None:
        bits = config.quant_dtype.bits
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1

        # Per-token scale
        scale = x.abs().max(dim=-1, keepdim=True)[0] / qmax
        scale = scale.clamp(min=1e-8)

        # Quantize and dequantize
        x_q = (x / scale).round().clamp(qmin, qmax)
        x_dq = x_q * scale

        return (x_dq,) + inputs[1:]

    return inputs


def _output_quantize_hook(module: nn.Module, inputs: tuple, output: torch.Tensor) -> torch.Tensor:
    """Forward hook for output quantization."""
    if not hasattr(module, "_output_quant_config"):
        return output

    config = module._output_quant_config
    if not config.is_enabled():
        return output

    # Simple per-token quantization
    if config.quant_dtype is not None:
        bits = config.quant_dtype.bits
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1

        # Per-token scale
        scale = output.abs().max(dim=-1, keepdim=True)[0] / qmax
        scale = scale.clamp(min=1e-8)

        # Quantize and dequantize
        y_q = (output / scale).round().clamp(qmin, qmax)
        y_dq = y_q * scale

        return y_dq

    return output
