# -*- coding: utf-8 -*-
"""SAM2 weight quantization."""

import gc
import typing as tp

import torch
import torch.nn as nn

from deepcompressor.quantizer.processor import Quantizer
from deepcompressor.utils import tools

from ..nn.struct import Sam2HieraBlockStruct, Sam2ModelStruct
from .config import Sam2QuantConfig

__all__ = ["quantize_sam2_weights", "load_sam2_weights_state_dict"]


def quantize_sam2_weights(
    model: Sam2ModelStruct,
    config: Sam2QuantConfig,
    quantizer_state_dict: dict | None = None,
    branch_state_dict: dict | None = None,
    return_with_scale_state_dict: bool = False,
) -> tuple[dict, dict, dict | None]:
    """Quantize SAM2 model weights.

    Args:
        model: SAM2 model structure.
        config: Quantization configuration.
        quantizer_state_dict: Pre-computed quantizer state dict.
        branch_state_dict: Pre-computed branch state dict for SVDQuant.
        return_with_scale_state_dict: Whether to return scale state dict.

    Returns:
        Tuple of (quantizer_state_dict, branch_state_dict, scale_state_dict).
    """
    logger = tools.logging.getLogger(__name__)

    if quantizer_state_dict is None:
        quantizer_state_dict = {}
    if branch_state_dict is None:
        branch_state_dict = {}

    scale_state_dict = {} if return_with_scale_state_dict else None

    wgts_config = config.wgts
    develop_dtype = config.develop_dtype

    for block_idx, block in enumerate(model.block_structs):
        logger.debug(f"- Quantizing block {block_idx}")

        # Quantize attention modules
        if block.attn_structs:
            for attn_struct in block.attn_structs:
                _quantize_attention_weights(
                    attn_struct,
                    wgts_config=wgts_config,
                    develop_dtype=develop_dtype,
                    quantizer_state_dict=quantizer_state_dict,
                    branch_state_dict=branch_state_dict,
                    scale_state_dict=scale_state_dict,
                )

        # Quantize FFN modules
        if block.ffn_struct is not None:
            _quantize_ffn_weights(
                block.ffn_struct,
                wgts_config=wgts_config,
                develop_dtype=develop_dtype,
                quantizer_state_dict=quantizer_state_dict,
                branch_state_dict=branch_state_dict,
                scale_state_dict=scale_state_dict,
            )

        gc.collect()
        torch.cuda.empty_cache()

    return quantizer_state_dict, branch_state_dict, scale_state_dict


def _quantize_attention_weights(
    attn_struct,
    *,
    wgts_config,
    develop_dtype: torch.dtype,
    quantizer_state_dict: dict,
    branch_state_dict: dict,
    scale_state_dict: dict | None,
) -> None:
    """Quantize attention module weights."""
    # Quantize QKV projection
    if attn_struct.q_proj is not None:
        _quantize_linear_weight(
            attn_struct.q_proj,
            name=attn_struct.q_proj_name,
            key=attn_struct.qkv_proj_key,
            config=wgts_config,
            develop_dtype=develop_dtype,
            quantizer_state_dict=quantizer_state_dict,
            branch_state_dict=branch_state_dict,
            scale_state_dict=scale_state_dict,
        )

    if attn_struct.k_proj is not None:
        _quantize_linear_weight(
            attn_struct.k_proj,
            name=attn_struct.k_proj_name,
            key=attn_struct.qkv_proj_key,
            config=wgts_config,
            develop_dtype=develop_dtype,
            quantizer_state_dict=quantizer_state_dict,
            branch_state_dict=branch_state_dict,
            scale_state_dict=scale_state_dict,
        )

    if attn_struct.v_proj is not None:
        _quantize_linear_weight(
            attn_struct.v_proj,
            name=attn_struct.v_proj_name,
            key=attn_struct.qkv_proj_key,
            config=wgts_config,
            develop_dtype=develop_dtype,
            quantizer_state_dict=quantizer_state_dict,
            branch_state_dict=branch_state_dict,
            scale_state_dict=scale_state_dict,
        )

    # Quantize output projection
    if attn_struct.o_proj is not None:
        _quantize_linear_weight(
            attn_struct.o_proj,
            name=attn_struct.o_proj_name,
            key=attn_struct.out_proj_key,
            config=wgts_config,
            develop_dtype=develop_dtype,
            quantizer_state_dict=quantizer_state_dict,
            branch_state_dict=branch_state_dict,
            scale_state_dict=scale_state_dict,
        )


def _quantize_ffn_weights(
    ffn_struct,
    *,
    wgts_config,
    develop_dtype: torch.dtype,
    quantizer_state_dict: dict,
    branch_state_dict: dict,
    scale_state_dict: dict | None,
) -> None:
    """Quantize FFN module weights."""
    # Quantize up projections
    for up_proj, up_name in zip(ffn_struct.up_projs, ffn_struct.up_proj_names):
        _quantize_linear_weight(
            up_proj,
            name=up_name,
            key=ffn_struct.up_proj_key,
            config=wgts_config,
            develop_dtype=develop_dtype,
            quantizer_state_dict=quantizer_state_dict,
            branch_state_dict=branch_state_dict,
            scale_state_dict=scale_state_dict,
        )

    # Quantize down projections
    for down_proj, down_name in zip(ffn_struct.down_projs, ffn_struct.down_proj_names):
        _quantize_linear_weight(
            down_proj,
            name=down_name,
            key=ffn_struct.down_proj_key,
            config=wgts_config,
            develop_dtype=develop_dtype,
            quantizer_state_dict=quantizer_state_dict,
            branch_state_dict=branch_state_dict,
            scale_state_dict=scale_state_dict,
        )


def _quantize_linear_weight(
    module: nn.Linear,
    *,
    name: str,
    key: str,
    config,
    develop_dtype: torch.dtype,
    quantizer_state_dict: dict,
    branch_state_dict: dict,
    scale_state_dict: dict | None,
) -> None:
    """Quantize a single linear layer weight."""
    if key in config.skips:
        return

    weight = module.weight.data
    weight_name = f"{name}.weight"

    # Check if we have cached state
    if weight_name in quantizer_state_dict:
        # Load from cache
        state = quantizer_state_dict[weight_name]
        if "scale" in state:
            if not hasattr(module, "weight_scale"):
                module.register_buffer("weight_scale", state["scale"])
            else:
                module.weight_scale.copy_(state["scale"])
        return

    # Compute quantization parameters
    weight_f32 = weight.to(develop_dtype)

    # Simple per-channel quantization
    if config.is_enabled():
        group_size = config.largest_group_shape[-1] if config.largest_group_shape else -1
        if group_size == -1:
            group_size = weight.shape[-1]

        # Reshape for group quantization
        orig_shape = weight_f32.shape
        if weight_f32.shape[-1] % group_size == 0:
            weight_grouped = weight_f32.reshape(-1, weight_f32.shape[-1] // group_size, group_size)
            scale = weight_grouped.abs().max(dim=-1, keepdim=True)[0] / (2 ** (config.quant_dtype.bits - 1) - 1)
            scale = scale.reshape(orig_shape[0], -1)
        else:
            scale = weight_f32.abs().max(dim=-1, keepdim=True)[0] / (2 ** (config.quant_dtype.bits - 1) - 1)

        # Store scale
        if not hasattr(module, "weight_scale"):
            module.register_buffer("weight_scale", scale.to(weight.dtype))
        else:
            module.weight_scale.copy_(scale.to(weight.dtype))

        # Store in state dict
        quantizer_state_dict[weight_name] = {"scale": scale.to(weight.dtype)}

        if scale_state_dict is not None:
            scale_state_dict[weight_name] = scale.to(weight.dtype)


def load_sam2_weights_state_dict(
    model: Sam2ModelStruct,
    config: Sam2QuantConfig,
    state_dict: dict,
    branch_state_dict: dict | None = None,
) -> None:
    """Load quantized weights from state dict.

    Args:
        model: SAM2 model structure.
        config: Quantization configuration.
        state_dict: Model state dict.
        branch_state_dict: Branch state dict for SVDQuant.
    """
    logger = tools.logging.getLogger(__name__)
    logger.info("Loading SAM2 weights from state dict")

    # Load state dict into model
    missing, unexpected = model.module.load_state_dict(state_dict, strict=False)

    if missing:
        logger.warning(f"Missing keys: {len(missing)}")
    if unexpected:
        logger.warning(f"Unexpected keys: {len(unexpected)}")
