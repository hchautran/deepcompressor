# -*- coding: utf-8 -*-
"""SAM2 smooth quantization."""

import gc

import torch
import torch.nn as nn

from deepcompressor.utils import tools

from ..nn.struct import Sam2ModelStruct
from .config import Sam2QuantConfig

__all__ = ["smooth_sam2"]


def smooth_sam2(
    model: Sam2ModelStruct,
    config: Sam2QuantConfig,
    smooth_cache: dict | None = None,
) -> dict:
    """Apply smooth quantization to SAM2 model.

    Smooth quantization reduces activation distribution variance by
    migrating quantization difficulty from activations to weights.

    Args:
        model: SAM2 model structure.
        config: Quantization configuration.
        smooth_cache: Pre-computed smooth scales.

    Returns:
        Dictionary of smooth scales.
    """
    logger = tools.logging.getLogger(__name__)

    if smooth_cache is not None:
        # Load from cache
        logger.info("Loading smooth scales from cache")
        _apply_smooth_scales(model, smooth_cache)
        return smooth_cache

    if not config.enabled_smooth:
        return {}

    logger.info("Computing smooth scales")
    smooth_cache = {}

    smooth_config = config.smooth

    for block_idx, block in enumerate(model.block_structs):
        logger.debug(f"- Smoothing block {block_idx}")

        # Smooth attention modules
        if smooth_config.enabled_proj and block.attn_structs:
            for attn_struct in block.attn_structs:
                # Smooth QKV projections
                if block.pre_attn_norms and attn_struct.q_proj is not None:
                    norm = block.pre_attn_norms[0] if block.pre_attn_norms else None
                    if norm is not None:
                        scales = _compute_smooth_scales(
                            norm,
                            [attn_struct.q_proj],
                            config=smooth_config.proj,
                        )
                        key = f"{attn_struct.q_proj_name}.smooth"
                        smooth_cache[key] = scales
                        _apply_smooth_to_layers(norm, [attn_struct.q_proj], scales)

                        if attn_struct.k_proj is not None:
                            key = f"{attn_struct.k_proj_name}.smooth"
                            smooth_cache[key] = scales
                            _apply_smooth_to_layers(None, [attn_struct.k_proj], scales)

                        if attn_struct.v_proj is not None:
                            key = f"{attn_struct.v_proj_name}.smooth"
                            smooth_cache[key] = scales
                            _apply_smooth_to_layers(None, [attn_struct.v_proj], scales)

        # Smooth FFN modules
        if smooth_config.enabled_proj and block.ffn_struct is not None:
            if block.pre_ffn_norm is not None:
                for up_proj, up_name in zip(block.ffn_struct.up_projs, block.ffn_struct.up_proj_names):
                    scales = _compute_smooth_scales(
                        block.pre_ffn_norm,
                        [up_proj],
                        config=smooth_config.proj,
                    )
                    key = f"{up_name}.smooth"
                    smooth_cache[key] = scales
                    _apply_smooth_to_layers(block.pre_ffn_norm, [up_proj], scales)

        gc.collect()
        torch.cuda.empty_cache()

    return smooth_cache


def _compute_smooth_scales(
    norm: nn.Module,
    modules: list[nn.Linear],
    *,
    config,
) -> torch.Tensor:
    """Compute smooth scales for a normalization layer and its consumers.

    Args:
        norm: Normalization layer (LayerNorm).
        modules: List of linear modules consuming the norm output.
        config: Smooth configuration.

    Returns:
        Smooth scales tensor.
    """
    # Get weight scales from linear layers
    weight_scales = []
    for module in modules:
        if isinstance(module, nn.Linear):
            w = module.weight.data.abs()
            w_scale = w.max(dim=0)[0]  # Per-input-channel max
            weight_scales.append(w_scale)

    if not weight_scales:
        return torch.ones(1)

    # Combine weight scales
    weight_scale = torch.stack(weight_scales).max(dim=0)[0]

    # Get activation scale from norm (use norm weight as proxy)
    if hasattr(norm, "weight") and norm.weight is not None:
        act_scale = norm.weight.data.abs()
    else:
        act_scale = torch.ones_like(weight_scale)

    # Compute smooth scale: s = (act_scale / weight_scale)^alpha
    alpha = config.alpha if hasattr(config, "alpha") else 0.5
    eps = 1e-8

    smooth_scale = (act_scale / (weight_scale + eps)).pow(alpha)
    smooth_scale = smooth_scale.clamp(min=eps, max=1.0 / eps)

    return smooth_scale


def _apply_smooth_scales(model: Sam2ModelStruct, smooth_cache: dict) -> None:
    """Apply cached smooth scales to model."""
    for name, scales in smooth_cache.items():
        if not name.endswith(".smooth"):
            continue

        module_name = name[:-7]  # Remove ".smooth" suffix
        try:
            module = _get_module_by_name(model.module, module_name)
            if isinstance(module, nn.Linear):
                # Scale the weight
                module.weight.data.mul_(scales.unsqueeze(0).to(module.weight.device))
        except AttributeError:
            pass


def _apply_smooth_to_layers(
    norm: nn.Module | None,
    modules: list[nn.Linear],
    scales: torch.Tensor,
) -> None:
    """Apply smooth scales to normalization and linear layers.

    Args:
        norm: Normalization layer to scale (or None).
        modules: Linear modules to inverse-scale.
        scales: Smooth scales.
    """
    device = modules[0].weight.device if modules else "cpu"
    scales = scales.to(device)

    # Scale normalization layer output (if applicable)
    if norm is not None and hasattr(norm, "weight") and norm.weight is not None:
        norm.weight.data.div_(scales)
        if hasattr(norm, "bias") and norm.bias is not None:
            norm.bias.data.div_(scales)

    # Inverse-scale linear layer inputs
    for module in modules:
        if isinstance(module, nn.Linear):
            # Weight: out_features x in_features
            # Scale input dimension (dim=1)
            module.weight.data.mul_(scales.unsqueeze(0))


def _get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    """Get a module by its name path."""
    parts = name.split(".")
    current = model
    for part in parts:
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current
