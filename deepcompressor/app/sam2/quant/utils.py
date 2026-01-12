# -*- coding: utf-8 -*-
"""SAM2 quantization utilities."""

import typing as tp

import torch
import torch.nn as nn

from deepcompressor.utils import tools

from ..nn.struct import Sam2HieraBlockStruct, Sam2ModelStruct

__all__ = [
    "get_sam2_block_quantizable_modules",
    "filter_sam2_block_modules_by_key",
]


def get_sam2_block_quantizable_modules(
    block: Sam2HieraBlockStruct,
    *,
    include_attn: bool = True,
    include_ffn: bool = True,
) -> dict[str, tuple[str, nn.Module]]:
    """Get quantizable modules from a SAM2 Hiera block.

    Args:
        block: SAM2 Hiera block structure.
        include_attn: Whether to include attention modules.
        include_ffn: Whether to include FFN modules.

    Returns:
        Dictionary mapping module keys to (name, module) tuples.
    """
    modules = {}

    if include_attn:
        for attn_struct in block.attn_structs:
            for key, name, module, parent, fname in attn_struct.named_key_modules():
                modules[name] = (key, module)

    if include_ffn and block.ffn_struct is not None:
        for key, name, module, parent, fname in block.ffn_struct.named_key_modules():
            modules[name] = (key, module)

    return modules


def filter_sam2_block_modules_by_key(
    block: Sam2HieraBlockStruct,
    *,
    skips: tp.Sequence[str] = (),
    includes: tp.Sequence[str] = (),
) -> dict[str, tuple[str, nn.Module]]:
    """Filter block modules by key.

    Args:
        block: SAM2 Hiera block structure.
        skips: Keys to skip.
        includes: Keys to include (if empty, include all).

    Returns:
        Filtered dictionary of modules.
    """
    all_modules = get_sam2_block_quantizable_modules(block)
    if not skips and not includes:
        return all_modules

    filtered = {}
    for name, (key, module) in all_modules.items():
        if includes and key not in includes:
            continue
        if key in skips:
            continue
        filtered[name] = (key, module)

    return filtered


def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    """Get a module by its name path."""
    parts = name.split(".")
    current = model
    for part in parts:
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def set_module_by_name(model: nn.Module, name: str, new_module: nn.Module) -> None:
    """Set a module by its name path."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)

    last_part = parts[-1]
    if last_part.isdigit():
        parent[int(last_part)] = new_module
    else:
        setattr(parent, last_part, new_module)
