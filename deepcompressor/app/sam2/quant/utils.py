import typing as tp

import torch.nn as nn

from ..nn.struct import Sam2ModelStruct
from .config import Sam2QuantConfig

__all__ = ["get_sam2_needs_inputs_fn", "get_sam2_needs_outputs_fn"]


def get_sam2_needs_inputs_fn(
    model: Sam2ModelStruct, config: Sam2QuantConfig
) -> tp.Callable[[str, nn.Module], bool]:
    """Return predicate for whether a module needs input caching."""

    needs_inputs_names = set()
    for module_key, module_name, _, _, _ in model.named_key_modules():
        if (config.enabled_wgts and config.wgts.is_enabled_for(module_key)) or (
            config.enabled_ipts and config.ipts.is_enabled_for(module_key)
        ):
            needs_inputs_names.add(module_name)
        if config.enabled_opts and config.opts.is_enabled_for(module_key):
            needs_inputs_names.add(module_name)

    def needs_inputs(name: str, module: nn.Module) -> bool:
        return name in needs_inputs_names

    return needs_inputs


def get_sam2_needs_outputs_fn(
    model: Sam2ModelStruct, config: Sam2QuantConfig
) -> tp.Callable[[str, nn.Module], bool]:
    """Return predicate for whether a module needs output caching."""

    if not config.enabled_opts:
        return lambda name, module: False
    needs_outputs_names = set()
    for module_key, module_name, _, _, _ in model.named_key_modules():
        if config.opts.is_enabled_for(module_key):
            needs_outputs_names.add(module_name)

    def needs_outputs(name: str, module: nn.Module) -> bool:
        return name in needs_outputs_names

    return needs_outputs
