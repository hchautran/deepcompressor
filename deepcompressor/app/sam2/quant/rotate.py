# -*- coding: utf-8 -*-
"""SAM2 rotation module."""

import gc

import torch
import torch.nn as nn

from deepcompressor.calib.rotate import hadamard_in_channels
from deepcompressor.utils import tools

from ..nn.struct import Sam2ModelStruct
from .config import Sam2QuantConfig

__all__ = ["rotate_sam2"]


@torch.inference_mode()
def rotate_sam2(model: Sam2ModelStruct, /, config: Sam2QuantConfig) -> None:
    """Apply rotation transforms to SAM2 linear modules."""
    if not isinstance(model, Sam2ModelStruct):
        model = Sam2ModelStruct.construct(model)
    assert isinstance(model, Sam2ModelStruct)

    if not config.rotation or not config.rotation.transforms:
        return

    logger = tools.logging.getLogger(f"{__name__}.Rotate")
    transforms = set(config.rotation.transforms)
    if "hadamard" not in transforms:
        logger.warning("Unsupported rotation transforms: %s", ", ".join(sorted(transforms)))
        return

    for module_key, module_name, module, _, _ in model.named_key_modules():
        if not isinstance(module, nn.Linear):
            continue
        if module_key in config.wgts.skips and module_key in config.ipts.skips:
            continue
        logger.debug("- Hadamard transform on %s", module_name)
        hadamard_in_channels([module])
        gc.collect()
        torch.cuda.empty_cache()
