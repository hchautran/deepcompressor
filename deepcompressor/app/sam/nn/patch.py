# -*- coding: utf-8 -*-
"""SAM model patches for quantization."""

import torch.nn as nn

__all__ = []


def patch_sam_attention(model: nn.Module) -> None:
    """Patch SAM attention modules for quantization compatibility.

    Args:
        model (`nn.Module`):
            The SAM model to patch.
    """
    # Add any necessary patches for SAM attention mechanisms
    # This can include replacing attention implementations or modifying forward passes
    pass


def patch_sam_normalization(model: nn.Module) -> None:
    """Patch SAM normalization layers for quantization.

    Args:
        model (`nn.Module`):
            The SAM model to patch.
    """
    # Add patches for LayerNorm or other normalization layers if needed
    pass
