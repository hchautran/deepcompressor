# -*- coding: utf-8 -*-
"""SAM2 model patches for quantization."""

import torch.nn as nn

__all__ = []


def patch_sam2_attention(model: nn.Module) -> None:
    """Patch SAM2 attention modules for quantization compatibility.

    Args:
        model (`nn.Module`):
            The SAM2 model to patch.
    """
    # Add any necessary patches for SAM2/Hiera attention mechanisms
    pass


def patch_sam2_normalization(model: nn.Module) -> None:
    """Patch SAM2 normalization layers for quantization.

    Args:
        model (`nn.Module`):
            The SAM2 model to patch.
    """
    # Add patches for LayerNorm or other normalization layers if needed
    pass
