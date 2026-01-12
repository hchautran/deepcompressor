# -*- coding: utf-8 -*-
"""SAM2 rotation quantization."""

import gc
import typing as tp

import torch
import torch.nn as nn

from deepcompressor.utils import tools

from ..nn.struct import Sam2HieraBlockStruct, Sam2ModelStruct
from .config import Sam2QuantConfig

__all__ = ["rotate_sam2"]


def rotate_sam2(
    model: Sam2ModelStruct,
    config: Sam2QuantConfig,
) -> None:
    """Apply rotation transforms to SAM2 model for better quantization.

    Rotation transforms (e.g., Hadamard) help distribute weight values
    more uniformly, improving quantization accuracy.

    Args:
        model: SAM2 model structure.
        config: Quantization configuration.
    """
    logger = tools.logging.getLogger(__name__)

    if not config.enabled_rotation:
        return

    rotation_config = config.rotation
    transforms = rotation_config.transforms

    logger.info(f"Applying rotation transforms: {transforms}")

    for block_idx, block in enumerate(model.block_structs):
        logger.debug(f"- Rotating block {block_idx}")

        # Apply rotations based on configuration
        if "hadamard" in transforms:
            _apply_hadamard_rotation(block)

        gc.collect()
        torch.cuda.empty_cache()


def _apply_hadamard_rotation(block: Sam2HieraBlockStruct) -> None:
    """Apply Hadamard rotation to block.

    The Hadamard transform is applied to:
    - QKV projection inputs
    - FFN up projection inputs

    Args:
        block: SAM2 Hiera block structure.
    """
    # Rotate attention QKV projections
    if block.attn_structs:
        for attn_struct in block.attn_structs:
            if attn_struct.q_proj is not None:
                _hadamard_in_channels(attn_struct.q_proj)
            if attn_struct.k_proj is not None:
                _hadamard_in_channels(attn_struct.k_proj)
            if attn_struct.v_proj is not None:
                _hadamard_in_channels(attn_struct.v_proj)

    # Rotate FFN up projections
    if block.ffn_struct is not None:
        for up_proj in block.ffn_struct.up_projs:
            _hadamard_in_channels(up_proj)


def _hadamard_in_channels(module: nn.Linear) -> None:
    """Apply Hadamard transform to input channels of a linear layer.

    Args:
        module: Linear layer to transform.
    """
    weight = module.weight.data
    in_features = weight.shape[1]

    # Generate Hadamard matrix
    H = _generate_hadamard_matrix(in_features, device=weight.device, dtype=weight.dtype)

    # Apply rotation: W' = W @ H
    with torch.no_grad():
        module.weight.data = weight @ H


def _generate_hadamard_matrix(
    size: int,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate a normalized Hadamard matrix.

    Args:
        size: Matrix size (must be power of 2 or will be padded).
        device: Device for the matrix.
        dtype: Data type for the matrix.

    Returns:
        Hadamard matrix of shape (size, size).
    """
    # Find next power of 2
    n = 1
    while n < size:
        n *= 2

    # Build Hadamard matrix recursively
    H = torch.tensor([[1.0]], device=device, dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat(
            [
                torch.cat([H, H], dim=1),
                torch.cat([H, -H], dim=1),
            ],
            dim=0,
        )

    # Normalize
    H = H / (n ** 0.5)

    # Truncate to actual size if needed
    if n > size:
        H = H[:size, :size]
        # Re-normalize after truncation (approximately)
        H = H * ((n / size) ** 0.5)

    return H


def _generate_random_rotation(
    size: int,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
) -> torch.Tensor:
    """Generate a random orthogonal rotation matrix.

    Args:
        size: Matrix size.
        device: Device for the matrix.
        dtype: Data type for the matrix.
        seed: Random seed.

    Returns:
        Random orthogonal matrix of shape (size, size).
    """
    torch.manual_seed(seed)

    # Generate random matrix
    A = torch.randn(size, size, device=device, dtype=dtype)

    # QR decomposition to get orthogonal matrix
    Q, R = torch.linalg.qr(A)

    # Make determinant positive
    d = torch.diagonal(R)
    ph = d.sign()
    Q = Q * ph.unsqueeze(0)

    return Q
