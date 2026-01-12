# -*- coding: utf-8 -*-
"""SAM2 model loading utilities."""

import typing as tp

import torch
import torch.nn as nn

__all__ = ["load_sam2_from_huggingface", "Sam2Config"]


class Sam2Config:
    """SAM2 configuration holder."""

    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        image_size: int = 1024,
        patch_size: int = 16,
        num_channels: int = 3,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels


def load_sam2_from_huggingface(
    model_name: str = "facebook/sam2-hiera-tiny",
    device: str | torch.device = "cuda",
    torch_dtype: torch.dtype = torch.float16,
) -> tuple[nn.Module, Sam2Config]:
    """Load SAM2 model from HuggingFace.

    Args:
        model_name (`str`, *optional*, defaults to `"facebook/sam2-hiera-tiny"`):
            HuggingFace model name. Available models:
            - facebook/sam2-hiera-tiny
            - facebook/sam2-hiera-small
            - facebook/sam2-hiera-base-plus
            - facebook/sam2-hiera-large
        device (`str` or `torch.device`, *optional*, defaults to `"cuda"`):
            Device to load model on.
        torch_dtype (`torch.dtype`, *optional*, defaults to `torch.float16`):
            Data type for model weights.

    Returns:
        tuple[nn.Module, Sam2Config]: Loaded model and configuration.
    """
    try:
        from transformers import Sam2Model, Sam2Processor
    except ImportError:
        raise ImportError(
            "transformers library is required to load SAM2 models. "
            "Install with: pip install transformers"
        )

    print(f"Loading SAM2 model: {model_name}")

    # Load model
    model = Sam2Model.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device if isinstance(device, str) else str(device),
    )

    # Extract configuration
    hf_config = model.config

    # Map HuggingFace config to our config
    vision_config = hf_config.vision_config if hasattr(hf_config, "vision_config") else hf_config

    config = Sam2Config(
        hidden_size=getattr(vision_config, "hidden_size", 768),
        intermediate_size=getattr(vision_config, "intermediate_size", 3072),
        num_hidden_layers=getattr(vision_config, "num_hidden_layers", 12),
        num_attention_heads=getattr(vision_config, "num_attention_heads", 12),
        image_size=getattr(vision_config, "image_size", 1024),
        patch_size=getattr(vision_config, "patch_size", 16),
        num_channels=getattr(vision_config, "num_channels", 3),
    )

    model.eval()

    print(f"Model loaded successfully:")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Num layers: {config.num_hidden_layers}")
    print(f"  - Attention heads: {config.num_attention_heads}")
    print(f"  - Image size: {config.image_size}")
    print(f"  - Device: {device}")
    print(f"  - Dtype: {torch_dtype}")

    return model, config


def get_sam2_processor(model_name: str = "facebook/sam2-hiera-tiny") -> tp.Any:
    """Get SAM2 processor for image preprocessing.

    Args:
        model_name (`str`, *optional*, defaults to `"facebook/sam2-hiera-tiny"`):
            HuggingFace model name.

    Returns:
        Sam2Processor: Processor for image preprocessing.
    """
    try:
        from transformers import Sam2Processor
    except ImportError:
        raise ImportError(
            "transformers library is required. "
            "Install with: pip install transformers"
        )

    return Sam2Processor.from_pretrained(model_name)


def get_sam2_image_encoder(model: nn.Module) -> nn.Module:
    """Extract image encoder from SAM2 model.

    Args:
        model (`nn.Module`):
            SAM2 model.

    Returns:
        `nn.Module`: Image encoder module.
    """
    # SAM2 typically has vision_encoder or image_encoder
    if hasattr(model, "vision_encoder"):
        return model.vision_encoder
    elif hasattr(model, "image_encoder"):
        return model.image_encoder
    elif hasattr(model, "sam"):
        if hasattr(model.sam, "vision_encoder"):
            return model.sam.vision_encoder
        elif hasattr(model.sam, "image_encoder"):
            return model.sam.image_encoder
    else:
        # Return the whole model if we can't find encoder
        return model
