# -*- coding: utf-8 -*-
"""SAM2 model loading utilities for HuggingFace models."""

import typing as tp

import torch
import torch.nn as nn

from .nn.struct import Sam2ConfigStruct, Sam2ModelStruct

__all__ = [
    "load_sam2_from_huggingface",
    "get_sam2_processor",
    "SAM2_MODELS",
]


# Available SAM2 models on HuggingFace
SAM2_MODELS = {
    "tiny": "facebook/sam2-hiera-tiny",
    "small": "facebook/sam2-hiera-small",
    "base": "facebook/sam2-hiera-base-plus",
    "large": "facebook/sam2-hiera-large",
}


def load_sam2_from_huggingface(
    model_name: str = "facebook/sam2-hiera-tiny",
    device: str | torch.device = "cuda",
    torch_dtype: torch.dtype = torch.float16,
) -> tuple[nn.Module, Sam2ModelStruct]:
    """Load SAM2 model from HuggingFace.

    Args:
        model_name (`str`, *optional*, defaults to `"facebook/sam2-hiera-tiny"`):
            HuggingFace model name or shortcut. Available models:
            - "tiny" or "facebook/sam2-hiera-tiny"
            - "small" or "facebook/sam2-hiera-small"
            - "base" or "facebook/sam2-hiera-base-plus"
            - "large" or "facebook/sam2-hiera-large"
        device (`str` or `torch.device`, *optional*, defaults to `"cuda"`):
            Device to load model on.
        torch_dtype (`torch.dtype`, *optional*, defaults to `torch.float16`):
            Data type for model weights.

    Returns:
        tuple[nn.Module, Sam2ModelStruct]: Loaded model and model structure.

    Examples:
        >>> model, model_struct = load_sam2_from_huggingface("tiny")
        >>> model, model_struct = load_sam2_from_huggingface("facebook/sam2-hiera-base-plus")
    """
    try:
        from transformers import Sam2Model
    except ImportError:
        raise ImportError(
            "transformers library is required to load SAM2 models. "
            "Install with: pip install transformers>=4.40.0"
        )

    # Handle shortcuts
    if model_name in SAM2_MODELS:
        model_name = SAM2_MODELS[model_name]

    print(f"Loading SAM2 model: {model_name}")
    print(f"  - Device: {device}")
    print(f"  - Dtype: {torch_dtype}")

    # Load model
    model = Sam2Model.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    )

    # Move to device
    if isinstance(device, str):
        device = torch.device(device)
    model = model.to(device)
    model.eval()

    # Extract configuration
    hf_config = model.config
    vision_config = hf_config.vision_config if hasattr(hf_config, "vision_config") else hf_config

    print(f"Model loaded successfully:")
    print(f"  - Hidden size: {getattr(vision_config, 'hidden_size', 768)}")
    print(f"  - Num layers: {getattr(vision_config, 'num_hidden_layers', 12)}")
    print(f"  - Attention heads: {getattr(vision_config, 'num_attention_heads', 12)}")
    print(f"  - Image size: {getattr(vision_config, 'image_size', 1024)}")

    # Construct model structure
    print(f"Constructing model structure...")
    model_struct = Sam2ModelStruct.construct(model)
    print(f"  - Total blocks: {model_struct.num_blocks}")
    print(f"  - Quantizable modules: {sum(1 for _ in model_struct.named_key_modules())}")

    return model, model_struct


def get_sam2_processor(model_name: str = "facebook/sam2-hiera-tiny") -> tp.Any:
    """Get SAM2 processor for image preprocessing.

    Args:
        model_name (`str`, *optional*, defaults to `"facebook/sam2-hiera-tiny"`):
            HuggingFace model name or shortcut.

    Returns:
        Sam2Processor: Processor for image preprocessing.

    Examples:
        >>> processor = get_sam2_processor("tiny")
        >>> inputs = processor(images=image, return_tensors="pt")
    """
    try:
        from transformers import Sam2Processor
    except ImportError:
        raise ImportError(
            "transformers library is required. "
            "Install with: pip install transformers>=4.40.0"
        )

    # Handle shortcuts
    if model_name in SAM2_MODELS:
        model_name = SAM2_MODELS[model_name]

    return Sam2Processor.from_pretrained(model_name)


def get_vision_encoder(model: nn.Module) -> nn.Module:
    """Extract vision encoder from SAM2 model.

    Args:
        model (`nn.Module`):
            SAM2 model.

    Returns:
        `nn.Module`: Vision encoder module (Hiera backbone).
    """
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


def print_model_info(model: nn.Module, model_struct: Sam2ModelStruct) -> None:
    """Print detailed model information.

    Args:
        model (`nn.Module`):
            SAM2 model.
        model_struct (`Sam2ModelStruct`):
            SAM2 model structure.
    """
    print("\n" + "=" * 80)
    print("SAM2 Model Information")
    print("=" * 80)

    # Configuration
    config = model_struct.config
    print(f"\nConfiguration:")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Intermediate size: {config.intermediate_size}")
    print(f"  - Num hidden layers: {config.num_hidden_layers}")
    print(f"  - Num attention heads: {config.num_attention_heads}")
    print(f"  - Image size: {config.image_size}")
    print(f"  - Patch size: {config.patch_size}")
    print(f"  - Embed dim: {config.embed_dim}")

    # Model structure
    print(f"\nModel Structure:")
    print(f"  - Total blocks: {model_struct.num_blocks}")
    print(f"  - Attention modules: {sum(1 for _ in model_struct.iter_attention_structs())}")
    print(f"  - FFN modules: {sum(1 for _ in model_struct.iter_ffn_structs())}")
    print(f"  - Quantizable modules: {sum(1 for _ in model_struct.named_key_modules())}")

    # Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nParameters:")
    print(f"  - Total: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"  - Trainable: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")

    print("=" * 80)
