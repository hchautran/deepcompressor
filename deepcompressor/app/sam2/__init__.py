# -*- coding: utf-8 -*-
"""SAM2 (Segment Anything Model 2) quantization from HuggingFace."""

from .cache import Sam2PtqCacheConfig, Sam2QuantCacheConfig
from .config import Sam2PtqConfig, Sam2PtqRunConfig
from .dataset import get_coco_calibration_loader, get_sa1b_calibration_loader
from .model import SAM2_MODELS, get_sam2_processor, get_vision_encoder, load_sam2_from_huggingface, print_model_info
from .nn import (
    Sam2AttentionStruct,
    Sam2ConfigStruct,
    Sam2FeedForwardStruct,
    Sam2HieraBlockStruct,
    Sam2ModelStruct,
    Sam2ModuleStruct,
    Sam2VisionEncoderStruct,
)
from .ptq import ptq
from .quant import (
    Sam2ActivationQuantizer,
    Sam2ActivationQuantizerConfig,
    Sam2GPTQConfig,
    Sam2ModuleQuantizer,
    Sam2ModuleQuantizerConfig,
    Sam2Quantizer,
    Sam2QuantizerConfig,
    Sam2WeightQuantizer,
    Sam2WeightQuantizerConfig,
    calibrate_sam2_activations,
    get_sam2_modules_for_quantization,
    load_sam2_weights_state_dict,
    quantize_sam2_activations,
    quantize_sam2_weights,
    rotate_sam2,
    smooth_sam2,
)
from .quant.config import Sam2CalibConfig, Sam2QuantConfig, Sam2RotationConfig, Sam2SmoothConfig

__all__ = [
    # PTQ
    "ptq",
    # Model Loading
    "load_sam2_from_huggingface",
    "get_sam2_processor",
    "get_vision_encoder",
    "print_model_info",
    "SAM2_MODELS",
    # Model Structure
    "Sam2ConfigStruct",
    "Sam2ModelStruct",
    "Sam2VisionEncoderStruct",
    "Sam2HieraBlockStruct",
    "Sam2AttentionStruct",
    "Sam2FeedForwardStruct",
    "Sam2ModuleStruct",
    # Quantizers
    "Sam2QuantizerConfig",
    "Sam2WeightQuantizerConfig",
    "Sam2ActivationQuantizerConfig",
    "Sam2ModuleQuantizerConfig",
    "Sam2GPTQConfig",
    "Sam2Quantizer",
    "Sam2WeightQuantizer",
    "Sam2ActivationQuantizer",
    "Sam2ModuleQuantizer",
    # Quantization Functions
    "quantize_sam2_weights",
    "calibrate_sam2_activations",
    "quantize_sam2_activations",
    "smooth_sam2",
    "rotate_sam2",
    "get_sam2_modules_for_quantization",
    "load_sam2_weights_state_dict",
    # Configuration
    "Sam2QuantConfig",
    "Sam2CalibConfig",
    "Sam2RotationConfig",
    "Sam2SmoothConfig",
    "Sam2PtqConfig",
    "Sam2PtqRunConfig",
    "Sam2PtqCacheConfig",
    "Sam2QuantCacheConfig",
    # Dataset
    "get_coco_calibration_loader",
    "get_sa1b_calibration_loader",
]
