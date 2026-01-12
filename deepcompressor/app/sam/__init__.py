# -*- coding: utf-8 -*-
"""SAM (Segment Anything Model) quantization utilities."""

from .cache import SamPtqCacheConfig, SamQuantCacheConfig
from .config import SamPtqConfig, SamPtqRunConfig
from .nn import (
    SamAttentionStruct,
    SamBackboneStruct,
    SamConfigStruct,
    SamDecoderStruct,
    SamFeedForwardStruct,
    SamModelStruct,
    SamModuleStruct,
    SamNunchakuBlockStruct,
)
from .ptq import ptq
from .quant import (
    SamActivationQuantizer,
    SamActivationQuantizerConfig,
    SamGPTQConfig,
    SamModuleQuantizer,
    SamModuleQuantizerConfig,
    SamQuantizer,
    SamQuantizerConfig,
    SamWeightQuantizer,
    SamWeightQuantizerConfig,
    calibrate_sam_activations,
    get_sam_modules_for_quantization,
    load_sam_weights_state_dict,
    quantize_sam_activations,
    quantize_sam_weights,
    rotate_sam,
    smooth_sam,
)
from .quant.config import SamCalibConfig, SamQuantConfig, SamRotationConfig, SamSmoothConfig

__all__ = [
    # PTQ
    "ptq",
    # Model Structure
    "SamConfigStruct",
    "SamModelStruct",
    "SamBackboneStruct",
    "SamNunchakuBlockStruct",
    "SamAttentionStruct",
    "SamFeedForwardStruct",
    "SamDecoderStruct",
    "SamModuleStruct",
    # Quantizers
    "SamQuantizerConfig",
    "SamWeightQuantizerConfig",
    "SamActivationQuantizerConfig",
    "SamModuleQuantizerConfig",
    "SamGPTQConfig",
    "SamQuantizer",
    "SamWeightQuantizer",
    "SamActivationQuantizer",
    "SamModuleQuantizer",
    # Quantization Functions
    "quantize_sam_weights",
    "calibrate_sam_activations",
    "quantize_sam_activations",
    "smooth_sam",
    "rotate_sam",
    "get_sam_modules_for_quantization",
    "load_sam_weights_state_dict",
    # Configuration
    "SamQuantConfig",
    "SamCalibConfig",
    "SamRotationConfig",
    "SamSmoothConfig",
    "SamPtqConfig",
    "SamPtqRunConfig",
    "SamPtqCacheConfig",
    "SamQuantCacheConfig",
]
