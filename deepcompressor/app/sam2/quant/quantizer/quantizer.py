# -*- coding: utf-8 -*-
"""SAM2 quantizer classes."""

from dataclasses import dataclass, field

import torch

from deepcompressor.data.common import TensorType
from deepcompressor.quantizer.processor import Quantizer

from .config import Sam2ActivationQuantizerConfig, Sam2WeightQuantizerConfig

__all__ = ["Sam2WeightQuantizer", "Sam2ActivationQuantizer"]


@dataclass
class Sam2WeightQuantizer(Quantizer):
    """SAM2 weight quantizer."""

    config: Sam2WeightQuantizerConfig = field(default=None)
    tensor_type: TensorType = field(init=False, default=TensorType.Weights)

    @classmethod
    def build(
        cls,
        config: Sam2WeightQuantizerConfig,
        weight: torch.Tensor,
        **kwargs,
    ) -> "Sam2WeightQuantizer":
        """Build a weight quantizer from configuration.

        Args:
            config: Weight quantizer configuration.
            weight: Weight tensor to quantize.

        Returns:
            Sam2WeightQuantizer instance.
        """
        if not config.is_enabled():
            return None

        quantizer = cls(
            config=config,
            channels_dim=0,
            develop_dtype=kwargs.get("develop_dtype", torch.float32),
        )
        return quantizer


@dataclass
class Sam2ActivationQuantizer(Quantizer):
    """SAM2 activation quantizer."""

    config: Sam2ActivationQuantizerConfig = field(default=None)
    tensor_type: TensorType = field(init=False, default=TensorType.Inputs)

    @classmethod
    def build(
        cls,
        config: Sam2ActivationQuantizerConfig,
        **kwargs,
    ) -> "Sam2ActivationQuantizer":
        """Build an activation quantizer from configuration.

        Args:
            config: Activation quantizer configuration.

        Returns:
            Sam2ActivationQuantizer instance.
        """
        if not config.is_enabled():
            return None

        quantizer = cls(
            config=config,
            channels_dim=-1,  # Last dimension for activations
            develop_dtype=kwargs.get("develop_dtype", torch.float32),
        )
        return quantizer
