# -*- coding: utf-8 -*-
"""SAM2 quantizer classes with SVDQuant support."""

import typing as tp
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from deepcompressor.calib.config import SkipBasedQuantLowRankCalibConfig
from deepcompressor.calib.lowrank import LowRankBranch, QuantLowRankCalibrator
from deepcompressor.data.cache import TensorsCache
from deepcompressor.data.common import TensorType
from deepcompressor.quantizer.processor import Quantizer

from .config import Sam2ActivationQuantizerConfig, Sam2WeightQuantizerConfig

__all__ = ["Sam2WeightQuantizer", "Sam2ActivationQuantizer"]


@dataclass
class Sam2WeightQuantizer(Quantizer):
    """SAM2 weight quantizer with SVDQuant support.

    Args:
        config (`Sam2WeightQuantizerConfig` or `None`):
            The quantizer configuration.
        key (`str`, *optional*, defaults to `""`):
            The key of the quantizer.
        tensor_type (`TensorType`, *optional*, defaults to `TensorType.Weights`):
            The type of the tensor to quantize.
        channels_dim (`int` or `None`, *optional*, defaults to `None`):
            The dimension of channels.
        develop_dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            The quantization development dtype.
    """

    config: Sam2WeightQuantizerConfig = field(default=None)
    tensor_type: TensorType = field(init=False, default=TensorType.Weights)
    low_rank: SkipBasedQuantLowRankCalibConfig | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.config is not None:
            self.low_rank = self.config.low_rank

    @classmethod
    def build(
        cls,
        config: Sam2WeightQuantizerConfig,
        weight: torch.Tensor,
        key: str = "",
        **kwargs,
    ) -> "Sam2WeightQuantizer":
        """Build a weight quantizer from configuration.

        Args:
            config: Weight quantizer configuration.
            weight: Weight tensor to quantize.
            key: The key/name of the module being quantized.

        Returns:
            Sam2WeightQuantizer instance.
        """
        if not config.is_enabled():
            return None

        quantizer = cls(
            config=config,
            channels_dim=0,
            key=key,
            develop_dtype=kwargs.get("develop_dtype", torch.float32),
        )
        return quantizer

    def calibrate_low_rank(
        self,
        input_quantizer: "Sam2ActivationQuantizer",
        modules: tp.Sequence[nn.Module],
        inputs: TensorsCache,
        weights: tp.Sequence[nn.Parameter] = None,
        eval_inputs: TensorsCache | None = None,
        eval_module: nn.Module | None = None,
        eval_kwargs: dict[str, tp.Any] | None = None,
        orig_inputs: TensorsCache | None = None,
        orig_eval_inputs: TensorsCache | None = None,
    ) -> LowRankBranch:
        """Calibrate the quantization low-rank branch for SVDQuant.

        This method uses the QuantLowRankCalibrator to find optimal low-rank
        compensation matrices that minimize quantization error.

        Args:
            input_quantizer: The activation quantizer for inputs.
            modules: The modules to calibrate.
            inputs: The cached input activations.
            weights: The weight parameters (optional, uses module weights if None).
            eval_inputs: Cached inputs for evaluation.
            eval_module: Module to evaluate quantization error.
            eval_kwargs: Keyword arguments for evaluation.
            orig_inputs: Original (unquantized) inputs for comparison.
            orig_eval_inputs: Original evaluation inputs.

        Returns:
            LowRankBranch: The calibrated low-rank branch.
        """
        if weights is None:
            weights = [module.weight for module in modules]
        return QuantLowRankCalibrator(
            config=self.low_rank,
            w_quantizer=self,
            x_quantizer=input_quantizer,
            develop_dtype=self.develop_dtype,
        ).calibrate(
            x_wgts=weights,
            x_acts=inputs,
            x_mods=modules,
            eval_inputs=eval_inputs,
            eval_module=eval_module,
            eval_kwargs=eval_kwargs,
            orig_x_acts=orig_inputs,
            orig_eval_inputs=orig_eval_inputs,
        )


@dataclass
class Sam2ActivationQuantizer(Quantizer):
    """SAM2 activation quantizer.

    Args:
        config (`Sam2ActivationQuantizerConfig` or `None`):
            The quantizer configuration.
        key (`str`, *optional*, defaults to `""`):
            The key of the quantizer.
        tensor_type (`TensorType`, *optional*, defaults to `TensorType.Inputs`):
            The type of the tensor to quantize.
        channels_dim (`int` or `None`, *optional*, defaults to `None`):
            The dimension of channels.
        develop_dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            The quantization development dtype.
    """

    config: Sam2ActivationQuantizerConfig = field(default=None)
    tensor_type: TensorType = field(init=False, default=TensorType.Inputs)

    @classmethod
    def build(
        cls,
        config: Sam2ActivationQuantizerConfig,
        key: str = "",
        **kwargs,
    ) -> "Sam2ActivationQuantizer":
        """Build an activation quantizer from configuration.

        Args:
            config: Activation quantizer configuration.
            key: The key/name of the module being quantized.

        Returns:
            Sam2ActivationQuantizer instance.
        """
        if not config.is_enabled():
            return None

        quantizer = cls(
            config=config,
            channels_dim=-1,  # Last dimension for activations
            key=key,
            develop_dtype=kwargs.get("develop_dtype", torch.float32),
        )
        return quantizer
