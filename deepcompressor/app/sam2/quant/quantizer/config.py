# -*- coding: utf-8 -*-
"""SAM2 Quantizer Configuration."""

from dataclasses import dataclass, field

import torch
from omniconfig import configclass

from deepcompressor.calib.config import SkipBasedDynamicRangeCalibConfig, SkipBasedQuantLowRankCalibConfig
from deepcompressor.data.dtype import QuantDataType
from deepcompressor.quantizer.config import QuantizerConfig
from deepcompressor.quantizer.kernel import QuantGptqConfig
from deepcompressor.utils.config import EnableConfig, IncludeBasedConfig, SkipBasedConfig

__all__ = [
    "Sam2QuantizerConfig",
    "Sam2WeightQuantizerConfig",
    "Sam2ActivationQuantizerConfig",
    "Sam2ModuleQuantizerConfig",
]


@configclass
@dataclass
class Sam2GPTQConfig(SkipBasedConfig, QuantGptqConfig):
    """Configuration for SAM2 GPTQ quantization.

    Args:
        damp_percentage (`float`, *optional*, defaults to `0.01`):
            The percentage of damping.
        block_size (`int`, *optional*, defaults to `128`):
            The block size of the GPTQ quantization.
        num_inv_tries (`int`, *optional*, defaults to `200`):
            The number of tries for the inverse.
        hessian_block_size (`int`, *optional*, defaults to `-1`):
            The block size when calculating the Hessian.
        skips (`list[str]`, *optional*, defaults to `[]`):
            List of module names to skip during GPTQ quantization.
    """

    pass


@configclass
@dataclass
class Sam2QuantizerConfig(QuantizerConfig):
    """SAM2 model quantizer configuration.

    Args:
        dtype (`QuantDataType` or `None`, *optional*, defaults to `None`):
            The quantization data type.
        zero_point (`ZeroPointDomain` or `None`, *optional*, defaults to `None`):
            The zero-point domain.
        group_shapes (`Sequence[Sequence[int]]`, *optional*, defaults to `((-1, -1),)`):
            The shapes for per-group quantization.
        scale_dtypes (`Sequence[torch.dtype | QuantDataType | None]`, *optional*, defaults to `(None,)`):
            The quantization scale data type for per-group quantization.
        static (`bool`, *optional*, defaults to `False`):
            Whether to use static quantization.
        kernel_gptq (`Sam2GPTQConfig` or `None`, *optional*, defaults to `None`):
            The GPTQ quantization configuration.
        low_rank (`SkipBasedQuantLowRankCalibConfig` or `None`, *optional*, defaults to `None`):
            The quantization low-rank branch calibration configuration.
        calib_range (`SkipBasedDynamicRangeCalibConfig` or `None`, *optional*, defaults to `None`):
            The quantizer dynamic range calibration configuration.
    """

    static: bool = False
    kernel_gptq: Sam2GPTQConfig | None = None
    low_rank: SkipBasedQuantLowRankCalibConfig | None = None
    calib_range: SkipBasedDynamicRangeCalibConfig | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.quant_dtype is None:
            self.static = False
            self.kernel_gptq = None
            self.low_rank = None
            self.calib_range = None
        if self.kernel_gptq is not None and not self.kernel_gptq.is_enabled():
            self.kernel_gptq = None
        if self.static and self.calib_range is None:
            self.calib_range = SkipBasedDynamicRangeCalibConfig()
        if self.low_rank is not None and not self.low_rank.is_enabled():
            self.low_rank = None

    @property
    def enabled_gptq(self) -> bool:
        """Whether GPTQ quantization kernel calibration is enabled."""
        return self.kernel_gptq is not None and self.kernel_gptq.is_enabled()

    @property
    def enabled_low_rank(self) -> bool:
        """Whether quantization low-rank (SVDQuant) calibration is enabled."""
        return self.low_rank is not None and self.low_rank.is_enabled()

    @property
    def enabled_calib_range(self) -> bool:
        """Whether quantization dynamic range calibration is enabled."""
        return self.calib_range is not None


@configclass
@dataclass
class Sam2WeightQuantizerConfig(Sam2QuantizerConfig):
    """SAM2 weight quantizer configuration.

    Args:
        dtype (`QuantDataType` or `None`, *optional*, defaults to `None`):
            The quantization data type.
        static (`bool`, *optional*, defaults to `True`):
            Whether to use static quantization for weights.
        skip_patch_embed (`bool`, *optional*, defaults to `True`):
            Whether to skip quantizing patch embedding layer.
        skip_first_block (`bool`, *optional*, defaults to `False`):
            Whether to skip quantizing the first transformer block.
        skip_last_block (`bool`, *optional*, defaults to `False`):
            Whether to skip quantizing the last transformer block.
        skip_decoder (`bool`, *optional*, defaults to `False`):
            Whether to skip quantizing the decoder.
    """

    static: bool = True
    skip_patch_embed: bool = True
    skip_first_block: bool = False
    skip_last_block: bool = False
    skip_decoder: bool = False


@configclass
@dataclass
class Sam2ActivationQuantizerConfig(Sam2QuantizerConfig):
    """SAM2 activation quantizer configuration.

    Args:
        dtype (`QuantDataType` or `None`, *optional*, defaults to `None`):
            The quantization data type.
        static (`bool`, *optional*, defaults to `False`):
            Whether to use static quantization for activations.
        per_token (`bool`, *optional*, defaults to `True`):
            Whether to use per-token quantization for activations.
    """

    static: bool = False
    per_token: bool = True


@configclass
@dataclass
class Sam2ModuleQuantizerConfig(EnableConfig):
    """SAM2 module quantizer configuration combining weights and activations.

    Args:
        wgts (`Sam2WeightQuantizerConfig` or `None`, *optional*, defaults to `None`):
            The weight quantizer configuration.
        ipts (`Sam2ActivationQuantizerConfig` or `None`, *optional*, defaults to `None`):
            The input activation quantizer configuration.
        opts (`Sam2ActivationQuantizerConfig` or `None`, *optional*, defaults to `None`):
            The output activation quantizer configuration.
    """

    wgts: Sam2WeightQuantizerConfig | None = None
    ipts: Sam2ActivationQuantizerConfig | None = None
    opts: Sam2ActivationQuantizerConfig | None = None

    def __post_init__(self) -> None:
        if self.wgts is None:
            self.wgts = Sam2WeightQuantizerConfig()
        if self.ipts is None:
            self.ipts = Sam2ActivationQuantizerConfig()
        if self.opts is None:
            self.opts = Sam2ActivationQuantizerConfig()

    def is_enabled(self) -> bool:
        """Check if any quantization is enabled."""
        return (
            (self.wgts is not None and self.wgts.is_enabled())
            or (self.ipts is not None and self.ipts.is_enabled())
            or (self.opts is not None and self.opts.is_enabled())
        )

    @property
    def enabled_wgts(self) -> bool:
        """Whether weight quantization is enabled."""
        return self.wgts is not None and self.wgts.is_enabled()

    @property
    def enabled_ipts(self) -> bool:
        """Whether input activation quantization is enabled."""
        return self.ipts is not None and self.ipts.is_enabled()

    @property
    def enabled_opts(self) -> bool:
        """Whether output activation quantization is enabled."""
        return self.opts is not None and self.opts.is_enabled()
