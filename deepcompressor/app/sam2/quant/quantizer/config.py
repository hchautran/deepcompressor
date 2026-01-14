# -*- coding: utf-8 -*-
"""SAM2 Quantizer config for SVDQuant W4A4."""

from dataclasses import dataclass, field

import torch
from omniconfig import configclass

from deepcompressor.calib.config import SkipBasedDynamicRangeCalibConfig, SkipBasedQuantLowRankCalibConfig
from deepcompressor.data.dtype import QuantDataType
from deepcompressor.quantizer.config import QuantizerConfig
from deepcompressor.quantizer.kernel import QuantGptqConfig
from deepcompressor.utils.config import EnableConfig, SkipBasedConfig

__all__ = [
    "SAM2QuantizerConfig",
    "SAM2WeightQuantizerConfig",
    "SAM2ActivationQuantizerConfig",
    "SAM2ModuleQuantizerConfig",
]


@configclass
@dataclass
class SAM2GPTQConfig(SkipBasedConfig, QuantGptqConfig):
    """Configuration for GPTQ quantization in SAM2."""

    pass


@configclass
@dataclass
class SAM2QuantizerConfig(QuantizerConfig):
    """SAM2 model quantizer configuration.

    Args:
        dtype (`QuantDataType` or `None`, *optional*, defaults to `None`):
            The quantization data type.
        zero_point (`ZeroPointDomain` or `None`, *optional*, defaults to `None`):
            The zero-point domain.
        group_shapes (`Sequence[Sequence[int]]`, *optional*, defaults to `((-1, -1, -1),)`):
            The shapes for per-group quantization.
        scale_dtypes (`Sequence[torch.dtype | QuantDataType | None]`, *optional*, defaults to `(None,)`):
            The quantization scale data type for per-group quantization.
        static (`bool`, *optional*, defaults to `False`):
            Whether to use static quantization.
        kernel_gptq (`SAM2GPTQConfig` or `None`, *optional*, defaults to `None`):
            The GPTQ quantization configuration.
        low_rank (`SkipBasedQuantLowRankCalibConfig` or `None`, *optional*, defaults to `None`):
            The SVDQuant low-rank branch calibration configuration.
        calib_range (`SkipBasedDynamicRangeCalibConfig` or `None`, *optional*, defaults to `None`):
            The dynamic range calibration configuration.
    """

    static: bool = False
    kernel_gptq: SAM2GPTQConfig | None = None
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
        """Whether GPTQ quantization is enabled."""
        return self.kernel_gptq is not None and self.kernel_gptq.is_enabled()

    @property
    def enabled_low_rank(self) -> bool:
        """Whether SVDQuant low-rank calibration is enabled."""
        return self.low_rank is not None and self.low_rank.is_enabled()

    @property
    def enabled_calib_range(self) -> bool:
        """Whether dynamic range calibration is enabled."""
        return self.calib_range is not None

    def generate_calib_dirname(self) -> str:
        """Generate the name for quantization calibration."""
        name = ""
        if self.static:
            name += ".static"
        if self.enabled_gptq:
            name += ".gptq"
        if self.enabled_low_rank:
            name += ".svdquant"
        if self.enabled_calib_range and (self.calib_range.needs_search or self.calib_range.ratio != 1):
            name += ".range"
        return name[1:] if name else ""


@configclass
@dataclass
class SkipBasedSAM2QuantizerConfig(SkipBasedConfig, SAM2QuantizerConfig):
    """SAM2 quantizer configuration with skip support."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.quant_dtype is None:
            self.skips.clear()


@configclass
@dataclass
class SAM2WeightQuantizerConfig(SkipBasedSAM2QuantizerConfig):
    """SAM2 weight quantizer configuration for W4 quantization.

    Default configuration for 4-bit weight quantization with SVDQuant.
    """

    static: bool = field(init=False, default=True)

    @property
    def needs_calib_data(self) -> bool:
        return self.enabled_calib_range and self.calib_range.needs_search


@configclass
@dataclass
class SAM2ActivationQuantizerConfig(SkipBasedSAM2QuantizerConfig):
    """SAM2 activation quantizer configuration for A4 quantization.

    Default configuration for 4-bit activation quantization.
    """

    kernel_gptq: None = field(init=False, default=None)
    low_rank: None = field(init=False, default=None)
    allow_unsigned: bool = False

    @property
    def needs_calib_data(self) -> bool:
        return self.enabled_calib_range and (self.calib_range.needs_search or self.static)

    def generate_dirnames(
        self,
        *,
        prefix: str = "",
        shape: torch.Size | tuple[int, ...] = (1024, 1024, 16, 16),
        default_dtype: torch.dtype = torch.float16,
        **kwargs,
    ) -> list[str]:
        names = super().generate_dirnames(prefix=prefix, shape=shape, default_dtype=default_dtype)
        if self.allow_unsigned:
            names[1] += ".u"
        return names

    def for_unsigned(self) -> "SAM2ActivationQuantizerConfig":
        """Get the quantizer configuration for unsigned activations."""
        if isinstance(self.dtype, QuantDataType) and self.allow_unsigned:
            return SAM2ActivationQuantizerConfig(
                dtype=self.dtype.to_unsigned(),
                zero_point=self.zero_point,
                group_shapes=self.group_shapes,
                scale_dtypes=self.scale_dtypes,
                skips=self.skips,
                static=self.static,
                calib_range=self.calib_range,
                allow_unsigned=self.allow_unsigned,
            )
        else:
            return self


@configclass
@dataclass(kw_only=True)
class SAM2ModuleQuantizerConfig(EnableConfig):
    """SAM2 module quantizer configuration for SVDQuant W4A4.

    Args:
        wgts (`SAM2WeightQuantizerConfig`):
            The weight quantization configuration (4-bit).
        ipts (`SAM2ActivationQuantizerConfig`):
            The input activation quantization configuration (4-bit).
        opts (`SAM2ActivationQuantizerConfig`):
            The output activation quantization configuration.
    """

    wgts: SAM2WeightQuantizerConfig
    ipts: SAM2ActivationQuantizerConfig
    opts: SAM2ActivationQuantizerConfig
    unsigned_ipts: SAM2ActivationQuantizerConfig = field(init=False)

    def is_enabled(self):
        return self.enabled_wgts or self.enabled_ipts or self.enabled_opts

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

    def __post_init__(self) -> None:
        if self.enabled_opts:
            raise NotImplementedError("Output activation quantization is not supported yet.")

    def generate_dirnames(
        self,
        *,
        prefix: str = "",
        shape: torch.Size | tuple[int, ...] = (1024, 1024, 16, 16),
        default_dtype: torch.dtype = torch.float16,
        **kwargs,
    ) -> list[str]:
        """Get the directory names of the quantization configuration."""
        wgts_names = self.wgts.generate_dirnames(prefix="w", shape=shape, default_dtype=default_dtype)
        ipts_names = self.ipts.generate_dirnames(prefix="x", shape=shape, default_dtype=default_dtype)
        opts_names = self.opts.generate_dirnames(prefix="y", shape=shape, default_dtype=default_dtype)
        names = [
            f"{wgts_name}-{ipts_name}-{opts_name}"
            for wgts_name, ipts_name, opts_name in zip(wgts_names, ipts_names, opts_names, strict=True)
        ]
        if prefix:
            names = [f"{prefix}.[{name}]" for name in names]
        return names

    def generate_calib_dirname(self) -> str:
        """Generate the name for quantization calibration."""
        name = ""
        if self.enabled_wgts:
            calib_name = self.wgts.generate_calib_dirname()
            if calib_name:
                name += f"-w.{calib_name}"
        if self.enabled_ipts:
            calib_name = self.ipts.generate_calib_dirname()
            if calib_name:
                name += f"-x.{calib_name}"
        if self.enabled_opts:
            calib_name = self.opts.generate_calib_dirname()
            if calib_name:
                name += f"-y.{calib_name}"
        return name[1:] if name else name
