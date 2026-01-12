# -*- coding: utf-8 -*-
"""SAM2 quantization configuration."""

import os
from dataclasses import dataclass, field

import torch
from omniconfig import configclass

from deepcompressor.calib.config import (
    QuantRotationConfig,
    SearchBasedCalibGranularity,
    SmoothTransfomerConfig,
)
from deepcompressor.data.utils.dtype import eval_dtype
from deepcompressor.quantizer.config import QuantLowRankConfig

from ..cache.config import Sam2QuantCacheConfig
from ..dataset.calib import Sam2CalibConfig
from .quantizer.config import Sam2ModuleQuantizerConfig

__all__ = ["Sam2QuantConfig"]


@configclass
@dataclass(kw_only=True)
class Sam2QuantConfig(Sam2ModuleQuantizerConfig):
    """SAM2 model quantization configuration.

    Args:
        wgts (`Sam2WeightQuantizerConfig`):
            The weight quantization configuration.
        ipts (`Sam2ActivationQuantizerConfig`):
            The input activation quantization configuration.
        opts (`Sam2ActivationQuantizerConfig`):
            The output activation quantization configuration.
        calib (`Sam2CalibConfig`):
            The calibration dataset configuration.
        rotation (`QuantRotationConfig` or `None`, *optional*, defaults to `None`):
            The rotation quantization configuration.
        smooth (`SmoothTransfomerConfig` or `None`, *optional*, defaults to `None`):
            The smooth quantization configuration.
        develop_dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            The development data type.
    """

    calib: Sam2CalibConfig
    rotation: QuantRotationConfig | None = None
    smooth: SmoothTransfomerConfig | None = None
    develop_dtype: torch.dtype = field(default_factory=lambda s=torch.float32: eval_dtype(s, with_quant_dtype=False))

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.rotation is not None and not self.rotation.transforms:
            self.rotation = None
        if self.smooth is not None:
            if not self.smooth.enabled_proj and not self.smooth.enabled_attn:
                self.smooth = None
        if self.enabled_smooth and self.smooth.enabled_proj and self.smooth.proj.allow_low_rank:
            if self.enabled_wgts:
                self.smooth.proj.allow_low_rank = self.wgts.enabled_low_rank
                if self.smooth.proj.allow_low_rank:
                    self.smooth.proj.granularity = SearchBasedCalibGranularity.Layer
            else:
                self.smooth.proj.allow_low_rank = False
        if self.enabled_ipts:
            if self.ipts.enabled_calib_range and self.ipts.calib_range.granularity == SearchBasedCalibGranularity.Group:
                self.ipts.calib_range.granularity = SearchBasedCalibGranularity.ChannelGroup
            if self.ipts.static:
                assert self.ipts.smallest_group_shape[0] == -1, "static quantization requires batch group size to be -1"
        if self.enabled_opts:
            if self.opts.enabled_calib_range and self.opts.calib_range.granularity == SearchBasedCalibGranularity.Group:
                self.opts.calib_range.granularity = SearchBasedCalibGranularity.ChannelGroup
            if self.opts.static:
                assert self.opts.smallest_group_shape[0] == -1, "static quantization requires batch group size to be -1"
        self.unsigned_ipts = self.ipts.for_unsigned()

    @property
    def enabled_rotation(self) -> bool:
        """Whether to enable rotation."""
        return self.rotation is not None and bool(self.rotation.transforms)

    @property
    def enabled_smooth(self) -> bool:
        """Whether to enable smooth quantization."""
        return self.smooth is not None

    @property
    def enabled_smooth_proj(self) -> bool:
        """Whether to enable smooth quantization for projections."""
        return self.enabled_smooth and self.smooth.enabled_proj

    @property
    def enabled_smooth_attn(self) -> bool:
        """Whether to enable smooth quantization for attentions."""
        return self.enabled_smooth and self.smooth.enabled_attn

    @property
    def needs_acts_quantizer_cache(self) -> bool:
        """Whether to cache the activations quantizer settings."""
        if self.enabled_ipts and self.ipts.needs_calib_data:
            return True
        if self.enabled_opts and self.opts.needs_calib_data:
            return True
        return False

    def generate_calib_dirname(self) -> str:
        name = ""
        if self.enabled_rotation:
            name += "-rotate"
            if self.rotation.random:
                name += ".rnd"
        if self.enabled_smooth:
            name += "-smooth"
            if self.enabled_smooth_proj:
                name += ".proj"
            if self.enabled_smooth_attn:
                name += ".attn"
        calib_name = super().generate_calib_dirname()
        if calib_name:
            name += f"-{calib_name}"
        return name[1:] if name else name

    def generate_cache_dirpath(
        self, *, root: str, default_dtype: torch.dtype = torch.float16
    ) -> Sam2QuantCacheConfig:
        """Generate the cache paths for the module quantization configuration."""
        quant_names = self.generate_dirnames(default_dtype=default_dtype)
        if self.enabled_wgts and self.wgts.enabled_low_rank:
            quant_names.extend(QuantLowRankConfig.generate_dirnames(self.wgts.low_rank, prefix="lowrank"))
        if self.enabled_rotation:
            quant_names.extend(self.rotation.generate_dirnames(prefix="rotate"))
        smooth_dirpath = ""
        if self.enabled_smooth:
            quant_names.extend(self.smooth.generate_dirnames(prefix="smooth"))
            smooth_dirpath = os.path.join("smooth", *quant_names)
        branch_dirpath = ""
        if self.enabled_wgts and self.wgts.enabled_low_rank:
            quant_names.extend(self.wgts.low_rank.generate_dirnames(prefix="lowrank"))
            branch_dirpath = os.path.join("branch", *quant_names)
        wgts_dirpath = ""
        if self.enabled_wgts and self.wgts.needs_calib_data:
            quant_names.extend(self.wgts.calib_range.generate_dirnames(prefix="w.range"))
            wgts_dirpath = os.path.join("wgts", *quant_names)
        if self.enabled_wgts and self.wgts.enabled_gptq:
            quant_names.extend(self.wgts.kernel_gptq.generate_dirnames(prefix="w.kernel"))
        acts_dirpath = ""
        if self.needs_acts_quantizer_cache:
            if self.enabled_ipts and self.ipts.needs_calib_data:
                quant_names.extend(self.ipts.calib_range.generate_dirnames(prefix="x.range"))
            if self.enabled_opts and self.opts.needs_calib_data:
                quant_names.extend(self.opts.calib_range.generate_dirnames(prefix="y.range"))
            acts_dirpath = os.path.join("acts", *quant_names)
        cache_dirpath = Sam2QuantCacheConfig(
            smooth=smooth_dirpath, branch=branch_dirpath, wgts=wgts_dirpath, acts=acts_dirpath
        )
        cache_dirpath = cache_dirpath.add_parent_dirs(*self.calib.generate_dirnames())
        cache_dirpath = cache_dirpath.add_parent_dirs(root, "sam2", "cache", "quant")
        return cache_dirpath

    def generate_default_dirname(self) -> str:
        """Generate output directory name for evaluating SAM2 model."""
        name = ""
        if self.enabled_wgts:
            if self.wgts.quant_dtype is not None:
                name += f"w{self.wgts.quant_dtype.bits}"
        if self.enabled_ipts:
            if self.ipts.quant_dtype is not None:
                name += f"a{self.ipts.quant_dtype.bits}"
        if self.enabled_rotation:
            name += "-rotate"
        if self.enabled_smooth:
            name += "-smooth"
        if self.enabled_wgts and self.wgts.enabled_low_rank:
            name += "-svdquant"
        if self.enabled_wgts and self.wgts.enabled_gptq:
            name += "-gptq"
        return name if name else "fp16"
