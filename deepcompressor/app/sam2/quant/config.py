# -*- coding: utf-8 -*-
"""SAM2 Quantization config for SVDQuant W4A4."""

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

from ..cache.config import SAM2QuantCacheConfig
from ..dataset.calib import SAM2CalibConfig
from .quantizer.config import SAM2ModuleQuantizerConfig

__all__ = ["SAM2QuantConfig"]


@configclass
@dataclass(kw_only=True)
class SAM2QuantConfig(SAM2ModuleQuantizerConfig):
    """SAM2 model quantization configuration for SVDQuant W4A4.

    This configuration is specifically designed for W4A4 quantization with SVDQuant
    low-rank compensation.

    Args:
        wgts (`SAM2WeightQuantizerConfig`):
            The weight quantization configuration (4-bit).
        ipts (`SAM2ActivationQuantizerConfig`):
            The input activation quantization configuration (4-bit).
        opts (`SAM2ActivationQuantizerConfig`):
            The output activation quantization configuration.
        calib (`SAM2CalibConfig`):
            The calibration dataset configuration.
        smooth (`SmoothTransfomerConfig` or `None`, *optional*, defaults to `None`):
            The smooth quantization configuration.
        rotation (`QuantRotationConfig` or `None`, *optional*, defaults to `None`):
            The rotation configuration for quantization.
        develop_dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            The development data type for calibration.
    """

    calib: SAM2CalibConfig
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
        self.organize()
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
    ) -> SAM2QuantCacheConfig:
        """Generate the cache paths for the module quantization configuration."""
        quant_names = self.generate_dirnames(default_dtype=default_dtype)
        if self.enabled_wgts and self.wgts.enabled_low_rank:
            from deepcompressor.quantizer.config import QuantLowRankConfig
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
            acts_dirpath = os.path.join("acts", *quant_names)
        cache_dirpath = SAM2QuantCacheConfig(
            smooth=smooth_dirpath, branch=branch_dirpath, wgts=wgts_dirpath, acts=acts_dirpath
        ).simplify(type(self)._key_map)
        cache_dirpath = cache_dirpath.add_parent_dirs(*self.calib.generate_dirnames())
        cache_dirpath = cache_dirpath.add_parent_dirs(root, "sam2", "cache", "quant")
        return cache_dirpath

    def generate_default_dirname(self) -> str:
        """Generate output directory name for SAM2 quantization."""
        key_map = type(self)._key_map

        def simplify_skips(skips):
            return set(
                SAM2QuantCacheConfig.simplify_path("skip.[{}]".format("+".join(skips)), key_map=key_map)[
                    6:-1
                ].split("+")
            )

        name = ""
        # Low-rank (SVDQuant) naming
        if self.enabled_wgts and self.wgts.enabled_low_rank:
            from deepcompressor.utils.common import num2str
            from deepcompressor.calib.config import SearchBasedCalibObjective

            name += f"-svdquant.r{num2str(self.wgts.low_rank.rank)}"
            if self.wgts.low_rank.num_iters > 1:
                name += f".i{num2str(self.wgts.low_rank.num_iters)}"
                if self.wgts.low_rank.early_stop:
                    name += ".e"
            if self.wgts.low_rank.exclusive:
                name += ".s"
            if self.wgts.low_rank.compensate:
                name += ".c"
            if self.wgts.low_rank.objective != SearchBasedCalibObjective.OutputsError:
                name += f".{self.wgts.low_rank.objective.name}"

        # Smooth quant naming
        if self.enabled_smooth:
            name += "-smooth"
            if self.enabled_smooth_proj:
                name += ".proj"
            if self.enabled_smooth_attn:
                name += ".attn"

        # GPTQ naming
        if self.enabled_wgts and self.wgts.enabled_gptq:
            name += "-gptq"

        name = name[1:] if name else "default"
        name += f"-{self.calib.generate_dirnames()[0]}"
        return name

    @classmethod
    def set_key_map(cls, key_map: dict[str, set[str]]) -> None:
        """Set the key map for the SAM2 quantization configuration."""
        cls._key_map = key_map

    def organize(self) -> dict[str, bool]:
        """Organize the flags for the SAM2 model quantization configuration."""
        key_map = getattr(type(self), '_key_map', {})
        if not key_map:
            return {}

        wgts_skip_set, ipts_skip_set = set(), set()
        if self.wgts is not None:
            wgts_skips = []
            for skip in self.wgts.skips:
                if skip in key_map:
                    wgts_skips.extend(list(key_map[skip]))
                else:
                    wgts_skips.append(skip)
            wgts_skip_set = set(wgts_skips)
            self.wgts.skips = sorted(wgts_skip_set)
            if self.wgts.kernel_gptq is not None:
                wgts_kernel_gptq_skips = []
                for skip in self.wgts.kernel_gptq.skips:
                    if skip in key_map:
                        wgts_kernel_gptq_skips.extend(list(key_map[skip]))
                    else:
                        wgts_kernel_gptq_skips.append(skip)
                self.wgts.kernel_gptq.skips = sorted(set(wgts_kernel_gptq_skips) - wgts_skip_set)
            if self.wgts.low_rank is not None:
                wgts_low_rank_skips = []
                for skip in self.wgts.low_rank.skips:
                    if skip in key_map:
                        wgts_low_rank_skips.extend(list(key_map[skip]))
                    else:
                        wgts_low_rank_skips.append(skip)
                self.wgts.low_rank.skips = sorted(set(wgts_low_rank_skips) - wgts_skip_set)
            if self.wgts.calib_range is not None:
                wgts_calib_range_skips = []
                for skip in self.wgts.calib_range.skips:
                    if skip in key_map:
                        wgts_calib_range_skips.extend(list(key_map[skip]))
                    else:
                        wgts_calib_range_skips.append(skip)
                self.wgts.calib_range.skips = sorted(set(wgts_calib_range_skips) - wgts_skip_set)
        if self.ipts is not None:
            ipts_skips = []
            for skip in self.ipts.skips:
                if skip in key_map:
                    ipts_skips.extend(list(key_map[skip]))
                else:
                    ipts_skips.append(skip)
            ipts_skip_set = set(ipts_skips)
            self.ipts.skips = sorted(ipts_skip_set)
            if self.ipts.calib_range is not None:
                ipts_calib_range_skips = []
                for skip in self.ipts.calib_range.skips:
                    if skip in key_map:
                        ipts_calib_range_skips.extend(list(key_map[skip]))
                    else:
                        ipts_calib_range_skips.append(skip)
                self.ipts.calib_range.skips = sorted(set(ipts_calib_range_skips) - ipts_skip_set)
        if self.smooth is not None and self.smooth.proj is not None:
            smooth_proj_skips = []
            for skip in self.smooth.proj.skips:
                if skip in key_map:
                    smooth_proj_skips.extend(list(key_map[skip]))
                else:
                    smooth_proj_skips.append(skip)
            self.smooth.proj.skips = sorted(set(smooth_proj_skips) - (wgts_skip_set & ipts_skip_set))
        if self.rotation is not None:
            rotation_transforms = []
            for transform in self.rotation.transforms:
                if transform in key_map:
                    rotation_transforms.extend(list(key_map[transform]))
                else:
                    rotation_transforms.append(transform)
            self.rotation.transforms = sorted(set(rotation_transforms))
        return {}
