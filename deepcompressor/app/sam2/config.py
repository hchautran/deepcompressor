# -*- coding: utf-8 -*-
"""Top-level config for SAM2 post-training quantization with SVDQuant W4A4."""

import os
from dataclasses import dataclass, field

import omniconfig
import torch
from omniconfig import ConfigParser, configclass

from deepcompressor.utils.config.output import OutputConfig

from .cache import SAM2PtqCacheConfig, SAM2QuantCacheConfig
from .nn.struct import SAM2ModelStruct
from .pipeline import SAM2PipelineConfig
from .quant import SAM2QuantConfig

__all__ = [
    "SAM2PtqRunConfig",
    "SAM2PtqCacheConfig",
    "SAM2QuantCacheConfig",
    "SAM2PipelineConfig",
    "SAM2QuantConfig",
]


@configclass
@dataclass
class SAM2PtqRunConfig:
    """Top-level config for SAM2 post-training quantization with SVDQuant W4A4.

    Args:
        cache (`SAM2PtqCacheConfig`):
            The cache configuration.
        output (`OutputConfig`):
            The output directory configuration.
        pipeline (`SAM2PipelineConfig`):
            The SAM2 pipeline configuration.
        quant (`SAM2QuantConfig`):
            The SVDQuant W4A4 quantization configuration.
        seed (`int`, *optional*, defaults to `12345`):
            The seed for reproducibility.
        load_from (`str`, *optional*, defaults to `""`):
            Directory path to load the quantization checkpoint.
        save_model (`str`, *optional*, defaults to `""`):
            Directory path to save the quantized model checkpoint.
        copy_on_save (`bool`, *optional*, defaults to `False`):
            Whether to copy the quantization cache on save.
    """

    cache: SAM2PtqCacheConfig | None
    output: OutputConfig
    pipeline: SAM2PipelineConfig
    quant: SAM2QuantConfig = field(metadata={omniconfig.ARGPARSE_KWARGS: {"prefix": ""}})
    seed: int = 12345
    load_from: str = ""
    save_model: str = ""
    copy_on_save: bool = False

    def __post_init__(self):
        # Setup calibration dataset path
        if self.quant.calib.path:
            self.quant.calib.path = os.path.abspath(os.path.expanduser(self.quant.calib.path))

        # Setup cache directory
        if self.cache is not None:
            if self.quant.enabled_wgts or self.quant.enabled_ipts or self.quant.enabled_opts:
                self.cache.dirpath = self.quant.generate_cache_dirpath(
                    root=self.cache.root, default_dtype=self.pipeline.dtype
                )
                self.cache.path = self.cache.dirpath.clone().add_children(f"{self.pipeline.name}.pt")
            else:
                self.cache.dirpath = self.cache.path = None

        # Setup output directory
        if self.output.dirname == "default":
            self.output.dirname = self.generate_default_dirname()
        calib_dirname = self.quant.generate_calib_dirname() or "-"
        self.output.dirpath = os.path.join(
            self.output.root,
            "sam2",
            self.pipeline.name,
            *self.quant.generate_dirnames(default_dtype=self.pipeline.dtype)[:-1],
            calib_dirname,
            self.output.dirname,
        )

        # Set random seed
        torch.manual_seed(self.seed)

    def generate_default_dirname(self) -> str:
        name = ""
        if self.quant.is_enabled():
            name += f"-{self.quant.generate_default_dirname()}"
        assert name[0] == "-" if name else True
        return name[1:] if name else "baseline"

    @classmethod
    def get_parser(cls) -> ConfigParser:
        """Get a parser for SAM2 post-training quantization.

        Returns:
            `ConfigParser`:
                A parser for SAM2 post-training quantization.
        """
        parser = ConfigParser("SAM2 SVDQuant W4A4 configuration")
        SAM2QuantConfig.set_key_map(SAM2ModelStruct._get_default_key_map())
        parser.add_config(cls)
        return parser
