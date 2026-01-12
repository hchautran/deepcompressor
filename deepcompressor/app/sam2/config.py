# -*- coding: utf-8 -*-
"""SAM2 configuration."""

from dataclasses import dataclass

from omniconfig import configclass

from .cache import Sam2PtqCacheConfig
from .quant.config import Sam2QuantConfig

__all__ = ["Sam2PtqRunConfig", "Sam2PtqConfig"]


@configclass
@dataclass
class Sam2PtqRunConfig:
    """SAM2 PTQ run configuration.

    Args:
        load_dirpath (`str`, *optional*, defaults to `""`):
            Directory path to load quantization checkpoint from.
        save_dirpath (`str`, *optional*, defaults to `""`):
            Directory path to save quantization checkpoint to.
        copy_on_save (`bool`, *optional*, defaults to `False`):
            Whether to copy cache files instead of symlinking.
        save_model (`bool`, *optional*, defaults to `False`):
            Whether to save the quantized model.
    """

    load_dirpath: str = ""
    save_dirpath: str = ""
    copy_on_save: bool = False
    save_model: bool = False


@configclass
@dataclass
class Sam2PtqConfig:
    """SAM2 PTQ configuration combining all settings.

    Args:
        quant (`Sam2QuantConfig`):
            Quantization configuration.
        cache (`Sam2PtqCacheConfig` or `None`, *optional*, defaults to `None`):
            Cache configuration.
        run (`Sam2PtqRunConfig` or `None`, *optional*, defaults to `None`):
            Run configuration.
    """

    quant: Sam2QuantConfig
    cache: Sam2PtqCacheConfig | None = None
    run: Sam2PtqRunConfig | None = None

    def __post_init__(self):
        """Initialize default configurations."""
        if self.cache is None:
            self.cache = Sam2PtqCacheConfig()
        if self.run is None:
            self.run = Sam2PtqRunConfig()
