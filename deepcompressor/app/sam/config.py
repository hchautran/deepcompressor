# -*- coding: utf-8 -*-
"""SAM configuration."""

from dataclasses import dataclass

from omniconfig import configclass

from .cache import SamPtqCacheConfig
from .quant.config import SamQuantConfig

__all__ = ["SamPtqRunConfig", "SamPtqConfig"]


@configclass
@dataclass
class SamPtqRunConfig:
    """SAM PTQ run configuration.

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
class SamPtqConfig:
    """SAM PTQ configuration combining all settings.

    Args:
        quant (`SamQuantConfig`):
            Quantization configuration.
        cache (`SamPtqCacheConfig` or `None`, *optional*, defaults to `None`):
            Cache configuration.
        run (`SamPtqRunConfig` or `None`, *optional*, defaults to `None`):
            Run configuration.
    """

    quant: SamQuantConfig
    cache: SamPtqCacheConfig | None = None
    run: SamPtqRunConfig | None = None

    def __post_init__(self):
        """Initialize default configurations."""
        if self.cache is None:
            self.cache = SamPtqCacheConfig()
        if self.run is None:
            self.run = SamPtqRunConfig()
