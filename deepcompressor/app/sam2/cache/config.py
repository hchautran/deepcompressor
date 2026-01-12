# -*- coding: utf-8 -*-
"""SAM2 cache configuration."""

import os
from dataclasses import dataclass

from omniconfig import configclass

__all__ = ["Sam2QuantCacheConfig", "Sam2PtqCacheConfig"]


@configclass
@dataclass
class Sam2QuantCacheConfig:
    """SAM2 quantization cache file paths.

    Args:
        smooth (`str`, *optional*, defaults to `""`):
            Path to smooth scales cache.
        branch (`str`, *optional*, defaults to `""`):
            Path to low-rank branch cache.
        wgts (`str`, *optional*, defaults to `""`):
            Path to weight quantization cache.
        acts (`str`, *optional*, defaults to `""`):
            Path to activation quantization cache.
    """

    smooth: str = ""
    branch: str = ""
    wgts: str = ""
    acts: str = ""


@configclass
@dataclass
class Sam2PtqCacheConfig:
    """SAM2 PTQ cache configuration.

    Args:
        dirpath (`str`, *optional*, defaults to `""`):
            Base directory path for cache files.
        path (`Sam2QuantCacheConfig` or `None`, *optional*, defaults to `None`):
            Specific cache file paths.
    """

    dirpath: str = ""
    path: Sam2QuantCacheConfig | None = None

    def __post_init__(self):
        """Initialize cache paths."""
        if self.path is None and self.dirpath:
            self.path = Sam2QuantCacheConfig(
                smooth=os.path.join(self.dirpath, "smooth.pt"),
                branch=os.path.join(self.dirpath, "branch.pt"),
                wgts=os.path.join(self.dirpath, "wgts.pt"),
                acts=os.path.join(self.dirpath, "acts.pt"),
            )
