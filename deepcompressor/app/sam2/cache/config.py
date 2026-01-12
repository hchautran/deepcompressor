# -*- coding: utf-8 -*-
"""SAM2 quantization cache configuration."""

import functools
import re
import typing as tp
from dataclasses import dataclass, field

from omniconfig import configclass

from deepcompressor.utils.config.path import BasePathConfig

from ..nn.struct import Sam2ModelStruct

__all__ = ["Sam2QuantCacheConfig", "Sam2PtqCacheConfig"]


@dataclass
class Sam2QuantCacheConfig(BasePathConfig):
    """SAM2 model quantization cache path.

    Args:
        smooth (`str`, *optional*, default=`""`):
            The smoothing scales cache path.
        branch (`str`, *optional*, default=`""`):
            The low-rank branches cache path.
        wgts (`str`, *optional*, default=`""`):
            The weight quantizers state dict cache path.
        acts (`str`, *optional*, default=`""`):
            The activation quantizers state dict cache path.
    """

    smooth: str = ""
    branch: str = ""
    wgts: str = ""
    acts: str = ""

    @staticmethod
    def simplify_path(path: str, key_map: dict[str, set[str]]) -> str:
        """Simplify the cache path."""
        to_replace = {}
        for part in re.finditer(r"(skip|include)\.\[[a-zA-Z0-9_\+]+\]", path):
            part = part.group(0)
            if part[0] == "s":
                prefix, keys = part[:4], part[6:-1]
            else:
                prefix, keys = part[:7], part[9:-1]
            keys = "+".join(
                (
                    "".join((s[0] for s in x.split("_")))
                    for x in Sam2ModelStruct._simplify_keys(keys.split("+"), key_map=key_map)
                )
            )
            to_replace[part] = f"{prefix}.[{keys}]"
        for key, value in to_replace.items():
            path = path.replace(key, value)
        return path

    def simplify(self, key_map: dict[str, set[str]]) -> tp.Self:
        """Simplify the cache paths."""
        return self.apply(functools.partial(self.simplify_path, key_map=key_map))


@configclass
@dataclass
class Sam2PtqCacheConfig:
    """SAM2 PTQ cache configuration.

    Args:
        root (`str`):
            The root directory for caching.
    """

    root: str
    dirpath: Sam2QuantCacheConfig = field(init=False)
    path: Sam2QuantCacheConfig = field(init=False)
