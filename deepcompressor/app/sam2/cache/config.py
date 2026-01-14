# -*- coding: utf-8 -*-
"""SAM2 quantization cache configuration."""

import functools
import re
import typing as tp
from dataclasses import dataclass, field

from omniconfig import configclass

from deepcompressor.utils.config.path import BasePathConfig

from ..nn.struct import SAM2ModelStruct

__all__ = ["SAM2QuantCacheConfig", "SAM2PtqCacheConfig"]


@dataclass
class SAM2QuantCacheConfig(BasePathConfig):
    """SAM2 model quantization cache path.

    Args:
        smooth (`str`, *optional*, default=`""`):
            The smoothing scales cache path.
        branch (`str`, *optional*, default=`""`):
            The SVDQuant low-rank branches cache path.
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
        if not key_map:
            return path
        to_replace = {}
        # Extract all parts matching the pattern "(skip|include).\[[a-zA-Z0-9_\+]+\]"
        for part in re.finditer(r"(skip|include)\.\[[a-zA-Z0-9_\+]+\]", path):
            part = part.group(0)
            if part[0] == "s":
                prefix, keys = part[:4], part[6:-1]
            else:
                prefix, keys = part[:7], part[9:-1]
            # Simplify the keys
            keys_list = keys.split("+")
            simplified = []
            for key in keys_list:
                # Try to find the key in key_map
                found = False
                for rkey, rkeys in key_map.items():
                    if key in rkeys:
                        simplified.append(rkey)
                        found = True
                        break
                if not found:
                    simplified.append(key)
            keys = "+".join(sorted(set(simplified)))
            to_replace[part] = f"{prefix}.[{keys}]"
        # Replace the parts
        for key, value in to_replace.items():
            path = path.replace(key, value)
        return path

    def simplify(self, key_map: dict[str, set[str]]) -> tp.Self:
        """Simplify the cache paths."""
        return self.apply(functools.partial(self.simplify_path, key_map=key_map))


@configclass
@dataclass
class SAM2PtqCacheConfig:
    """SAM2 post-training quantization cache configuration.

    Args:
        root (`str`):
            The root directory for cache storage.
    """

    root: str
    dirpath: SAM2QuantCacheConfig = field(init=False)
    path: SAM2QuantCacheConfig = field(init=False)
