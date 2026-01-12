# -*- coding: utf-8 -*-
"""SAM Quantizers."""

import typing as tp
from dataclasses import dataclass

from deepcompressor.quantizer import Quantizer

from .config import SamActivationQuantizerConfig, SamModuleQuantizerConfig, SamWeightQuantizerConfig

__all__ = ["SamQuantizer", "SamWeightQuantizer", "SamActivationQuantizer", "SamModuleQuantizer"]


@dataclass
class SamQuantizer(Quantizer):
    """Base SAM quantizer class."""

    config: SamWeightQuantizerConfig | SamActivationQuantizerConfig

    def __post_init__(self):
        super().__post_init__()


@dataclass
class SamWeightQuantizer(SamQuantizer):
    """SAM weight quantizer.

    Args:
        config (`SamWeightQuantizerConfig`):
            The weight quantizer configuration.
        key (`str`):
            The key/name of the module being quantized.
    """

    config: SamWeightQuantizerConfig

    def should_skip(self, module_name: str) -> bool:
        """Determine if this module should be skipped for quantization.

        Args:
            module_name (`str`):
                The name of the module.

        Returns:
            `bool`: Whether to skip quantization for this module.
        """
        if not self.is_enabled():
            return True

        # Skip patch embedding if configured
        if self.config.skip_patch_embed and "patch_embed" in module_name:
            return True

        # Skip first block if configured
        if self.config.skip_first_block and "blocks.0" in module_name:
            return True

        # Skip last block if configured
        # This requires knowing the total number of blocks, handled at pipeline level
        if self.config.skip_last_block and module_name.startswith("blocks.") and "blocks." in module_name:
            # Will be handled by checking block index in the pipeline
            pass

        # Skip decoder if configured
        if self.config.skip_decoder and "mask_decoder" in module_name:
            return True

        return False


@dataclass
class SamActivationQuantizer(SamQuantizer):
    """SAM activation quantizer.

    Args:
        config (`SamActivationQuantizerConfig`):
            The activation quantizer configuration.
        key (`str`):
            The key/name of the module being quantized.
    """

    config: SamActivationQuantizerConfig

    def __post_init__(self):
        super().__post_init__()
        # Set up per-token quantization if enabled
        if self.config.per_token and len(self.config.group_shapes) > 0:
            # Ensure group shape is set for per-token quantization
            # Typically (1, -1) for per-token along sequence dimension
            pass


@dataclass
class SamModuleQuantizer:
    """SAM module quantizer combining weight and activation quantizers.

    Args:
        config (`SamModuleQuantizerConfig`):
            The module quantizer configuration.
        key (`str`):
            The key/name of the module being quantized.
    """

    config: SamModuleQuantizerConfig
    key: str
    wgts: SamWeightQuantizer | None = None
    ipts: SamActivationQuantizer | None = None
    opts: SamActivationQuantizer | None = None

    def __post_init__(self):
        """Initialize weight and activation quantizers."""
        if self.config.enabled_wgts:
            self.wgts = SamWeightQuantizer(
                config=self.config.wgts,
                key=f"{self.key}.weight",
            )

        if self.config.enabled_ipts:
            self.ipts = SamActivationQuantizer(
                config=self.config.ipts,
                key=f"{self.key}.input",
            )

        if self.config.enabled_opts:
            self.opts = SamActivationQuantizer(
                config=self.config.opts,
                key=f"{self.key}.output",
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
