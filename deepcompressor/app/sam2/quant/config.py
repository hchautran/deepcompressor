# -*- coding: utf-8 -*-
"""SAM2 quantization configuration."""

from dataclasses import dataclass

from omniconfig import configclass

from deepcompressor.utils.config import EnableConfig

from .quantizer import Sam2ActivationQuantizerConfig, Sam2ModuleQuantizerConfig, Sam2WeightQuantizerConfig

__all__ = [
    "Sam2CalibConfig",
    "Sam2QuantConfig",
    "Sam2RotationConfig",
    "Sam2SmoothConfig",
]


@configclass
@dataclass
class Sam2CalibConfig(EnableConfig):
    """SAM2 calibration configuration.

    Args:
        num_samples (`int`, *optional*, defaults to `128`):
            Number of calibration samples.
        batch_size (`int`, *optional*, defaults to `1`):
            Batch size for calibration.
    """

    num_samples: int = 128
    batch_size: int = 1

    def is_enabled(self) -> bool:
        """Check if calibration is enabled."""
        return self.num_samples > 0


@configclass
@dataclass
class Sam2RotationConfig(EnableConfig):
    """SAM2 rotation configuration for quantization.

    Args:
        enabled (`bool`, *optional*, defaults to `False`):
            Whether to enable rotation.
    """

    enabled: bool = False

    def is_enabled(self) -> bool:
        """Check if rotation is enabled."""
        return self.enabled


@configclass
@dataclass
class Sam2SmoothConfig(EnableConfig):
    """SAM2 smooth quantization configuration.

    Args:
        enabled (`bool`, *optional*, defaults to `False`):
            Whether to enable smooth quantization.
        alpha (`float`, *optional*, defaults to `0.5`):
            Smooth factor (0-1).
    """

    enabled: bool = False
    alpha: float = 0.5

    def is_enabled(self) -> bool:
        """Check if smooth quantization is enabled."""
        return self.enabled


@configclass
@dataclass
class Sam2QuantConfig(EnableConfig):
    """SAM2 quantization configuration.

    Args:
        calib (`Sam2CalibConfig` or `None`, *optional*, defaults to `None`):
            Calibration configuration.
        wgts (`Sam2WeightQuantizerConfig` or `None`, *optional*, defaults to `None`):
            Weight quantization configuration.
        ipts (`Sam2ActivationQuantizerConfig` or `None`, *optional*, defaults to `None`):
            Input activation quantization configuration.
        opts (`Sam2ActivationQuantizerConfig` or `None`, *optional*, defaults to `None`):
            Output activation quantization configuration.
        rotation (`Sam2RotationConfig` or `None`, *optional*, defaults to `None`):
            Rotation configuration.
        smooth (`Sam2SmoothConfig` or `None`, *optional*, defaults to `None`):
            Smooth quantization configuration.
    """

    calib: Sam2CalibConfig | None = None
    wgts: Sam2WeightQuantizerConfig | None = None
    ipts: Sam2ActivationQuantizerConfig | None = None
    opts: Sam2ActivationQuantizerConfig | None = None
    rotation: Sam2RotationConfig | None = None
    smooth: Sam2SmoothConfig | None = None

    def __post_init__(self):
        """Initialize default configurations."""
        if self.calib is None:
            self.calib = Sam2CalibConfig()
        if self.wgts is None:
            self.wgts = Sam2WeightQuantizerConfig()
        if self.ipts is None:
            self.ipts = Sam2ActivationQuantizerConfig()
        if self.opts is None:
            self.opts = Sam2ActivationQuantizerConfig()
        if self.rotation is None:
            self.rotation = Sam2RotationConfig()
        if self.smooth is None:
            self.smooth = Sam2SmoothConfig()

    def is_enabled(self) -> bool:
        """Check if any quantization is enabled."""
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

    @property
    def enabled_rotation(self) -> bool:
        """Whether rotation is enabled."""
        return self.rotation is not None and self.rotation.is_enabled()

    @property
    def enabled_smooth(self) -> bool:
        """Whether smooth quantization is enabled."""
        return self.smooth is not None and self.smooth.is_enabled()
