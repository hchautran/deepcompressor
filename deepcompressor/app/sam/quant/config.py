# -*- coding: utf-8 -*-
"""SAM quantization configuration."""

from dataclasses import dataclass

from omniconfig import configclass

from deepcompressor.utils.config import EnableConfig

from .quantizer import SamActivationQuantizerConfig, SamModuleQuantizerConfig, SamWeightQuantizerConfig

__all__ = [
    "SamCalibConfig",
    "SamQuantConfig",
    "SamRotationConfig",
    "SamSmoothConfig",
]


@configclass
@dataclass
class SamCalibConfig(EnableConfig):
    """SAM calibration configuration.

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
class SamRotationConfig(EnableConfig):
    """SAM rotation configuration for quantization.

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
class SamSmoothConfig(EnableConfig):
    """SAM smooth quantization configuration.

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
class SamQuantConfig(EnableConfig):
    """SAM quantization configuration.

    Args:
        calib (`SamCalibConfig` or `None`, *optional*, defaults to `None`):
            Calibration configuration.
        wgts (`SamWeightQuantizerConfig` or `None`, *optional*, defaults to `None`):
            Weight quantization configuration.
        ipts (`SamActivationQuantizerConfig` or `None`, *optional*, defaults to `None`):
            Input activation quantization configuration.
        opts (`SamActivationQuantizerConfig` or `None`, *optional*, defaults to `None`):
            Output activation quantization configuration.
        rotation (`SamRotationConfig` or `None`, *optional*, defaults to `None`):
            Rotation configuration.
        smooth (`SamSmoothConfig` or `None`, *optional*, defaults to `None`):
            Smooth quantization configuration.
    """

    calib: SamCalibConfig | None = None
    wgts: SamWeightQuantizerConfig | None = None
    ipts: SamActivationQuantizerConfig | None = None
    opts: SamActivationQuantizerConfig | None = None
    rotation: SamRotationConfig | None = None
    smooth: SamSmoothConfig | None = None

    def __post_init__(self):
        """Initialize default configurations."""
        if self.calib is None:
            self.calib = SamCalibConfig()
        if self.wgts is None:
            self.wgts = SamWeightQuantizerConfig()
        if self.ipts is None:
            self.ipts = SamActivationQuantizerConfig()
        if self.opts is None:
            self.opts = SamActivationQuantizerConfig()
        if self.rotation is None:
            self.rotation = SamRotationConfig()
        if self.smooth is None:
            self.smooth = SamSmoothConfig()

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
