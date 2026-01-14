# -*- coding: utf-8 -*-
"""SAM2 pipeline configuration module."""

import typing as tp
from dataclasses import dataclass, field

import torch
from omniconfig import configclass

from deepcompressor.data.utils.dtype import eval_dtype
from deepcompressor.utils import tools

__all__ = ["SAM2PipelineConfig"]


# Model ID to config file mappings
SAM2_MODEL_CONFIGS = {
    "sam2-hiera-tiny": "configs/sam2/sam2_hiera_t.yaml",
    "sam2-hiera-small": "configs/sam2/sam2_hiera_s.yaml",
    "sam2-hiera-base-plus": "configs/sam2/sam2_hiera_b+.yaml",
    "sam2-hiera-large": "configs/sam2/sam2_hiera_l.yaml",
    "sam2.1-hiera-tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2.1-hiera-small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2.1-hiera-base-plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "sam2.1-hiera-large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}

# HuggingFace model IDs
HF_MODEL_IDS = {
    "sam2-hiera-tiny": "facebook/sam2-hiera-tiny",
    "sam2-hiera-small": "facebook/sam2-hiera-small",
    "sam2-hiera-base-plus": "facebook/sam2-hiera-base-plus",
    "sam2-hiera-large": "facebook/sam2-hiera-large",
    "sam2.1-hiera-tiny": "facebook/sam2.1-hiera-tiny",
    "sam2.1-hiera-small": "facebook/sam2.1-hiera-small",
    "sam2.1-hiera-base-plus": "facebook/sam2.1-hiera-base-plus",
    "sam2.1-hiera-large": "facebook/sam2.1-hiera-large",
}


@configclass
@dataclass
class SAM2PipelineConfig:
    """SAM2 pipeline configuration.

    Args:
        name (`str`):
            The name/variant of the SAM2 model.
        checkpoint (`str`, *optional*):
            Path to the model checkpoint. If not provided, downloads from HuggingFace.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            The data type of the model.
        device (`str`, *optional*, defaults to `"cuda"`):
            The device to load the model on.
    """

    name: str
    checkpoint: str = ""
    dtype: torch.dtype = field(
        default_factory=lambda s=torch.float32: eval_dtype(s, with_quant_dtype=False, with_none=False)
    )
    device: str = "cuda"
    family: str = field(init=False)

    def __post_init__(self):
        self.family = "sam2"
        # Normalize name
        if self.name.startswith("facebook/"):
            # Convert HF model ID to short name
            for short_name, hf_id in HF_MODEL_IDS.items():
                if hf_id == self.name:
                    self.name = short_name
                    break

    def build(
        self,
        *,
        dtype: str | torch.dtype | None = None,
        device: str | torch.device | None = None,
        mode: str = "eval",
    ) -> tp.Any:
        """Build the SAM2 model.

        Args:
            dtype (`str` or `torch.dtype`, *optional*):
                The data type of the model.
            device (`str` or `torch.device`, *optional*):
                The device to load the model on.
            mode (`str`, *optional*, defaults to `"eval"`):
                Whether to set model to eval mode.

        Returns:
            SAM2 model instance.
        """
        logger = tools.logging.getLogger(__name__)

        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device

        # Try to import sam2
        try:
            from sam2.build_sam import build_sam2, build_sam2_hf
        except ImportError:
            raise ImportError(
                "sam2 package not found. Please install it from "
                "https://github.com/facebookresearch/sam2"
            )

        if self.checkpoint:
            # Load from local checkpoint
            if self.name not in SAM2_MODEL_CONFIGS:
                raise ValueError(
                    f"Unknown SAM2 model: {self.name}. "
                    f"Available models: {list(SAM2_MODEL_CONFIGS.keys())}"
                )
            config_file = SAM2_MODEL_CONFIGS[self.name]
            logger.info(f"Loading SAM2 model {self.name} from {self.checkpoint}")
            model = build_sam2(
                config_file=config_file,
                ckpt_path=self.checkpoint,
                device=device,
                mode=mode,
            )
        else:
            # Load from HuggingFace
            if self.name not in HF_MODEL_IDS:
                raise ValueError(
                    f"Unknown SAM2 model: {self.name}. "
                    f"Available models: {list(HF_MODEL_IDS.keys())}"
                )
            hf_model_id = HF_MODEL_IDS[self.name]
            logger.info(f"Loading SAM2 model from HuggingFace: {hf_model_id}")
            model = build_sam2_hf(
                model_id=hf_model_id,
                device=device,
                mode=mode,
            )

        # Convert to specified dtype
        if dtype != torch.float32:
            model = model.to(dtype=dtype)

        return model

    def get_backbone(self, model: tp.Any) -> tp.Any:
        """Extract the Hiera backbone from a SAM2 model.

        Args:
            model: SAM2 model instance.

        Returns:
            Hiera backbone module.
        """
        if hasattr(model, "model"):
            # SAM2VideoPredictor or SAM2ImagePredictor
            model = model.model

        if hasattr(model, "image_encoder"):
            image_encoder = model.image_encoder
            if hasattr(image_encoder, "trunk"):
                return image_encoder.trunk
            return image_encoder
        elif hasattr(model, "trunk"):
            return model.trunk
        elif hasattr(model, "blocks"):
            return model
        else:
            raise ValueError(f"Cannot find Hiera backbone in model: {type(model)}")
