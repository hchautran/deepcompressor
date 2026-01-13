# -*- coding: utf-8 -*-
"""SAM2 quantization run configuration."""

import os
from dataclasses import dataclass, field

import torch
from omniconfig import configclass

from deepcompressor.utils import tools
from deepcompressor.utils.config.output import  OutputConfig
# from deepcompressor.utils.config import FieldRegistry, OutputConfig

from .cache.config import Sam2PtqCacheConfig
from .nn.struct import Sam2ModelStruct
from .quant.config import Sam2QuantConfig

__all__ = ["Sam2PtqRunConfig", "Sam2ModelConfig"]


SAM2_MODELS = {
    "tiny": "facebook/sam2.1-hiera-tiny",
    "small": "facebook/sam2.1-hiera-small",
    "base": "facebook/sam2.1-hiera-base-plus",
    "base-plus": "facebook/sam2.1-hiera-base-plus",
    "large": "facebook/sam2.1-hiera-large",
}


@configclass
@dataclass
class Sam2ModelConfig:
    """SAM2 model configuration.

    Args:
        name (`str`, *optional*, defaults to `"tiny"`):
            The model name or shortcut (tiny, small, base, large).
        path (`str`, *optional*, defaults to `""`):
            The HuggingFace model path (overrides name).
        device (`str`, *optional*, defaults to `"cuda"`):
            The device to load the model on.
        dtype (`str`, *optional*, defaults to `"float16"`):
            The data type for the model.
    """

    name: str = "tiny"
    path: str = ""
    device: str = "cuda"
    dtype: str = "float16"

    def __post_init__(self) -> None:
        if not self.path:
            self.path = SAM2_MODELS.get(self.name.lower(), self.name)

    @property
    def torch_dtype(self) -> torch.dtype:
        """Get the torch dtype."""
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
        }
        return dtype_map.get(self.dtype.lower(), torch.float16)

    def build(self) -> tuple:
        """Build the SAM2 model and processor.

        Returns:
            Tuple of (model, processor, model_struct).
        """
        logger = tools.logging.getLogger(__name__)
        logger.info(f"Loading SAM2 model from {self.path}")

        # try:
        from transformers import Sam2Model, Sam2Processor

        model = Sam2Model.from_pretrained(
            self.path,
            torch_dtype=self.torch_dtype,
        ).to(self.device)
        model.eval()

        processor = Sam2Processor.from_pretrained(self.path)

        model_struct = Sam2ModelStruct.construct(model)

        return model, processor, model_struct

        # except ImportError as e:
            # logger.error(f"Failed to import transformers: {e}")
            # raise
        # except Exception as e:
            # logger.error(f"Failed to load model: {e}")
            # raise


@configclass
@dataclass
class Sam2PtqRunConfig:
    """SAM2 post-training quantization run configuration.

    Args:
        model (`Sam2ModelConfig`):
            The model configuration.
        quant (`Sam2QuantConfig`):
            The quantization configuration.
        cache (`Sam2PtqCacheConfig`):
            The cache configuration.
        output (`OutputConfig`):
            The output configuration.
        seed (`int`, *optional*, defaults to `42`):
            The random seed.
        load_from (`str`, *optional*, defaults to `""`):
            The directory path to load the quantization checkpoint.
        save_model (`str`, *optional*, defaults to `""`):
            The path to save the quantized model.
        copy_on_save (`bool`, *optional*, defaults to `False`):
            Whether to copy the cache to the save directory.
    """

    model: Sam2ModelConfig
    quant: Sam2QuantConfig
    cache: Sam2PtqCacheConfig
    output: OutputConfig
    seed: int = 42
    load_from: str = ""
    save_model: str = ""
    copy_on_save: bool = False

    def __post_init__(self) -> None:
        # Set random seeds
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        if self.quant.is_enabled():
            self.cache.dirpath = self.quant.generate_cache_dirpath(
                root=self.cache.root, default_dtype=self.model.torch_dtype
            )
            self.cache.path = self.cache.dirpath.clone().add_children(f"{self.model.name}.pt")
        else:
            self.cache.dirpath = self.cache.path = None

    def main(self) -> None:
        """Run the PTQ pipeline."""
        from .ptq import ptq

        logger = tools.logging.getLogger(__name__)

        # Build model
        logger.info("=== Building SAM2 Model ===")
        model, processor, model_struct = self.model.build()

        logger.info(f"Model has {model_struct.num_blocks} blocks")

        # Run PTQ
        logger.info("=== Running Post-Training Quantization ===")

        save_dirpath = ""
        if self.save_model:
            if isinstance(self.save_model, str):
                save_model_value = self.save_model.lower()
            else:
                save_model_value = self.save_model
            if save_model_value in ("false", "none", "null", "nil", False):
                save_model = False
            elif save_model_value in ("true", "default", True):
                save_dirpath = os.path.join(self.output.running_job_dirpath, "model")
                save_model = True
            else:
                save_dirpath = self.save_model
                save_model = True
        else:
            save_model = False

        model_struct = ptq(
            model_struct,
            self.quant,
            cache=self.cache,
            load_dirpath=self.load_from,
            save_dirpath=save_dirpath if save_dirpath else os.path.join(self.output.running_job_dirpath, "cache"),
            copy_on_save=self.copy_on_save,
            save_model=save_model,
        )

        logger.info("=== PTQ Complete ===")

        return model_struct

    @classmethod
    def get_parser(cls):
        """Get the argument parser for this configuration."""
        from omniconfig import OmniConfig

        return OmniConfig.from_dataclass(cls)


# Register field converters
# FieldRegistry.register("Sam2ModelConfig", Sam2ModelConfig)
# FieldRegistry.register("Sam2QuantConfig", Sam2QuantConfig)
# FieldRegistry.register("Sam2PtqCacheConfig", Sam2PtqCacheConfig)
