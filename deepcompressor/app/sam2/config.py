# -*- coding: utf-8 -*-
"""Top-level config of post-training quantization for SAM2."""

import os
from dataclasses import dataclass, field

import omniconfig
import torch
import torch.nn as nn
from omniconfig import ConfigParser, configclass

from deepcompressor.data.utils.dtype import eval_dtype
from deepcompressor.utils.config.output import OutputConfig

from .cache import Sam2PtqCacheConfig, Sam2QuantCacheConfig
from .nn.struct import Sam2ModelStruct
from .quant import Sam2QuantConfig

__all__ = [
    "Sam2ModelConfig",
    "Sam2PtqRunConfig",
    "Sam2PtqCacheConfig",
    "Sam2QuantCacheConfig",
    "Sam2QuantConfig",
]


SAM2_NAME_TO_HF_ID = {
    "tiny": "facebook/sam2.1-hiera-tiny",
    "small": "facebook/sam2.1-hiera-small",
    "base-plus": "facebook/sam2.1-hiera-base-plus",
    "base": "facebook/sam2.1-hiera-base-plus",
    "large": "facebook/sam2.1-hiera-large",
}


@configclass
@dataclass
class Sam2ModelConfig:
    """SAM2 model configuration."""

    name: str
    device: str = "cuda"
    dtype: torch.dtype = field(
        default_factory=lambda s=torch.float16: eval_dtype(s, with_quant_dtype=False, with_none=False)
    )
    hf_id: str = ""
    config: str = ""
    ckpt_path: str = ""
    image_size: int = 1024

    def resolve_hf_id(self) -> str:
        if self.hf_id:
            return self.hf_id
        return SAM2_NAME_TO_HF_ID.get(self.name, self.name)

    def build(self) -> nn.Module:
        def import_builders():
            from sam2.build_sam import build_sam2, build_sam2_hf

            return build_sam2, build_sam2_hf

        def init_hydra_if_needed():
            from hydra.core.global_hydra import GlobalHydra
            from hydra import initialize_config_dir

            if GlobalHydra().is_initialized():
                return
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
            sam2_root = os.path.join(repo_root, "sam2")
            config_dir = os.path.join(sam2_root, "sam2")
            initialize_config_dir(config_dir=config_dir, version_base=None, job_name="sam2_ptq")

        try:
            build_sam2, build_sam2_hf = import_builders()
        except RuntimeError as exc:
            msg = str(exc)
            if "parent directory of the sam2 repository" not in msg:
                raise
            # Force-import the sam2 package from the repo's package dir to avoid shadowing.
            import importlib
            import sys

            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
            sam2_pkg = os.path.join(repo_root, "sam2", "sam2")
            if sam2_pkg not in sys.path:
                sys.path.insert(0, sam2_pkg)
            sys.modules.pop("sam2", None)
            sys.modules.pop("sam2.build_sam", None)
            build_sam2, build_sam2_hf = import_builders()
        init_hydra_if_needed()

        if self.config or self.ckpt_path:
            model = build_sam2(
                config_file=self.config,
                ckpt_path=self.ckpt_path or None,
                device=self.device,
                mode="eval",
            )
        else:
            model = build_sam2_hf(self.resolve_hf_id(), device=self.device, mode="eval")
        model = model.to(device=self.device, dtype=self.dtype)
        return model

    def get_quant_model(self, model: nn.Module) -> nn.Module:
        return model.image_encoder.trunk


@configclass
@dataclass
class Sam2PtqRunConfig:
    """Top-level config of post-training quantization for SAM2."""

    output: OutputConfig
    model: Sam2ModelConfig
    quant: Sam2QuantConfig = field(metadata={omniconfig.ARGPARSE_KWARGS: {"prefix": ""}})
    cache: Sam2PtqCacheConfig = None
    seed: int = 12345
    load_from: str = ""
    save_model: str = ""
    copy_on_save: bool = False

    def __post_init__(self) -> None:
        if self.quant.calib.path:
            self.quant.calib.path = os.path.abspath(os.path.expanduser(self.quant.calib.path))
        self.quant.calib.image_size = self.model.image_size
        if self.cache is not None:
            if self.quant.is_enabled():
                self.cache.dirpath = self.quant.generate_cache_dirpath(
                    root=self.cache.root, shift=False, default_dtype=self.model.dtype
                )
                self.cache.path = self.cache.dirpath.clone().add_children(f"{self.model.name}.pt")
            else:
                self.cache.dirpath = self.cache.path = None
        if self.output.dirname == "default":
            self.output.dirname = self.model.name
        self.output.dirpath = os.path.join(self.output.root, self.output.dirname)
        torch.manual_seed(self.seed)

    @classmethod
    def get_parser(cls) -> ConfigParser:
        parser = ConfigParser("SAM2 Run configuration")
        Sam2QuantConfig.set_key_map(Sam2ModelStruct._get_default_key_map())
        parser.add_config(cls)
        return parser
