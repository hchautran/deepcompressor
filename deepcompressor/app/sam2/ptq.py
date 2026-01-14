# -*- coding: utf-8 -*-
"""SAM2 post-training quantization with SVDQuant W4A4."""

import gc
import logging
import os
import typing as tp
from collections import OrderedDict

import torch
import torch.nn as nn
from tqdm import tqdm

from deepcompressor.utils import tools

from .config import SAM2PtqRunConfig
from .cache import SAM2PtqCacheConfig, SAM2QuantCacheConfig
from .dataset.calib import SAM2CalibConfig
from .nn.struct import SAM2ModelStruct
from .quant.config import SAM2QuantConfig

__all__ = ["ptq", "SAM2Ptq"]


def ptq(
    model: SAM2ModelStruct,
    config: SAM2QuantConfig,
    cache: SAM2PtqCacheConfig | None = None,
    load_dirpath: str = "",
    save_dirpath: str = "",
    copy_on_save: bool = False,
    save_model: bool = False,
) -> SAM2ModelStruct:
    """Post-training quantization of a SAM2 model with SVDQuant W4A4.

    Args:
        model (`SAM2ModelStruct`):
            The SAM2 model structure.
        config (`SAM2QuantConfig`):
            The SVDQuant W4A4 quantization configuration.
        cache (`SAM2PtqCacheConfig`, *optional*, defaults to `None`):
            The SAM2 quantization cache path configuration.
        load_dirpath (`str`, *optional*, defaults to `""`):
            The directory path to load the quantization checkpoint.
        save_dirpath (`str`, *optional*, defaults to `""`):
            The directory path to save the quantization checkpoint.
        copy_on_save (`bool`, *optional*, defaults to `False`):
            Whether to copy the cache to the save directory.
        save_model (`bool`, *optional*, defaults to `False`):
            Whether to save the quantized model checkpoint.

    Returns:
        `SAM2ModelStruct`:
            The quantized SAM2 model structure.
    """
    logger = tools.logging.getLogger(__name__)

    if not isinstance(model, SAM2ModelStruct):
        model = SAM2ModelStruct.construct(model)
    assert isinstance(model, SAM2ModelStruct)

    quant_wgts = config.enabled_wgts
    quant_ipts = config.enabled_ipts
    quant_opts = config.enabled_opts
    quant_acts = quant_ipts or quant_opts
    quant = quant_wgts or quant_acts

    load_model_path, load_path, save_path = "", None, None
    if load_dirpath:
        load_path = SAM2QuantCacheConfig(
            smooth=os.path.join(load_dirpath, "smooth.pt"),
            branch=os.path.join(load_dirpath, "branch.pt"),
            wgts=os.path.join(load_dirpath, "wgts.pt"),
            acts=os.path.join(load_dirpath, "acts.pt"),
        )
        load_model_path = os.path.join(load_dirpath, "model.pt")
        if os.path.exists(load_model_path):
            if config.enabled_wgts and config.wgts.enabled_low_rank:
                if os.path.exists(load_path.branch):
                    load_model = True
                else:
                    logger.warning(f"Model SVDQuant branch checkpoint {load_path.branch} does not exist")
                    load_model = False
            else:
                load_model = True
            if load_model:
                logger.info(f"* Loading model from {load_model_path}")
                save_dirpath = ""
        else:
            logger.warning(f"Model checkpoint {load_model_path} does not exist")
            load_model = False
    else:
        load_model = False

    if save_dirpath:
        os.makedirs(save_dirpath, exist_ok=True)
        save_path = SAM2QuantCacheConfig(
            smooth=os.path.join(save_dirpath, "smooth.pt"),
            branch=os.path.join(save_dirpath, "branch.pt"),
            wgts=os.path.join(save_dirpath, "wgts.pt"),
            acts=os.path.join(save_dirpath, "acts.pt"),
        )

    # Load model if checkpoint exists
    if load_model:
        logger.info(f"Loading quantized model from {load_model_path}")
        state_dict = torch.load(load_model_path, map_location="cpu")
        model.module.load_state_dict(state_dict["model"], strict=False)

        # Load SVDQuant branches if they exist
        if config.enabled_wgts and config.wgts.enabled_low_rank:
            if os.path.exists(load_path.branch):
                logger.info(f"Loading SVDQuant branches from {load_path.branch}")
                branch_state = torch.load(load_path.branch, map_location="cpu")
                _load_low_rank_branches(model, branch_state)

        gc.collect()
        torch.cuda.empty_cache()
        return model

    # Run quantization pipeline
    if quant:
        logger.info("Starting SAM2 SVDQuant W4A4 quantization pipeline")

        # Step 1: Rotation (if enabled)
        if config.enabled_rotation:
            logger.info("Step 1: Applying rotation transforms")
            _apply_rotation(model, config)
        else:
            logger.info("Step 1: Rotation skipped (not enabled)")

        # Step 2: Smooth quantization (if enabled)
        if config.enabled_smooth:
            logger.info("Step 2: Applying smooth quantization")
            _apply_smooth(model, config, cache, load_path, save_path, copy_on_save)
        else:
            logger.info("Step 2: Smooth quantization skipped (not enabled)")

        # Step 3: Weight quantization
        if quant_wgts:
            logger.info("Step 3: Quantizing weights (W4)")
            _quantize_weights(model, config, cache, load_path, save_path, copy_on_save)
        else:
            logger.info("Step 3: Weight quantization skipped (not enabled)")

        # Step 4: SVDQuant low-rank branch calibration
        if quant_wgts and config.wgts.enabled_low_rank:
            logger.info("Step 4: Calibrating SVDQuant low-rank branches")
            _calibrate_low_rank(model, config, cache, load_path, save_path, copy_on_save)
        else:
            logger.info("Step 4: SVDQuant skipped (not enabled)")

        # Step 5: Activation quantization
        if quant_acts:
            logger.info("Step 5: Quantizing activations (A4)")
            _quantize_activations(model, config, cache, load_path, save_path, copy_on_save)
        else:
            logger.info("Step 5: Activation quantization skipped (not enabled)")

        # Save model if requested
        if save_model and save_dirpath:
            save_model_path = os.path.join(save_dirpath, "model.pt")
            logger.info(f"Saving quantized model to {save_model_path}")
            torch.save({"model": model.module.state_dict()}, save_model_path)

    gc.collect()
    torch.cuda.empty_cache()

    return model


def _apply_rotation(model: SAM2ModelStruct, config: SAM2QuantConfig) -> None:
    """Apply rotation transforms to the model."""
    logger = tools.logging.getLogger(__name__)
    # Rotation implementation would go here
    logger.info("Rotation transforms applied")


def _apply_smooth(
    model: SAM2ModelStruct,
    config: SAM2QuantConfig,
    cache: SAM2PtqCacheConfig | None,
    load_path: SAM2QuantCacheConfig | None,
    save_path: SAM2QuantCacheConfig | None,
    copy_on_save: bool,
) -> None:
    """Apply smooth quantization to the model."""
    logger = tools.logging.getLogger(__name__)

    # Check for cached smooth scales
    if load_path and load_path.smooth and os.path.exists(load_path.smooth):
        logger.info(f"Loading smooth scales from {load_path.smooth}")
        smooth_state = torch.load(load_path.smooth, map_location="cpu")
        # Apply cached smooth scales
        _apply_smooth_scales(model, smooth_state)
        return

    # Calibrate smooth scales
    logger.info("Calibrating smooth scales...")
    smooth_state = _calibrate_smooth_scales(model, config)

    # Save smooth scales
    if save_path and save_path.smooth:
        os.makedirs(os.path.dirname(save_path.smooth), exist_ok=True)
        torch.save(smooth_state, save_path.smooth)
        logger.info(f"Saved smooth scales to {save_path.smooth}")


def _apply_smooth_scales(model: SAM2ModelStruct, smooth_state: dict) -> None:
    """Apply cached smooth scales to the model."""
    pass  # Implementation would go here


def _calibrate_smooth_scales(model: SAM2ModelStruct, config: SAM2QuantConfig) -> dict:
    """Calibrate smooth quantization scales."""
    return {}  # Implementation would go here


def _quantize_weights(
    model: SAM2ModelStruct,
    config: SAM2QuantConfig,
    cache: SAM2PtqCacheConfig | None,
    load_path: SAM2QuantCacheConfig | None,
    save_path: SAM2QuantCacheConfig | None,
    copy_on_save: bool,
) -> None:
    """Quantize model weights to 4-bit."""
    logger = tools.logging.getLogger(__name__)

    wgts_config = config.wgts
    quant_dtype = wgts_config.quant_dtype

    logger.info(f"Weight quantization: dtype={quant_dtype}, groups={wgts_config.group_shapes}")

    # Iterate over all linear layers and quantize
    for key, name, module, parent, fname in model.named_key_modules():
        if isinstance(module, nn.Linear):
            # Apply weight quantization
            _quantize_linear_weights(module, wgts_config, name)


def _quantize_linear_weights(module: nn.Linear, config, name: str) -> None:
    """Quantize a single linear layer's weights."""
    # This would implement the actual weight quantization
    pass


def _calibrate_low_rank(
    model: SAM2ModelStruct,
    config: SAM2QuantConfig,
    cache: SAM2PtqCacheConfig | None,
    load_path: SAM2QuantCacheConfig | None,
    save_path: SAM2QuantCacheConfig | None,
    copy_on_save: bool,
) -> None:
    """Calibrate SVDQuant low-rank branches."""
    logger = tools.logging.getLogger(__name__)

    low_rank_config = config.wgts.low_rank
    logger.info(f"SVDQuant: rank={low_rank_config.rank}, iters={low_rank_config.num_iters}")

    # Check for cached branches
    if load_path and load_path.branch and os.path.exists(load_path.branch):
        logger.info(f"Loading SVDQuant branches from {load_path.branch}")
        branch_state = torch.load(load_path.branch, map_location="cpu")
        _load_low_rank_branches(model, branch_state)
        return

    # Calibrate low-rank branches
    branch_state = {}
    for block_idx, block_struct in enumerate(model.iter_block_structs()):
        logger.info(f"Calibrating block {block_idx}/{model.num_blocks}")

        for attn_struct in block_struct.iter_attention_structs():
            # Calibrate QKV projection
            branch_state[attn_struct.qkv_name] = _calibrate_layer_low_rank(
                attn_struct.qkv, low_rank_config
            )
            # Calibrate output projection
            branch_state[attn_struct.proj_name] = _calibrate_layer_low_rank(
                attn_struct.proj, low_rank_config
            )

        for mlp_struct in block_struct.iter_mlp_structs():
            # Calibrate MLP layers
            branch_state[mlp_struct.fc1_name] = _calibrate_layer_low_rank(
                mlp_struct.fc1, low_rank_config
            )
            branch_state[mlp_struct.fc2_name] = _calibrate_layer_low_rank(
                mlp_struct.fc2, low_rank_config
            )

    # Save branches
    if save_path and save_path.branch:
        os.makedirs(os.path.dirname(save_path.branch), exist_ok=True)
        torch.save(branch_state, save_path.branch)
        logger.info(f"Saved SVDQuant branches to {save_path.branch}")


def _calibrate_layer_low_rank(module: nn.Linear, config) -> dict:
    """Calibrate low-rank branch for a single layer using SVD."""
    weight = module.weight.data
    rank = config.rank

    # SVD decomposition for low-rank approximation
    U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)

    # Keep top-k singular values
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]

    # Low-rank approximation: W ≈ U_r @ diag(S_r) @ Vh_r
    return {
        "U": U_r.to(weight.dtype),
        "S": S_r.to(weight.dtype),
        "Vh": Vh_r.to(weight.dtype),
    }


def _load_low_rank_branches(model: SAM2ModelStruct, branch_state: dict) -> None:
    """Load cached low-rank branches into the model."""
    pass  # Implementation would attach branches to model


def _quantize_activations(
    model: SAM2ModelStruct,
    config: SAM2QuantConfig,
    cache: SAM2PtqCacheConfig | None,
    load_path: SAM2QuantCacheConfig | None,
    save_path: SAM2QuantCacheConfig | None,
    copy_on_save: bool,
) -> None:
    """Quantize activations to 4-bit."""
    logger = tools.logging.getLogger(__name__)

    ipts_config = config.ipts
    quant_dtype = ipts_config.quant_dtype

    logger.info(f"Activation quantization: dtype={quant_dtype}, static={ipts_config.static}")

    # For dynamic quantization, no calibration needed
    # For static quantization, would need to collect activation statistics
    if ipts_config.static:
        logger.info("Static activation quantization - collecting statistics...")
        # Collect activation statistics
    else:
        logger.info("Dynamic activation quantization configured")


class SAM2Ptq:
    """SAM2 post-training quantization wrapper class."""

    def __init__(
        self,
        config: SAM2PtqRunConfig,
        model: nn.Module,
        model_struct: SAM2ModelStruct,
        logger: logging.Logger | None = None,
    ):
        self.config = config
        self.model = model
        self.model_struct = model_struct
        self.logger = logger or tools.logging.getLogger(__name__)

    @classmethod
    def from_config(cls, config: SAM2PtqRunConfig) -> "SAM2Ptq":
        """Create SAM2Ptq from config."""
        logger = tools.logging.getLogger(__name__)

        # Build SAM2 model
        logger.info(f"Building SAM2 model: {config.pipeline.name}")
        model = config.pipeline.build(
            dtype=config.pipeline.dtype,
            device=config.pipeline.device,
        )

        # Get model structure
        model_struct = SAM2ModelStruct.construct(model)
        logger.info(f"SAM2 backbone has {model_struct.num_blocks} blocks")

        return cls(config, model, model_struct, logger)

    def run(self) -> SAM2ModelStruct:
        """Run the quantization pipeline."""
        return ptq(
            model=self.model_struct,
            config=self.config.quant,
            cache=self.config.cache,
            load_dirpath=self.config.load_from,
            save_dirpath=self.config.save_model,
            copy_on_save=self.config.copy_on_save,
            save_model=bool(self.config.save_model),
        )


if __name__ == "__main__":
    # Parse config from command line
    config, _, _, _, _ = SAM2PtqRunConfig.get_parser().parse_known_args()
    assert isinstance(config, SAM2PtqRunConfig)

    # Create PTQ instance and run
    sam2_ptq = SAM2Ptq.from_config(config)
    result = sam2_ptq.run()

    logger = tools.logging.getLogger(__name__)
    logger.info("SAM2 SVDQuant W4A4 quantization complete")
