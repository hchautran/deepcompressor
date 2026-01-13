# -*- coding: utf-8 -*-
"""SAM2 post-training quantization."""

import gc
import os

import torch

from deepcompressor.utils import tools

from .cache.config import Sam2PtqCacheConfig, Sam2QuantCacheConfig
from .nn.struct import Sam2ModelStruct
from .quant import (
    load_sam2_weights_state_dict,
    quantize_sam2_activations,
    quantize_sam2_weights,
    rotate_sam2,
    smooth_sam2,
)
from .quant.config import Sam2QuantConfig

__all__ = ["ptq"]


def ptq(
    model: Sam2ModelStruct,
    config: Sam2QuantConfig,
    cache: Sam2PtqCacheConfig | None = None,
    load_dirpath: str = "",
    save_dirpath: str = "",
    copy_on_save: bool = False,
    save_model: bool = False,
) -> Sam2ModelStruct:
    """Post-training quantization of a SAM2 model.

    Args:
        model (`Sam2ModelStruct`):
            The SAM2 model structure.
        config (`Sam2QuantConfig`):
            The SAM2 model post-training quantization configuration.
        cache (`Sam2PtqCacheConfig`, *optional*, defaults to `None`):
            The SAM2 model quantization cache path configuration.
        load_dirpath (`str`, *optional*, defaults to `""`):
            The directory path to load the quantization checkpoint.
        save_dirpath (`str`, *optional*, defaults to `""`):
            The directory path to save the quantization checkpoint.
        copy_on_save (`bool`, *optional*, defaults to `False`):
            Whether to copy the cache to the save directory.
        save_model (`bool`, *optional*, defaults to `False`):
            Whether to save the quantized model checkpoint.

    Returns:
        `Sam2ModelStruct`:
            The quantized SAM2 model.
    """
    logger = tools.logging.getLogger(__name__)

    if not isinstance(model, Sam2ModelStruct):
        model = Sam2ModelStruct.construct(model)
    assert isinstance(model, Sam2ModelStruct)

    quant_wgts = config.enabled_wgts
    quant_ipts = config.enabled_ipts
    quant_opts = config.enabled_opts
    quant_acts = quant_ipts or quant_opts
    quant = quant_wgts or quant_acts

    load_model_path, load_path, save_path = "", None, None
    if load_dirpath:
        load_path = Sam2QuantCacheConfig(
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
                    logger.warning(f"Model low-rank branch checkpoint {load_path.branch} does not exist")
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
        save_path = Sam2QuantCacheConfig(
            smooth=os.path.join(save_dirpath, "smooth.pt"),
            branch=os.path.join(save_dirpath, "branch.pt"),
            wgts=os.path.join(save_dirpath, "wgts.pt"),
            acts=os.path.join(save_dirpath, "acts.pt"),
        )
    else:
        save_model = False

    # Rotation phase
    if quant and config.enabled_rotation:
        logger.info("* Rotating model for quantization")
        tools.logging.Formatter.indent_inc()
        rotate_sam2(model, config=config)
        tools.logging.Formatter.indent_dec()
        gc.collect()
        torch.cuda.empty_cache()

    # Smooth quantization phase
    if quant and config.enabled_smooth:
        logger.info("* Smoothing model for quantization")
        tools.logging.Formatter.indent_inc()
        load_from = ""
        if load_path and os.path.exists(load_path.smooth):
            load_from = load_path.smooth
        elif cache and cache.path.smooth and os.path.exists(cache.path.smooth):
            load_from = cache.path.smooth
        if load_from:
            logger.info(f"- Loading smooth scales from {load_from}")
            smooth_cache = torch.load(load_from)
            smooth_sam2(model, config, smooth_cache=smooth_cache)
        else:
            logger.info("- Generating smooth scales")
            smooth_cache = smooth_sam2(model, config)
            if cache and cache.path.smooth:
                logger.info(f"- Saving smooth scales to {cache.path.smooth}")
                os.makedirs(cache.dirpath.smooth, exist_ok=True)
                torch.save(smooth_cache, cache.path.smooth)
                load_from = cache.path.smooth
        if save_path:
            if not copy_on_save and load_from:
                logger.info(f"- Linking smooth scales to {save_path.smooth}")
                os.symlink(os.path.relpath(load_from, save_dirpath), save_path.smooth)
            else:
                logger.info(f"- Saving smooth scales to {save_path.smooth}")
                torch.save(smooth_cache, save_path.smooth)
        del smooth_cache
        tools.logging.Formatter.indent_dec()
        gc.collect()
        torch.cuda.empty_cache()

    # Collect original state dict for activation quantization
    if config.needs_acts_quantizer_cache:
        if load_path and os.path.exists(load_path.acts):
            orig_state_dict = None
        elif cache and cache.path.acts and os.path.exists(cache.path.acts):
            orig_state_dict = None
        else:
            orig_state_dict: dict[str, torch.Tensor] = {
                name: param.detach().clone() for name, param in model.module.named_parameters() if param.ndim > 1
            }
    else:
        orig_state_dict = None

    # Load model if checkpoint exists
    if load_model:
        logger.info(f"* Loading model checkpoint from {load_model_path}")
        load_sam2_weights_state_dict(
            model,
            config,
            state_dict=torch.load(load_model_path),
            branch_state_dict=torch.load(load_path.branch) if os.path.exists(load_path.branch) else None,
        )
        gc.collect()
        torch.cuda.empty_cache()
    elif quant_wgts:
        logger.info("* Quantizing weights")
        tools.logging.Formatter.indent_inc()
        quantizer_state_dict, quantizer_load_from = None, ""
        if load_path and os.path.exists(load_path.wgts):
            quantizer_load_from = load_path.wgts
        elif cache and cache.path.wgts and os.path.exists(cache.path.wgts):
            quantizer_load_from = cache.path.wgts
        if quantizer_load_from:
            logger.info(f"- Loading weight settings from {quantizer_load_from}")
            quantizer_state_dict = torch.load(quantizer_load_from)
        branch_state_dict, branch_load_from = None, ""
        if load_path and os.path.exists(load_path.branch):
            branch_load_from = load_path.branch
        elif cache and cache.path.branch and os.path.exists(cache.path.branch):
            branch_load_from = cache.path.branch
        if branch_load_from:
            logger.info(f"- Loading branch settings from {branch_load_from}")
            branch_state_dict = torch.load(branch_load_from)
        if not quantizer_load_from:
            logger.info("- Generating weight settings")
        if not branch_load_from:
            logger.info("- Generating branch settings")
        quantizer_state_dict, branch_state_dict, scale_state_dict = quantize_sam2_weights(
            model,
            config,
            quantizer_state_dict=quantizer_state_dict,
            branch_state_dict=branch_state_dict,
            return_with_scale_state_dict=bool(save_dirpath),
        )
        if not quantizer_load_from and cache and cache.dirpath.wgts:
            logger.info(f"- Saving weight settings to {cache.path.wgts}")
            os.makedirs(cache.dirpath.wgts, exist_ok=True)
            torch.save(quantizer_state_dict, cache.path.wgts)
            quantizer_load_from = cache.path.wgts
        if not branch_load_from and cache and cache.dirpath.branch:
            logger.info(f"- Saving branch settings to {cache.path.branch}")
            os.makedirs(cache.dirpath.branch, exist_ok=True)
            torch.save(branch_state_dict, cache.path.branch)
            branch_load_from = cache.path.branch
        if save_path:
            if not copy_on_save and quantizer_load_from:
                logger.info(f"- Linking weight settings to {save_path.wgts}")
                os.symlink(os.path.relpath(quantizer_load_from, save_dirpath), save_path.wgts)
            else:
                logger.info(f"- Saving weight settings to {save_path.wgts}")
                torch.save(quantizer_state_dict, save_path.wgts)
            if not copy_on_save and branch_load_from:
                logger.info(f"- Linking branch settings to {save_path.branch}")
                os.symlink(os.path.relpath(branch_load_from, save_dirpath), save_path.branch)
            else:
                logger.info(f"- Saving branch settings to {save_path.branch}")
                torch.save(branch_state_dict, save_path.branch)
        if save_model:
            logger.info(f"- Saving model to {save_dirpath}")
            torch.save(scale_state_dict, os.path.join(save_dirpath, "scale.pt"))
            torch.save(model.module.state_dict(), os.path.join(save_dirpath, "model.pt"))
        del quantizer_state_dict, branch_state_dict, scale_state_dict
        tools.logging.Formatter.indent_dec()
        gc.collect()
        torch.cuda.empty_cache()

    # Activation quantization phase
    if quant_acts:
        logger.info("* Quantizing activations")
        tools.logging.Formatter.indent_inc()
        breakpoint()
        if config.needs_acts_quantizer_cache:
            load_from = ""
            if load_path and os.path.exists(load_path.acts):
                load_from = load_path.acts
            elif cache and cache.path.acts and os.path.exists(cache.path.acts):
                load_from = cache.path.acts
            if load_from:
                logger.info(f"- Loading activation settings from {load_from}")
                quantizer_state_dict = torch.load(load_from)
                quantize_sam2_activations(
                    model, config, quantizer_state_dict=quantizer_state_dict, orig_state_dict=orig_state_dict
                )
            else:
                logger.info("- Generating activation settings")
                quantizer_state_dict = quantize_sam2_activations(model, config, orig_state_dict=orig_state_dict)
                if cache and cache.dirpath.acts and quantizer_state_dict is not None:
                    logger.info(f"- Saving activation settings to {cache.path.acts}")
                    os.makedirs(cache.dirpath.acts, exist_ok=True)
                    torch.save(quantizer_state_dict, cache.path.acts)
                load_from = cache.path.acts
            if save_dirpath:
                if not copy_on_save and load_from:
                    logger.info(f"- Linking activation quantizer settings to {save_path.acts}")
                    os.symlink(os.path.relpath(load_from, save_dirpath), save_path.acts)
                else:
                    logger.info(f"- Saving activation quantizer settings to {save_path.acts}")
                    torch.save(quantizer_state_dict, save_path.acts)
            del quantizer_state_dict
        else:
            logger.info("- No need to generate/load activation quantizer settings")
            quantize_sam2_activations(model, config, orig_state_dict=orig_state_dict)
        tools.logging.Formatter.indent_dec()
        del orig_state_dict
        gc.collect()
        torch.cuda.empty_cache()

    return model
