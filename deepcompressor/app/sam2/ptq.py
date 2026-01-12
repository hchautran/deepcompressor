# -*- coding: utf-8 -*-
"""SAM2 Post-Training Quantization Pipeline."""

import gc
import os
import typing as tp

import torch
from tqdm import tqdm

from deepcompressor.utils import tools

from .cache import Sam2PtqCacheConfig, Sam2QuantCacheConfig
from .nn.struct import Sam2ModelStruct
from .quant import (
    Sam2ActivationQuantizer,
    Sam2WeightQuantizer,
    calibrate_sam_activations,
    quantize_sam_activations,
    quantize_sam_weights,
    rotate_sam,
    smooth_sam,
)
from .quant.config import Sam2QuantConfig

__all__ = ["ptq"]


def ptq(
    model: Sam2ModelStruct | torch.nn.Module,
    config: Sam2QuantConfig,
    calib_loader: tp.Iterable | None = None,
    cache: Sam2PtqCacheConfig | None = None,
    load_dirpath: str = "",
    save_dirpath: str = "",
    copy_on_save: bool = False,
    save_model: bool = False,
) -> Sam2ModelStruct:
    """Post-training quantization of a SAM2 model.

    Args:
        model (`Sam2ModelStruct` or `torch.nn.Module`):
            The SAM2 model to quantize.
        config (`Sam2QuantConfig`):
            The quantization configuration.
        calib_loader (`Iterable` or `None`, *optional*, defaults to `None`):
            Calibration data loader for activation quantization.
        cache (`Sam2PtqCacheConfig` or `None`, *optional*, defaults to `None`):
            Cache configuration for saving/loading intermediate results.
        load_dirpath (`str`, *optional*, defaults to `""`):
            Directory path to load quantization checkpoint from.
        save_dirpath (`str`, *optional*, defaults to `""`):
            Directory path to save quantization checkpoint to.
        copy_on_save (`bool`, *optional*, defaults to `False`):
            Whether to copy cache files instead of symlinking.
        save_model (`bool`, *optional*, defaults to `False`):
            Whether to save the quantized model checkpoint.

    Returns:
        `Sam2ModelStruct`: The quantized SAM2 model.
    """
    logger = tools.logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("SAM2 Post-Training Quantization")
    logger.info("=" * 80)

    # Convert to Sam2ModelStruct if needed
    if not isinstance(model, Sam2ModelStruct):
        logger.info("Constructing SAM2 model structure...")
        model = Sam2ModelStruct.construct(model)
    assert isinstance(model, Sam2ModelStruct)

    # Check what quantization is enabled
    quant_wgts = config.enabled_wgts
    quant_ipts = config.enabled_ipts
    quant_opts = config.enabled_opts
    quant_acts = quant_ipts or quant_opts
    quant = quant_wgts or quant_acts

    if not quant:
        logger.warning("No quantization enabled!")
        return model

    # Setup cache paths
    load_model_path = ""
    load_path = None
    save_path = None

    if load_dirpath:
        load_path = Sam2QuantCacheConfig(
            smooth=os.path.join(load_dirpath, "smooth.pt"),
            branch=os.path.join(load_dirpath, "branch.pt"),
            wgts=os.path.join(load_dirpath, "wgts.pt"),
            acts=os.path.join(load_dirpath, "acts.pt"),
        )
        load_model_path = os.path.join(load_dirpath, "model.pt")
        if os.path.exists(load_model_path):
            logger.info(f"Loading model from {load_model_path}")
            model.module.load_state_dict(torch.load(load_model_path))
            save_dirpath = ""  # Don't save if loading
        else:
            logger.warning(f"Model checkpoint {load_model_path} not found")

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

    # =========================================================================
    # Step 1: Rotation (optional)
    # =========================================================================
    if quant and config.enabled_rotation:
        logger.info("")
        logger.info("Step 1: Rotating model for quantization")
        logger.info("-" * 80)
        tools.logging.Formatter.indent_inc()
        rotate_sam(model)
        tools.logging.Formatter.indent_dec()
        gc.collect()
        torch.cuda.empty_cache()

    # =========================================================================
    # Step 2: Smooth Quantization (optional)
    # =========================================================================
    if quant and config.enabled_smooth:
        logger.info("")
        logger.info("Step 2: Smooth quantization")
        logger.info("-" * 80)
        tools.logging.Formatter.indent_inc()

        load_from = ""
        if load_path and os.path.exists(load_path.smooth):
            load_from = load_path.smooth
        elif cache and cache.path and cache.path.smooth and os.path.exists(cache.path.smooth):
            load_from = cache.path.smooth

        if load_from:
            logger.info(f"Loading smooth scales from {load_from}")
            smooth_cache = torch.load(load_from)
            smooth_sam(model, alpha=config.smooth.alpha, smooth_cache=smooth_cache)
        else:
            logger.info("Computing smooth scales")
            smooth_cache = smooth_sam(model, alpha=config.smooth.alpha)
            if cache and cache.path and cache.path.smooth:
                logger.info(f"Saving smooth scales to {cache.path.smooth}")
                os.makedirs(os.path.dirname(cache.path.smooth), exist_ok=True)
                torch.save(smooth_cache, cache.path.smooth)
                load_from = cache.path.smooth

        if save_path:
            if not copy_on_save and load_from:
                logger.info(f"Linking smooth scales to {save_path.smooth}")
                os.symlink(os.path.relpath(load_from, save_dirpath), save_path.smooth)
            else:
                logger.info(f"Saving smooth scales to {save_path.smooth}")
                torch.save(smooth_cache, save_path.smooth)

        del smooth_cache
        tools.logging.Formatter.indent_dec()
        gc.collect()
        torch.cuda.empty_cache()

    # =========================================================================
    # Step 3: Weight Quantization
    # =========================================================================
    if quant_wgts:
        logger.info("")
        logger.info("Step 3: Weight quantization")
        logger.info("-" * 80)
        tools.logging.Formatter.indent_inc()

        # Create weight quantizers for all modules
        weight_quantizers = {}
        for key, module_name, module, parent_struct, field_name in model.named_key_modules():
            if not isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                continue

            quantizer = Sam2WeightQuantizer(
                config=config.wgts,
                key=key,
            )

            if not quantizer.should_skip(module_name):
                weight_quantizers[key] = quantizer

        logger.info(f"Created {len(weight_quantizers)} weight quantizers")

        # Check if we should load from cache
        load_from = ""
        if load_path and os.path.exists(load_path.wgts):
            load_from = load_path.wgts
        elif cache and cache.path and cache.path.wgts and os.path.exists(cache.path.wgts):
            load_from = cache.path.wgts

        if load_from:
            logger.info(f"Loading weight quantization from {load_from}")
            wgts_state_dict = torch.load(load_from)
            # Apply cached quantization parameters
            for key, params in wgts_state_dict.items():
                if key in weight_quantizers:
                    quantizer = weight_quantizers[key]
                    if "scale" in params:
                        quantizer.scale = params["scale"]
                    if "zero" in params:
                        quantizer.zero = params["zero"]
        else:
            # Quantize weights
            logger.info("Quantizing weights...")
            wgts_state_dict = quantize_sam_weights(
                model,
                weight_quantizers,
                develop_dtype=torch.float32,
            )

            if cache and cache.path and cache.path.wgts:
                logger.info(f"Saving weight quantization to {cache.path.wgts}")
                os.makedirs(os.path.dirname(cache.path.wgts), exist_ok=True)
                torch.save(wgts_state_dict, cache.path.wgts)
                load_from = cache.path.wgts

        if save_path:
            if not copy_on_save and load_from:
                logger.info(f"Linking weight quantization to {save_path.wgts}")
                os.symlink(os.path.relpath(load_from, save_dirpath), save_path.wgts)
            else:
                logger.info(f"Saving weight quantization to {save_path.wgts}")
                torch.save(wgts_state_dict, save_path.wgts)

        del wgts_state_dict
        tools.logging.Formatter.indent_dec()
        gc.collect()
        torch.cuda.empty_cache()

    # =========================================================================
    # Step 4: Activation Quantization
    # =========================================================================
    if quant_acts:
        if calib_loader is None:
            logger.error("Calibration loader required for activation quantization!")
            raise ValueError("calib_loader must be provided for activation quantization")

        logger.info("")
        logger.info("Step 4: Activation quantization")
        logger.info("-" * 80)
        tools.logging.Formatter.indent_inc()

        # Create activation quantizers
        act_quantizers = {}
        for key, module_name, module, parent_struct, field_name in model.named_key_modules():
            if not isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                continue

            if quant_ipts:
                ipt_quantizer = Sam2ActivationQuantizer(
                    config=config.ipts,
                    key=f"{key}.input",
                )
                act_quantizers[f"{key}.input"] = ipt_quantizer

            if quant_opts:
                opt_quantizer = Sam2ActivationQuantizer(
                    config=config.opts,
                    key=f"{key}.output",
                )
                act_quantizers[f"{key}.output"] = opt_quantizer

        logger.info(f"Created {len(act_quantizers)} activation quantizers")

        # Calibrate activations
        logger.info("Calibrating activations...")
        acts_calib_dict = calibrate_sam_activations(
            model,
            act_quantizers,
            calib_loader,
            num_samples=config.calib.num_samples,
        )

        # Apply activation quantizers
        logger.info("Applying activation quantizers...")
        quantize_sam_activations(model, act_quantizers)

        if cache and cache.path and cache.path.acts:
            logger.info(f"Saving activation calibration to {cache.path.acts}")
            os.makedirs(os.path.dirname(cache.path.acts), exist_ok=True)
            torch.save(acts_calib_dict, cache.path.acts)

        if save_path and save_path.acts:
            logger.info(f"Saving activation calibration to {save_path.acts}")
            torch.save(acts_calib_dict, save_path.acts)

        del acts_calib_dict
        tools.logging.Formatter.indent_dec()
        gc.collect()
        torch.cuda.empty_cache()

    # =========================================================================
    # Save Model
    # =========================================================================
    if save_model and save_dirpath:
        logger.info("")
        logger.info("Saving quantized model")
        logger.info("-" * 80)
        model_save_path = os.path.join(save_dirpath, "model.pt")
        logger.info(f"Saving model to {model_save_path}")
        torch.save(model.module.state_dict(), model_save_path)

    logger.info("")
    logger.info("=" * 80)
    logger.info("Quantization Complete!")
    logger.info("=" * 80)

    return model
