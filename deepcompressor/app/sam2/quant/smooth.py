# -*- coding: utf-8 -*-
"""SAM2 smooth quantization module."""

import typing as tp

import torch
import torch.nn as nn
from tqdm import tqdm

from deepcompressor.calib.smooth import ActivationSmoother, smooth_linear_modules
from deepcompressor.data.cache import IOTensorsCache
from deepcompressor.quantizer import Quantizer
from deepcompressor.utils import tools

from ..nn.struct import Sam2BlockStruct, Sam2ModelStruct, Sam2ModuleStruct
from .config import Sam2QuantConfig
from .utils import get_sam2_needs_inputs_fn

__all__ = ["smooth_sam2"]


@torch.inference_mode()
def smooth_sam2(
    model: Sam2ModelStruct,
    config: Sam2QuantConfig,
    smooth_cache: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    if not isinstance(model, Sam2ModelStruct):
        model = Sam2ModelStruct.construct(model)
    assert isinstance(model, Sam2ModelStruct)
    smooth_cache = smooth_cache or {}

    if not config.enabled_smooth or not config.smooth.enabled_proj:
        return smooth_cache

    logger = tools.logging.getLogger(f"{__name__}.SmoothQuant")
    needs_cache = not smooth_cache

    def iter_layers():
        if not needs_cache:
            for _, layer in model.get_named_layers(skip_pre_modules=False, skip_post_modules=False).items():
                yield layer, {}, {}
            return
        for _, (layer, layer_cache, layer_kwargs) in config.calib.build_loader().iter_layer_activations(
            model,
            needs_inputs_fn=get_sam2_needs_inputs_fn(model, config),
            skip_pre_modules=False,
            skip_post_modules=False,
        ):
            yield layer, layer_cache, layer_kwargs

    for layer, layer_cache, layer_kwargs in tqdm(
        iter_layers(),
        desc="smoothing weights",
        leave=False,
        total=model.num_blocks + int(bool(model.get_prev_module_keys())),
        dynamic_ncols=True,
    ):
        assert isinstance(layer, (Sam2BlockStruct, Sam2ModuleStruct))
        for module_key, module_name, module, _, _ in layer.named_key_modules():
            if not isinstance(module, nn.Linear):
                continue
            needs_quant = config.enabled_wgts and config.wgts.is_enabled_for(module_key)
            needs_quant = needs_quant or (config.enabled_ipts and config.ipts.is_enabled_for(module_key))
            if not needs_quant or not config.smooth.proj.is_enabled_for(module_key):
                continue
            cache_key = module_name
            if cache_key not in smooth_cache:
                cache = layer_cache.get(module_name, None)
                if cache is None or cache.inputs is None:
                    logger.debug("- Skipping smooth for %s (missing cached inputs)", module_name)
                    continue
                config_wgts = config.wgts
                if config.enabled_extra_wgts and config.extra_wgts.is_enabled_for(module_key):
                    config_wgts = config.extra_wgts
                smooth_cache[cache_key] = smooth_linear_modules(
                    None,
                    module,
                    scale=None,
                    config=config.smooth.proj,
                    weight_quantizer=Quantizer(config_wgts, key=module_key, low_rank=config.wgts.low_rank),
                    input_quantizer=Quantizer(config.ipts, channels_dim=-1, key=module_key),
                    inputs=cache.inputs,
                    eval_inputs=cache.inputs,
                    eval_module=module,
                    eval_kwargs=layer_kwargs,
                    develop_dtype=config.develop_dtype,
                )
            ActivationSmoother(
                smooth_cache[cache_key], channels_dim=-1, develop_dtype=config.develop_dtype
            ).as_hook().register(module)
            module.in_smooth_cache_key = cache_key
    return smooth_cache
