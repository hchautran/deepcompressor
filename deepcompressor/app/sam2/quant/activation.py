# -*- coding: utf-8 -*-
"""SAM2 activation quantization module."""

import gc
import typing as tp

import torch
import torch.nn as nn
from tqdm import tqdm

from deepcompressor.data.cache import IOTensorsCache
from deepcompressor.data.common import TensorType
from deepcompressor.utils import tools

from ..nn.struct import Sam2BlockStruct, Sam2ModelStruct, Sam2ModuleStruct
from .config import Sam2QuantConfig
from .quantizer import Sam2ActivationQuantizer
from .utils import get_sam2_needs_inputs_fn, get_sam2_needs_outputs_fn

__all__ = ["quantize_sam2_activations"]


@torch.inference_mode()
def quantize_sam2_block_activations(
    layer: Sam2ModuleStruct | Sam2BlockStruct,
    config: Sam2QuantConfig,
    quantizer_state_dict: dict[str, dict[str, torch.Tensor | float | None]],
    layer_cache: dict[str, IOTensorsCache] | None = None,
    layer_kwargs: dict[str, tp.Any] | None = None,
    orig_state_dict: dict[str, torch.Tensor] | None = None,
) -> dict[str, Sam2ActivationQuantizer]:
    logger = tools.logging.getLogger(f"{__name__}.ActivationQuant")
    layer_cache = layer_cache or {}
    layer_kwargs = layer_kwargs or {}
    orig_state_dict = orig_state_dict or {}

    quantizers: dict[str, Sam2ActivationQuantizer] = {}
    tools.logging.Formatter.indent_inc()
    for module_key, module_name, module, _, _ in layer.named_key_modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        channels_dim = -1 if isinstance(module, nn.Linear) else 1
        cache_key = f"{module_name}.input"
        quantizer_config = config.unsigned_ipts if getattr(module, "unsigned", False) else config.ipts
        quantizer = Sam2ActivationQuantizer(
            quantizer_config,
            channels_dim=channels_dim,
            develop_dtype=config.develop_dtype,
            key=module_key,
            tensor_type=TensorType.Inputs,
        )
        if quantizer.is_enabled():
            if cache_key not in quantizer_state_dict:
                logger.debug("- Calibrating %s", cache_key)
                orig_wgts = None
                if module_name in orig_state_dict:
                    orig_wgts = [(module.weight, orig_state_dict[f"{module_name}.weight"])]
                cache = layer_cache.get(module_name, None)
                if cache is None or cache.inputs is None:
                    logger.debug("- Skipping %s (missing cached inputs)", cache_key)
                    continue
                quantizer.calibrate_dynamic_range(
                    modules=[module],
                    activations=cache.inputs,
                    eval_module=module,
                    eval_inputs=cache.inputs,
                    eval_kwargs=layer_kwargs,
                    orig_weights=orig_wgts,
                )
                quantizer_state_dict[cache_key] = quantizer.state_dict()
                gc.collect()
                torch.cuda.empty_cache()
            else:
                quantizer.load_state_dict(quantizer_state_dict[cache_key], device=module.weight.device)
            quantizers[cache_key] = quantizer
        del quantizer
    tools.logging.Formatter.indent_dec()
    return quantizers


@torch.inference_mode()
def quantize_sam2_activations(
    model: nn.Module | Sam2ModelStruct,
    config: Sam2QuantConfig,
    quantizer_state_dict: dict[str, dict[str, torch.Tensor | float | None]] | None = None,
    orig_state_dict: dict[str, torch.Tensor] | None = None,
) -> dict[str, dict[str, torch.Tensor | float | None]]:
    logger = tools.logging.getLogger(f"{__name__}.ActivationQuant")
    if not isinstance(model, Sam2ModelStruct):
        model = Sam2ModelStruct.construct(model)
    assert isinstance(model, Sam2ModelStruct)
    quantizer_state_dict = quantizer_state_dict or {}
    quantizers: dict[str, Sam2ActivationQuantizer] = {}

    skip_pre_modules = all(key in config.ipts.skips for key in model.get_prev_module_keys())
    skip_post_modules = all(key in config.ipts.skips for key in model.get_post_module_keys())
    if not quantizer_state_dict and config.needs_acts_quantizer_cache:
        with tools.logging.redirect_tqdm():
            for _, (layer, layer_cache, layer_kwargs) in tqdm(
                config.calib.build_loader().iter_layer_activations(
                    model,
                    needs_inputs_fn=get_sam2_needs_inputs_fn(model, config=config),
                    needs_outputs_fn=get_sam2_needs_outputs_fn(model, config=config),
                    skip_pre_modules=skip_pre_modules,
                    skip_post_modules=skip_post_modules,
                ),
                desc="quantizing activations",
                leave=False,
                total=model.num_blocks + int(not skip_post_modules) + int(not skip_pre_modules),
                dynamic_ncols=True,
            ):
                block_quantizers = quantize_sam2_block_activations(
                    layer=layer,
                    config=config,
                    quantizer_state_dict=quantizer_state_dict,
                    layer_cache=layer_cache,
                    layer_kwargs=layer_kwargs,
                    orig_state_dict=orig_state_dict,
                )
                quantizers.update(block_quantizers)
    else:
        for _, layer in model.get_named_layers(
            skip_pre_modules=skip_pre_modules, skip_post_modules=skip_post_modules
        ).items():
            block_quantizers = quantize_sam2_block_activations(
                layer=layer,
                config=config,
                quantizer_state_dict=quantizer_state_dict,
                orig_state_dict=orig_state_dict,
            )
            quantizers.update(block_quantizers)

    for _, module_name, module, _, _ in model.named_key_modules():
        ipts_quantizer = quantizers.get(f"{module_name}.input", None)
        opts_quantizer = quantizers.get(f"{module_name}.output", None)
        needs_quant_ipts = ipts_quantizer is not None and ipts_quantizer.is_enabled()
        needs_quant_opts = opts_quantizer is not None and opts_quantizer.is_enabled()
        if needs_quant_ipts or needs_quant_opts:
            logger.debug(
                "- Quantizing %s (%s)",
                module_name,
                ("inputs" if needs_quant_ipts else "")
                + (" and " if needs_quant_ipts and needs_quant_opts else "")
                + ("outputs" if needs_quant_opts else ""),
            )
            if needs_quant_ipts:
                ipts_quantizer.as_hook(is_output=False).register(module)
            if needs_quant_opts:
                opts_quantizer.as_hook(is_output=True).register(module)
    return quantizer_state_dict
