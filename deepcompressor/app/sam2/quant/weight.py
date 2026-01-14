# -*- coding: utf-8 -*-
"""SAM2 weight quantization calibration module."""

import gc
import typing as tp

import torch
import torch.nn as nn
from tqdm import tqdm

from deepcompressor.data.cache import IOTensorsCache
from deepcompressor.data.zero import ZeroPointDomain
from deepcompressor.nn.patch.lowrank import LowRankBranch
from deepcompressor.utils import tools

from ..nn.struct import Sam2BlockStruct, Sam2ModelStruct, Sam2ModuleStruct
from .config import Sam2QuantConfig
from .quantizer import Sam2ActivationQuantizer, Sam2WeightQuantizer
from .utils import get_sam2_needs_inputs_fn

__all__ = ["quantize_sam2_weights", "load_sam2_weights_state_dict"]


@torch.inference_mode()
def calibrate_sam2_block_low_rank_branch(
    layer: Sam2ModuleStruct | Sam2BlockStruct,
    config: Sam2QuantConfig,
    branch_state_dict: dict[str, dict[str, torch.Tensor]],
    layer_cache: dict[str, IOTensorsCache] | None = None,
    layer_kwargs: dict[str, tp.Any] | None = None,
) -> None:
    assert config.wgts.low_rank is not None
    logger = tools.logging.getLogger(f"{__name__}.WeightQuantSVD")
    logger.debug("- Calibrating low-rank branches of block %s", layer.name)
    layer_cache = layer_cache or {}
    for module_key, module_name, module, _, _ in layer.named_key_modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        config_wgts = config.wgts
        if config.enabled_extra_wgts and config.extra_wgts.is_enabled_for(module_key):
            config_wgts = config.extra_wgts
        quantizer = Sam2WeightQuantizer(config_wgts, develop_dtype=config.develop_dtype, key=module_key)
        if not (quantizer.is_enabled() and quantizer.is_enabled_low_rank()):
            continue
        cache = layer_cache.get(module_name, None)
        if cache is None or cache.inputs is None:
            logger.debug("- Skipping low-rank branch for %s (missing cached inputs)", module_name)
            continue
        if module_name not in branch_state_dict:
            logger.debug("- Calibrating low-rank branch for %s", module_name)
            tools.logging.Formatter.indent_inc()
            channels_dim = -1 if isinstance(module, nn.Linear) else 1
            branch_state_dict[module_name] = quantizer.calibrate_low_rank(
                input_quantizer=Sam2ActivationQuantizer(config.ipts, key=module_key, channels_dim=channels_dim),
                modules=[module],
                inputs=cache.inputs,
                eval_inputs=cache.inputs,
                eval_module=module,
                eval_kwargs=layer_kwargs,
            ).state_dict()
            tools.logging.Formatter.indent_dec()
            gc.collect()
            torch.cuda.empty_cache()
        branch = LowRankBranch(
            in_features=module.weight.shape[1],
            out_features=module.weight.shape[0],
            rank=config.wgts.low_rank.rank,
        )
        branch.to(device=module.weight.device, dtype=module.weight.dtype)
        branch.load_state_dict(branch_state_dict[module_name])
        logger.debug("  + Adding low-rank branch to %s", module_name)
        module.weight.data.sub_(branch.get_effective_weight().view(module.weight.data.shape))
        branch.as_hook().register(module)
        del branch
        gc.collect()
        torch.cuda.empty_cache()


@torch.inference_mode()
def update_sam2_block_weight_quantizer_state_dict(
    layer: Sam2ModuleStruct | Sam2BlockStruct,
    config: Sam2QuantConfig,
    quantizer_state_dict: dict[str, dict[str, torch.Tensor | float | None]],
    layer_cache: dict[str, IOTensorsCache],
    layer_kwargs: dict[str, tp.Any],
):
    logger = tools.logging.getLogger(f"{__name__}.WeightQuant")
    logger.debug("- Calibrating weights: block %s", layer.name)
    tools.logging.Formatter.indent_inc()
    for module_key, module_name, module, _, _ in layer.named_key_modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        config_wgts = config.wgts
        if config.enabled_extra_wgts and config.extra_wgts.is_enabled_for(module_key):
            config_wgts = config.extra_wgts
        quantizer = Sam2WeightQuantizer(config_wgts, develop_dtype=config.develop_dtype, key=module_key)
        if quantizer.is_enabled():
            if module_name not in quantizer_state_dict:
                logger.debug("- Calibrating %s.weight quantizer", module_name)
                cache = layer_cache.get(module_name, None)
                if cache is None or cache.inputs is None:
                    logger.debug("- Skipping %s (missing cached inputs)", module_name)
                    continue
                quantizer.calibrate_dynamic_range(
                    module=module,
                    inputs=cache.inputs,
                    eval_inputs=cache.inputs,
                    eval_module=module,
                    eval_kwargs=layer_kwargs,
                )
                quantizer_state_dict[module_name] = quantizer.state_dict()
                gc.collect()
                torch.cuda.empty_cache()
            else:
                logger.debug("- Loading %s.weight quantizer", module_name)
        else:
            logger.debug("- Skipping %s.weight", module_name)
            if module_name in quantizer_state_dict:
                quantizer_state_dict.pop(module_name)
    tools.logging.Formatter.indent_dec()


@torch.inference_mode()
def quantize_sam2_block_weights(
    layer: Sam2ModuleStruct | Sam2BlockStruct,
    config: Sam2QuantConfig,
    quantizer_state_dict: dict[str, dict[str, torch.Tensor | float | None]],
    layer_cache: dict[str, IOTensorsCache] | None = None,
    return_with_scale_state_dict: bool = False,
) -> dict[str, torch.Tensor | float | None]:
    logger = tools.logging.getLogger(f"{__name__}.WeightQuant")
    logger.debug("- Quantizing weights: block %s", layer.name)
    layer_cache = layer_cache or {}

    scale_state_dict: dict[str, torch.Tensor | float | None] = {}

    tools.logging.Formatter.indent_inc()
    for module_key, module_name, module, _, _ in layer.named_key_modules():
        if module_name in quantizer_state_dict:
            param_name = f"{module_name}.weight"
            logger.debug("- Quantizing %s", param_name)
            config_wgts = config.wgts
            if config.enabled_extra_wgts and config.extra_wgts.is_enabled_for(module_key):
                config_wgts = config.extra_wgts
            quantizer = Sam2WeightQuantizer(config_wgts, develop_dtype=config.develop_dtype, key=module_key)
            quantizer.load_state_dict(quantizer_state_dict[module_name], device=module.weight.device)
            cache = layer_cache.get(module_name, None)
            result = quantizer.quantize(
                module.weight.data,
                inputs=cache.inputs.front() if cache is not None and cache.inputs is not None else None,
                return_with_dequant=True,
                return_with_quant=return_with_scale_state_dict,
            )
            if (
                config.wgts.enabled_low_rank
                and config.wgts.low_rank.is_enabled_for(module_key)
                and config.wgts.low_rank.compensate
                and config.wgts.low_rank.num_iters <= 1
            ):
                logger.debug("- Adding compensate low-rank branch to %s (side)", module_name)
                LowRankBranch(
                    in_features=module.weight.shape[1],
                    out_features=module.weight.shape[0],
                    rank=config.wgts.low_rank.rank,
                    weight=module.weight.data - result.data,
                ).as_hook().register(module)
            module.weight.data = result.data
            if return_with_scale_state_dict:
                scale_state_dict.update(result.scale.state_dict(f"{param_name}.scale"))
                zero_name = "scaled_zero" if config.wgts.zero_point is ZeroPointDomain.PostScale else "zero"
                if isinstance(result.zero, torch.Tensor):
                    scale_state_dict[f"{param_name}.{zero_name}"] = result.zero.to("cpu")
                else:
                    scale_state_dict[f"{param_name}.{zero_name}"] = result.zero
            del result
            gc.collect()
            torch.cuda.empty_cache()
    tools.logging.Formatter.indent_dec()
    return scale_state_dict


@torch.inference_mode()
def quantize_sam2_weights(
    model: nn.Module | Sam2ModelStruct,
    config: Sam2QuantConfig,
    quantizer_state_dict: dict[str, dict[str, torch.Tensor | float | None]] | None = None,
    branch_state_dict: dict[str, dict[str, torch.Tensor]] | None = None,
    return_with_scale_state_dict: bool = False,
) -> tuple[
    dict[str, dict[str, torch.Tensor | float | None]],
    dict[str, dict[str, torch.Tensor]],
    dict[str, torch.Tensor | float | None],
]:
    logger = tools.logging.getLogger(f"{__name__}.WeightQuant")
    if not isinstance(model, Sam2ModelStruct):
        model = Sam2ModelStruct.construct(model)
    assert isinstance(model, Sam2ModelStruct)
    quantizer_state_dict = quantizer_state_dict or {}
    branch_state_dict = branch_state_dict or {}

    if config.wgts.enabled_low_rank and (not config.wgts.low_rank.compensate or config.wgts.low_rank.num_iters > 1):
        logger.info("* Adding low-rank branches to weights")
        tools.logging.Formatter.indent_inc()
        with tools.logging.redirect_tqdm():
            if branch_state_dict:
                for _, layer in tqdm(
                    model.get_named_layers(skip_pre_modules=True, skip_post_modules=True).items(),
                    desc="adding low-rank branches",
                    leave=False,
                    dynamic_ncols=True,
                ):
                    calibrate_sam2_block_low_rank_branch(
                        layer=layer, config=config, branch_state_dict=branch_state_dict
                    )
            else:
                for _, (layer, layer_cache, layer_kwargs) in tqdm(
                    config.calib.build_loader().iter_layer_activations(
                        model,
                        needs_inputs_fn=get_sam2_needs_inputs_fn(model, config),
                        skip_pre_modules=True,
                        skip_post_modules=True,
                    ),
                    desc="calibrating low-rank branches",
                    leave=False,
                    total=model.num_blocks,
                    dynamic_ncols=True,
                ):
                    calibrate_sam2_block_low_rank_branch(
                        layer=layer,
                        config=config,
                        branch_state_dict=branch_state_dict,
                        layer_cache=layer_cache,
                        layer_kwargs=layer_kwargs,
                    )
        tools.logging.Formatter.indent_dec()

    skip_pre_modules = all(key in config.wgts.skips for key in model.get_prev_module_keys())
    skip_post_modules = all(key in config.wgts.skips for key in model.get_post_module_keys())
    with tools.logging.redirect_tqdm():
        if not quantizer_state_dict:
            if config.wgts.needs_calib_data:
                iterable = config.calib.build_loader().iter_layer_activations(
                    model,
                    needs_inputs_fn=get_sam2_needs_inputs_fn(model, config),
                    skip_pre_modules=skip_pre_modules,
                    skip_post_modules=skip_post_modules,
                )
            else:
                iterable = map(
                    lambda kv: (kv[0], (kv[1], {}, {})),
                    model.get_named_layers(
                        skip_pre_modules=skip_pre_modules, skip_post_modules=skip_post_modules
                    ).items(),
                )
            for _, (layer, layer_cache, layer_kwargs) in tqdm(
                iterable,
                desc="calibrating weight quantizers",
                leave=False,
                total=model.num_blocks + int(not skip_post_modules) + int(not skip_pre_modules),
                dynamic_ncols=True,
            ):
                update_sam2_block_weight_quantizer_state_dict(
                    layer=layer,
                    config=config,
                    quantizer_state_dict=quantizer_state_dict,
                    layer_cache=layer_cache,
                    layer_kwargs=layer_kwargs,
                )
    scale_state_dict: dict[str, torch.Tensor | float | None] = {}
    if config.wgts.enabled_gptq:
        iterable = config.calib.build_loader().iter_layer_activations(
            model,
            needs_inputs_fn=get_sam2_needs_inputs_fn(model, config),
            skip_pre_modules=skip_pre_modules,
            skip_post_modules=skip_post_modules,
        )
    else:
        iterable = map(
            lambda kv: (kv[0], (kv[1], {}, {})),
            model.get_named_layers(skip_pre_modules=skip_pre_modules, skip_post_modules=skip_post_modules).items(),
        )
    for _, (layer, layer_cache, _) in tqdm(
        iterable,
        desc="quantizing weights",
        leave=False,
        total=model.num_blocks + int(not skip_post_modules) + int(not skip_pre_modules),
        dynamic_ncols=True,
    ):
        layer_scale_state_dict = quantize_sam2_block_weights(
            layer=layer,
            config=config,
            layer_cache=layer_cache,
            quantizer_state_dict=quantizer_state_dict,
            return_with_scale_state_dict=return_with_scale_state_dict,
        )
        scale_state_dict.update(layer_scale_state_dict)
    return quantizer_state_dict, branch_state_dict, scale_state_dict


@torch.inference_mode()
def load_sam2_weights_state_dict(
    model: nn.Module | Sam2ModelStruct,
    config: Sam2QuantConfig,
    state_dict: dict[str, torch.Tensor],
    branch_state_dict: dict[str, dict[str, torch.Tensor]] | None = None,
) -> None:
    if not isinstance(model, Sam2ModelStruct):
        model = Sam2ModelStruct.construct(model)
    assert isinstance(model, Sam2ModelStruct)
    if config.enabled_wgts and config.wgts.enabled_low_rank:
        assert branch_state_dict is not None
        for _, layer in tqdm(
            model.get_named_layers(skip_pre_modules=True, skip_post_modules=True).items(),
            desc="adding low-rank branches",
            leave=False,
            dynamic_ncols=True,
        ):
            calibrate_sam2_block_low_rank_branch(layer=layer, config=config, branch_state_dict=branch_state_dict)
    model.module.load_state_dict(state_dict)
    gc.collect()
    torch.cuda.empty_cache()
