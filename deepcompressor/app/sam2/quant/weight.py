# -*- coding: utf-8 -*-
"""SAM2 weight quantization with SVDQuant support."""

import gc
import typing as tp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepcompressor.data.cache import IOTensorsCache, TensorCache, TensorsCache
from deepcompressor.nn.patch.lowrank import LowRankBranch
from deepcompressor.utils import tools

from ..nn.struct import Sam2ModelStruct
from .config import Sam2QuantConfig
from .quantizer.quantizer import Sam2ActivationQuantizer, Sam2WeightQuantizer
from .utils import ActivationStatsCollector

__all__ = ["quantize_sam2_weights", "load_sam2_weights_state_dict"]


@torch.inference_mode()
def quantize_sam2_weights(
    model: Sam2ModelStruct,
    config: Sam2QuantConfig,
    quantizer_state_dict: dict | None = None,
    branch_state_dict: dict | None = None,
    activation_stats: ActivationStatsCollector | None = None,
    calib_dataloader: DataLoader | None = None,
    return_with_scale_state_dict: bool = False,
) -> tuple[dict, dict, dict | None]:
    """Quantize SAM2 model weights with SVDQuant support.

    Args:
        model: SAM2 model structure.
        config: Quantization configuration.
        quantizer_state_dict: Pre-computed quantizer state dict.
        branch_state_dict: Pre-computed branch state dict for SVDQuant.
        activation_stats: Activation statistics from calibration.
        calib_dataloader: DataLoader for calibration data (for GPTQ/SVDQuant).
        return_with_scale_state_dict: Whether to return scale state dict.

    Returns:
        Tuple of (quantizer_state_dict, branch_state_dict, scale_state_dict).
    """
    logger = tools.logging.getLogger(__name__)

    if quantizer_state_dict is None:
        quantizer_state_dict = {}
    if branch_state_dict is None:
        branch_state_dict = {}

    scale_state_dict = {} if return_with_scale_state_dict else None

    wgts_config = config.wgts
    develop_dtype = config.develop_dtype
    device = next(model.module.parameters()).device

    # Check if SVDQuant (low-rank) is enabled
    use_svdquant = wgts_config.enabled_low_rank and calib_dataloader is not None

    if use_svdquant:
        logger.info("Using SVDQuant for weight quantization")
        # Collect activation caches for low-rank calibration
        layer_caches = _collect_layer_caches(
            model=model,
            calib_dataloader=calib_dataloader,
            device=device,
            max_samples=wgts_config.kernel_gptq.block_size if wgts_config.kernel_gptq else 128,
        )
    else:
        layer_caches = {}
        if wgts_config.kernel_gptq is not None and calib_dataloader is not None:
            logger.info("Using GPTQ for weight quantization")
        else:
            logger.info("Using RTN (Round-to-Nearest) quantization")

    # Process block by block
    for block_idx, block in enumerate(tqdm(model.block_structs, desc="Quantizing blocks")):
        logger.debug(f"- Quantizing block {block_idx}")

        # Collect modules to quantize in this block
        modules_to_quantize = {}

        # Attention modules
        for attn_struct in block.attn_structs:
            if attn_struct.q_proj is not None:
                modules_to_quantize[attn_struct.q_proj_name] = attn_struct.q_proj
            if attn_struct.k_proj is not None:
                modules_to_quantize[attn_struct.k_proj_name] = attn_struct.k_proj
            if attn_struct.v_proj is not None:
                modules_to_quantize[attn_struct.v_proj_name] = attn_struct.v_proj
            if attn_struct.o_proj is not None:
                modules_to_quantize[attn_struct.o_proj_name] = attn_struct.o_proj

        # FFN modules
        if block.ffn_struct is not None:
            for up_proj, up_name in zip(block.ffn_struct.up_projs, block.ffn_struct.up_proj_names):
                modules_to_quantize[up_name] = up_proj
            for down_proj, down_name in zip(block.ffn_struct.down_projs, block.ffn_struct.down_proj_names):
                modules_to_quantize[down_name] = down_proj

        # Quantize each module
        for name, module in modules_to_quantize.items():
            if name in wgts_config.skips:
                continue

            # Build weight quantizer
            quantizer = Sam2WeightQuantizer.build(
                config=wgts_config,
                weight=module.weight,
                key=name,
                develop_dtype=develop_dtype,
            )

            if quantizer is None:
                continue

            # Check if we have cached quantizer state
            if name not in quantizer_state_dict:
                quantizer_state_dict[name] = {}

            # SVDQuant: calibrate low-rank branch
            if use_svdquant and wgts_config.low_rank.is_enabled_for(name):
                if name not in branch_state_dict:
                    logger.debug(f"  - Calibrating low-rank branch for {name}")
                    tools.logging.Formatter.indent_inc()

                    # Get cached inputs for this module
                    layer_cache = layer_caches.get(name)
                    if layer_cache is not None:
                        input_quantizer = Sam2ActivationQuantizer.build(
                            config=config.ipts,
                            key=name,
                            develop_dtype=develop_dtype,
                        )

                        branch = quantizer.calibrate_low_rank(
                            input_quantizer=input_quantizer,
                            modules=[module],
                            inputs=layer_cache.inputs if layer_cache else None,
                        )
                        branch_state_dict[name] = branch.state_dict()
                    tools.logging.Formatter.indent_dec()

                # Apply low-rank branch
                if name in branch_state_dict:
                    logger.debug(f"  - Adding low-rank branch to {name}")
                    branch = LowRankBranch(
                        in_features=module.weight.shape[1],
                        out_features=module.weight.shape[0],
                        rank=wgts_config.low_rank.rank,
                    )
                    branch.to(device=module.weight.device, dtype=module.weight.dtype)
                    branch.load_state_dict(branch_state_dict[name])

                    # Subtract low-rank component from weight (will be added back during inference)
                    module.weight.data.sub_(branch.get_effective_weight().view(module.weight.data.shape))

                    # Register branch as hook
                    branch.as_hook().register(module)

            # Quantize weights
            result = _quantize_linear_weight(
                module=module,
                name=name,
                config=wgts_config,
                develop_dtype=develop_dtype,
                quantizer=quantizer,
                quantizer_state_dict=quantizer_state_dict,
                scale_state_dict=scale_state_dict,
                layer_cache=layer_caches.get(name),
            )

        gc.collect()
        torch.cuda.empty_cache()

    return quantizer_state_dict, branch_state_dict, scale_state_dict


def _collect_layer_caches(
    model: Sam2ModelStruct,
    calib_dataloader: DataLoader,
    device: torch.device,
    max_samples: int = 128,
) -> dict[str, IOTensorsCache]:
    """Collect input activation caches for all linear layers."""
    logger = tools.logging.getLogger(__name__)
    logger.info("Collecting activation caches for SVDQuant calibration")

    layer_caches: dict[str, IOTensorsCache] = {}
    hooks = []

    # Collect all linear modules
    linear_modules = {}
    for block in model.block_structs:
        for attn_struct in block.attn_structs:
            if attn_struct.q_proj is not None:
                linear_modules[attn_struct.q_proj_name] = attn_struct.q_proj
            if attn_struct.k_proj is not None:
                linear_modules[attn_struct.k_proj_name] = attn_struct.k_proj
            if attn_struct.v_proj is not None:
                linear_modules[attn_struct.v_proj_name] = attn_struct.v_proj
            if attn_struct.o_proj is not None:
                linear_modules[attn_struct.o_proj_name] = attn_struct.o_proj
        if block.ffn_struct is not None:
            for up_proj, up_name in zip(block.ffn_struct.up_projs, block.ffn_struct.up_proj_names):
                linear_modules[up_name] = up_proj
            for down_proj, down_name in zip(block.ffn_struct.down_projs, block.ffn_struct.down_proj_names):
                linear_modules[down_name] = down_proj

    def create_hook(name: str):
        def hook(module: nn.Module, inputs: tuple, output):
            if not inputs or inputs[0] is None:
                return

            inp = inputs[0]
            if isinstance(inp, tuple):
                inp = inp[0]

            if not isinstance(inp, torch.Tensor):
                return

            if name not in layer_caches:
                # Create IOTensorsCache with a TensorCache for inputs
                input_cache = TensorCache(channels_dim=-1)
                layer_caches[name] = IOTensorsCache(inputs=input_cache)

            # Store input tensor (detached, on CPU to save GPU memory)
            inp_cpu = inp.detach().cpu()
            layer_caches[name].inputs.data.append(inp_cpu)
            layer_caches[name].inputs.num_cached += 1
            layer_caches[name].inputs.num_total += 1
            layer_caches[name].inputs.num_samples += inp_cpu.shape[0]

        return hook

    # Register hooks
    for name, module in linear_modules.items():
        hook = module.register_forward_hook(create_hook(name))
        hooks.append(hook)

    # Run calibration
    model.module.eval()
    samples_processed = 0

    with torch.no_grad():
        for batch in tqdm(calib_dataloader, desc="Collecting activations", leave=False):
            if samples_processed >= max_samples:
                break

            pixel_values = batch["pixel_values"]
            if isinstance(pixel_values, torch.Tensor):
                pixel_values = pixel_values.to(device)

            try:
                if hasattr(model.module, "vision_encoder"):
                    model.module.vision_encoder(pixel_values)
                elif hasattr(model.module, "image_encoder"):
                    model.module.image_encoder(pixel_values)
                else:
                    model.module(pixel_values)
            except Exception:
                continue

            samples_processed += pixel_values.shape[0]

    # Remove hooks
    for hook in hooks:
        hook.remove()

    logger.info(f"Collected activations for {len(layer_caches)} layers")
    return layer_caches


def _quantize_linear_weight(
    module: nn.Linear,
    name: str,
    config,
    develop_dtype: torch.dtype,
    quantizer: Sam2WeightQuantizer,
    quantizer_state_dict: dict,
    scale_state_dict: dict | None,
    layer_cache: IOTensorsCache | None = None,
) -> None:
    """Quantize a linear layer weight using the quantizer."""
    weight = module.weight.data
    weight_name = f"{name}.weight"

    # Check if we have cached state
    if weight_name in quantizer_state_dict and "scale" in quantizer_state_dict.get(weight_name, {}):
        state = quantizer_state_dict[weight_name]
        if "scale" in state:
            if not hasattr(module, "weight_scale"):
                module.register_buffer("weight_scale", state["scale"])
            else:
                module.weight_scale.copy_(state["scale"])
        return

    if not config.is_enabled():
        return

    # Use GPTQ if available and we have calibration data
    if config.kernel_gptq is not None and layer_cache is not None:
        _quantize_with_gptq(
            module=module,
            name=name,
            config=config,
            develop_dtype=develop_dtype,
            quantizer_state_dict=quantizer_state_dict,
            scale_state_dict=scale_state_dict,
            layer_cache=layer_cache,
        )
    else:
        _quantize_with_rtn(
            module=module,
            name=name,
            config=config,
            develop_dtype=develop_dtype,
            quantizer_state_dict=quantizer_state_dict,
            scale_state_dict=scale_state_dict,
        )


def _quantize_with_gptq(
    module: nn.Linear,
    name: str,
    config,
    develop_dtype: torch.dtype,
    quantizer_state_dict: dict,
    scale_state_dict: dict | None,
    layer_cache: IOTensorsCache,
) -> None:
    """Quantize a linear layer weight using GPTQ algorithm."""
    weight = module.weight.data.clone()
    weight_name = f"{name}.weight"

    # Get quantization parameters
    group_size = config.largest_group_shape[-1] if config.largest_group_shape else -1
    if group_size == -1:
        group_size = weight.shape[-1]

    n_bits = config.quant_dtype.bits
    gptq_config = config.kernel_gptq
    damp_percentage = gptq_config.damp_percentage if hasattr(gptq_config, 'damp_percentage') else 0.01

    # Compute Hessian from cached inputs
    H = None
    if layer_cache is not None and layer_cache.inputs:
        inputs_list = layer_cache.inputs.data if hasattr(layer_cache.inputs, 'data') else layer_cache.inputs
        if inputs_list:
            for inp in inputs_list:
                inp_flat = inp.float().reshape(-1, inp.shape[-1])
                h = inp_flat.t() @ inp_flat
                if H is None:
                    H = h
                else:
                    H = H + h
            H = H / len(inputs_list)

    if H is None:
        # Fallback to RTN
        _quantize_with_rtn(module, name, config, develop_dtype, quantizer_state_dict, scale_state_dict)
        return

    # Work in float32
    W = weight.to(develop_dtype)
    H = H.to(develop_dtype).to(W.device)

    # Add damping to Hessian diagonal
    diag = torch.diag(H)
    damp = damp_percentage * torch.mean(diag)
    H = H + damp * torch.eye(H.shape[0], device=H.device, dtype=H.dtype)

    # Compute inverse Hessian using Cholesky decomposition
    try:
        L = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(L)
    except RuntimeError:
        # Fallback if Cholesky fails
        H_inv = torch.linalg.pinv(H)

    # GPTQ quantization
    out_features, in_features = W.shape
    num_groups = (in_features + group_size - 1) // group_size

    scales = torch.zeros(out_features, num_groups, device=W.device, dtype=W.dtype)
    Q = torch.zeros_like(W)

    for group_idx in range(num_groups):
        start_idx = group_idx * group_size
        end_idx = min(start_idx + group_size, in_features)

        # Get group weights
        W_group = W[:, start_idx:end_idx].clone()

        # Compute group scale
        max_val = W_group.abs().max(dim=1, keepdim=True)[0]
        scale = max_val / (2 ** (n_bits - 1) - 1)
        scale = scale.clamp(min=1e-8)
        scales[:, group_idx:group_idx + 1] = scale

        # Quantize and dequantize
        W_q = torch.clamp(
            torch.round(W_group / scale),
            -(2 ** (n_bits - 1)),
            2 ** (n_bits - 1) - 1,
        )
        W_dq = W_q * scale

        # GPTQ error compensation for remaining weights
        error = W_group - W_dq
        if end_idx < in_features:
            # Get relevant part of inverse Hessian
            H_inv_block = H_inv[start_idx:end_idx, end_idx:]
            # Compensate remaining weights
            W[:, end_idx:] -= error @ H_inv_block

        Q[:, start_idx:end_idx] = W_dq

    # Update module weight
    module.weight.data.copy_(Q.to(weight.dtype))

    # Store scale
    scale_flat = scales.reshape(out_features, -1)
    if not hasattr(module, "weight_scale"):
        module.register_buffer("weight_scale", scale_flat.to(weight.dtype))
    else:
        module.weight_scale.copy_(scale_flat.to(weight.dtype))

    # Store in state dicts with proper format for nunchaku conversion
    quantizer_state_dict[name] = {"scale": scale_flat.to(weight.dtype)}
    if scale_state_dict is not None:
        # Use the proper .scale.0 format expected by nunchaku conversion
        scale_4d = scale_flat.view(out_features, 1, num_groups, 1)
        scale_state_dict[f"{weight_name}.scale.0"] = scale_4d.to(weight.dtype)


def _quantize_with_rtn(
    module: nn.Linear,
    name: str,
    config,
    develop_dtype: torch.dtype,
    quantizer_state_dict: dict,
    scale_state_dict: dict | None,
) -> None:
    """Quantize a linear layer weight using RTN (Round-to-Nearest)."""
    weight = module.weight.data
    weight_name = f"{name}.weight"

    if not config.is_enabled():
        return

    # Compute quantization parameters
    weight_f32 = weight.to(develop_dtype)

    group_size = config.largest_group_shape[-1] if config.largest_group_shape else -1
    if group_size == -1:
        group_size = weight.shape[-1]

    n_bits = config.quant_dtype.bits

    # Reshape for group quantization
    out_features, in_features = weight_f32.shape
    if in_features % group_size == 0:
        num_groups = in_features // group_size
        weight_grouped = weight_f32.reshape(out_features, num_groups, group_size)
        scale = weight_grouped.abs().max(dim=-1, keepdim=True)[0] / (2 ** (n_bits - 1) - 1)
        scale = scale.clamp(min=1e-8)

        # Quantize and dequantize
        W_q = torch.clamp(
            torch.round(weight_grouped / scale),
            -(2 ** (n_bits - 1)),
            2 ** (n_bits - 1) - 1,
        )
        W_dq = W_q * scale
        module.weight.data.copy_(W_dq.reshape(out_features, in_features).to(weight.dtype))

        scale = scale.squeeze(-1)
    else:
        num_groups = 1
        scale = weight_f32.abs().max(dim=-1, keepdim=True)[0] / (2 ** (n_bits - 1) - 1)
        scale = scale.clamp(min=1e-8)

        # Quantize and dequantize
        W_q = torch.clamp(
            torch.round(weight_f32 / scale),
            -(2 ** (n_bits - 1)),
            2 ** (n_bits - 1) - 1,
        )
        W_dq = W_q * scale
        module.weight.data.copy_(W_dq.to(weight.dtype))

    # Store scale
    if not hasattr(module, "weight_scale"):
        module.register_buffer("weight_scale", scale.to(weight.dtype))
    else:
        module.weight_scale.copy_(scale.to(weight.dtype))

    # Store in state dicts with proper format for nunchaku conversion
    quantizer_state_dict[name] = {"scale": scale.to(weight.dtype)}
    if scale_state_dict is not None:
        # Use the proper .scale.0 format expected by nunchaku conversion
        scale_4d = scale.view(out_features, 1, num_groups, 1)
        scale_state_dict[f"{weight_name}.scale.0"] = scale_4d.to(weight.dtype)


def load_sam2_weights_state_dict(
    model: Sam2ModelStruct,
    config: Sam2QuantConfig,
    state_dict: dict,
    branch_state_dict: dict | None = None,
) -> None:
    """Load quantized weights from state dict.

    Args:
        model: SAM2 model structure.
        config: Quantization configuration.
        state_dict: Model state dict.
        branch_state_dict: Branch state dict for SVDQuant.
    """
    logger = tools.logging.getLogger(__name__)
    logger.info("Loading SAM2 weights from state dict")

    # Load state dict into model
    missing, unexpected = model.module.load_state_dict(state_dict, strict=False)

    if missing:
        logger.warning(f"Missing keys: {len(missing)}")
    if unexpected:
        logger.warning(f"Unexpected keys: {len(unexpected)}")

    # Load low-rank branches if available
    if branch_state_dict and config.wgts.enabled_low_rank:
        logger.info("Loading SVDQuant low-rank branches")
        for block in model.block_structs:
            for attn_struct in block.attn_structs:
                _load_branch_for_module(attn_struct.q_proj, attn_struct.q_proj_name, branch_state_dict, config)
                _load_branch_for_module(attn_struct.k_proj, attn_struct.k_proj_name, branch_state_dict, config)
                _load_branch_for_module(attn_struct.v_proj, attn_struct.v_proj_name, branch_state_dict, config)
                _load_branch_for_module(attn_struct.o_proj, attn_struct.o_proj_name, branch_state_dict, config)
            if block.ffn_struct is not None:
                for up_proj, up_name in zip(block.ffn_struct.up_projs, block.ffn_struct.up_proj_names):
                    _load_branch_for_module(up_proj, up_name, branch_state_dict, config)
                for down_proj, down_name in zip(block.ffn_struct.down_projs, block.ffn_struct.down_proj_names):
                    _load_branch_for_module(down_proj, down_name, branch_state_dict, config)


def _load_branch_for_module(
    module: nn.Module | None,
    name: str,
    branch_state_dict: dict,
    config: Sam2QuantConfig,
) -> None:
    """Load a low-rank branch for a module."""
    if module is None or name not in branch_state_dict:
        return

    branch = LowRankBranch(
        in_features=module.weight.shape[1],
        out_features=module.weight.shape[0],
        rank=config.wgts.low_rank.rank,
    )
    branch.to(device=module.weight.device, dtype=module.weight.dtype)
    branch.load_state_dict(branch_state_dict[name])
    branch.as_hook().register(module)
