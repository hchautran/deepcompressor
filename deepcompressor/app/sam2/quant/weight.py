# -*- coding: utf-8 -*-
"""SAM2 weight quantization."""

import gc
import typing as tp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepcompressor.utils import tools

from ..nn.struct import Sam2HieraBlockStruct, Sam2ModelStruct
from .config import Sam2QuantConfig
from .utils import ActivationStatsCollector

__all__ = ["quantize_sam2_weights", "load_sam2_weights_state_dict"]


def quantize_sam2_weights(
    model: Sam2ModelStruct,
    config: Sam2QuantConfig,
    quantizer_state_dict: dict | None = None,
    branch_state_dict: dict | None = None,
    activation_stats: ActivationStatsCollector | None = None,
    calib_dataloader: DataLoader | None = None,
    return_with_scale_state_dict: bool = False,
) -> tuple[dict, dict, dict | None]:
    """Quantize SAM2 model weights.

    Args:
        model: SAM2 model structure.
        config: Quantization configuration.
        quantizer_state_dict: Pre-computed quantizer state dict.
        branch_state_dict: Pre-computed branch state dict for SVDQuant.
        activation_stats: Activation statistics from calibration.
        calib_dataloader: DataLoader for calibration data (for GPTQ).
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

    # Check if we should use GPTQ
    use_gptq = (
        wgts_config.kernel_gptq is not None
        and calib_dataloader is not None
        and len(quantizer_state_dict) == 0  # Only if not loading from cache
    )

    if use_gptq:
        logger.info("Using GPTQ for weight quantization")
        quantizer_state_dict, branch_state_dict, scale_state_dict = _quantize_weights_gptq(
            model=model,
            config=config,
            calib_dataloader=calib_dataloader,
            activation_stats=activation_stats,
            return_with_scale_state_dict=return_with_scale_state_dict,
        )
    else:
        # Simple RTN (Round-to-Nearest) quantization
        if calib_dataloader is not None:
            logger.info("Using RTN with activation statistics")
        else:
            logger.info("Using RTN (Round-to-Nearest) quantization")

        for block_idx, block in enumerate(model.block_structs):
            logger.debug(f"- Quantizing block {block_idx}")

            # Quantize attention modules
            if block.attn_structs:
                for attn_struct in block.attn_structs:
                    _quantize_attention_weights(
                        attn_struct,
                        wgts_config=wgts_config,
                        develop_dtype=develop_dtype,
                        quantizer_state_dict=quantizer_state_dict,
                        branch_state_dict=branch_state_dict,
                        scale_state_dict=scale_state_dict,
                        activation_stats=activation_stats,
                    )

            # Quantize FFN modules
            if block.ffn_struct is not None:
                _quantize_ffn_weights(
                    block.ffn_struct,
                    wgts_config=wgts_config,
                    develop_dtype=develop_dtype,
                    quantizer_state_dict=quantizer_state_dict,
                    branch_state_dict=branch_state_dict,
                    scale_state_dict=scale_state_dict,
                    activation_stats=activation_stats,
                )

            gc.collect()
            torch.cuda.empty_cache()

    return quantizer_state_dict, branch_state_dict, scale_state_dict


def _quantize_weights_gptq(
    model: Sam2ModelStruct,
    config: Sam2QuantConfig,
    calib_dataloader: DataLoader,
    activation_stats: ActivationStatsCollector | None = None,
    return_with_scale_state_dict: bool = False,
) -> tuple[dict, dict, dict | None]:
    """Quantize weights using GPTQ algorithm.

    GPTQ uses the Hessian (X^T @ X) computed from calibration data to
    minimize quantization error.
    """
    logger = tools.logging.getLogger(__name__)

    quantizer_state_dict = {}
    branch_state_dict = {}
    scale_state_dict = {} if return_with_scale_state_dict else None

    wgts_config = config.wgts
    gptq_config = wgts_config.kernel_gptq
    develop_dtype = config.develop_dtype

    device = next(model.module.parameters()).device

    # Process block by block to save memory
    for block_idx, block in enumerate(tqdm(model.block_structs, desc="GPTQ Blocks")):
        logger.debug(f"- GPTQ quantizing block {block_idx}")

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

        # Collect Hessians for this block
        hessians = _collect_block_hessians(
            model=model,
            block_idx=block_idx,
            modules=modules_to_quantize,
            calib_dataloader=calib_dataloader,
            device=device,
            max_samples=gptq_config.block_size if hasattr(gptq_config, 'block_size') else 128,
        )

        # Quantize each module with GPTQ
        for name, module in modules_to_quantize.items():
            if name in wgts_config.skips:
                continue

            H = hessians.get(name)
            if H is None:
                # Fallback to RTN if no Hessian available
                _quantize_linear_weight_rtn(
                    module=module,
                    name=name,
                    config=wgts_config,
                    develop_dtype=develop_dtype,
                    quantizer_state_dict=quantizer_state_dict,
                    scale_state_dict=scale_state_dict,
                )
            else:
                _quantize_linear_weight_gptq(
                    module=module,
                    name=name,
                    hessian=H,
                    config=wgts_config,
                    gptq_config=gptq_config,
                    develop_dtype=develop_dtype,
                    quantizer_state_dict=quantizer_state_dict,
                    scale_state_dict=scale_state_dict,
                )

        gc.collect()
        torch.cuda.empty_cache()

    return quantizer_state_dict, branch_state_dict, scale_state_dict


def _collect_block_hessians(
    model: Sam2ModelStruct,
    block_idx: int,
    modules: dict[str, nn.Module],
    calib_dataloader: DataLoader,
    device: torch.device,
    max_samples: int = 128,
) -> dict[str, torch.Tensor]:
    """Collect Hessian approximations (X^T @ X) for modules in a block."""
    hessians: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    hooks = []

    def create_hook(name: str):
        def hook(module: nn.Module, inputs: tuple, output):
            if not inputs or inputs[0] is None:
                return

            inp = inputs[0]
            if isinstance(inp, tuple):
                inp = inp[0]

            if not isinstance(inp, torch.Tensor):
                return

            # Flatten to (N, in_features)
            inp_flat = inp.detach().float().reshape(-1, inp.shape[-1])

            # Compute Hessian approximation: X^T @ X
            h = inp_flat.t() @ inp_flat

            if name not in hessians:
                hessians[name] = h
                counts[name] = inp_flat.shape[0]
            else:
                hessians[name] = hessians[name] + h
                counts[name] += inp_flat.shape[0]

        return hook

    # Register hooks
    for name, module in modules.items():
        hook = module.register_forward_hook(create_hook(name))
        hooks.append(hook)

    # Run calibration
    model.module.eval()
    samples_processed = 0

    with torch.no_grad():
        for batch in calib_dataloader:
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

    # Normalize Hessians
    for name in hessians:
        if counts[name] > 0:
            hessians[name] = hessians[name] / counts[name]

    return hessians


def _quantize_linear_weight_gptq(
    module: nn.Linear,
    name: str,
    hessian: torch.Tensor,
    config,
    gptq_config,
    develop_dtype: torch.dtype,
    quantizer_state_dict: dict,
    scale_state_dict: dict | None,
) -> None:
    """Quantize a linear layer weight using GPTQ algorithm."""
    weight = module.weight.data.clone()
    weight_name = f"{name}.weight"

    # Get quantization parameters
    group_size = config.largest_group_shape[-1] if config.largest_group_shape else -1
    if group_size == -1:
        group_size = weight.shape[-1]

    n_bits = config.quant_dtype.bits
    damp_percentage = gptq_config.damp_percentage if hasattr(gptq_config, 'damp_percentage') else 0.01

    # Work in float32
    W = weight.to(develop_dtype)
    H = hessian.to(develop_dtype).to(W.device)

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
        group_len = end_idx - start_idx

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

    # Store in state dicts
    quantizer_state_dict[weight_name] = {"scale": scale_flat.to(weight.dtype)}
    if scale_state_dict is not None:
        scale_state_dict[weight_name] = scale_flat.to(weight.dtype)


def _quantize_linear_weight_rtn(
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

    # Check if we have cached state
    if weight_name in quantizer_state_dict:
        state = quantizer_state_dict[weight_name]
        if "scale" in state:
            if not hasattr(module, "weight_scale"):
                module.register_buffer("weight_scale", state["scale"])
            else:
                module.weight_scale.copy_(state["scale"])
        return

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

    # Store in state dict
    quantizer_state_dict[weight_name] = {"scale": scale.to(weight.dtype)}
    if scale_state_dict is not None:
        scale_state_dict[weight_name] = scale.to(weight.dtype)


def _quantize_attention_weights(
    attn_struct,
    *,
    wgts_config,
    develop_dtype: torch.dtype,
    quantizer_state_dict: dict,
    branch_state_dict: dict,
    scale_state_dict: dict | None,
    activation_stats: ActivationStatsCollector | None = None,
) -> None:
    """Quantize attention module weights."""
    # Quantize QKV projections
    if attn_struct.q_proj is not None:
        _quantize_linear_weight_rtn(
            attn_struct.q_proj,
            name=attn_struct.q_proj_name,
            config=wgts_config,
            develop_dtype=develop_dtype,
            quantizer_state_dict=quantizer_state_dict,
            scale_state_dict=scale_state_dict,
        )

    if attn_struct.k_proj is not None:
        _quantize_linear_weight_rtn(
            attn_struct.k_proj,
            name=attn_struct.k_proj_name,
            config=wgts_config,
            develop_dtype=develop_dtype,
            quantizer_state_dict=quantizer_state_dict,
            scale_state_dict=scale_state_dict,
        )

    if attn_struct.v_proj is not None:
        _quantize_linear_weight_rtn(
            attn_struct.v_proj,
            name=attn_struct.v_proj_name,
            config=wgts_config,
            develop_dtype=develop_dtype,
            quantizer_state_dict=quantizer_state_dict,
            scale_state_dict=scale_state_dict,
        )

    # Quantize output projection
    if attn_struct.o_proj is not None:
        _quantize_linear_weight_rtn(
            attn_struct.o_proj,
            name=attn_struct.o_proj_name,
            config=wgts_config,
            develop_dtype=develop_dtype,
            quantizer_state_dict=quantizer_state_dict,
            scale_state_dict=scale_state_dict,
        )


def _quantize_ffn_weights(
    ffn_struct,
    *,
    wgts_config,
    develop_dtype: torch.dtype,
    quantizer_state_dict: dict,
    branch_state_dict: dict,
    scale_state_dict: dict | None,
    activation_stats: ActivationStatsCollector | None = None,
) -> None:
    """Quantize FFN module weights."""
    # Quantize up projections
    for up_proj, up_name in zip(ffn_struct.up_projs, ffn_struct.up_proj_names):
        _quantize_linear_weight_rtn(
            up_proj,
            name=up_name,
            config=wgts_config,
            develop_dtype=develop_dtype,
            quantizer_state_dict=quantizer_state_dict,
            scale_state_dict=scale_state_dict,
        )

    # Quantize down projections
    for down_proj, down_name in zip(ffn_struct.down_projs, ffn_struct.down_proj_names):
        _quantize_linear_weight_rtn(
            down_proj,
            name=down_name,
            config=wgts_config,
            develop_dtype=develop_dtype,
            quantizer_state_dict=quantizer_state_dict,
            scale_state_dict=scale_state_dict,
        )


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
