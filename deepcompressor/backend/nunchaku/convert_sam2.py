# -*- coding: utf-8 -*-
"""Converts a DeepCompressor SAM2 state dict to a Nunchaku state dict."""

import argparse
import os

import safetensors.torch
import torch
import tqdm

from .utils import convert_to_nunchaku_w4x4y16_linear_weight

__all__ = [
    "convert_to_nunchaku_sam2_linear_state_dict",
    "convert_to_nunchaku_sam2_block_state_dict",
    "convert_to_nunchaku_sam2_state_dicts",
]


DEFAULT_LORA_RANK = 32


def convert_to_nunchaku_sam2_linear_state_dict(
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    smooth: torch.Tensor | None = None,
    lora: tuple[torch.Tensor, torch.Tensor] | None = None,
    smooth_fused: bool = False,
    float_point: bool = False,
    subscale: torch.Tensor | None = None,
    default_lora_rank: int = DEFAULT_LORA_RANK,
) -> dict[str, torch.Tensor]:
    """Convert a linear layer to nunchaku format.

    Args:
        weight: Weight tensor.
        scale: Scale tensor for quantization.
        bias: Optional bias tensor.
        smooth: Optional smooth tensor.
        lora: Optional LoRA weights (down, up).
        smooth_fused: Whether smooth is fused.
        float_point: Whether to use float-point 4-bit.
        subscale: Optional subscale tensor.
        default_lora_rank: Default low-rank dimension when lora is None.

    Returns:
        Dictionary with nunchaku format tensors compatible with SVDQW4A4Linear:
        - qweight: Packed quantized weights (int8)
        - wscales/wcscales/wtscale: Weight scales (bf16/fp16)
        - bias: Bias tensor (bf16/fp16)
        - smooth_factor: Smooth factors (bf16/fp16)
        - smooth_factor_orig: Original smooth factors (bf16/fp16)
        - proj_down: Low-rank down projection (bf16/fp16)
        - proj_up: Low-rank up projection (bf16/fp16)
    """
    if weight.ndim > 2:
        assert weight.numel() == weight.shape[0] * weight.shape[1]
        weight = weight.view(weight.shape[0], weight.shape[1])

    if scale.numel() > 1:
        assert scale.ndim == weight.ndim * 2
        assert scale.numel() == scale.shape[0] * scale.shape[2]
        scale = scale.view(scale.shape[0], 1, scale.shape[2], 1)
        scale_key = "wcscales" if scale.shape[2] == 1 else "wscales"
    else:
        scale_key = "wtscale"

    if subscale is None:
        subscale_key = ""
    else:
        assert subscale.ndim == weight.ndim * 2
        assert subscale.numel() == subscale.shape[0] * subscale.shape[2]
        subscale = subscale.view(subscale.shape[0], 1, subscale.shape[2], 1)
        subscale_key = "wcscales" if subscale.shape[2] == 1 else "wscales"

    weight, scale, bias, smooth, lora, subscale = convert_to_nunchaku_w4x4y16_linear_weight(
        weight, scale=scale, bias=bias, smooth=smooth, lora=lora, float_point=float_point, subscale=subscale
    )

    state_dict: dict[str, torch.Tensor] = {}
    state_dict["qweight"] = weight
    state_dict[scale_key] = scale
    if subscale is not None:
        state_dict[subscale_key] = subscale
    state_dict["bias"] = bias
    # Use SVDQW4A4Linear-compatible naming
    state_dict["smooth_factor_orig"] = smooth
    state_dict["smooth_factor"] = torch.ones_like(smooth) if smooth_fused else smooth.clone()

    # Handle low-rank projections (proj_down/proj_up)
    # SVDQW4A4Linear expects these even if SVDQuant wasn't used
    if lora is not None:
        state_dict["proj_down"] = lora[0]
        state_dict["proj_up"] = lora[1]
    else:
        # Create default empty low-rank projections for compatibility
        # These will be zero tensors that don't affect the output
        oc = weight.shape[0]  # output channels
        ic = smooth.shape[0] if smooth.numel() > 1 else weight.shape[1] * 2  # input channels (weight is packed)
        state_dict["proj_down"] = torch.zeros((ic, default_lora_rank), dtype=bias.dtype, device="cpu")
        state_dict["proj_up"] = torch.zeros((oc, default_lora_rank), dtype=bias.dtype, device="cpu")

    return state_dict


def convert_to_nunchaku_sam2_block_state_dict(
    state_dict: dict[str, torch.Tensor],
    scale_dict: dict[str, torch.Tensor],
    smooth_dict: dict[str, torch.Tensor],
    branch_dict: dict[str, torch.Tensor],
    block_name: str,
    float_point: bool = False,
) -> dict[str, torch.Tensor]:
    """Convert a SAM2 Hiera block to nunchaku format.

    Args:
        state_dict: Model state dict.
        scale_dict: Scale state dict.
        smooth_dict: Smooth scales dict.
        branch_dict: Branch (SVDQuant) dict.
        block_name: Block name prefix.
        float_point: Whether to use float-point 4-bit.

    Returns:
        Converted state dict for the block.
    """
    converted: dict[str, torch.Tensor] = {}

    # SAM2 Hiera block structure:
    # - attn.qkv (fused Q, K, V projection)
    # - attn.proj (output projection)
    # - mlp.fc1 (up projection)
    # - mlp.fc2 (down projection)
    # - norm1, norm2 (layer norms - not quantized)

    local_name_map = {
        "attn.qkv": "attn.qkv",
        "attn.proj": "attn.proj",
        "mlp.fc1": "mlp.fc1",
        "mlp.fc2": "mlp.fc2",
    }

    smooth_name_map = {
        "attn.qkv": "attn.qkv",
        "attn.proj": "attn.proj",
        "mlp.fc1": "mlp.fc1",
        "mlp.fc2": "mlp.fc2",
    }

    branch_name_map = {
        "attn.qkv": "attn.qkv",
        "attn.proj": "attn.proj",
        "mlp.fc1": "mlp.fc1",
        "mlp.fc2": "mlp.fc2",
    }

    # Extract parameters for this block
    candidates: dict[str, torch.Tensor] = {
        param_name: param for param_name, param in state_dict.items() if param_name.startswith(block_name)
    }

    for converted_local_name, candidate_local_name in local_name_map.items():
        candidate_name = f"{block_name}.{candidate_local_name}"
        weight_name = f"{candidate_name}.weight"
        bias_name = f"{candidate_name}.bias"

        if weight_name not in candidates:
            continue

        weight = candidates[weight_name]
        bias = candidates.get(bias_name, None)
        # Try new format first ({weight_name}.scale.0), then fall back to old format ({weight_name})
        scale = scale_dict.get(f"{weight_name}.scale.0", None)
        if scale is None:
            # Try old format where scale is stored directly under weight_name
            scale = scale_dict.get(weight_name, None)
            if scale is not None:
                # Reshape scale to expected 4D format [oc, 1, num_groups, 1]
                oc = weight.shape[0]
                if scale.ndim == 1:
                    scale = scale.view(oc, 1, 1, 1)
                elif scale.ndim == 2:
                    scale = scale.view(oc, 1, scale.shape[1], 1)
        subscale = scale_dict.get(f"{weight_name}.scale.1", None)
        # Try different smooth key formats: {module_name}.smooth or {module_name}
        smooth_key_base = f"{block_name}.{smooth_name_map.get(converted_local_name, '')}"
        smooth = smooth_dict.get(f"{smooth_key_base}.smooth", None)
        if smooth is None:
            smooth = smooth_dict.get(smooth_key_base, None)
        # Also try the weight name format
        if smooth is None:
            smooth = smooth_dict.get(f"{weight_name}.smooth", None)
        if smooth is None:
            smooth = smooth_dict.get(candidate_name, None)
        # Try different branch key formats
        branch_key_base = f"{block_name}.{branch_name_map.get(converted_local_name, '')}"
        branch = branch_dict.get(branch_key_base, None)
        if branch is None:
            branch = branch_dict.get(candidate_name, None)

        if branch is not None:
            branch = (branch["a.weight"], branch["b.weight"])

        if scale is None:
            # Copy without conversion
            converted[f"{converted_local_name}.weight"] = weight.clone().cpu()
            if bias is not None:
                converted[f"{converted_local_name}.bias"] = bias.clone().cpu()
            continue

        # Convert to nunchaku format
        layer_state_dict = convert_to_nunchaku_sam2_linear_state_dict(
            weight=weight,
            scale=scale,
            bias=bias,
            smooth=smooth,
            lora=branch,
            smooth_fused=False,
            float_point=float_point,
            subscale=subscale,
        )

        for key, value in layer_state_dict.items():
            converted[f"{converted_local_name}.{key}"] = value

    # Copy normalization layers (not quantized)
    for norm_name in ["norm1", "norm2"]:
        norm_prefix = f"{block_name}.{norm_name}"
        for param_name, param in candidates.items():
            if param_name.startswith(norm_prefix):
                local_key = param_name[len(block_name) + 1 :]
                converted[local_key] = param.clone().cpu()

    return converted


def convert_to_nunchaku_sam2_state_dicts(
    state_dict: dict[str, torch.Tensor],
    scale_dict: dict[str, torch.Tensor],
    smooth_dict: dict[str, torch.Tensor],
    branch_dict: dict[str, torch.Tensor],
    float_point: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Convert SAM2 model to nunchaku format.

    Args:
        state_dict: Model state dict.
        scale_dict: Scale state dict.
        smooth_dict: Smooth scales dict.
        branch_dict: Branch (SVDQuant) dict.
        float_point: Whether to use float-point 4-bit.

    Returns:
        Tuple of (converted_state_dict, other_state_dict).
    """
    # Find all Hiera blocks
    block_names: set[str] = set()
    other: dict[str, torch.Tensor] = {}

    for param_name in state_dict.keys():
        # SAM2 Hiera blocks: vision_encoder.backbone.blocks.X
        if "backbone.blocks." in param_name:
            parts = param_name.split(".")
            # Find the blocks.X part
            for i, part in enumerate(parts):
                if part == "blocks" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    block_name = ".".join(parts[: i + 2])
                    block_names.add(block_name)
                    break
            else:
                other[param_name] = state_dict[param_name]
        else:
            other[param_name] = state_dict[param_name]

    block_names = sorted(block_names, key=lambda x: int(x.split(".")[-1]))
    print(f"Converting {len(block_names)} SAM2 Hiera blocks...")

    converted: dict[str, torch.Tensor] = {}
    for block_name in tqdm.tqdm(block_names, desc="Converting blocks"):
        block_state_dict = convert_to_nunchaku_sam2_block_state_dict(
            state_dict=state_dict,
            scale_dict=scale_dict,
            smooth_dict=smooth_dict,
            branch_dict=branch_dict,
            block_name=block_name,
            float_point=float_point,
        )

        for key, value in block_state_dict.items():
            converted[f"{block_name}.{key}"] = value

    return converted, other


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SAM2 quantized model to Nunchaku format")
    parser.add_argument("--quant-path", type=str, required=True, help="Path to the quantization checkpoint directory.")
    parser.add_argument("--output-root", type=str, default="", help="Root to the output checkpoint directory.")
    parser.add_argument("--model-name", type=str, default=None, help="Name of the model.")
    parser.add_argument("--float-point", action="store_true", help="Use float-point 4-bit quantization.")
    args = parser.parse_args()

    if not args.output_root:
        args.output_root = args.quant_path
    if args.model_name is None:
        args.model_name = "sam2_quantized"

    model_name = args.model_name
    state_dict_path = os.path.join(args.quant_path, "model.pt")
    scale_dict_path = os.path.join(args.quant_path, "scale.pt")
    smooth_dict_path = os.path.join(args.quant_path, "smooth.pt")
    branch_dict_path = os.path.join(args.quant_path, "branch.pt")

    map_location = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"

    print(f"Loading model from {args.quant_path}...")
    state_dict = torch.load(state_dict_path, map_location=map_location)
    scale_dict = torch.load(scale_dict_path, map_location="cpu")
    smooth_dict = torch.load(smooth_dict_path, map_location=map_location) if os.path.exists(smooth_dict_path) else {}
    branch_dict = torch.load(branch_dict_path, map_location=map_location) if os.path.exists(branch_dict_path) else {}

    converted_state_dict, other_state_dict = convert_to_nunchaku_sam2_state_dicts(
        state_dict=state_dict,
        scale_dict=scale_dict,
        smooth_dict=smooth_dict,
        branch_dict=branch_dict,
        float_point=args.float_point,
    )

    output_dirpath = os.path.join(args.output_root, model_name)
    os.makedirs(output_dirpath, exist_ok=True)

    safetensors.torch.save_file(converted_state_dict, os.path.join(output_dirpath, "hiera_blocks.safetensors"))
    safetensors.torch.save_file(other_state_dict, os.path.join(output_dirpath, "unquantized_layers.safetensors"))

    print(f"Quantized SAM2 model saved to {output_dirpath}")
