# -*- coding: utf-8 -*-
"""SAM2 quantization utilities."""

import typing as tp
from collections import defaultdict
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepcompressor.utils import tools

from ..nn.struct import Sam2HieraBlockStruct, Sam2ModelStruct

__all__ = [
    "get_sam2_block_quantizable_modules",
    "filter_sam2_block_modules_by_key",
    "ActivationStatsCollector",
    "collect_activation_statistics",
]


def get_sam2_block_quantizable_modules(
    block: Sam2HieraBlockStruct,
    *,
    include_attn: bool = True,
    include_ffn: bool = True,
) -> dict[str, tuple[str, nn.Module]]:
    """Get quantizable modules from a SAM2 Hiera block.

    Args:
        block: SAM2 Hiera block structure.
        include_attn: Whether to include attention modules.
        include_ffn: Whether to include FFN modules.

    Returns:
        Dictionary mapping module keys to (name, module) tuples.
    """
    modules = {}

    if include_attn:
        for attn_struct in block.attn_structs:
            for key, name, module, parent, fname in attn_struct.named_key_modules():
                modules[name] = (key, module)

    if include_ffn and block.ffn_struct is not None:
        for key, name, module, parent, fname in block.ffn_struct.named_key_modules():
            modules[name] = (key, module)

    return modules


def filter_sam2_block_modules_by_key(
    block: Sam2HieraBlockStruct,
    *,
    skips: tp.Sequence[str] = (),
    includes: tp.Sequence[str] = (),
) -> dict[str, tuple[str, nn.Module]]:
    """Filter block modules by key.

    Args:
        block: SAM2 Hiera block structure.
        skips: Keys to skip.
        includes: Keys to include (if empty, include all).

    Returns:
        Filtered dictionary of modules.
    """
    all_modules = get_sam2_block_quantizable_modules(block)
    if not skips and not includes:
        return all_modules

    filtered = {}
    for name, (key, module) in all_modules.items():
        if includes and key not in includes:
            continue
        if key in skips:
            continue
        filtered[name] = (key, module)

    return filtered


def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    """Get a module by its name path."""
    parts = name.split(".")
    current = model
    for part in parts:
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def set_module_by_name(model: nn.Module, name: str, new_module: nn.Module) -> None:
    """Set a module by its name path."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)

    last_part = parts[-1]
    if last_part.isdigit():
        parent[int(last_part)] = new_module
    else:
        setattr(parent, last_part, new_module)


class ActivationStatsCollector:
    """Collects activation statistics during forward passes.

    Tracks per-channel min, max, mean, and variance for each registered module's
    input and output activations.
    """

    def __init__(self):
        self.stats: dict[str, dict[str, torch.Tensor]] = {}
        self.hooks: list = []
        self._counts: dict[str, int] = defaultdict(int)

    def _create_input_hook(self, name: str):
        """Create a forward hook to collect input activation statistics."""

        def hook(module: nn.Module, inputs: tuple, output) -> None:
            if not inputs or inputs[0] is None:
                return

            inp = inputs[0]
            if isinstance(inp, tuple):
                inp = inp[0]

            if not isinstance(inp, torch.Tensor):
                return

            # Flatten to (batch * seq, features)
            inp_flat = inp.detach().float().reshape(-1, inp.shape[-1])

            key = f"{name}.input"
            self._update_stats(key, inp_flat)

        return hook

    def _create_output_hook(self, name: str):
        """Create a forward hook to collect output activation statistics."""

        def hook(module: nn.Module, inputs: tuple, output) -> None:
            out = output
            if isinstance(out, tuple):
                out = out[0]

            if not isinstance(out, torch.Tensor):
                return

            # Flatten to (batch * seq, features)
            out_flat = out.detach().float().reshape(-1, out.shape[-1])

            key = f"{name}.output"
            self._update_stats(key, out_flat)

        return hook

    def _update_stats(self, key: str, tensor: torch.Tensor) -> None:
        """Update running statistics for a given key."""
        # tensor shape: (N, C) where N is flattened batch*seq, C is channels
        batch_min = tensor.min(dim=0)[0]
        batch_max = tensor.max(dim=0)[0]
        batch_mean = tensor.mean(dim=0)
        batch_var = tensor.var(dim=0)
        batch_abs_max = tensor.abs().max(dim=0)[0]
        n = tensor.shape[0]

        if key not in self.stats:
            self.stats[key] = {
                "min": batch_min,
                "max": batch_max,
                "mean": batch_mean,
                "var": batch_var,
                "abs_max": batch_abs_max,
                "count": n,
            }
            self._counts[key] = 1
        else:
            # Running update for min/max
            self.stats[key]["min"] = torch.minimum(self.stats[key]["min"], batch_min)
            self.stats[key]["max"] = torch.maximum(self.stats[key]["max"], batch_max)
            self.stats[key]["abs_max"] = torch.maximum(self.stats[key]["abs_max"], batch_abs_max)

            # Welford's online algorithm for mean and variance
            old_count = self.stats[key]["count"]
            new_count = old_count + n
            delta = batch_mean - self.stats[key]["mean"]

            self.stats[key]["mean"] = self.stats[key]["mean"] + delta * n / new_count

            # Update variance using parallel algorithm
            m_a = self.stats[key]["var"] * old_count
            m_b = batch_var * n
            m2 = m_a + m_b + delta.pow(2) * old_count * n / new_count
            self.stats[key]["var"] = m2 / new_count

            self.stats[key]["count"] = new_count
            self._counts[key] += 1

    def register_module(
        self,
        module: nn.Module,
        name: str,
        collect_input: bool = True,
        collect_output: bool = True,
    ) -> None:
        """Register a module for activation statistics collection.

        Args:
            module: The module to monitor.
            name: Name identifier for the module.
            collect_input: Whether to collect input statistics.
            collect_output: Whether to collect output statistics.
        """
        if collect_input:
            hook = module.register_forward_hook(self._create_input_hook(name))
            self.hooks.append(hook)

        if collect_output:
            hook = module.register_forward_hook(self._create_output_hook(name))
            self.hooks.append(hook)

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get_input_stats(self, name: str) -> dict[str, torch.Tensor] | None:
        """Get input activation statistics for a module."""
        key = f"{name}.input"
        return self.stats.get(key)

    def get_output_stats(self, name: str) -> dict[str, torch.Tensor] | None:
        """Get output activation statistics for a module."""
        key = f"{name}.output"
        return self.stats.get(key)

    def get_input_scale(self, name: str) -> torch.Tensor | None:
        """Get the activation scale (abs_max) for a module's input."""
        stats = self.get_input_stats(name)
        if stats is None:
            return None
        return stats["abs_max"]

    def get_output_scale(self, name: str) -> torch.Tensor | None:
        """Get the activation scale (abs_max) for a module's output."""
        stats = self.get_output_stats(name)
        if stats is None:
            return None
        return stats["abs_max"]

    def to_state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        """Convert collected statistics to a saveable state dict."""
        state_dict = {}
        for key, stats in self.stats.items():
            state_dict[key] = {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in stats.items()
            }
        return state_dict

    @classmethod
    def from_state_dict(cls, state_dict: dict) -> "ActivationStatsCollector":
        """Load statistics from a state dict."""
        collector = cls()
        collector.stats = state_dict
        return collector


def collect_activation_statistics(
    model: Sam2ModelStruct,
    dataloader: DataLoader,
    device: str | torch.device = "cuda",
    collect_input: bool = True,
    collect_output: bool = True,
    max_samples: int | None = None,
) -> ActivationStatsCollector:
    """Collect activation statistics by running inference on calibration data.

    Args:
        model: SAM2 model structure.
        dataloader: DataLoader providing calibration images.
        device: Device to run inference on.
        collect_input: Whether to collect input statistics.
        collect_output: Whether to collect output statistics.
        max_samples: Maximum number of samples to process.

    Returns:
        ActivationStatsCollector with collected statistics.
    """
    logger = tools.logging.getLogger(__name__)

    collector = ActivationStatsCollector()

    # Register hooks for all quantizable modules
    for block_idx, block in enumerate(model.block_structs):
        # Register attention modules
        for attn_struct in block.attn_structs:
            if attn_struct.q_proj is not None:
                collector.register_module(
                    attn_struct.q_proj,
                    attn_struct.q_proj_name,
                    collect_input=collect_input,
                    collect_output=collect_output,
                )
            if attn_struct.k_proj is not None:
                collector.register_module(
                    attn_struct.k_proj,
                    attn_struct.k_proj_name,
                    collect_input=collect_input,
                    collect_output=collect_output,
                )
            if attn_struct.v_proj is not None:
                collector.register_module(
                    attn_struct.v_proj,
                    attn_struct.v_proj_name,
                    collect_input=collect_input,
                    collect_output=collect_output,
                )
            if attn_struct.o_proj is not None:
                collector.register_module(
                    attn_struct.o_proj,
                    attn_struct.o_proj_name,
                    collect_input=collect_input,
                    collect_output=collect_output,
                )

        # Register FFN modules
        if block.ffn_struct is not None:
            for up_proj, up_name in zip(block.ffn_struct.up_projs, block.ffn_struct.up_proj_names):
                collector.register_module(
                    up_proj,
                    up_name,
                    collect_input=collect_input,
                    collect_output=collect_output,
                )
            for down_proj, down_name in zip(block.ffn_struct.down_projs, block.ffn_struct.down_proj_names):
                collector.register_module(
                    down_proj,
                    down_name,
                    collect_input=collect_input,
                    collect_output=collect_output,
                )

    # Run inference on calibration data
    logger.info(f"Collecting activation statistics from {len(dataloader)} batches")

    model.module.eval()
    samples_processed = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Calibration")):
            if max_samples is not None and samples_processed >= max_samples:
                break

            pixel_values = batch["pixel_values"]
            if isinstance(pixel_values, torch.Tensor):
                pixel_values = pixel_values.to(device)

            # Run through vision encoder only
            try:
                # SAM2 vision encoder forward
                if hasattr(model.module, "vision_encoder"):
                    model.module.vision_encoder(pixel_values)
                elif hasattr(model.module, "image_encoder"):
                    model.module.image_encoder(pixel_values)
                else:
                    # Try full forward with dummy inputs
                    model.module(pixel_values)
            except Exception as e:
                logger.warning(f"Batch {batch_idx} failed: {e}")
                continue

            samples_processed += pixel_values.shape[0]

    # Clean up hooks
    collector.remove_hooks()

    logger.info(f"Collected statistics from {samples_processed} samples")
    logger.info(f"Statistics available for {len(collector.stats)} tensors")

    return collector


@contextmanager
def collect_hessians(
    model: Sam2ModelStruct,
    modules: dict[str, nn.Module],
):
    """Context manager to collect Hessian approximations for GPTQ.

    Collects X^T @ X for each module's input activations, which approximates
    the Hessian for weight quantization optimization.

    Args:
        model: SAM2 model structure.
        modules: Dictionary of module names to modules to collect for.

    Yields:
        Dictionary of module names to Hessian tensors.
    """
    hessians: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = defaultdict(int)
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
            else:
                hessians[name] = hessians[name] + h

            counts[name] += inp_flat.shape[0]

        return hook

    # Register hooks
    for name, module in modules.items():
        hook = module.register_forward_hook(create_hook(name))
        hooks.append(hook)

    try:
        yield hessians, counts
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Normalize Hessians by count
        for name in hessians:
            if counts[name] > 0:
                hessians[name] = hessians[name] / counts[name]
