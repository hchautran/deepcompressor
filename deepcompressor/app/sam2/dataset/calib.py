# -*- coding: utf-8 -*-
"""Calibration dataset for SAM2 models."""

import typing as tp
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn
from omniconfig import configclass

from deepcompressor.data.cache import IOTensorsCache, ModuleForwardInput, TensorCache, TensorsCache
from deepcompressor.data.utils.reshape import LinearReshapeFn
from deepcompressor.dataset.action import CacheAction, ConcatCacheAction
from deepcompressor.dataset.cache import BaseCalibCacheLoader
from deepcompressor.dataset.config import BaseDataLoaderConfig

from ..nn.struct import SAM2BlockStruct, SAM2ModelStruct
from .base import SAM2ImageDataset

__all__ = [
    "SAM2CalibConfig",
    "SAM2CalibDataset",
    "SAM2CalibCacheLoader",
]


@configclass
@dataclass(kw_only=True)
class SAM2CalibConfig(BaseDataLoaderConfig):
    """Configuration for SAM2 calibration dataset.

    Args:
        data (`str`):
            Dataset name (e.g., "COCO").
        num_samples (`int`):
            Number of calibration samples.
        batch_size (`int`):
            Batch size for calibration.
        path (`str`):
            Path to the calibration images directory.
        num_workers (`int`):
            Number of workers for data loading.
        image_size (`int`):
            Image size for SAM2 processing.
    """

    path: str = ""
    num_workers: int = 4
    image_size: int = 1024

    def build_dataset(self) -> "SAM2CalibDataset":
        """Build the calibration dataset."""
        return SAM2CalibDataset(
            self.path,
            num_samples=self.num_samples,
            image_size=self.image_size,
        )

    def build_loader(self) -> "SAM2CalibCacheLoader":
        """Build the calibration cache loader."""
        return SAM2CalibCacheLoader(self)


class SAM2CalibDataset(SAM2ImageDataset):
    """Calibration dataset for SAM2 models.

    Loads images from COCO or other image datasets for calibration.
    """

    pass


class SAM2ConcatCacheAction(ConcatCacheAction):
    """Cache action for SAM2 calibration that concatenates tensors."""

    def info(
        self,
        name: str,
        module: nn.Module,
        tensors: dict[int | str, torch.Tensor],
        cache: TensorsCache,
    ) -> None:
        """Update cache information based on tensor shapes."""
        for key, tensor in tensors.items():
            if key in cache.tensors:
                tensor_cache = cache.tensors[key]
                if tensor_cache.channels_dim is None:
                    # For SAM2, hidden states are typically (B, H, W, C) or (B, N, C)
                    if tensor.ndim == 4:
                        tensor_cache.channels_dim = -1  # (B, H, W, C)
                    else:
                        tensor_cache.channels_dim = -1  # (B, N, C)
                    tensor_cache.reshape = LinearReshapeFn()
        return super().info(name, module, tensors, cache)


class SAM2CalibCacheLoader(BaseCalibCacheLoader):
    """Calibration cache loader for SAM2 models."""

    config: SAM2CalibConfig
    dataset: SAM2CalibDataset

    def __init__(self, config: SAM2CalibConfig) -> None:
        super().__init__(dataset=config.build_dataset(), batch_size=config.batch_size)
        self.config = config

    def _init_cache(self, name: str, module: nn.Module) -> IOTensorsCache:
        """Initialize cache for a module."""
        # For SAM2 blocks, we cache hidden states
        return IOTensorsCache(
            inputs=TensorsCache(
                OrderedDict(
                    hidden_states=TensorCache(channels_dim=-1, reshape=LinearReshapeFn()),
                )
            ),
            outputs=TensorCache(channels_dim=-1, reshape=LinearReshapeFn()),
        )

    def iter_samples(self) -> tp.Generator[ModuleForwardInput, None, None]:
        """Iterate over calibration samples."""
        dataloader = self.dataset.build_loader(
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.config.num_workers,
        )
        for batch in dataloader:
            yield ModuleForwardInput(
                args=(),
                kwargs={"images": batch["image"]},
            )

    def iter_layer_activations(
        self,
        model: nn.Module | SAM2ModelStruct,
        *args,
        needs_inputs_fn: tp.Callable[[str, nn.Module], bool],
        needs_outputs_fn: tp.Callable[[str, nn.Module], bool] | None = None,
        action: CacheAction | None = None,
        skip_pre_modules: bool = True,
        skip_post_modules: bool = True,
        **kwargs,
    ) -> tp.Generator[
        tuple[
            str,
            tuple[
                SAM2BlockStruct | nn.Module,
                dict[str, IOTensorsCache],
                dict[str, tp.Any],
            ],
        ],
        None,
        None,
    ]:
        """Iterate over model block activations for calibration.

        Args:
            model: SAM2 model or model structure.
            needs_inputs_fn: Function to determine if inputs should be cached.
            needs_outputs_fn: Function to determine if outputs should be cached.
            action: Cache action to use.
            skip_pre_modules: Whether to skip pre-modules.
            skip_post_modules: Whether to skip post-modules.

        Yields:
            Tuple of (layer_name, (layer_struct, layer_cache, layer_kwargs))
        """
        if not isinstance(model, SAM2ModelStruct):
            model_struct = SAM2ModelStruct.construct(model)
        else:
            model_struct = model
            model = model_struct.module

        action = SAM2ConcatCacheAction("cpu") if action is None else action

        layers, layer_structs, recomputes, use_prev_layer_outputs = model_struct.get_iter_layer_activations_args(
            skip_pre_modules=skip_pre_modules,
            skip_post_modules=skip_post_modules,
        )

        for layer_idx, (layer_name, (layer, layer_cache, layer_inputs)) in enumerate(
            self._iter_layer_activations(
                model,
                *args,
                action=action,
                layers=layers,
                needs_inputs_fn=needs_inputs_fn,
                needs_outputs_fn=needs_outputs_fn,
                recomputes=recomputes,
                use_prev_layer_outputs=use_prev_layer_outputs,
                **kwargs,
            )
        ):
            layer_kwargs = {}
            layer_struct = layer_structs[layer_idx]
            if isinstance(layer_struct, SAM2BlockStruct):
                assert layer_struct.name == layer_name
                assert layer is layer_struct.module
            yield layer_name, (layer_struct, layer_cache, layer_kwargs)
