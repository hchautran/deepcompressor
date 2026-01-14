# -*- coding: utf-8 -*-
"""Calibration dataset for SAM2 Hiera."""

import os
import random
import typing as tp
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.utils.data
from omniconfig import configclass
from PIL import Image

from deepcompressor.data.cache import IOTensorsCache, ModuleForwardInput
from deepcompressor.dataset.action import CacheAction, ConcatCacheAction
from deepcompressor.dataset.cache import BaseCalibCacheLoader
from deepcompressor.dataset.config import BaseDataLoaderConfig

from ..nn.struct import Sam2BlockStruct, Sam2ModelStruct

try:  # Optional import for runtime.
    from sam2.utils.transforms import SAM2Transforms
except Exception:  # pragma: no cover
    SAM2Transforms = None

__all__ = [
    "Sam2CalibCacheLoaderConfig",
    "Sam2CalibDataset",
    "Sam2CalibCacheLoader",
]


@configclass
@dataclass(kw_only=True)
class Sam2CalibCacheLoaderConfig(BaseDataLoaderConfig):
    """Configuration for SAM2 calibration dataset.

    Args:
        data (`str`): Dataset name.
        num_samples (`int`): Number of samples.
        batch_size (`int`): Batch size.
        path (`str`): Path to image directory.
        image_size (`int`): Resize resolution for SAM2.
        num_workers (`int`): DataLoader workers.
        seed (`int`): Shuffle seed.
    """

    data: str = "coco"
    path: str
    image_size: int = 1024
    num_workers: int = 8
    seed: int = 42

    def __post_init__(self) -> None:
        self.path = os.path.abspath(os.path.expanduser(self.path)) if self.path else ""

    def build_dataset(self) -> "Sam2CalibDataset":
        return Sam2CalibDataset(
            self.path,
            num_samples=self.num_samples,
            seed=self.seed,
            image_size=self.image_size,
        )

    def build_loader(self) -> "Sam2CalibCacheLoader":
        return Sam2CalibCacheLoader(self)


class Sam2CalibDataset(torch.utils.data.Dataset):
    """Simple image folder dataset for SAM2 calibration."""

    def __init__(self, path: str, num_samples: int, seed: int, image_size: int) -> None:
        if not path or not os.path.isdir(path):
            raise ValueError(f"Invalid calibration path: {path}")
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        filenames = [f for f in sorted(os.listdir(path)) if f.lower().endswith(exts)]
        if num_samples > 0 and num_samples < len(filenames):
            random.Random(seed).shuffle(filenames)
            filenames = sorted(filenames[:num_samples])
        if not filenames:
            raise ValueError(f"No images found under: {path}")
        self.filepaths = [os.path.join(path, name) for name in filenames]
        if SAM2Transforms is None:
            raise RuntimeError("SAM2Transforms not available; ensure sam2 is installed")
        self.transforms = SAM2Transforms(resolution=image_size, mask_threshold=0.0)

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        with Image.open(self.filepaths[idx]) as img:
            image = img.convert("RGB")
        return self.transforms(image).to(dtype=torch.float16)

    def build_loader(self, **kwargs) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self, **kwargs)


class Sam2CalibCacheLoader(BaseCalibCacheLoader):
    config: Sam2CalibCacheLoaderConfig
    dataset: Sam2CalibDataset

    def __init__(self, config: Sam2CalibCacheLoaderConfig) -> None:
        super().__init__(dataset=config.build_dataset(), batch_size=config.batch_size)
        self.batch_size = min(config.batch_size, len(self.dataset))
        self.config = config

    def iter_samples(self) -> tp.Generator[ModuleForwardInput, None, None]:
        dataloader = self.dataset.build_loader(
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.config.num_workers,
        )
        for batch in dataloader:
            yield ModuleForwardInput(args=[batch], kwargs={})

    def iter_layer_activations(  # noqa: C901
        self,
        model: nn.Module | Sam2ModelStruct,
        *args,
        needs_inputs_fn: tp.Callable[[str, nn.Module], bool],
        needs_outputs_fn: tp.Callable[[str, nn.Module], bool] | None = None,
        action: CacheAction | None = None,
        skip_pre_modules: bool = False,
        skip_post_modules: bool = False,
        **kwargs,
    ) -> tp.Generator[
        tuple[
            str,
            tuple[
                Sam2BlockStruct | nn.Module,
                dict[str, IOTensorsCache],
                dict[str, tp.Any],
            ],
        ],
        None,
        None,
    ]:
        if not isinstance(model, Sam2ModelStruct):
            model_struct = Sam2ModelStruct.construct(model)
        else:
            model_struct = model
            model = model_struct.module
        assert isinstance(model_struct, Sam2ModelStruct)
        assert isinstance(model, nn.Module)
        layers, layer_structs, recomputes, use_prev_layer_outputs = model_struct.get_iter_layer_activations_args(
            skip_pre_modules=skip_pre_modules,
            skip_post_modules=skip_post_modules,
        )
        action = ConcatCacheAction("cpu") if action is None else action
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
            layer_kwargs = {k: v for k, v in layer_inputs[0].kwargs.items()}  # noqa: C416
            layer_struct = layer_structs[layer_idx]
            if isinstance(layer_struct, Sam2BlockStruct):
                assert layer_struct.name == layer_name
                assert layer is layer_struct.module
            yield layer_name, (layer_struct, layer_cache, layer_kwargs)
