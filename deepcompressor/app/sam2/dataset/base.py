# -*- coding: utf-8 -*-
"""Dataset for SAM2 calibration."""

import os
import random
import typing as tp

import torch
import torch.utils.data
from PIL import Image

from deepcompressor.utils.common import tree_collate

__all__ = ["SAM2Dataset", "SAM2ImageDataset"]


class SAM2Dataset(torch.utils.data.Dataset):
    """Base dataset class for SAM2 calibration."""

    path: str
    filenames: list[str]
    filepaths: list[str]

    def __init__(
        self,
        path: str,
        num_samples: int = -1,
        seed: int = 0,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
    ) -> None:
        if not os.path.exists(path):
            raise ValueError(f"Invalid data path: {path}")

        self.path = path
        filenames = []
        for f in sorted(os.listdir(path)):
            if any(f.lower().endswith(ext) for ext in extensions):
                filenames.append(f)

        if num_samples > 0 and num_samples < len(filenames):
            random.Random(seed).shuffle(filenames)
            filenames = sorted(filenames[:num_samples])

        self.filenames = filenames
        self.filepaths = [os.path.join(path, f) for f in filenames]

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> dict[str, tp.Any]:
        raise NotImplementedError

    def build_loader(self, **kwargs) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self, collate_fn=tree_collate, **kwargs)


class SAM2ImageDataset(SAM2Dataset):
    """Dataset for loading images for SAM2 calibration.

    Returns raw images that will be processed by SAM2's image encoder.
    """

    def __init__(
        self,
        path: str,
        num_samples: int = -1,
        seed: int = 0,
        image_size: int = 1024,
    ) -> None:
        super().__init__(path, num_samples=num_samples, seed=seed)
        self.image_size = image_size

    def __getitem__(self, idx: int) -> dict[str, tp.Any]:
        filepath = self.filepaths[idx]
        filename = self.filenames[idx]

        # Load image
        image = Image.open(filepath).convert("RGB")

        return {
            "filename": filename,
            "filepath": filepath,
            "image": image,
        }

    @staticmethod
    def collate_fn(batch: list[dict[str, tp.Any]]) -> dict[str, tp.Any]:
        """Custom collate function for image batches."""
        return {
            "filename": [item["filename"] for item in batch],
            "filepath": [item["filepath"] for item in batch],
            "image": [item["image"] for item in batch],
        }

    def build_loader(self, **kwargs) -> torch.utils.data.DataLoader:
        kwargs.setdefault("collate_fn", self.collate_fn)
        return torch.utils.data.DataLoader(self, **kwargs)
