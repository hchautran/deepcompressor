# -*- coding: utf-8 -*-
"""SAM2 calibration dataset."""

import os
import random
import typing as tp
from dataclasses import dataclass, field

import torch
from omniconfig import configclass
from PIL import Image
from torch.utils.data import DataLoader, Dataset

__all__ = ["Sam2CalibConfig", "Sam2CalibDataset"]


@configclass
@dataclass
class Sam2CalibConfig:
    """SAM2 calibration dataset configuration.

    Args:
        path (`str`):
            The path to the calibration dataset (directory of images).
        num_samples (`int`, *optional*, defaults to `128`):
            The number of samples to use for calibration.
        batch_size (`int`, *optional*, defaults to `1`):
            The batch size for calibration.
        image_size (`int`, *optional*, defaults to `1024`):
            The image size for SAM2 input.
        seed (`int`, *optional*, defaults to `42`):
            The random seed for sampling.
    """

    path: str = ""
    num_samples: int = 128
    batch_size: int = 1
    image_size: int = 1024
    seed: int = 42

    def generate_dirnames(self) -> list[str]:
        """Generate directory names for caching."""
        names = []
        if self.path:
            dataset_name = os.path.basename(self.path.rstrip("/"))
            names.append(f"data.{dataset_name}")
        names.append(f"n{self.num_samples}")
        return names

    def build_dataloader(self) -> DataLoader:
        """Build a dataloader for calibration."""
        dataset = Sam2CalibDataset(self)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )


class Sam2CalibDataset(Dataset):
    """SAM2 calibration dataset.

    Loads images from a directory for calibration purposes.
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, config: Sam2CalibConfig):
        """Initialize the calibration dataset.

        Args:
            config: Calibration configuration.
        """
        self.config = config
        self.image_size = config.image_size

        # Collect image paths
        self.image_paths = self._collect_images(config.path)

        # Sample images
        random.seed(config.seed)
        if len(self.image_paths) > config.num_samples:
            self.image_paths = random.sample(self.image_paths, config.num_samples)
        elif len(self.image_paths) < config.num_samples:
            # If not enough images, repeat
            self.image_paths = self.image_paths * (config.num_samples // len(self.image_paths) + 1)
            self.image_paths = self.image_paths[: config.num_samples]

        self._processor = None

    def _collect_images(self, path: str) -> list[str]:
        """Collect image paths from directory."""
        if not path or not os.path.exists(path):
            return []

        image_paths = []
        for root, dirs, files in os.walk(path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in self.SUPPORTED_EXTENSIONS:
                    image_paths.append(os.path.join(root, file))

        return sorted(image_paths)

    @property
    def processor(self):
        """Lazy load the SAM2 processor."""
        if self._processor is None:
            try:
                from transformers import Sam2Processor

                self._processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-tiny")
            except Exception:
                self._processor = None
        return self._processor

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a calibration sample.

        Returns:
            Dictionary with 'pixel_values' tensor.
        """
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.processor is not None:
            inputs = self.processor(images=image, return_tensors="pt")
            return {"pixel_values": inputs["pixel_values"].squeeze(0)}
        else:
            # Fallback: simple resize and normalize
            from torchvision import transforms

            transform = transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            pixel_values = transform(image)
            return {"pixel_values": pixel_values}


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate function for DataLoader."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    return {"pixel_values": pixel_values}
