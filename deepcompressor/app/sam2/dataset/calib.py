# -*- coding: utf-8 -*-
"""SAM2 calibration dataset utilities."""

import os
import typing as tp
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

__all__ = [
    "Sam2CalibDataset",
    "COCOCalibDataset",
    "SA1BCalibDataset",
    "get_coco_calibration_loader",
    "get_sa1b_calibration_loader",
]


class Sam2CalibDataset(Dataset):
    """Base calibration dataset for SAM2."""

    def __init__(
        self,
        image_paths: list[str],
        processor: tp.Any = None,
        image_size: int = 1024,
    ):
        """
        Args:
            image_paths: List of paths to images
            processor: SAM2 processor for preprocessing
            image_size: Target image size
        """
        self.image_paths = image_paths
        self.processor = processor
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Load and preprocess image."""
        image_path = self.image_paths[idx]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Preprocess with processor if available
        if self.processor is not None:
            inputs = self.processor(images=image, return_tensors="pt")
            # Remove batch dimension added by processor
            return {k: v.squeeze(0) if v.ndim > 0 else v for k, v in inputs.items()}
        else:
            # Simple preprocessing without processor
            import torchvision.transforms as transforms

            transform = transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            pixel_values = transform(image)
            return {"pixel_values": pixel_values}


class COCOCalibDataset(Sam2CalibDataset):
    """COCO calibration dataset for SAM2."""

    def __init__(
        self,
        coco_root: str,
        split: str = "val2017",
        num_samples: int = 128,
        processor: tp.Any = None,
        image_size: int = 1024,
    ):
        """
        Args:
            coco_root: Root directory of COCO dataset
            split: Dataset split (train2017, val2017)
            num_samples: Number of samples to use
            processor: SAM2 processor
            image_size: Target image size
        """
        images_dir = os.path.join(coco_root, split)

        # Get image paths
        if os.path.exists(images_dir):
            image_files = sorted([f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png"))])
            image_paths = [os.path.join(images_dir, f) for f in image_files[:num_samples]]
        else:
            raise ValueError(f"COCO directory not found: {images_dir}")

        super().__init__(image_paths, processor, image_size)


class SA1BCalibDataset(Sam2CalibDataset):
    """SA-1B calibration dataset for SAM2."""

    def __init__(
        self,
        sa1b_root: str,
        num_samples: int = 128,
        processor: tp.Any = None,
        image_size: int = 1024,
    ):
        """
        Args:
            sa1b_root: Root directory of SA-1B dataset
            num_samples: Number of samples to use
            processor: SAM2 processor
            image_size: Target image size
        """
        # Collect image paths from SA-1B structure
        image_paths = []
        sa1b_path = Path(sa1b_root)

        if sa1b_path.exists():
            # SA-1B has subdirectories with images
            for img_path in sa1b_path.rglob("*.jpg"):
                image_paths.append(str(img_path))
                if len(image_paths) >= num_samples:
                    break

            if len(image_paths) == 0:
                raise ValueError(f"No images found in SA-1B directory: {sa1b_root}")
        else:
            raise ValueError(f"SA-1B directory not found: {sa1b_root}")

        super().__init__(image_paths[:num_samples], processor, image_size)


def get_coco_calibration_loader(
    coco_root: str,
    split: str = "val2017",
    num_samples: int = 128,
    batch_size: int = 1,
    processor: tp.Any = None,
    image_size: int = 1024,
    num_workers: int = 4,
) -> DataLoader:
    """Get COCO calibration data loader.

    Args:
        coco_root: Root directory of COCO dataset
        split: Dataset split (train2017, val2017)
        num_samples: Number of calibration samples
        batch_size: Batch size
        processor: SAM2 processor
        image_size: Target image size
        num_workers: Number of data loading workers

    Returns:
        DataLoader for calibration
    """
    dataset = COCOCalibDataset(
        coco_root=coco_root,
        split=split,
        num_samples=num_samples,
        processor=processor,
        image_size=image_size,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def get_sa1b_calibration_loader(
    sa1b_root: str,
    num_samples: int = 128,
    batch_size: int = 1,
    processor: tp.Any = None,
    image_size: int = 1024,
    num_workers: int = 4,
) -> DataLoader:
    """Get SA-1B calibration data loader.

    Args:
        sa1b_root: Root directory of SA-1B dataset
        num_samples: Number of calibration samples
        batch_size: Batch size
        processor: SAM2 processor
        image_size: Target image size
        num_workers: Number of data loading workers

    Returns:
        DataLoader for calibration
    """
    dataset = SA1BCalibDataset(
        sa1b_root=sa1b_root,
        num_samples=num_samples,
        processor=processor,
        image_size=image_size,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
