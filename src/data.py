"""Dataset loading and preprocessing for the Indian Dance Forms images.

Images are decoded once into a uint8 array so that every epoch reads from RAM
instead of re-decoding JPEGs. That keeps training fast on a CPU-only machine.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = (
    "bharatanatyam",
    "kathak",
    "kathakali",
    "kuchipudi",
    "manipuri",
    "mohiniyattam",
    "odissi",
    "sattriya",
)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


@dataclass
class Split:
    images: np.ndarray  # uint8, (N, size, size, 3)
    labels: np.ndarray  # int64, (N,)
    paths: list[str]


def _load_one(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as handle:
        image = handle.convert("RGB").resize((size, size), Image.BILINEAR)
        return np.asarray(image, dtype=np.uint8)


def load_split(root: Path, split: str, size: int, labelled: bool = True) -> Split:
    """Load one split. Returns uint8 images in NHWC order."""
    directory = root / split
    if not directory.is_dir():
        raise FileNotFoundError(f"split directory not found: {directory}")

    images: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []

    if labelled:
        for class_index, class_name in enumerate(CLASSES):
            class_dir = directory / class_name
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.iterdir()):
                if path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                images.append(_load_one(path, size))
                labels.append(class_index)
                paths.append(str(path.relative_to(root)))
    else:
        for path in sorted(directory.rglob("*")):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            images.append(_load_one(path, size))
            labels.append(-1)
            paths.append(str(path.relative_to(root)))

    if not images:
        raise RuntimeError(f"no images found under {directory}")

    return Split(
        images=np.stack(images),
        labels=np.asarray(labels, dtype=np.int64),
        paths=paths,
    )


def to_float(batch: np.ndarray) -> np.ndarray:
    """uint8 NHWC -> float32 NCHW in [0, 1]."""
    return batch.astype(np.float32).transpose(0, 3, 1, 2) / 255.0


def summarise(split: Split, name: str) -> None:
    counts = np.bincount(split.labels[split.labels >= 0], minlength=len(CLASSES))
    present = [f"{CLASSES[i]}={counts[i]}" for i in range(len(CLASSES)) if counts[i]]
    print(f"{name}: {len(split.images)} images ({', '.join(present) or 'unlabelled'})")
