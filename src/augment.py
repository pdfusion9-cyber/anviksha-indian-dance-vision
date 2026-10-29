"""On-the-fly augmentation for the dance images.

The dataset has roughly 1,200 training images, which is small enough that the
network memorises quickly. These transforms are cheap, vectorised over a batch,
and applied to uint8 arrays before they are converted to tensors.

Every transform preserves the semantic content of a dance photograph: the
horizontal flip is safe because no dance form is defined by left-right
orientation, and the crop range keeps the dancer in frame.
"""
from __future__ import annotations

import numpy as np


def random_hflip(images: np.ndarray, rng: np.random.Generator, p: float = 0.5) -> np.ndarray:
    """Flip a random subset of the batch left-to-right."""
    flip = rng.random(len(images)) < p
    if flip.any():
        images[flip] = images[flip][:, :, ::-1]
    return images


def random_crop(images: np.ndarray, rng: np.random.Generator, pad: int = 4) -> np.ndarray:
    """Pad by `pad` pixels then take a random crop of the original size."""
    if pad <= 0:
        return images
    count, height, width, channels = images.shape
    padded = np.pad(images, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="reflect")
    out = np.empty_like(images)
    span = 2 * pad + 1
    offsets_y = rng.integers(0, span, size=count)
    offsets_x = rng.integers(0, span, size=count)
    for i in range(count):
        y, x = offsets_y[i], offsets_x[i]
        out[i] = padded[i, y : y + height, x : x + width]
    return out


def colour_jitter(
    images: np.ndarray,
    rng: np.random.Generator,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.15,
) -> np.ndarray:
    """Brightness, contrast and saturation jitter, applied per image."""
    work = images.astype(np.float32)
    count = len(work)

    brightness_factor = 1.0 + rng.uniform(-brightness, brightness, size=count)
    work *= brightness_factor[:, None, None, None]

    grey = work.mean(axis=3, keepdims=True)
    contrast_factor = 1.0 + rng.uniform(-contrast, contrast, size=count)
    work = (work - grey) * contrast_factor[:, None, None, None] + grey

    saturation_factor = 1.0 + rng.uniform(-saturation, saturation, size=count)
    work = grey + (work - grey) * saturation_factor[:, None, None, None]

    return np.clip(work, 0, 255).astype(np.uint8)


def augment_batch(images: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply the full augmentation chain. Input is modified in place and returned."""
    images = random_hflip(images, rng)
    images = random_crop(images, rng, pad=4)
    images = colour_jitter(images, rng)
    return images
