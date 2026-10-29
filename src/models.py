"""Model definitions: a convolutional autoencoder and a dance-form classifier.

Both are small enough to train from scratch on a CPU in minutes, which is the
point — the dataset has roughly 1,200 training images, so a large backbone would
memorise rather than learn.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """Symmetric convolutional autoencoder with a linear bottleneck.

    Four stride-2 convolutions take a 96x96 image to a 6x6 feature map, which
    the linear layers compress to `latent_dim` numbers. The decoder mirrors it.
    """

    def __init__(self, latent_dim: int = 128, image_size: int = 96, base: int = 24):
        super().__init__()
        if image_size % 16 != 0:
            raise ValueError("image_size must be divisible by 16")
        reduced = image_size // 16
        channels = base * 4  # deepest encoder width

        self.encoder = nn.Sequential(
            nn.Conv2d(3, base, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base * 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 3, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 3, channels, 4, 2, 1),
            nn.ReLU(inplace=True),
        )
        flat = channels * reduced * reduced
        self.to_latent = nn.Linear(flat, latent_dim)
        self.from_latent = nn.Linear(latent_dim, flat)
        self.channels = channels
        self.reduced = reduced
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels, base * 3, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base * 3, base * 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        return self.to_latent(features.flatten(1))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(images)
        features = self.from_latent(latent).view(
            -1, self.channels, self.reduced, self.reduced
        )
        return self.decoder(features), latent


class DanceClassifier(nn.Module):
    """Small VGG-style CNN with batch norm and global average pooling."""

    def __init__(
        self,
        num_classes: int = 8,
        dropout: float = 0.3,
        widths: tuple[int, int, int] = (24, 48, 96),
    ):
        super().__init__()

        def block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        w1, w2, w3 = widths
        self.features = nn.Sequential(
            block(3, w1),    # /2
            block(w1, w2),   # /4
            block(w2, w3),   # /8
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(widths[2], num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(images))
