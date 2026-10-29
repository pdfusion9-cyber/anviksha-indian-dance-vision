"""Build the figures that go into results/figures.

Reads the checkpoints and the evaluation JSON produced by the training and
evaluation scripts, runs inference, and writes PNGs. Nothing here invents
numbers; every plotted value comes from a stored metric or a live forward pass.

Example:
    python src/make_report.py --out results/figures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import CLASSES, load_split, to_float  # noqa: E402
from models import ConvAutoencoder, DanceClassifier  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/dance_raw/data", type=Path)
    parser.add_argument("--classifier", default="runs/classifier/best.pt", type=Path)
    parser.add_argument("--autoencoder", default="runs/autoencoder/best.pt", type=Path)
    parser.add_argument("--evaluation", default="results/evaluation.json", type=Path)
    parser.add_argument("--out", default="results/figures", type=Path)
    return parser.parse_args()


def plot_training(out: Path) -> None:
    classifier = json.loads(Path("runs/classifier/metrics.json").read_text())
    autoencoder = json.loads(Path("runs/autoencoder/metrics.json").read_text())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor="white")

    history = classifier["history"]
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="valid")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy")
    axes[0].set_title("Classifier loss")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, [row["val_accuracy"] * 100 for row in history],
                 color="#d62728", lw=2)
    axes[1].axhline(12.5, ls="--", lw=1, color="grey", label="chance (8 classes)")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation accuracy (%)")
    axes[1].set_title("Classifier accuracy")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)

    ae_history = autoencoder["history"]
    axes[2].plot([r["epoch"] for r in ae_history], [r["train_loss"] for r in ae_history],
                 label="train")
    axes[2].plot([r["epoch"] for r in ae_history], [r["val_loss"] for r in ae_history],
                 label="valid")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("MSE")
    axes[2].set_title("Autoencoder loss")
    axes[2].legend(frameon=False)
    axes[2].grid(alpha=0.25)

    fig.suptitle("Training on the Indian Dance Forms dataset", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(evaluation: dict, out: Path) -> None:
    matrix = np.asarray(evaluation["confusion_matrix"], dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(matrix, row_sums, out=np.zeros_like(matrix),
                           where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(7.5, 6.5), facecolor="white")
    image = ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    classes = evaluation["classes"]
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    for i in range(len(classes)):
        for j in range(len(classes)):
            value = normalised[i, j]
            if value > 0:
                ax.text(j, i, f"{value * 100:.0f}", ha="center", va="center",
                        fontsize=8, color="white" if value > 0.5 else "#222")
    ax.set_title(
        f"Confusion matrix — validation accuracy "
        f"{evaluation['validation_accuracy'] * 100:.1f}% (n={evaluation['n_validation']})"
    )
    fig.colorbar(image, ax=ax, fraction=0.046, label="fraction of true class")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_reconstructions(args: argparse.Namespace, out: Path) -> None:
    checkpoint = torch.load(args.autoencoder, weights_only=False)
    size = checkpoint.get("image_size", 64)
    model = ConvAutoencoder(latent_dim=checkpoint["latent"], image_size=size)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    valid = load_split(args.data, "valid", size)
    picks = np.random.RandomState(0).choice(len(valid.images), 6, replace=False)
    batch = torch.from_numpy(to_float(valid.images[picks]))
    with torch.no_grad():
        reconstruction, _ = model(batch)

    originals = batch.numpy().transpose(0, 2, 3, 1)
    rebuilt = reconstruction.numpy().transpose(0, 2, 3, 1)

    gap = 4
    tile = size
    canvas = np.ones((2 * tile + 3 * gap, len(picks) * (tile + gap) + gap, 3))
    for column in range(len(picks)):
        x = gap + column * (tile + gap)
        canvas[gap : gap + tile, x : x + tile] = originals[column]
        canvas[2 * gap + tile : 2 * gap + 2 * tile, x : x + tile] = rebuilt[column]

    fig, ax = plt.subplots(figsize=(13, 5), facecolor="white")
    ax.imshow(np.clip(canvas, 0, 1), interpolation="nearest")
    ax.axis("off")
    ax.set_title(
        f"Autoencoder — top: originals, bottom: reconstructions "
        f"({checkpoint['latent']}-d latent, {size}x{size}, "
        f"{size * size * 3 / checkpoint['latent']:.0f}:1 compression)"
    )
    fig.tight_layout()
    fig.savefig(out / "reconstructions.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_predictions(args: argparse.Namespace, out: Path) -> None:
    checkpoint = torch.load(args.classifier, weights_only=False)
    size = checkpoint.get("image_size", 64)
    classes = checkpoint.get("classes", CLASSES)
    model = DanceClassifier(num_classes=len(classes))
    model.load_state_dict(checkpoint["model"])
    model.eval()

    valid = load_split(args.data, "valid", size)
    picks = np.random.RandomState(1).choice(len(valid.images), 8, replace=False)
    batch = torch.from_numpy(to_float(valid.images[picks]))
    with torch.no_grad():
        probabilities = model(batch).softmax(1)
    guesses = probabilities.argmax(1).tolist()
    confidences = probabilities.max(1).values.tolist()

    fig, axes = plt.subplots(2, 4, figsize=(13, 7), facecolor="white")
    for ax, index, guess, confidence in zip(
        axes.ravel(), picks, guesses, confidences
    ):
        ax.imshow(valid.images[index])
        truth = classes[valid.labels[index]]
        predicted = classes[guess]
        colour = "#1a7f37" if predicted == truth else "#c62828"
        ax.set_title(f"true: {truth}\npred: {predicted} ({confidence * 100:.0f}%)",
                     fontsize=9, color=colour)
        ax.axis("off")
    fig.suptitle("Held-out validation predictions", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "predictions.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plot_training(args.out)
    plot_confusion(json.loads(args.evaluation.read_text()), args.out)
    plot_reconstructions(args, args.out)
    plot_predictions(args, args.out)
    print(f"wrote figures to {args.out}")


if __name__ == "__main__":
    main()
