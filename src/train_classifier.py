"""Train the 8-class Indian classical dance classifier.

Example:
    python src/train_classifier.py --epochs 40 --out runs/classifier

Resumable: the checkpoint stores model weights, optimiser state and the epoch
counter, so an interrupted run continues without the moment estimates resetting
to zero (which causes a loss spike on the first epoch after a restart).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import CLASSES, load_split, summarise, to_float  # noqa: E402
from models import DanceClassifier  # noqa: E402
from optim import Adam  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/dance_raw/data", type=Path)
    parser.add_argument("--out", default="runs/classifier", type=Path)
    parser.add_argument("--size", default=96, type=int)
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--batch", default=64, type=int)
    parser.add_argument("--lr", default=2e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--threads", default=6, type=int)
    parser.add_argument(
        "--time-budget",
        default=0.0,
        type=float,
        help="stop cleanly after this many seconds and leave a resumable checkpoint",
    )
    return parser.parse_args()


def batches(count: int, batch: int, rng: np.random.Generator):
    order = rng.permutation(count)
    for start in range(0, count, batch):
        yield order[start : start + batch]


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    train = load_split(args.data, "train", args.size)
    valid = load_split(args.data, "valid", args.size)
    summarise(train, "train")
    summarise(valid, "valid")

    model = DanceClassifier(num_classes=len(CLASSES))
    optimiser = Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    parameters = sum(p.numel() for p in model.parameters())
    print(f"parameters: {parameters:,}")

    args.out.mkdir(parents=True, exist_ok=True)
    last_path = args.out / "last.pt"
    best_path = args.out / "best.pt"

    start_epoch = 0
    history: list[dict] = []
    best_accuracy = 0.0
    if last_path.exists():
        checkpoint = torch.load(last_path, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimiser.load_state_dict(checkpoint["optimiser"])
        start_epoch = checkpoint["epoch"]
        history = checkpoint["history"]
        best_accuracy = checkpoint["best_accuracy"]
        print(f"resumed at epoch {start_epoch}, best val acc {best_accuracy:.4f}")

    def evaluate() -> tuple[float, float]:
        model.eval()
        correct = 0
        loss_total = 0.0
        with torch.no_grad():
            for start in range(0, len(valid.images), args.batch):
                chunk = slice(start, start + args.batch)
                inputs = torch.from_numpy(to_float(valid.images[chunk]))
                targets = torch.from_numpy(valid.labels[chunk])
                logits = model(inputs)
                loss_total += F.cross_entropy(logits, targets).item() * len(targets)
                correct += (logits.argmax(1) == targets).sum().item()
        return loss_total / len(valid.images), correct / len(valid.images)

    started = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = 0.0
        epoch_rng = np.random.default_rng(epoch)
        for index in batches(len(train.images), args.batch, epoch_rng):
            inputs = torch.from_numpy(to_float(train.images[index]))
            targets = torch.from_numpy(train.labels[index])
            optimiser.zero_grad()
            loss = F.cross_entropy(model(inputs), targets)
            loss.backward()
            optimiser.step()
            running += loss.item() * len(index)
        train_loss = running / len(train.images)

        val_loss, val_accuracy = evaluate()
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )
        print(
            f"epoch {epoch + 1:3d}/{args.epochs}  train {train_loss:.4f}  "
            f"val {val_loss:.4f}  acc {val_accuracy * 100:.2f}%  "
            f"({time.time() - started:.0f}s)",
            flush=True,
        )

        torch.save(
            {
                "model": model.state_dict(),
                "optimiser": optimiser.state_dict(),
                "epoch": epoch + 1,
                "history": history,
                "best_accuracy": max(best_accuracy, val_accuracy),
                "classes": CLASSES,
                "image_size": args.size,
            },
            last_path,
        )
        if val_accuracy >= best_accuracy:
            best_accuracy = val_accuracy
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_accuracy": val_accuracy,
                    "classes": CLASSES,
                    "image_size": args.size,
                },
                best_path,
            )
            print(f"  new best val accuracy {val_accuracy * 100:.2f}%")

        if args.time_budget and time.time() - started > args.time_budget:
            print("time budget reached; rerun to resume")
            break

    metrics = {
        "parameters": parameters,
        "epochs_completed": len(history),
        "best_val_accuracy": best_accuracy,
        "final_val_accuracy": history[-1]["val_accuracy"] if history else 0.0,
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "history": history,
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"best validation accuracy: {best_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
