"""Train the convolutional autoencoder on the dance images.

The autoencoder is not the classifier. It exists to answer a separate question:
how much of a dance image survives compression into a small latent code, and
does that code preserve the information needed to tell the dance forms apart.

Example:
    python src/train_autoencoder.py --epochs 30 --latent 128 --out runs/autoencoder
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

from data import load_split, summarise, to_float  # noqa: E402
from models import ConvAutoencoder  # noqa: E402
from optim import Adam  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/dance_raw/data", type=Path)
    parser.add_argument("--out", default="runs/autoencoder", type=Path)
    parser.add_argument("--size", default=96, type=int)
    parser.add_argument("--latent", default=128, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--batch", default=32, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--threads", default=6, type=int)
    parser.add_argument("--time-budget", default=0.0, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    train = load_split(args.data, "train", args.size)
    valid = load_split(args.data, "valid", args.size)
    summarise(train, "train")
    summarise(valid, "valid")

    model = ConvAutoencoder(latent_dim=args.latent, image_size=args.size)
    optimiser = Adam(model.parameters(), lr=args.lr)
    parameters = sum(p.numel() for p in model.parameters())
    ratio = (args.size * args.size * 3) / args.latent
    print(f"parameters: {parameters:,}  compression: {ratio:.0f}:1")

    args.out.mkdir(parents=True, exist_ok=True)
    last_path = args.out / "last.pt"
    best_path = args.out / "best.pt"

    start_epoch = 0
    history: list[dict] = []
    best_loss = float("inf")
    if last_path.exists():
        checkpoint = torch.load(last_path, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimiser.load_state_dict(checkpoint["optimiser"])
        start_epoch = checkpoint["epoch"]
        history = checkpoint["history"]
        best_loss = checkpoint["best_loss"]
        print(f"resumed at epoch {start_epoch}, best val loss {best_loss:.5f}")

    def evaluate() -> float:
        model.eval()
        total = 0.0
        with torch.no_grad():
            for start in range(0, len(valid.images), args.batch):
                chunk = slice(start, start + args.batch)
                inputs = torch.from_numpy(to_float(valid.images[chunk]))
                reconstruction, _ = model(inputs)
                total += F.mse_loss(reconstruction, inputs).item() * len(inputs)
        return total / len(valid.images)

    started = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        order = np.random.default_rng(epoch).permutation(len(train.images))
        running = 0.0
        for start in range(0, len(order), args.batch):
            index = order[start : start + args.batch]
            inputs = torch.from_numpy(to_float(train.images[index]))
            optimiser.zero_grad()
            reconstruction, _ = model(inputs)
            loss = F.mse_loss(reconstruction, inputs)
            loss.backward()
            optimiser.step()
            running += loss.item() * len(index)
        train_loss = running / len(train.images)

        val_loss = evaluate()
        psnr = float(10 * np.log10(1.0 / max(val_loss, 1e-12)))
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "psnr_db": psnr,
            }
        )
        print(
            f"epoch {epoch + 1:3d}/{args.epochs}  train {train_loss:.5f}  "
            f"val {val_loss:.5f}  psnr {psnr:.2f} dB  ({time.time() - started:.0f}s)",
            flush=True,
        )

        torch.save(
            {
                "model": model.state_dict(),
                "optimiser": optimiser.state_dict(),
                "epoch": epoch + 1,
                "history": history,
                "best_loss": float(min(best_loss, val_loss)),
                "latent": args.latent,
                "image_size": args.size,
            },
            last_path,
        )
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "latent": args.latent,
                    "image_size": args.size,
                },
                best_path,
            )
            print(f"  new best val loss {val_loss:.5f}")

        if args.time_budget and time.time() - started > args.time_budget:
            print("time budget reached; rerun to resume")
            break

    metrics = {
        "parameters": parameters,
        "latent": args.latent,
        "image_size": args.size,
        "compression_ratio": ratio,
        "epochs_completed": len(history),
        "best_val_loss": best_loss,
        "best_psnr_db": 10 * np.log10(1.0 / max(best_loss, 1e-12)),
        "history": history,
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"best val loss {best_loss:.5f} ({metrics['best_psnr_db']:.2f} dB)")


if __name__ == "__main__":
    main()
