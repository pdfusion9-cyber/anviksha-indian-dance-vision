"""Evaluate the trained classifier.

Produces, from real inference only:
  - validation accuracy, macro precision/recall/F1
  - a confusion matrix (JSON and PNG)
  - predictions for the unlabelled test split, which has no ground truth

Example:
    python src/evaluate.py --checkpoint runs/classifier/best.pt --out results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import CLASSES, load_split, to_float  # noqa: E402
from models import DanceClassifier  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/dance_raw/data", type=Path)
    parser.add_argument("--checkpoint", default="runs/classifier/best.pt", type=Path)
    parser.add_argument("--out", default="results", type=Path)
    parser.add_argument("--batch", default=64, type=int)
    parser.add_argument("--threads", default=6, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)

    checkpoint = torch.load(args.checkpoint, weights_only=False)
    classes = checkpoint.get("classes", CLASSES)
    size = checkpoint.get("image_size", 96)
    model = DanceClassifier(num_classes=len(classes))
    model.load_state_dict(checkpoint["model"])
    model.eval()

    valid = load_split(args.data, "valid", size)
    predictions: list[int] = []
    with torch.no_grad():
        for start in range(0, len(valid.images), args.batch):
            chunk = slice(start, start + args.batch)
            logits = model(torch.from_numpy(to_float(valid.images[chunk])))
            predictions.extend(logits.argmax(1).tolist())
    predicted = np.asarray(predictions)
    truth = valid.labels

    accuracy = float((predicted == truth).mean())
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for actual, guess in zip(truth, predicted):
        matrix[actual, guess] += 1

    per_class = {}
    for index, name in enumerate(classes):
        tp = int(matrix[index, index])
        predicted_positive = int(matrix[:, index].sum())
        support = int(matrix[index].sum())
        precision = tp / predicted_positive if predicted_positive else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    macro = {
        key: float(np.mean([per_class[name][key] for name in classes]))
        for key in ("precision", "recall", "f1")
    }

    print(f"validation accuracy: {accuracy * 100:.2f}%")
    print(f"macro precision {macro['precision'] * 100:.2f}%  "
          f"recall {macro['recall'] * 100:.2f}%  f1 {macro['f1'] * 100:.2f}%")
    for name in classes:
        stats = per_class[name]
        print(f"  {name:16s} precision {stats['precision'] * 100:5.1f}%  "
              f"recall {stats['recall'] * 100:5.1f}%  n={stats['support']}")

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(args.checkpoint),
        "epoch": checkpoint.get("epoch"),
        "validation_accuracy": accuracy,
        "macro": macro,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "classes": list(classes),
        "n_validation": int(len(truth)),
    }

    test_dir = args.data / "test"
    if test_dir.is_dir():
        test = load_split(args.data, "test", size, labelled=False)
        test_predictions: list[int] = []
        with torch.no_grad():
            for start in range(0, len(test.images), args.batch):
                chunk = slice(start, start + args.batch)
                logits = model(torch.from_numpy(to_float(test.images[chunk])))
                test_predictions.extend(logits.argmax(1).tolist())
        counts = np.bincount(test_predictions, minlength=len(classes))
        payload["test_split"] = {
            "n_images": int(len(test.images)),
            "note": "the test split ships without labels, so these are "
                    "predictions only and cannot be scored",
            "predicted_counts": {
                name: int(counts[i]) for i, name in enumerate(classes)
            },
            "predictions": [
                {"path": path, "predicted": classes[pred]}
                for path, pred in zip(test.paths, test_predictions)
            ],
        }
        print(f"test split: {len(test.images)} unlabelled images predicted")

    (args.out / "evaluation.json").write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out / 'evaluation.json'}")


if __name__ == "__main__":
    main()
