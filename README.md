# Anviksha — Indian Dance Vision

Recognising the eight classical Indian dance forms from photographs, trained
from scratch on an openly licensed Indian image dataset.

This is not a benchmark-chasing project. It is a small, fully reproducible
reference implementation: every number below was produced by the scripts in
this repository, on CPU, from the public dataset linked below.

![Validation predictions](results/figures/predictions.png)

## Dataset

**[tanmaykm/indian_dance_forms](https://huggingface.co/datasets/tanmaykm/indian_dance_forms)** — Apache-2.0.

| Split | Images | Labels |
|---|---|---|
| train | 1,188 | 8 classes, ~150 each |
| valid | 364 | 8 classes, 36–50 each |
| test | 100 | **none** — ships unlabelled |

Classes: `bharatanatyam`, `kathak`, `kathakali`, `kuchipudi`, `manipuri`,
`mohiniyattam`, `odissi`, `sattriya`.

The dataset is not vendored into this repository. `src/download_data.py` fetches
it from the Hugging Face Hub and is safe to re-run after an interruption.

## Results

All figures below are validation scores on the 364 held-out images, from the
best checkpoint (epoch 21 of 24).

| Metric | Value |
|---|---|
| Validation accuracy | **73.90 %** |
| Chance level (8 classes) | 12.50 % |
| Macro precision | 76.25 % |
| Macro recall | 74.01 % |
| Macro F1 | 73.68 % |
| Classifier parameters | 162,800 |
| Input resolution | 64 × 64 RGB |
| Training time | ~17 min on 2 CPU cores |

Per class, on the same held-out split:

| Class | Precision | Recall | F1 | n |
|---|---|---|---|---|
| bharatanatyam | 84.0 % | 44.7 % | 58.3 % | 47 |
| kathak | 63.5 % | 90.9 % | 74.8 % | 44 |
| kathakali | 82.1 % | 97.9 % | 89.3 % | 47 |
| kuchipudi | 62.2 % | 60.9 % | 61.5 % | 46 |
| manipuri | 96.4 % | 75.0 % | 84.4 % | 36 |
| mohiniyattam | 88.9 % | 80.0 % | 84.2 % | 50 |
| odissi | 73.9 % | 69.4 % | 71.6 % | 49 |
| sattriya | 58.9 % | 73.3 % | 65.3 % | 45 |

![Confusion matrix](results/figures/confusion_matrix.png)

The confusion matrix is the interesting part. Kathakali is close to solved
(97.9 % recall) because of its distinctive green-and-red makeup and large
headdress. Bharatanatyam has the highest precision but the lowest recall — it is
rarely predicted wrongly, but it is often missed, and it leaks into kuchipudi
and odissi, which share costume silhouettes and dark stage lighting.

## Autoencoder

A separate convolutional autoencoder answers a different question: how much of a
dance photograph survives compression into a small latent code?

| Metric | Value |
|---|---|
| Latent dimension | 128 |
| Compression ratio | 96 : 1 (64×64×3 → 128) |
| Parameters | 766,211 |
| Best validation MSE | 0.01795 |
| Reconstruction PSNR | 17.46 dB |
| Epochs | 24 |

![Reconstructions](results/figures/reconstructions.png)

At 96:1 compression the model keeps pose, costume colour, stage lighting and
overall silhouette, and loses facial detail and fabric texture. That is the
expected ceiling for this bottleneck size, not a training failure.

![Training curves](results/figures/training_curves.png)

## Project layout

```
src/download_data.py      fetch the dataset from the Hugging Face Hub
src/data.py               decode + resize images into uint8 arrays
src/models.py             ConvAutoencoder and DanceClassifier
src/optim.py              Adam (see note below)
src/train_classifier.py   train the 8-class classifier
src/train_autoencoder.py  train the autoencoder
src/evaluate.py           per-class metrics, confusion matrix, test predictions
src/make_report.py        build the figures in results/figures
configs/default.json      the exact settings used for the published run
results/evaluation.json   full evaluation output
```

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/download_data.py --out data/dance_raw
python src/train_classifier.py --data data/dance_raw/data --size 64 --batch 32 \
    --lr 1e-3 --epochs 24 --out runs/classifier
python src/train_autoencoder.py --data data/dance_raw/data --size 64 --latent 128 \
    --batch 32 --epochs 24 --out runs/autoencoder
python src/evaluate.py --data data/dance_raw/data \
    --checkpoint runs/classifier/best.pt --out results
python src/make_report.py --data data/dance_raw/data
```

Training is resumable. Both training scripts checkpoint the model, the optimiser
state and the epoch counter, and accept `--time-budget` to stop cleanly. Resuming
restores the Adam moment estimates, so the loss does not spike on the first epoch
after a restart.

## Note on the optimiser

`src/optim.py` contains a small Adam implementation instead of using
`torch.optim.Adam`. Importing `torch.optim` pulls in `torch._dynamo`, which
exceeds the 2 GB memory limit of the container this was trained in and kills
training before the first step. The maths is standard Adam; it also carries
`state_dict`/`load_state_dict` so runs resume exactly.

## Honest limits

- **1,188 training images is small.** The model memorises costume and stage
  cues well; it does not learn choreography or mudra vocabulary. It would not
  generalise to dancers photographed in everyday clothing.
- **The validation split is also small** (364 images), so a single image is
  0.27 percentage points. Validation accuracy swung between 57 % and 74 % across
  the last few epochs. Treat 73.9 % as the best observed point, not a stable
  converged estimate.
- **The test split has no labels.** `evaluate.py` predicts on it and records the
  distribution, but nothing here can score it.
- **No pretrained weights.** Everything is trained from scratch on 64×64 inputs.
  A pretrained backbone would beat this easily; that is a different experiment.

## Licence

Code in this repository is Apache-2.0. The dataset is Apache-2.0 by
`tanmaykm` and is downloaded, not redistributed, by this project.
