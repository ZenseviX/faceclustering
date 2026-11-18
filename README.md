# Face Clustering — Optimized

Clustering grayscale images with **Fuzzy C-Means (FCM)**. All original pixel-based clustering functionality is retained, with vectorized distance calculations, stable updates, a reproducible random seed, input validation, and CSV export.

> **Scope:** This is unsupervised pixel clustering, **not** face identity recognition. Results can be strongly affected by alignment, lighting and backgrounds.

## Requirements

Python 3.10+ is recommended. Install:

```powershell
python -m pip install -r requirements.txt
```

Put your own `.jpg`, `.jpeg`, `.png`, or `.bmp` images in a `Dataset/` folder beside `faceclustering.py`. Images are not included in this package.

## Run

```powershell
python faceclustering.py --dataset Dataset --clusters 3
```

Run without opening a graph window, and save CSV plus a result image:

```powershell
python faceclustering.py --dataset Dataset --clusters 3 --csv results/clusters.csv --save results/montage.png --no-show
```

Additional options: `--width`, `--height`, `--seed`, `--fuzziness`, `--tolerance`, `--max-iterations`, `--max-preview`, `--save`, `--csv`, `--no-show`. Use `python faceclustering.py --help` for help.

The CSV includes each filename, winning cluster, winning membership score and membership to every cluster. Cluster numbers start at 1. A large image collection only displays up to 60 montage thumbnails by default; CSV still includes every image.

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Improvements

- Memory-efficient matrix squared distances instead of materializing an `images × clusters × pixels` distance tensor.
- Robust normalization, near-zero distance handling, and an explicit one-cluster fast path.
- Protection against zero-weight cluster division in center calculations.
- Input and parameter validation, deterministic seed, readable errors for missing Dataset.
- Configurable membership CSV output and optional visualization; cap montage size to avoid enormous figures.

Performance improvement depends on dataset size and hardware; no speedup percentage is claimed without a paired benchmark.

## GitHub

Copy `faceclustering.py`, `README.md`, `requirements.txt`, `.gitignore`, and `tests/` into your **existing cloned repository**. Do not overwrite or delete `.git/`. Review changed files with `git status` and commit/push using your own Git identity.
