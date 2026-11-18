# Face Clustering (Optimized)

Fuzzy C-Means clustering of grayscale images. Place your images in `Dataset/` beside `faceclustering.py`.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python faceclustering.py
python faceclustering.py --dataset Dataset --clusters 3 --save results.png --no-show
```

Options: `--width`, `--height`, `--seed`, `--max-iterations`, `--save`, `--no-show`.

## Improvements

- Matrix-based squared distances reduce extra memory from O(N*C*D) to O(N*C) for N images, C clusters and D pixels.
- Stable membership updates via log-weights and exact-zero handling.
- Deterministic seed; input validation; proper centers for final membership.
- Image reading validation and optional result export.

Note: Pixel-based clustering is sensitive to image alignment, lighting, and background. It does not identify people and does not use face embeddings.
