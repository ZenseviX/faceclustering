"""Face-image clustering with memory-efficient Fuzzy C-Means.

Install: pip install numpy opencv-python matplotlib
Run: python faceclustering_optimized.py --dataset Dataset --clusters 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp"}


def read_dataset(directory: Path, size: tuple[int, int] = (32, 32)):
    """Load grayscale images and normalized float32 feature vectors."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์ Dataset: {directory}")
    if len(size) != 2 or any(x <= 0 for x in size):
        raise ValueError("size ต้องเป็น (width, height) ที่มากกว่า 0")

    pictures, filenames, features = [], [], []
    for image_path in sorted(directory.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_TYPES:
            continue
        picture = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if picture is None:
            print(f"คำเตือน: อ่านภาพไม่ได้ ข้าม {image_path.name}")
            continue
        resized = cv2.resize(picture, size, interpolation=cv2.INTER_AREA)
        pictures.append(picture)
        filenames.append(image_path.name)
        features.append(resized.reshape(-1).astype(np.float32) / 255.0)

    if not features:
        raise FileNotFoundError(f"ไม่พบไฟล์ภาพที่อ่านได้ใน {directory}")
    return pictures, filenames, np.stack(features)


def fuzzy_c_means(
    data: np.ndarray,
    clusters: int,
    fuzziness: float = 2.0,
    tolerance: float = 1e-5,
    max_iterations: int = 300,
    seed: int = 42,
):
    """Return (centers, memberships) using a vectorized FCM update.

    Distance is computed via dot products, avoiding an N x C x D array.
    Exact matches to centers get equal membership among matching centers.
    """
    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 2 or not x.shape[0] or not x.shape[1]:
        raise ValueError("data ต้องเป็นอาร์เรย์ 2 มิติที่ไม่ว่าง")
    if not np.isfinite(x).all():
        raise ValueError("data มี NaN หรือ Infinity")
    if not isinstance(clusters, (int, np.integer)) or not 1 <= clusters <= len(x):
        raise ValueError("clusters ต้องอยู่ระหว่าง 1 ถึงจำนวนภาพ")
    if not np.isfinite(fuzziness) or fuzziness <= 1:
        raise ValueError("fuzziness ต้องมากกว่า 1")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance ต้องมากกว่า 0")
    if max_iterations < 1:
        raise ValueError("max_iterations ต้องมากกว่า 0")

    rng = np.random.default_rng(seed)
    membership = rng.random((len(x), clusters))
    membership /= membership.sum(axis=1, keepdims=True)
    x_norm2 = np.einsum("ij,ij->i", x, x)[:, None]
    exponent = -1.0 / (fuzziness - 1.0)  # squared distances

    def calculate_centers(u: np.ndarray) -> np.ndarray:
        weights = u ** fuzziness
        denom = weights.sum(axis=0)
        return (weights.T @ x) / denom[:, None]

    for iteration in range(max_iterations):
        centers = calculate_centers(membership)
        # O(N*C) auxiliary memory rather than O(N*C*D).
        dist2 = x_norm2 + np.einsum("ij,ij->i", centers, centers)[None, :] - 2.0 * (x @ centers.T)
        np.maximum(dist2, 0.0, out=dist2)
        # Dot-product distance can suffer cancellation near exact matches.
        # Recompute potentially zero distances directly to distinguish them.
        near_zero = dist2 <= 1e-12 * np.maximum(x_norm2, 1.0)
        if near_zero.any():
            rr, cc = np.nonzero(near_zero)
            delta = x[rr] - centers[cc]
            dist2[rr, cc] = np.einsum("ij,ij->i", delta, delta)

        exact = dist2 == 0.0
        exact_rows = exact.any(axis=1)
        updated = np.empty_like(membership)
        if exact_rows.any():
            exact_matches = exact[exact_rows]
            updated[exact_rows] = exact_matches / exact_matches.sum(axis=1, keepdims=True)

        ordinary_rows = ~exact_rows
        if ordinary_rows.any():
            d = dist2[ordinary_rows]
            # Log-sum-exp style normalization avoids power overflow/underflow.
            log_w = exponent * np.log(d)
            log_w -= log_w.max(axis=1, keepdims=True)
            w = np.exp(log_w)
            updated[ordinary_rows] = w / w.sum(axis=1, keepdims=True)

        change = np.max(np.abs(updated - membership))
        membership = updated
        if change < tolerance:
            break
    else:
        print(f"คำเตือน: ยังไม่ converged ภายใน {max_iterations} รอบ")

    # Return centers consistent with the final membership (not the previous step).
    return calculate_centers(membership), membership


def show_results(images, filenames, labels, cluster_count: int, save_path: Path | None = None, show: bool = True):
    if not images:
        raise ValueError("ไม่มีภาพสำหรับแสดงผล")
    columns = min(max(cluster_count, 1), 3)
    rows = (len(images) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 4, rows * 3), squeeze=False)

    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index < len(images):
            group = int(labels[index]) + 1
            print(f"{filenames[index]} -> Cluster {group}")
            axis.imshow(images[index], cmap="gray", vmin=0, vmax=255)
            axis.set_title(f"{filenames[index]}\nCluster {group}")

    figure.tight_layout()
    if save_path is not None:
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target, dpi=150, bbox_inches="tight")
        print(f"บันทึกกราฟ: {target}")
    if show:
        plt.show()
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Cluster face images using Fuzzy C-Means")
    parser.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parent / "Dataset")
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--save", type=Path, help="Save montage image (PNG, etc.)")
    parser.add_argument("--no-show", action="store_true", help="Do not open a plot window")
    args = parser.parse_args()

    images, filenames, data = read_dataset(args.dataset, (args.width, args.height))
    centers, membership = fuzzy_c_means(
        data, clusters=args.clusters, seed=args.seed, max_iterations=args.max_iterations
    )
    labels = np.argmax(membership, axis=1)
    print(f"ภาพทั้งหมด {len(images)} ภาพ, {len(centers)} กลุ่ม")
    show_results(images, filenames, labels, args.clusters, args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
