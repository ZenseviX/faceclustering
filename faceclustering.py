from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp"}


def read_dataset(directory: Path, size: tuple[int, int] = (32, 32)):
    """อ่านภาพและแปลงแต่ละภาพเป็นเวกเตอร์ที่มีค่าอยู่ระหว่าง 0 ถึง 1"""
    pictures = []
    filenames = []
    features = []

    for image_path in sorted(directory.iterdir()):
        if image_path.suffix.lower() not in IMAGE_TYPES:
            continue

        picture = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if picture is None:
            continue

        pictures.append(picture)
        filenames.append(image_path.name)
        resized = cv2.resize(picture, size)
        features.append(resized.astype(np.float32).reshape(-1) / 255.0)

    if not features:
        raise FileNotFoundError(f"ไม่พบไฟล์ภาพที่รองรับใน {directory}")

    return pictures, filenames, np.asarray(features)


def fuzzy_c_means(
    data: np.ndarray,
    clusters: int,
    fuzziness: float = 2.0,
    tolerance: float = 1e-5,
    max_iterations: int = 300,
):
    """แบ่งข้อมูลเป็นกลุ่มด้วย Fuzzy C-Means และคืนค่า center กับ membership"""
    if len(data) < clusters:
        raise ValueError("จำนวนภาพต้องไม่น้อยกว่าจำนวนกลุ่ม")
    if fuzziness <= 1:
        raise ValueError("fuzziness ต้องมากกว่า 1")

    generator = np.random.default_rng(42)
    membership = generator.random((len(data), clusters))
    membership /= membership.sum(axis=1, keepdims=True)

    for _ in range(max_iterations):
        previous = membership.copy()
        weights = membership**fuzziness
        centers = (weights.T @ data) / weights.sum(axis=0)[:, None]

        distances = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
        zero_distance = distances == 0

        if zero_distance.any():
            membership.fill(0)
            for row in range(len(data)):
                exact_centers = zero_distance[row]
                if exact_centers.any():
                    membership[row, exact_centers] = 1 / exact_centers.sum()
                else:
                    inverse = distances[row] ** (-2 / (fuzziness - 1))
                    membership[row] = inverse / inverse.sum()
        else:
            inverse = distances ** (-2 / (fuzziness - 1))
            membership = inverse / inverse.sum(axis=1, keepdims=True)

        if np.max(np.abs(membership - previous)) < tolerance:
            break
    else:
        print(f"คำเตือน: ยังไม่ converged ภายใน {max_iterations} รอบ")

    return centers, membership


def show_results(images, filenames, labels, cluster_count: int):
    columns = min(cluster_count, 3)
    rows = int(np.ceil(len(images) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 4, rows * 3))
    axes = np.atleast_1d(axes).ravel()

    for index, axis in enumerate(axes):
        axis.axis("off")
        if index >= len(images):
            continue

        group = labels[index] + 1
        print(f"{filenames[index]} -> Cluster {group}")
        axis.imshow(images[index], cmap="gray")
        axis.set_title(f"{filenames[index]}\nCluster {group}")

    figure.tight_layout()
    plt.show()


def main():
    dataset_path = Path(__file__).resolve().parent / "Dataset"
    images, filenames, data = read_dataset(dataset_path)

    number_of_clusters = 3
    _, membership = fuzzy_c_means(data, clusters=number_of_clusters)
    labels = np.argmax(membership, axis=1)
    show_results(images, filenames, labels, number_of_clusters)


if __name__ == "__main__":
    main()
