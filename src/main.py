from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from enum import Enum
import os
from PIL import Image, ImageDraw
from time import time
from functools import wraps

SEED = 1234


def timer(func):
    """Decorator to time the execution of a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        print(f"Execution time of {func.__name__}: {end_time - start_time:.2f} seconds")
        return result

    return wrapper


# had other types during development, but now only images are used
# May include other types in the future, e.g. tabular data.
class DatasetTypes(Enum):
    IMAGES = "images"


def generate_dataset(
    dataset_type: DatasetTypes,
    save_data: bool = False,
    n_total: int = 200,
    priors: np.ndarray = np.array([0.80, 0.15, 0.05]),
    image_size: int = 28,
    edge_bounding_box: int = 12,
):
    rng = np.random.default_rng(SEED)
    n_per_class = (priors * n_total).astype(int)
    n_per_class[0] += n_total - n_per_class.sum()

    if dataset_type == DatasetTypes.IMAGES:
        X, y = generate_image_shapes(
            rng=rng,
            n_per_class=n_per_class,
            image_size=image_size,
            edge_bounding_box=edge_bounding_box,
        )

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=SEED
    )

    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )

    # X_train, y_train  -> model training
    # X_cal,   y_cal    -> split-CP calibration
    # X_test,  y_test   -> final evaluation

    plot_dataset_distribution(
        dataset_type=dataset_type,
        n_per_class=n_per_class,
        X=X,
        y=y,
        y_train=y_train,
        y_cal=y_cal,
        y_test=y_test,
    )

    if dataset_type == DatasetTypes.IMAGES:
        dump_split("train", X_train, y_train)
        dump_split("calibration", X_cal, y_cal)
        dump_split("test", X_test, y_test)

    if save_data:
        np.savez_compressed(
            "gaussian_blobs.npz",
            X_train=X_train,
            y_train=y_train,
            X_cal=X_cal,
            y_cal=y_cal,
            X_test=X_test,
            y_test=y_test,
        )


def plot_dataset_distribution(dataset_type, n_per_class, X, y, y_train, y_cal, y_test):
    class_names = ["Circle", "Square", "Triangle"]
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    counts = [np.sum(y == i) for i in range(len(n_per_class))]
    ax1.bar(class_names, counts)
    ax1.set_ylabel("Count")
    ax1.set_title("Class Distribution in Dataset")
    for i, count in enumerate(counts):
        ax1.text(i, count + 2, str(count), ha="center")
    # Add legend showing total samples
    ax1.legend([f"Total: {sum(counts)}"], loc="upper right")

    splits = ["Train", "Calibration", "Test"]
    split_counts = [len(y_train), len(y_cal), len(y_test)]
    ax2.bar(splits, split_counts)
    ax2.set_ylabel("Count")
    ax2.set_title("Dataset Split Distribution")
    for i, count in enumerate(split_counts):
        ax2.text(i, count + 2, str(count), ha="center")
    # Add legend showing total samples
    ax2.legend([f"Total: {sum(split_counts)}"], loc="upper right")

    plt.tight_layout()
    output_path = f"assets/dataset_statistics_{sum(split_counts)}.png"
    plt.savefig(output_path, dpi=100)
    print(f"Dataset statistics plot saved to {output_path=}")

    # Sample images for each class
    if dataset_type == DatasetTypes.IMAGES:
        _, axes = plt.subplots(1, 3, figsize=(12, 4))
        for i in range(3):
            idx = np.where(y == i)[0][0]
            axes[i].imshow(X[idx].squeeze(), cmap="gray")
            axes[i].set_title(class_names[i])
            axes[i].axis("off")
        plt.tight_layout()
        output_path = f"assets/sample_images_{sum(split_counts)}.png"
        plt.savefig(output_path, dpi=100)
        print(f"Sample images saved to {output_path=}")


@timer
def generate_image_shapes(
    rng: np.random.Generator,
    n_per_class: np.ndarray,
    image_size: int,
    edge_bounding_box: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate images of simple shapes (circle, square, triangle) to test our hypothesis.

    Args:
        rng (np.random.Generator): random number generator
        n_per_class (np.ndarray): the number of samples per class.
        image_size (int, optional): height and width of image. Defaults to 28 (same as MNIST).
        edge_bounding_box (int, optional): bounding box for the shapes. Defaults to 12 (=approx 40% of image size with image_size=28).

    Returns:
        tuple[np.ndarray, np.ndarray]: X (images) and y (labels)
    """
    print(
        f"Generating image shapes with {n_per_class=}, {image_size=}, {edge_bounding_box=}"
    )

    def draw_circle():
        img = Image.new("L", (image_size, image_size), 0)
        draw = ImageDraw.Draw(img)
        x0 = rng.integers(0, image_size - edge_bounding_box)
        y0 = rng.integers(0, image_size - edge_bounding_box)
        bbox = (x0, y0, x0 + edge_bounding_box, y0 + edge_bounding_box)
        draw.ellipse(bbox, fill=255)
        return np.asarray(img, dtype=np.uint8)

    def draw_square():
        img = Image.new("L", (image_size, image_size), 0)
        draw = ImageDraw.Draw(img)
        x0 = rng.integers(0, image_size - edge_bounding_box)
        y0 = rng.integers(0, image_size - edge_bounding_box)
        bbox = (x0, y0, x0 + edge_bounding_box, y0 + edge_bounding_box)
        draw.rectangle(bbox, fill=255)
        return np.asarray(img, dtype=np.uint8)

    def draw_triangle():
        img = Image.new("L", (image_size, image_size), 0)
        draw = ImageDraw.Draw(img)
        x0 = rng.integers(0, image_size - edge_bounding_box)
        y0 = rng.integers(0, image_size - edge_bounding_box)
        pts = [
            (x0 + edge_bounding_box // 2, y0),
            (x0, y0 + edge_bounding_box),
            (x0 + edge_bounding_box, y0 + edge_bounding_box),
        ]
        draw.polygon(pts, fill=255)
        return np.asarray(img, dtype=np.uint8)

    generators = [draw_circle, draw_square, draw_triangle]

    Xs, ys = [], []
    for k, n_k in enumerate(n_per_class):
        for _ in range(n_k):
            Xs.append(generators[k]())
            ys.append(k)

    X = np.stack(Xs)[:, None, :, :]  # shape (N, 1, 28, 28) == PyTorch ready
    y = np.array(ys, dtype=np.int64)

    return X, y


def dump_split(split_name, X_split, y_split, root="data"):
    root = Path(root) / split_name
    for cls in np.unique(y_split):
        (root / f"class_{cls}").mkdir(parents=True, exist_ok=True)

    for i, (img_arr, label) in enumerate(zip(X_split, y_split)):
        out_path = root / f"class_{label}" / f"{i:06d}.png"
        plt.imsave(out_path, img_arr.squeeze(), cmap="gray")

    print(f"Saved {len(X_split)} images to {root=}")


if __name__ == "__main__":
    os.makedirs("assets", exist_ok=True)

    generate_dataset(
        dataset_type=DatasetTypes.IMAGES,
        n_total=200,
        priors=np.array([0.80, 0.15, 0.05]),
        image_size=28,
        edge_bounding_box=12,
    )
