#!/usr/bin/env python
"""
Download MNIST dataset using only stdlib and numpy.

Downloads to data/mnist/ directory.
"""

import gzip
import os
import struct
import urllib.request
from pathlib import Path

import numpy as np


MNIST_URL = "http://yann.lecun.com/exdb/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def download_file(url: str, dest: Path) -> None:
    """Download a file if it doesn't exist."""
    if dest.exists():
        print(f"  {dest.name} already exists, skipping.")
        return
    print(f"  Downloading {url}...")
    urllib.request.urlretrieve(url, dest)


def read_idx_images(path: Path) -> np.ndarray:
    """Read IDX image file format."""
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows * cols).astype(np.float32) / 255.0


def read_idx_labels(path: Path) -> np.ndarray:
    """Read IDX label file format."""
    with gzip.open(path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic}")
        return np.frombuffer(f.read(), dtype=np.uint8)


def download_mnist(data_dir: Path = None) -> None:
    """Download MNIST to data_dir."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "mnist"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading MNIST...")
    for key, filename in FILES.items():
        url = MNIST_URL + filename
        dest = data_dir / filename
        download_file(url, dest)

    print("MNIST download complete.")


def load_mnist(data_dir: Path = None, subset: int = None) -> dict:
    """
    Load MNIST dataset.

    Parameters
    ----------
    data_dir : Path
        Directory containing MNIST files.
    subset : int
        If provided, only load this many samples from train/test.

    Returns
    -------
    data : dict
        - 'X_train': (N, 784) training images
        - 'y_train': (N,) training labels
        - 'X_test': (M, 784) test images
        - 'y_test': (M,) test labels
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "mnist"

    X_train = read_idx_images(data_dir / FILES["train_images"])
    y_train = read_idx_labels(data_dir / FILES["train_labels"])
    X_test = read_idx_images(data_dir / FILES["test_images"])
    y_test = read_idx_labels(data_dir / FILES["test_labels"])

    if subset is not None:
        X_train = X_train[:subset]
        y_train = y_train[:subset]
        X_test = X_test[:min(subset, len(X_test))]
        y_test = y_test[:min(subset, len(y_test))]

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
    }


if __name__ == "__main__":
    download_mnist()
    data = load_mnist(subset=1000)
    print(f"Loaded shapes: X_train={data['X_train'].shape}, y_train={data['y_train'].shape}")
