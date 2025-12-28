#!/usr/bin/env python
"""
Download CIFAR-10 dataset using only stdlib and numpy.

Downloads to data/cifar10/ directory.
"""

import os
import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np


CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def download_file(url: str, dest: Path) -> None:
    """Download a file if it doesn't exist."""
    if dest.exists():
        print(f"  {dest.name} already exists, skipping.")
        return
    print(f"  Downloading {url}...")
    urllib.request.urlretrieve(url, dest)


def download_cifar10(data_dir: Path = None) -> None:
    """Download and extract CIFAR-10."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "cifar10"
    data_dir.mkdir(parents=True, exist_ok=True)

    tar_path = data_dir / "cifar-10-python.tar.gz"

    # Check if already extracted
    if (data_dir / "cifar-10-batches-py").exists():
        print("CIFAR-10 already extracted.")
        return

    print("Downloading CIFAR-10...")
    download_file(CIFAR10_URL, tar_path)

    print("Extracting...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(data_dir)

    print("CIFAR-10 download complete.")


def load_cifar10_batch(path: Path) -> tuple:
    """Load a single CIFAR-10 batch file."""
    with open(path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    images = batch[b'data'].astype(np.float32) / 255.0
    labels = np.array(batch[b'labels'])
    return images, labels


def load_cifar10(data_dir: Path = None, subset: int = None, flatten: bool = True) -> dict:
    """
    Load CIFAR-10 dataset.

    Parameters
    ----------
    data_dir : Path
        Directory containing CIFAR-10 files.
    subset : int
        If provided, only load this many samples.
    flatten : bool
        If True, flatten images to (N, 3072). Otherwise (N, 32, 32, 3).

    Returns
    -------
    data : dict
        - 'X_train': training images
        - 'y_train': training labels
        - 'X_test': test images
        - 'y_test': test labels
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "cifar10"

    batch_dir = data_dir / "cifar-10-batches-py"

    # Load training batches
    X_train_list = []
    y_train_list = []
    for i in range(1, 6):
        X, y = load_cifar10_batch(batch_dir / f"data_batch_{i}")
        X_train_list.append(X)
        y_train_list.append(y)

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    # Load test batch
    X_test, y_test = load_cifar10_batch(batch_dir / "test_batch")

    if subset is not None:
        X_train = X_train[:subset]
        y_train = y_train[:subset]
        X_test = X_test[:min(subset, len(X_test))]
        y_test = y_test[:min(subset, len(y_test))]

    if not flatten:
        # Reshape to (N, 32, 32, 3)
        X_train = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        X_test = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
    }


if __name__ == "__main__":
    download_cifar10()
    data = load_cifar10(subset=1000)
    print(f"Loaded shapes: X_train={data['X_train'].shape}, y_train={data['y_train'].shape}")
