"""
Data loader module for CIFAR-10 dataset.
Reads pickle batches, parses channel-first 3072-byte vectors,
verifies data integrity, performs stratified splitting, and computes normalization statistics.
"""

import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split


def load_pickle_file(filepath):
    """Load a pickle file with bytes encoding."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    with open(filepath, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data


def load_cifar_meta(meta_path):
    """Load class names from batches.meta."""
    meta = load_pickle_file(meta_path)
    label_names_bytes = meta.get(b'label_names', meta.get('label_names'))
    if label_names_bytes is None:
        raise KeyError("Could not find 'label_names' in meta file.")
    class_names = [
        name.decode('utf-8') if isinstance(name, bytes) else str(name)
        for name in label_names_bytes
    ]
    return class_names


def load_cifar_raw(data_dir='dataset'):
    """
    Load raw training and test batches from dataset directory.
    Returns:
        train_data: np.ndarray shape (50000, 32, 32, 3) in uint8 [0, 255]
        train_labels: np.ndarray shape (50000,) int64
        test_data: np.ndarray shape (10000, 32, 32, 3) in uint8 [0, 255]
        test_labels: np.ndarray shape (10000,) int64
        class_names: list of str (length 10)
    """
    meta_path = os.path.join(data_dir, 'batches.meta')
    class_names = load_cifar_meta(meta_path)

    # Load 5 training batches
    train_data_list = []
    train_labels_list = []

    for i in range(1, 6):
        batch_filename = f'data_batch_{i}'
        batch_path = os.path.join(data_dir, batch_filename)
        batch = load_pickle_file(batch_path)
        raw_data = batch[b'data']  # (10000, 3072)
        raw_labels = batch[b'labels']  # list of 10000 ints
        train_data_list.append(raw_data)
        train_labels_list.extend(raw_labels)

    train_data_raw = np.vstack(train_data_list)  # (50000, 3072)
    train_labels = np.array(train_labels_list, dtype=np.int64)

    # Load test batch
    test_path = os.path.join(data_dir, 'test_batch')
    test_batch = load_pickle_file(test_path)
    test_data_raw = test_batch[b'data']  # (10000, 3072)
    test_labels = np.array(test_batch[b'labels'], dtype=np.int64)

    # Convert CIFAR raw format (channel-first: 3 x 32 x 32) to HWC (32, 32, 3)
    # The first 1024 entries are Red, next 1024 are Green, final 1024 are Blue
    train_data = train_data_raw.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_data = test_data_raw.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    return train_data, train_labels, test_data, test_labels, class_names


def stratified_split(train_data, train_labels, val_ratio=0.1, random_seed=42):
    """
    Split train dataset into training and validation sets stratified by class labels.
    Prevents data leakage and ensures balanced representation.
    """
    train_idx, val_idx = train_test_split(
        np.arange(len(train_labels)),
        test_size=val_ratio,
        stratify=train_labels,
        random_state=random_seed
    )
    
    x_train, y_train = train_data[train_idx], train_labels[train_idx]
    x_val, y_val = train_data[val_idx], train_labels[val_idx]
    
    return x_train, y_train, x_val, y_val


def compute_normalization_stats(train_images):
    """
    Compute per-channel mean and standard deviation strictly on the training set (0-1 scaled).
    train_images: np.ndarray shape (N, 32, 32, 3) with values in [0, 255]
    Returns:
        mean: tuple of 3 floats
        std: tuple of 3 floats
    """
    # Scale to [0, 1]
    images_scaled = train_images.astype(np.float32) / 255.0
    # Mean across samples and pixels (N, H, W) for each channel
    mean = tuple(images_scaled.mean(axis=(0, 1, 2)).tolist())
    std = tuple(images_scaled.std(axis=(0, 1, 2)).tolist())
    return mean, std
