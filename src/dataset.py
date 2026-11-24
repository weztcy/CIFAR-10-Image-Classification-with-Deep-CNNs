"""
PyTorch Dataset and transformation pipelines for CIFAR-10.
Includes data augmentation exclusively for training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class CIFAR10Dataset(Dataset):
    """
    CIFAR-10 PyTorch Dataset.
    images: np.ndarray shape (N, 32, 32, 3) in uint8
    labels: np.ndarray shape (N,)
    transform: torchvision transforms
    """
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = int(self.labels[idx])

        # Convert to PIL Image for torchvision transforms compatibility
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def get_data_transforms(mean, std):
    """
    Return training and evaluation (validation / test) transforms.
    
    Training pipeline:
    - RandomCrop (32x32 with padding=4, reflect mode): teaches shift invariance
    - RandomHorizontalFlip (p=0.5): teaches horizontal reflection invariance
    - ToTensor: converts PIL (H, W, C) in [0, 255] to Tensor (C, H, W) in [0.0, 1.0]
    - Normalize: applies channel-wise (x - mean) / std strictly from train split
    
    Evaluation pipeline:
    - ToTensor
    - Normalize
    (No augmentations applied to validation/test!)
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transform, eval_transform
