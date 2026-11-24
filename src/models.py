"""
CNN Model architectures for CIFAR-10 Image Classification.

1. BaselineCNN:
   - 3-stage VGG-style Convolutional Neural Network
   - Stacked Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d blocks
   - Dropout for regularization
   - Fully-connected classification head: Flatten -> Linear(2048, 256) -> ReLU -> Linear(256, 10)

2. ResNetCIFAR (Improved Deep CNN):
   - Modern Deep Residual Network specifically tailored for CIFAR-10 (He et al., 2016)
   - CIFAR-10 Stem: 3x3 Conv with stride 1, padding 1, and no initial max-pooling
     (crucial to avoid destroying 32x32 spatial resolution too early)
   - 3 Residual Stages:
     * Stage 1: 16 channels, 3 residual blocks (32x32 feature maps)
     * Stage 2: 32 channels, 3 residual blocks (16x16 feature maps, stride 2 shortcut)
     * Stage 3: 64 channels, 3 residual blocks (8x8 feature maps, stride 2 shortcut)
   - Global Average Pooling (GAP): compresses 8x8 spatial grid into 1x1, eliminating parameter bloat
   - Dense Head: Linear(64, 10) with Dropout(0.2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):
    """
    Baseline Convolutional Neural Network.
    Architecture:
    - Conv Block 1: 3 -> 32 -> 32, MaxPool(2x2), Dropout(0.2) [Output: 16x16]
    - Conv Block 2: 32 -> 64 -> 64, MaxPool(2x2), Dropout(0.25) [Output: 8x8]
    - Conv Block 3: 64 -> 128, MaxPool(2x2), Dropout(0.3) [Output: 4x4]
    - Classifier: Flatten -> Dense(2048, 256) -> BatchNorm1d -> ReLU -> Dropout(0.4) -> Dense(256, 10)
    """
    def __init__(self, num_classes=10):
        super(BaselineCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1 (32x32 -> 16x16)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            # Block 2 (16x16 -> 8x8)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 3 (8x8 -> 4x4)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual Block with two 3x3 convs, Batch Normalization, and identity shortcut.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class ResNetCIFAR(nn.Module):
    """
    Improved Deep CNN: Custom ResNet-20 for CIFAR-10.
    Follows He et al. (2016) Section 4.2:
    - Initial Stem: Conv 3x3, stride 1, padding 1, BN, ReLU (16 filters)
    - 3 Stages of 2x Residual Blocks each:
      * Stage 1: 16 channels, output 32x32 (2 blocks)
      * Stage 2: 32 channels, output 16x16 (2 blocks, stride 2)
      * Stage 3: 64 channels, output 8x8 (2 blocks, stride 2)
    - Global Average Pooling -> 1x1x64
    - Linear(64, num_classes)
    """
    def __init__(self, num_classes=10):
        super(ResNetCIFAR, self).__init__()
        
        # CIFAR-10 Initial Stem (no maxpool, preserves 32x32 details)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # Residual Stages (2 blocks per stage for balanced depth and training speed)
        self.stage1 = self._make_layer(16, 16, num_blocks=2, stride=1)
        self.stage2 = self._make_layer(16, 32, num_blocks=2, stride=2)
        self.stage3 = self._make_layer(32, 64, num_blocks=2, stride=2)

        # Global Average Pooling and Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [ResidualBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.gap(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


# Alias for compatibility
CIFARResNet = ResNetCIFAR


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    b_model = BaselineCNN()
    r_model = ResNetCIFAR()
    
    b_tot, b_trn = count_parameters(b_model)
    r_tot, r_trn = count_parameters(r_model)
    
    print(f"BaselineCNN: {b_tot:,} params")
    print(f"ResNetCIFAR: {r_tot:,} params")
    
    dummy = torch.randn(4, 3, 32, 32)
    b_out = b_model(dummy)
    r_out = r_model(dummy)
    print(f"Baseline output shape: {b_out.shape}")
    print(f"ResNet output shape: {r_out.shape}")
