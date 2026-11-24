"""
Training module for CIFAR-10 Deep Learning models.
Includes stratified train/val split, training history logging to CSV,
Cosine Annealing LR scheduling, and model checkpointing.
"""

import os
import time
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loader import load_cifar_raw, stratified_split, compute_normalization_stats
from dataset import CIFAR10Dataset, get_data_transforms
from models import BaselineCNN, ResNetCIFAR, count_parameters


def set_seed(seed=42):
    """Set random seed for reproducibility across random, numpy, and torch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Run one training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate_val(model, dataloader, criterion, device):
    """Run validation evaluation."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def train_model(
    model_name='resnet',
    epochs=15,
    batch_size=128,
    lr=0.001,
    weight_decay=1e-4,
    random_seed=42,
    val_ratio=0.1,
    data_dir='dataset',
    checkpoints_dir='outputs/checkpoints',
    metrics_dir='outputs/metrics'
):
    """Full training pipeline with checkpointing and history logging."""
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    set_seed(random_seed)
    torch.set_num_threads(12)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=======================================================")
    print(f"Training Model: {model_name.upper()} on Device: {device}")
    print(f"Settings: Epochs={epochs}, BatchSize={batch_size}, LR={lr}, Seed={random_seed}")
    print(f"=======================================================\n")

    # 1. Load data & stratified split
    raw_train_images, raw_train_labels, _, _, class_names = load_cifar_raw(data_dir)
    x_train, y_train, x_val, y_val = stratified_split(
        raw_train_images, raw_train_labels, val_ratio=val_ratio, random_seed=random_seed
    )

    # 2. Compute normalization stats strictly from training split
    mean, std = compute_normalization_stats(x_train)
    norm_stats = {
        'mean': [float(m) for m in mean],
        'std': [float(s) for s in std],
        'class_names': class_names
    }
    with open(os.path.join(metrics_dir, 'normalization_stats.json'), 'w') as f:
        json.dump(norm_stats, f, indent=2)

    # 3. Create datasets and dataloaders
    train_transform, val_transform = get_data_transforms(mean, std)
    train_dataset = CIFAR10Dataset(x_train, y_train, transform=train_transform)
    val_dataset = CIFAR10Dataset(x_val, y_val, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 4. Initialize model
    if model_name.lower() in ['baseline', 'baseline_cnn']:
        model = BaselineCNN(num_classes=len(class_names))
        model_key = 'baseline'
    elif model_name.lower() in ['resnet', 'resnet_cifar', 'improved']:
        model = ResNetCIFAR(num_classes=len(class_names))
        model_key = 'resnet'
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    model = model.to(device)
    tot_params, trn_params = count_parameters(model)
    print(f"Model: {model_key} | Total Parameters: {tot_params:,} | Trainable: {trn_params:,}")

    # 5. Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # 6. Training Loop
    history = []
    best_val_acc = 0.0
    best_epoch = 0
    checkpoint_path = os.path.join(checkpoints_dir, f"{model_key}_best.pt")

    start_train_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_val(model, val_loader, criterion, device)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        history.append({
            'epoch': epoch,
            'train_loss': float(round(train_loss, 4)),
            'train_acc': float(round(train_acc, 4)),
            'val_loss': float(round(val_loss, 4)),
            'val_acc': float(round(val_acc, 4)),
            'lr': float(round(current_lr, 6)),
            'epoch_time_sec': float(round(epoch_time, 2))
        })

        is_best = val_acc > best_val_acc
        best_marker = ""
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch
            best_marker = " -> [BEST MODEL CHECKPOINT SAVED]"
            torch.save({
                'epoch': epoch,
                'model_name': model_key,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': float(val_acc),
                'val_loss': float(val_loss),
                'train_acc': float(train_acc),
                'train_loss': float(train_loss),
                'class_names': class_names,
                'mean': mean,
                'std': std,
                'hyperparameters': {
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'lr': lr,
                    'weight_decay': weight_decay,
                    'random_seed': random_seed,
                    'val_ratio': val_ratio
                }
            }, checkpoint_path)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
              f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s{best_marker}")

    total_train_time = time.time() - start_train_time
    print(f"\nTraining completed in {total_train_time/60:.2f} minutes.")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}% at Epoch {best_epoch}")

    # 7. Save history to CSV
    history_df = pd.DataFrame(history)
    history_path = os.path.join(metrics_dir, f"{model_key}_history.csv")
    history_df.to_csv(history_path, index=False)
    print(f"Saved training history to {history_path}")

    return history_df, best_val_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train CIFAR-10 CNN Models")
    parser.add_argument('--model', type=str, default='baseline', choices=['baseline', 'resnet'], help='Model architecture')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    train_model(model_name=args.model, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
