"""
Exploratory Data Analysis (EDA) and Data Inspection Script for CIFAR-10.
Generates:
1. figures/dataset_samples.png
2. figures/class_distribution.png
3. figures/rgb_pixel_distribution.png
Outputs json summary of data inspection.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_cifar_raw, stratified_split, compute_normalization_stats

def run_eda(data_dir='dataset', output_dir='figures', metrics_dir='outputs/metrics'):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    print("=== TAHAP A: DATA INSPECTION & EXPLORATORY DATA ANALYSIS ===")
    
    # 1. File inspection
    required_files = [
        'batches.meta', 'data_batch_1', 'data_batch_2', 'data_batch_3', 
        'data_batch_4', 'data_batch_5', 'test_batch'
    ]
    file_info = {}
    for f in required_files:
        p = os.path.join(data_dir, f)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required dataset file: {p}")
        size_bytes = os.path.getsize(p)
        file_info[f] = {
            'size_bytes': size_bytes,
            'size_mb': round(size_bytes / (1024 * 1024), 2)
        }
        print(f"File: {f} | Size: {file_info[f]['size_mb']} MB")

    # 2. Load dataset
    train_images, train_labels, test_images, test_labels, class_names = load_cifar_raw(data_dir)
    print(f"\nLoaded Train Images: {train_images.shape}, dtype: {train_images.dtype}")
    print(f"Loaded Train Labels: {train_labels.shape}, unique classes: {len(np.unique(train_labels))}")
    print(f"Loaded Test Images: {test_images.shape}, dtype: {test_images.dtype}")
    print(f"Loaded Test Labels: {test_labels.shape}, unique classes: {len(np.unique(test_labels))}")
    print(f"Class Names: {class_names}")

    # Check value ranges & nulls
    train_min, train_max = int(train_images.min()), int(train_images.max())
    test_min, test_max = int(test_images.min()), int(test_images.max())
    has_nan_train = bool(np.isnan(train_images).any())
    has_nan_test = bool(np.isnan(test_images).any())

    # Check duplicates across images (hash-based quick check)
    print("Checking for duplicate images...")
    train_flat = train_images.reshape(len(train_images), -1)
    unique_train_count = len(np.unique(train_flat, axis=0))
    duplicate_train_count = len(train_images) - unique_train_count
    print(f"Total training images: {len(train_images)}, Unique images: {unique_train_count}, Duplicates found: {duplicate_train_count}")

    # 3. Stratified split (90% train, 10% val)
    x_train, y_train, x_val, y_val = stratified_split(train_images, train_labels, val_ratio=0.1, random_seed=42)
    print(f"Stratified Split -> Train: {x_train.shape}, Val: {x_val.shape}")

    # Compute training stats strictly on x_train
    mean, std = compute_normalization_stats(x_train)
    print(f"Strict Training Set Normalization Stats:")
    print(f"Mean (R, G, B): {[round(m, 4) for m in mean]}")
    print(f"Std  (R, G, B): {[round(s, 4) for s in std]}")

    # Class distributions
    train_class_counts = {class_names[i]: int(np.sum(y_train == i)) for i in range(10)}
    val_class_counts = {class_names[i]: int(np.sum(y_val == i)) for i in range(10)}
    test_class_counts = {class_names[i]: int(np.sum(test_labels == i)) for i in range(10)}

    eda_summary = {
        'files': file_info,
        'shapes': {
            'train_raw_total': list(train_images.shape),
            'train_split': list(x_train.shape),
            'val_split': list(x_val.shape),
            'test': list(test_images.shape)
        },
        'pixel_value_range': {
            'train_min': train_min,
            'train_max': train_max,
            'test_min': test_min,
            'test_max': test_max,
            'has_nan_train': has_nan_train,
            'has_nan_test': has_nan_test
        },
        'duplicate_images_in_train': duplicate_train_count,
        'normalization_stats_train': {
            'mean': [float(round(m, 5)) for m in mean],
            'std': [float(round(s, 5)) for s in std]
        },
        'class_names': class_names,
        'class_counts': {
            'train': train_class_counts,
            'val': val_class_counts,
            'test': test_class_counts
        }
    }

    with open(os.path.join(metrics_dir, 'eda_summary.json'), 'w') as f:
        json.dump(eda_summary, f, indent=2)
    print(f"Saved EDA summary to {os.path.join(metrics_dir, 'eda_summary.json')}")

    # 4. Generate Visualizations
    # Set style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

    # VISUALIZATION 1: dataset_samples.png (Montage of 10 classes x 5 samples)
    print("Generating figures/dataset_samples.png...")
    fig, axes = plt.subplots(10, 6, figsize=(12, 18))
    fig.suptitle("CIFAR-10 Dataset Sample Montage (10 Classes × 6 Samples)", fontsize=16, fontweight='bold', y=0.995)

    for class_idx, class_name in enumerate(class_names):
        match_indices = np.where(train_labels == class_idx)[0][:6]
        for sample_col, img_idx in enumerate(match_indices):
            ax = axes[class_idx, sample_col]
            ax.imshow(train_images[img_idx])
            ax.axis('off')
            if sample_col == 0:
                ax.set_title(f"Class {class_idx}: {class_name}", fontsize=11, fontweight='bold', loc='left')

    plt.tight_layout()
    fig_samples_path = os.path.join(output_dir, 'dataset_samples.png')
    plt.savefig(fig_samples_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {fig_samples_path}")

    # VISUALIZATION 2: class_distribution.png (Bar chart across train, val, test)
    print("Generating figures/class_distribution.png...")
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(class_names))
    width = 0.25

    rects1 = ax.bar(x - width, [train_class_counts[c] for c in class_names], width, label='Training Set (90%)', color='#2b5c8f')
    rects2 = ax.bar(x, [val_class_counts[c] for c in class_names], width, label='Validation Set (10%)', color='#46a575')
    rects3 = ax.bar(x + width, [test_class_counts[c] for c in class_names], width, label='Test Set', color='#e26d5c')

    ax.set_ylabel('Jumlah Sampel Citra', fontsize=12, fontweight='bold')
    ax.set_title('Distribusi Jumlah Sampel per Kelas pada CIFAR-10 Splits', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha='right', fontsize=11)
    ax.legend(frameon=True, fontsize=11)
    ax.set_ylim(0, 5200)

    # Add count labels on bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for rect in rects3:
        height = rect.get_height()
        ax.annotate(f'{height}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    fig_dist_path = os.path.join(output_dir, 'class_distribution.png')
    plt.savefig(fig_dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {fig_dist_path}")

    # VISUALIZATION 3: rgb_pixel_distribution.png
    print("Generating figures/rgb_pixel_distribution.png...")
    # Sample 5000 images for fast and accurate density estimation
    np.random.seed(42)
    sample_sub = x_train[np.random.choice(len(x_train), 5000, replace=False)]
    r_vals = sample_sub[:, :, :, 0].flatten()
    g_vals = sample_sub[:, :, :, 1].flatten()
    b_vals = sample_sub[:, :, :, 2].flatten()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Histogram / Density plot
    ax1.hist(r_vals, bins=50, color='red', alpha=0.5, density=True, label=f'Red Channel (μ={mean[0]*255:.1f}, σ={std[0]*255:.1f})')
    ax1.hist(g_vals, bins=50, color='green', alpha=0.5, density=True, label=f'Green Channel (μ={mean[1]*255:.1f}, σ={std[1]*255:.1f})')
    ax1.hist(b_vals, bins=50, color='blue', alpha=0.5, density=True, label=f'Blue Channel (μ={mean[2]*255:.1f}, σ={std[2]*255:.1f})')
    ax1.set_title('Distribusi Intensitas Piksel per Kanal RGB (Training Set)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Nilai Intensitas Piksel [0 - 255]', fontsize=11)
    ax1.set_ylabel('Densitas Frekuensi', fontsize=11)
    ax1.legend(frameon=True, fontsize=10)

    # Boxplot of RGB channels
    data_to_plot = [r_vals[::10], g_vals[::10], b_vals[::10]]
    box = ax2.boxplot(data_to_plot, patch_artist=True, tick_labels=['Red', 'Green', 'Blue'])
    colors = ['#ff9999', '#99ff99', '#9999ff']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax2.set_title('Sebaran dan Kuartil Nilai Piksel Kanal RGB', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Nilai Piksel [0 - 255]', fontsize=11)

    plt.tight_layout()
    fig_rgb_path = os.path.join(output_dir, 'rgb_pixel_distribution.png')
    plt.savefig(fig_rgb_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {fig_rgb_path}")

    print("=== TAHAP A COMPLETE ===")

if __name__ == '__main__':
    run_eda()
