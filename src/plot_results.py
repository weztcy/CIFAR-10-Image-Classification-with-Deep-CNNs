"""
Plotting and Visualization Module for CIFAR-10 experiments.
Generates all remaining required figures:
4. figures/training_loss.png
5. figures/training_accuracy.png
6. figures/model_comparison.png
7. figures/confusion_matrix.png
8. figures/per_class_metrics.png
9. figures/prediction_examples.png
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_cifar_raw


def generate_all_plots(
    metrics_dir='outputs/metrics',
    predictions_dir='outputs/predictions',
    output_dir='figures',
    data_dir='dataset'
):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

    # Load histories
    baseline_hist_path = os.path.join(metrics_dir, 'baseline_history.csv')
    resnet_hist_path = os.path.join(metrics_dir, 'resnet_history.csv')

    has_baseline_hist = os.path.exists(baseline_hist_path)
    has_resnet_hist = os.path.exists(resnet_hist_path)

    df_base = pd.read_csv(baseline_hist_path) if has_baseline_hist else None
    df_res = pd.read_csv(resnet_hist_path) if has_resnet_hist else None

    # Load evals
    base_eval_path = os.path.join(metrics_dir, 'baseline_eval.json')
    res_eval_path = os.path.join(metrics_dir, 'resnet_eval.json')

    eval_base = json.load(open(base_eval_path)) if os.path.exists(base_eval_path) else None
    eval_res = json.load(open(res_eval_path)) if os.path.exists(res_eval_path) else None

    # Determine best model
    if eval_base and eval_res:
        best_model_name = 'resnet' if eval_res['test_accuracy'] >= eval_base['test_accuracy'] else 'baseline'
        best_eval = eval_res if best_model_name == 'resnet' else eval_base
    elif eval_res:
        best_model_name = 'resnet'
        best_eval = eval_res
    elif eval_base:
        best_model_name = 'baseline'
        best_eval = eval_base
    else:
        raise FileNotFoundError("No evaluation files found to plot.")

    print(f"Generating plots. Best model identified: {best_model_name.upper()}")

    # 4. TRAINING LOSS PLOT (figures/training_loss.png)
    print("Generating figures/training_loss.png...")
    plt.figure(figsize=(10, 6))
    if df_base is not None:
        plt.plot(df_base['epoch'], df_base['train_loss'], label='Baseline Train Loss', color='#4a90e2', linestyle='--', linewidth=2)
        plt.plot(df_base['epoch'], df_base['val_loss'], label='Baseline Val Loss', color='#1f4e79', linewidth=2.5)
    if df_res is not None:
        plt.plot(df_res['epoch'], df_res['train_loss'], label='ResNetCIFAR Train Loss', color='#f5a623', linestyle='--', linewidth=2)
        plt.plot(df_res['epoch'], df_res['val_loss'], label='ResNetCIFAR Val Loss', color='#d0021b', linewidth=2.5)

    plt.title('Kurva Training dan Validation Loss per Epoch', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.legend(frameon=True, fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_loss.png'), dpi=300)
    plt.close()

    # 5. TRAINING ACCURACY PLOT (figures/training_accuracy.png)
    print("Generating figures/training_accuracy.png...")
    plt.figure(figsize=(10, 6))
    if df_base is not None:
        plt.plot(df_base['epoch'], df_base['train_acc'] * 100, label='Baseline Train Acc', color='#4a90e2', linestyle='--', linewidth=2)
        plt.plot(df_base['epoch'], df_base['val_acc'] * 100, label='Baseline Val Acc', color='#1f4e79', linewidth=2.5)
    if df_res is not None:
        plt.plot(df_res['epoch'], df_res['train_acc'] * 100, label='ResNetCIFAR Train Acc', color='#f5a623', linestyle='--', linewidth=2)
        plt.plot(df_res['epoch'], df_res['val_acc'] * 100, label='ResNetCIFAR Val Acc', color='#d0021b', linewidth=2.5)

    plt.title('Kurva Training dan Validation Accuracy per Epoch', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(frameon=True, fontsize=11, loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_accuracy.png'), dpi=300)
    plt.close()

    # 6. MODEL COMPARISON PLOT (figures/model_comparison.png)
    print("Generating figures/model_comparison.png...")
    if eval_base and eval_res:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        models = ['Baseline CNN', 'ResNetCIFAR']
        
        # Test Accuracy & Macro F1
        acc_vals = [eval_base['test_accuracy'] * 100, eval_res['test_accuracy'] * 100]
        f1_vals = [eval_base['macro_avg_f1'] * 100, eval_res['macro_avg_f1'] * 100]
        
        x = np.arange(len(models))
        width = 0.35
        axes[0].bar(x - width/2, acc_vals, width, label='Test Accuracy (%)', color='#2b5c8f')
        axes[0].bar(x + width/2, f1_vals, width, label='Macro F1 (%)', color='#46a575')
        axes[0].set_title('Performa Prediksi Test Set', fontweight='bold', fontsize=12)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, fontweight='bold')
        axes[0].set_ylim(0, 100)
        axes[0].legend(loc='lower right')
        for i in range(len(models)):
            axes[0].text(x[i] - width/2, acc_vals[i] + 1.5, f"{acc_vals[i]:.2f}%", ha='center', fontsize=10, fontweight='bold')
            axes[0].text(x[i] + width/2, f1_vals[i] + 1.5, f"{f1_vals[i]:.2f}%", ha='center', fontsize=10, fontweight='bold')

        # Parameters & Checkpoint Size
        param_k = [eval_base['total_params'] / 1000, eval_res['total_params'] / 1000]
        axes[1].bar(models, param_k, color=['#e26d5c', '#38b000'], width=0.5)
        axes[1].set_title('Kompleksitas Model (Jumlah Parameter)', fontweight='bold', fontsize=12)
        axes[1].set_ylabel('Parameter (Ribu / K)', fontsize=11)
        for i in range(len(models)):
            axes[1].text(i, param_k[i] + 10, f"{param_k[i]:.1f}K", ha='center', fontsize=10, fontweight='bold')

        # Inference Throughput (FPS)
        fps_vals = [eval_base['throughput_fps'], eval_res['throughput_fps']]
        axes[2].bar(models, fps_vals, color=['#7209b7', '#4361ee'], width=0.5)
        axes[2].set_title('Efisiensi Inferensi CPU (Frames Per Second)', fontweight='bold', fontsize=12)
        axes[2].set_ylabel('Inference Throughput (FPS)', fontsize=11)
        for i in range(len(models)):
            axes[2].text(i, fps_vals[i] + 1, f"{fps_vals[i]:.1f} FPS", ha='center', fontsize=10, fontweight='bold')

        plt.suptitle('Perbandingan Komprehensif: Baseline CNN vs ResNetCIFAR', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300)
        plt.close()

    # 7. CONFUSION MATRIX (figures/confusion_matrix.png)
    print("Generating figures/confusion_matrix.png...")
    conf_matrix = np.array(best_eval['confusion_matrix'])
    class_names = list(best_eval['per_class'].keys())
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={'label': 'Jumlah Sampel Citra'}
    )
    plt.title(f'Confusion Matrix pada Test Batch ({best_model_name.upper()})', fontsize=14, fontweight='bold')
    plt.xlabel('Label Prediksi Model', fontsize=12, fontweight='bold')
    plt.ylabel('Label Sebenarnya (Ground Truth)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()

    # 8. PER-CLASS METRICS (figures/per_class_metrics.png)
    print("Generating figures/per_class_metrics.png...")
    class_metrics = best_eval['per_class']
    precisions = [class_metrics[c]['precision'] * 100 for c in class_names]
    recalls = [class_metrics[c]['recall'] * 100 for c in class_names]
    f1s = [class_metrics[c]['f1_score'] * 100 for c in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    plt.figure(figsize=(14, 6))
    plt.bar(x - width, precisions, width, label='Precision (%)', color='#3a86ff')
    plt.bar(x, recalls, width, label='Recall (%)', color='#ffbe0b')
    plt.bar(x + width, f1s, width, label='F1-Score (%)', color='#fb5607')

    plt.title(f'Metrik Per-Kelas pada Test Set ({best_model_name.upper()})', fontsize=14, fontweight='bold')
    plt.xlabel('Kelas CIFAR-10', fontsize=12, fontweight='bold')
    plt.ylabel('Skor (%)', fontsize=12, fontweight='bold')
    plt.xticks(x, class_names, rotation=30, ha='right', fontsize=11)
    plt.ylim(0, 105)
    plt.legend(frameon=True, fontsize=11, loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'), dpi=300)
    plt.close()

    # 9. PREDICTION EXAMPLES (figures/prediction_examples.png)
    print("Generating figures/prediction_examples.png...")
    # Load test raw images
    _, _, test_images, test_labels, _ = load_cifar_raw(data_dir)
    pred_path = os.path.join(predictions_dir, f"{best_model_name}_predictions.json")
    if os.path.exists(pred_path):
        preds_data = json.load(open(pred_path))
        correct_samples = preds_data['correct'][:5]
        error_samples = preds_data['errors'][:5]

        fig, axes = plt.subplots(2, 5, figsize=(15, 7))
        fig.suptitle(f"Contoh Prediksi Model {best_model_name.upper()} pada Test Batch (Atas: Benar | Bawah: Salah)", fontsize=15, fontweight='bold')

        # Top row: Correct predictions
        for i, sample in enumerate(correct_samples):
            ax = axes[0, i]
            img = test_images[sample['index']]
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(
                f"True: {sample['true_label']}\nPred: {sample['pred_label']}\nConf: {sample['confidence']*100:.1f}%",
                color='darkgreen', fontsize=10, fontweight='bold'
            )

        # Bottom row: Error predictions
        for i, sample in enumerate(error_samples):
            ax = axes[1, i]
            img = test_images[sample['index']]
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(
                f"True: {sample['true_label']}\nPred: {sample['pred_label']}\nConf: {sample['confidence']*100:.1f}%",
                color='darkred', fontsize=10, fontweight='bold'
            )

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'prediction_examples.png'), dpi=300)
        plt.close()

    print("=== ALL PLOTS GENERATED SUCCESSFULLY ===")


if __name__ == '__main__':
    generate_all_plots()
