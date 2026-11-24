"""
Evaluation module for CIFAR-10 models on test_batch.
Computes test accuracy, loss, precision, recall, F1-score per class,
confusion matrix, inference latency / FPS, model parameters, and error analysis.
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
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import load_cifar_raw
from dataset import CIFAR10Dataset, get_data_transforms
from models import BaselineCNN, ResNetCIFAR, count_parameters


def evaluate_model_on_test(
    model_name='baseline',
    data_dir='dataset',
    checkpoints_dir='outputs/checkpoints',
    metrics_dir='outputs/metrics',
    predictions_dir='outputs/predictions'
):
    """Run rigorous final evaluation on the isolated test_batch."""
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(12)

    checkpoint_path = os.path.join(checkpoints_dir, f"{model_name}_best.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint['class_names']
    mean = checkpoint['mean']
    std = checkpoint['std']

    # Initialize model
    if model_name == 'baseline':
        model = BaselineCNN(num_classes=len(class_names))
    elif model_name == 'resnet':
        model = ResNetCIFAR(num_classes=len(class_names))
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Model complexity
    total_params, trainable_params = count_parameters(model)
    checkpoint_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)

    # Load test batch
    _, _, test_images, test_labels, _ = load_cifar_raw(data_dir)
    _, test_transform = get_data_transforms(mean, std)
    test_dataset = CIFAR10Dataset(test_images, test_labels, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    # Evaluate test metrics
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    test_loss = total_loss / len(test_labels)
    test_accuracy = float(np.mean(all_preds == all_targets))

    # Classification report & Confusion Matrix
    clf_report_dict = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True)
    conf_matrix = confusion_matrix(all_targets, all_preds)

    # Save confusion matrix numpy
    np.save(os.path.join(metrics_dir, f"{model_name}_confusion_matrix.npy"), conf_matrix)

    # Convert classification report to DataFrame and CSV
    report_df = pd.DataFrame(clf_report_dict).transpose()
    report_csv_path = os.path.join(metrics_dir, f"{model_name}_classification_report.csv")
    report_df.to_csv(report_csv_path)

    # Latency / FPS Benchmark on 1000 test samples
    benchmark_samples = min(1000, len(test_images))
    benchmark_subset = torch.stack([test_dataset[i][0] for i in range(benchmark_samples)]).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(benchmark_subset[:10])

    t0 = time.time()
    with torch.no_grad():
        for i in range(benchmark_samples):
            single_img = benchmark_subset[i:i+1]
            _ = model(single_img)
    total_bench_time = time.time() - t0
    latency_per_img_ms = (total_bench_time / benchmark_samples) * 1000.0
    fps = benchmark_samples / total_bench_time

    # Error Analysis: top correct and top misclassified
    confidences = np.max(all_probs, axis=1)
    correct_mask = (all_preds == all_targets)
    incorrect_mask = ~correct_mask

    # Correct samples with high confidence
    top_correct_indices = np.where(correct_mask)[0]
    top_correct_sorted = top_correct_indices[np.argsort(-confidences[top_correct_indices])[:15]]

    # Error samples with high confidence
    top_error_indices = np.where(incorrect_mask)[0]
    top_error_sorted = top_error_indices[np.argsort(-confidences[top_error_indices])[:15]]

    prediction_samples = {
        'correct': [
            {
                'index': int(idx),
                'true_label': class_names[all_targets[idx]],
                'pred_label': class_names[all_preds[idx]],
                'confidence': float(round(confidences[idx], 4)),
                'all_probs': [float(round(p, 4)) for p in all_probs[idx]]
            }
            for idx in top_correct_sorted
        ],
        'errors': [
            {
                'index': int(idx),
                'true_label': class_names[all_targets[idx]],
                'pred_label': class_names[all_preds[idx]],
                'confidence': float(round(confidences[idx], 4)),
                'all_probs': [float(round(p, 4)) for p in all_probs[idx]]
            }
            for idx in top_error_sorted
        ]
    }

    pred_samples_path = os.path.join(predictions_dir, f"{model_name}_predictions.json")
    with open(pred_samples_path, 'w') as f:
        json.dump(prediction_samples, f, indent=2)

    # Per-class summary
    per_class_metrics = {}
    for c in class_names:
        per_class_metrics[c] = {
            'precision': float(round(clf_report_dict[c]['precision'], 4)),
            'recall': float(round(clf_report_dict[c]['recall'], 4)),
            'f1_score': float(round(clf_report_dict[c]['f1-score'], 4)),
            'support': int(clf_report_dict[c]['support'])
        }

    eval_summary = {
        'model_name': model_name,
        'test_accuracy': float(round(test_accuracy, 4)),
        'test_loss': float(round(test_loss, 4)),
        'macro_avg_f1': float(round(clf_report_dict['macro avg']['f1-score'], 4)),
        'weighted_avg_f1': float(round(clf_report_dict['weighted avg']['f1-score'], 4)),
        'total_params': int(total_params),
        'trainable_params': int(trainable_params),
        'checkpoint_size_mb': float(round(checkpoint_size_mb, 2)),
        'latency_per_img_ms': float(round(latency_per_img_ms, 3)),
        'throughput_fps': float(round(fps, 1)),
        'per_class': per_class_metrics,
        'confusion_matrix': conf_matrix.tolist()
    }

    eval_json_path = os.path.join(metrics_dir, f"{model_name}_eval.json")
    with open(eval_json_path, 'w') as f:
        json.dump(eval_summary, f, indent=2)

    print(f"\n=======================================================")
    print(f"EVALUATION RESULTS FOR: {model_name.upper()}")
    print(f"Test Accuracy: {test_accuracy*100:.2f}% | Test Loss: {test_loss:.4f}")
    print(f"Macro F1-Score: {clf_report_dict['macro avg']['f1-score']:.4f}")
    print(f"Parameters: {total_params:,} | Checkpoint Size: {checkpoint_size_mb:.2f} MB")
    print(f"Latency: {latency_per_img_ms:.2f} ms/image | FPS: {fps:.1f}")
    print(f"=======================================================\n")

    return eval_summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='baseline', choices=['baseline', 'resnet'])
    args = parser.parse_args()
    evaluate_model_on_test(args.model)
