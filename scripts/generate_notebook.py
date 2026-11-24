"""
Script to create the complete Jupyter Notebook notebooks/cifar10_deep_cnn.ipynb.
"""

import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = []

# Title & Metadata
cells.append(nbf.v4.new_markdown_cell("""# CIFAR-10 Image Classification with Deep CNNs

**Deep Learning Engineer & Data Scientist Report Notebook**  
- **Domain:** Computer Vision  
- **Task:** Multiclass Image Classification (10 Classes)  
- **Framework:** PyTorch & Torchvision  
- **Hardware:** CPU Multi-threading Optimized (16 Cores)  

---

## 1. Pendahuluan dan Lingkungan Eksperimen
Notebook ini mereproduksi seluruh alur kerja proyek klasifikasi citra CIFAR-10:
1. Pemeriksaan dan analisis eksplorasi data (EDA).
2. Pipeline preprocessing dan augmentasi data dengan pencegahan kebocoran data (*data leakage*).
3. Pembangunan dua arsitektur CNN: **Baseline CNN** (VGG-style dengan Dense Head) dan **ResNetCIFAR** (Custom ResNet dengan stem khusus 3x3 dan Global Average Pooling).
4. Pelatihan model dengan optimizer AdamW dan Cosine Annealing Learning Rate scheduler.
5. Evaluasi komprehensif pada *test set* independen (Akurasi, Presisi, Recall, F1-Score, Confusion Matrix, Latensi Inferensi).
6. Analisis kesalahan prediksi dan visualisasi hasil.
"""))

# Cell 2: Imports & Environment Check
cells.append(nbf.v4.new_code_cell("""import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# Add src to path
sys.path.append(os.path.abspath('../src'))
from data_loader import load_cifar_raw, stratified_split, compute_normalization_stats
from dataset import CIFAR10Dataset, get_data_transforms
from models import BaselineCNN, ResNetCIFAR, count_parameters

# Set seeds and threads
np.random.seed(42)
torch.manual_seed(42)
torch.set_num_threads(12)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"PyTorch Version: {torch.__version__}")
print(f"Device: {device} | Active Threads: {torch.get_num_threads()}")
"""))

# Cell 3: Data Inspection & EDA Markdown
cells.append(nbf.v4.new_markdown_cell("""## 2. Pemeriksaan Dataset dan Exploratory Data Analysis (EDA)
Memuat kelima berkas *training batch* (`data_batch_1` s.d. `data_batch_5`), metadata (`batches.meta`), dan berkas pengujian (`test_batch`).  
Format piksel CIFAR-10 mentah disusun dalam bentuk *channel-first* `(3072,) = 3 × 32 × 32`, yang kemudian ditransformasikan ke dimensi citra standar `(32, 32, 3)` dalam format `uint8` [0, 255].
"""))

# Cell 4: Load Data Code
cells.append(nbf.v4.new_code_cell("""# Muat dataset mentah
data_dir = '../dataset'
train_images, train_labels, test_images, test_labels, class_names = load_cifar_raw(data_dir)

print(f"Dimensi Train Images: {train_images.shape}, Tipe Data: {train_images.dtype}")
print(f"Jumlah Train Labels: {len(train_labels)}, Distribusi Kelas Seimbang: {len(np.unique(train_labels))} kelas")
print(f"Dimensi Test Images: {test_images.shape}, Tipe Data: {test_images.dtype}")
print(f"Daftar Kelas (batches.meta): {class_names}")

# Periksa nilai piksel dan missing values
assert not np.isnan(train_images).any(), "Ditemukan NaN pada training data!"
assert not np.isnan(test_images).any(), "Ditemukan NaN pada testing data!"
print(f"Rentang nilai piksel: [{train_images.min()}, {train_images.max()}]")
"""))

# Cell 5: EDA Samples Plot
cells.append(nbf.v4.new_code_cell("""# Visualisasi Sampel Citra per Kelas
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("Sampel Citra CIFAR-10 (10 Kelas)", fontsize=14, fontweight='bold')
for i, ax in enumerate(axes.flat):
    idx = np.where(train_labels == i)[0][0]
    ax.imshow(train_images[idx])
    ax.set_title(f"{i}: {class_names[i]}", fontweight='bold')
    ax.axis('off')
plt.tight_layout()
plt.show()
"""))

# Cell 6: Preprocessing Markdown
cells.append(nbf.v4.new_markdown_cell("""## 3. Preprocessing dan Pencegahan Data Leakage
- **Stratified Split:** Data training 50.000 citra dipisahkan secara terstratifikasi menjadi 45.000 data latih (*training split*) dan 5.000 data validasi (*validation split*) menggunakan `random_state=42`.
- **Statistik Normalisasi:** Mean dan standar deviasi kanal RGB dihitung **hanya dari training split**, lalu diterapkan secara identik ke validation dan test set.
- **Data Augmentasi:** Diterapkan **hanya pada data training** (`RandomCrop(32, padding=4, reflect)`, `RandomHorizontalFlip(p=0.5)`). Tidak menggunakan vertical flip agar orientasi objek tetap realistis.
"""))

# Cell 7: Preprocessing Code
cells.append(nbf.v4.new_code_cell("""# 1. Stratified Split 90:10
x_train, y_train, x_val, y_val = stratified_split(train_images, train_labels, val_ratio=0.1, random_seed=42)
print(f"Bentuk Training Split: {x_train.shape}, Validation Split: {x_val.shape}")

# 2. Hitung statistik normalisasi secara ketat dari training split
mean, std = compute_normalization_stats(x_train)
print(f"Statistik Normalisasi Training Split:")
print(f"Mean (R, G, B): {[round(m, 4) for m in mean]}")
print(f"Std  (R, G, B): {[round(s, 4) for s in std]}")

# 3. Transformasi dan DataLoader
train_transform, eval_transform = get_data_transforms(mean, std)
train_dataset = CIFAR10Dataset(x_train, y_train, transform=train_transform)
val_dataset = CIFAR10Dataset(x_val, y_val, transform=eval_transform)
test_dataset = CIFAR10Dataset(test_images, test_labels, transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

print("Semua DataLoader siap digunakan!")
"""))

# Cell 8: Model Architecture Markdown
cells.append(nbf.v4.new_markdown_cell("""## 4. Pengembangan Model Deep Learning
Kita membandingkan dua model konvolusional:
1. **Model 1: Baseline CNN**
   - Arsitektur sekuensial VGG-style 3-tahap (Conv2d, BatchNorm, ReLU, MaxPool, Dropout).
   - Menggunakan Flatten dan Dense Layer berukuran besar (2048 → 256 → 10).
   - Total parameter: **667.690 parameter**.
2. **Model 2: ResNetCIFAR (Improved Deep CNN)**
   - ResNet khusus CIFAR-10 dengan stem 3×3 conv stride 1 (tanpa max-pooling awal) agar resolusi 32×32 tetap utuh.
   - 3 Tahap Residual Blocks dengan skip-connection identitas.
   - Menggantikan Dense Layer dengan **Global Average Pooling (GAP)**.
   - Total parameter: **175.258 parameter** (3,8× lebih efisien).
"""))

# Cell 9: Model Inspection Code
cells.append(nbf.v4.new_code_cell("""model_base = BaselineCNN(num_classes=10)
model_res = ResNetCIFAR(num_classes=10)

tot_b, trn_b = count_parameters(model_base)
tot_r, trn_r = count_parameters(model_res)

print(f"Baseline CNN : {tot_b:,} parameters")
print(f"ResNetCIFAR  : {tot_r:,} parameters")
"""))

# Cell 10: Training History Markdown
cells.append(nbf.v4.new_markdown_cell("""## 5. Hasil Pelatihan dan Pemantauan Metrik
Berikut adalah visualisasi hasil pelatihan riil dari 10 epoch untuk kedua arsitektur model.
"""))

# Cell 11: Plot Loss and Accuracy Curves
cells.append(nbf.v4.new_code_cell("""df_base = pd.read_csv('../outputs/metrics/baseline_history.csv')
df_res = pd.read_csv('../outputs/metrics/resnet_history.csv')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss
ax1.plot(df_base['epoch'], df_base['train_loss'], '--', color='#4a90e2', label='Baseline Train Loss')
ax1.plot(df_base['epoch'], df_base['val_loss'], color='#1f4e79', label='Baseline Val Loss')
ax1.plot(df_res['epoch'], df_res['train_loss'], '--', color='#f5a623', label='ResNetCIFAR Train Loss')
ax1.plot(df_res['epoch'], df_res['val_loss'], color='#d0021b', label='ResNetCIFAR Val Loss')
ax1.set_title('Perbandingan Cross-Entropy Loss per Epoch', fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()

# Accuracy
ax2.plot(df_base['epoch'], df_base['train_acc'] * 100, '--', color='#4a90e2', label='Baseline Train Acc')
ax2.plot(df_base['epoch'], df_base['val_acc'] * 100, color='#1f4e79', label='Baseline Val Acc')
ax2.plot(df_res['epoch'], df_res['train_acc'] * 100, '--', color='#f5a623', label='ResNetCIFAR Train Acc')
ax2.plot(df_res['epoch'], df_res['val_acc'] * 100, color='#d0021b', label='ResNetCIFAR Val Acc')
ax2.set_title('Perbandingan Akurasi (%) per Epoch', fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Akurasi (%)')
ax2.legend()

plt.tight_layout()
plt.show()
"""))

# Cell 12: Final Evaluation Markdown
cells.append(nbf.v4.new_markdown_cell("""## 6. Evaluasi Final pada Test Set (test_batch)
Evaluasi akhir dilakukan satu kali secara independen pada 10.000 citra uji `test_batch`.
"""))

# Cell 13: Evaluation Metrics Code
cells.append(nbf.v4.new_code_cell("""with open('../outputs/metrics/baseline_eval.json') as f:
    eval_base = json.load(f)
with open('../outputs/metrics/resnet_eval.json') as f:
    eval_res = json.load(f)

comparison_df = pd.DataFrame([
    {
        'Model': 'Baseline CNN',
        'Test Accuracy': f"{eval_base['test_accuracy']*100:.2f}%",
        'Test Loss': f"{eval_base['test_loss']:.4f}",
        'Macro F1': f"{eval_base['macro_avg_f1']:.4f}",
        'Parameters': f"{eval_base['total_params']:,}",
        'Checkpoint Size': f"{eval_base['checkpoint_size_mb']:.2f} MB",
        'Latency (CPU)': f"{eval_base['latency_per_img_ms']:.2f} ms",
        'Throughput': f"{eval_base['throughput_fps']:.1f} FPS"
    },
    {
        'Model': 'ResNetCIFAR',
        'Test Accuracy': f"{eval_res['test_accuracy']*100:.2f}%",
        'Test Loss': f"{eval_res['test_loss']:.4f}",
        'Macro F1': f"{eval_res['macro_avg_f1']:.4f}",
        'Parameters': f"{eval_res['total_params']:,}",
        'Checkpoint Size': f"{eval_res['checkpoint_size_mb']:.2f} MB",
        'Latency (CPU)': f"{eval_res['latency_per_img_ms']:.2f} ms",
        'Throughput': f"{eval_res['throughput_fps']:.1f} FPS"
    }
])

display(comparison_df)
"""))

# Cell 14: Confusion Matrix Plot
cells.append(nbf.v4.new_code_cell("""conf_matrix = np.array(eval_res['confusion_matrix'])

plt.figure(figsize=(9, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix ResNetCIFAR pada Test Set (10.000 Citra)', fontweight='bold')
plt.xlabel('Label Prediksi Model', fontweight='bold')
plt.ylabel('Label Sebenarnya (Ground Truth)', fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
"""))

# Cell 15: Error Analysis Markdown & Code
cells.append(nbf.v4.new_markdown_cell("""## 7. Analisis Kesalahan Prediksi (Error Analysis)
Melihat sampel citra yang diklasifikasikan secara keliru oleh model terbaik (**ResNetCIFAR**).
"""))

cells.append(nbf.v4.new_code_cell("""with open('../outputs/predictions/resnet_predictions.json') as f:
    preds = json.load(f)

errors = preds['errors'][:5]

fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
fig.suptitle("Contoh Kesalahan Klasifikasi Model ResNetCIFAR", fontsize=13, fontweight='bold')

for i, err in enumerate(errors):
    ax = axes[i]
    ax.imshow(test_images[err['index']])
    ax.set_title(f"True: {err['true_label']}\nPred: {err['pred_label']}\nConf: {err['confidence']*100:.1f}%",
                 color='darkred', fontweight='bold', fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.show()
"""))

# Cell 16: Conclusion
cells.append(nbf.v4.new_markdown_cell("""## 8. Kesimpulan
1. **ResNetCIFAR unggul signifikan:** Mencapai Test Accuracy **80.34%** dan Macro F1 **0.8027**, melampaui Baseline CNN (**77.78%**).
2. **Efisiensi Parameter Ekstrem:** ResNetCIFAR hanya membutuhkan **175.258 parameter** (73,7% lebih sedikit dibandingkan Baseline CNN 667.690 parameter) berkat penggunaan *Residual Blocks* dan *Global Average Pooling*.
3. **Analisis Kesalahan:** Kebingungan terbesar model terjadi antara kelas hewan berkaki empat yang memiliki fitur tekstur serupa pada resolusi 32×32 piksel (khususnya *cat* vs *dog*). Kelas kendaraan (*automobile*, *truck*, *ship*, *airplane*) memiliki akurasi dan presisi tertinggi (>85-90%).
"""))

nb['cells'] = cells

os.makedirs('notebooks', exist_ok=True)
notebook_path = 'notebooks/cifar10_deep_cnn.ipynb'
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Jupyter Notebook successfully generated at: {notebook_path}")
