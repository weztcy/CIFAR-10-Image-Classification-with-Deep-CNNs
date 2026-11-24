# Portfolio Project Data — CIFAR-10 Image Classification with Deep CNNs

- **Category:** `machine-learning` (`Machine Learning`)
- **Accent Color:** `indigo`
- **Target Assets Folder:** `public/images/projects/cifar_10_image_classification/`
- **Required Local Images:** `banner.webp`, `image01.webp` .. `image08.webp`

## TypeScript Data (`ProjectCaseStudy`)

```typescript
{
  "id": "cifar-10-image-classification-with-deep-cnns",
  "code": "PRO-39 // ML-24",
  "title": "CIFAR-10 Image Classification with Deep CNNs",
  "category": "machine-learning",
  "categoryLabel": "Deep Learning",
  "year": "2024",
  "status": "Completed",
  "accentColor": "indigo",
  "repoUrl": "https://github.com/weztcy/CIFAR-10-Image-Classification-with-Deep-CNNs",
  "license": "MIT License",
  "banner": "/images/projects/cifar_10_image_classification/banner.webp",
  "summary": "PyTorch CIFAR-10 classifier comparing BaselineCNN against a compact ResNetCIFAR achieving 80.34% test accuracy with 73.75% fewer parameters.",
  "projectDescription": "Built a leak-free PyTorch pipeline for 10-class CIFAR-10 image classification, contrasting a 667K-parameter BaselineCNN with a 175K-parameter ResNetCIFAR. Using a stride-1 stem, residual connections, and Global Average Pooling, ResNetCIFAR reached 80.34% test accuracy and 347.5 FPS on CPU.",
  "architectureSummary": "Raw CIFAR-10 Pickles -> Stratified 90:10 Split & Train-Only Normalization -> ResNetCIFAR (3 Residual Stages + GAP) -> Test Evaluation & Streamlit Inference UI",
  "techStack": [
    "Python",
    "PyTorch",
    "Torchvision",
    "NumPy",
    "Pandas",
    "Scikit-Learn",
    "Matplotlib",
    "Seaborn",
    "Pillow",
    "Altair",
    "Streamlit"
  ],
  "telemetryMatrix": {
    "projectType": "Deep CNN Image Classification Pipeline",
    "targetPlatform": "PyTorch Runtime & Streamlit Web UI",
    "primaryDomain": "Computer Vision & Visual Recognition",
    "architecturePipeline": "CIFAR Stem -> Residual Stages -> GAP Head",
    "testingAndAudit": "Verified",
    "projectStatus": "Completed",
    "projectYear": "2024"
  },
  "mlProjectInfo": {
    "projectType": "Deep Image Classification Pipeline",
    "learningParadigm": "Deep Learning",
    "modelDomain": "Computer Vision Object Recognition",
    "projectStatus": "Completed",
    "developmentYear": "2024",
    "capabilityBadges": [
      "Residual Skip Connections",
      "Global Average Pooling",
      "Leak-Free Normalization"
    ]
  },
  "mlProblemContext": "Classifying 32x32 natural RGB images across 10 diverse CIFAR-10 categories challenges CNNs because early pooling destroys fine spatial structure while large dense heads overfit. Designing a compact residual network with a resolution-preserving stem resolves gradient degradation and cuts parameter bloat.",
  "mlDatasetOverview": {
    "profileBadge": "60K Balanced RGB Images",
    "source": "Local CIFAR-10 Python Pickle Batches (dataset/original/)",
    "modality": "Computer Vision (32x32 RGB Imagery)",
    "scale": "60,000 Images (45,000 Train / 5,000 Val / 10,000 Test)",
    "inputShape": "3 x 32 x 32 Normalized Float32 Tensor",
    "dataFormat": "Python Pickle (Channel-First uint8 Vectors)",
    "description": "Sixty thousand 32x32 RGB images evenly distributed across 10 classes (5,000 train+val and 1,000 test per class). Automated inspection verified zero missing values and zero duplicate training images.",
    "provenanceNote": "Curated CIFAR-10 benchmark stored as local binary batches (batches.meta, data_batch_1..5, test_batch) with zero external API dependencies."
  },
  "mlFeaturesAndDataCharacteristics": {
    "spaceBadge": "Spatial RGB Tensor Space",
    "spaceNote": "Channel-first 3,072-byte vectors reshaped to (32, 32, 3) and standardized per channel",
    "objectiveOutput": {
      "label": "Target Output",
      "name": "CIFAR-10 Object Class Label",
      "type": "Multi-Class (10 Classes)",
      "distribution": "Perfectly balanced: 4,500 train, 500 validation, and 1,000 test samples per class"
    },
    "inputSignals": [
      {
        "name": "Red Channel Pixel Intensity",
        "type": "Spatial Tensor (32x32)",
        "importanceOrRole": "Train Mean 0.4911, Std 0.2469",
        "description": "Captures warm surface reflectance, vehicle paint hues, and terrestrial foreground contrast across the 32x32 grid."
      },
      {
        "name": "Green Channel Pixel Intensity",
        "type": "Spatial Tensor (32x32)",
        "importanceOrRole": "Train Mean 0.4821, Std 0.2434",
        "description": "Encodes natural foliage backgrounds and mid-spectrum luminance separating frogs and deer from urban backgrounds."
      },
      {
        "name": "Blue Channel Pixel Intensity",
        "type": "Spatial Tensor (32x32)",
        "importanceOrRole": "Train Mean 0.4464, Std 0.2616",
        "description": "Provides sky and water context cues for airplane and ship classes while exhibiting the highest channel variance."
      }
    ],
    "dataCharacteristicsSummary": "Each 32x32 RGB sample spans [0, 255] uint8 intensities with perfectly balanced 10-class counts across train, validation, and test splits. Because low spatial resolution blurs fine textures, global silhouette geometry and background color context strongly drive class separability."
  },
  "mlLearningApproach": {
    "paradigm": "Deep Learning (Supervised Multi-Class CNN Classification)",
    "methodologyDescription": "Implemented and benchmarked two end-to-end PyTorch CNNs under identical stratified splits and cosine annealing schedules. BaselineCNN stacks three Conv-BN-ReLU-MaxPool stages with a 2,048-unit dense head, whereas ResNetCIFAR uses residual blocks and Global Average Pooling.",
    "justification": "ResNetCIFAR preserves 32x32 spatial detail via a stride-1 stem without initial max-pooling, stabilizes gradient flow through identity and 1x1 projection shortcuts, and eliminates 524,288 dense-head parameters via Global Average Pooling to improve generalization."
  },
  "mlDataPreparation": [
    {
      "step": "01",
      "title": "Binary Unpickling & Integrity Audit",
      "description": "Decoded CIFAR-10 batches, reshaped 3,072-byte vectors to (32, 32, 3) RGB arrays, and verified zero NaN entries and zero training duplicates.",
      "badge": "Data Audit"
    },
    {
      "step": "02",
      "title": "Stratified Train-Validation Split",
      "description": "Partitioned the 50,000 raw training images 90:10 (45,000 train, 5,000 validation) using scikit-learn train_test_split with stratify and random_state=42.",
      "badge": "90:10 Split"
    },
    {
      "step": "03",
      "title": "Train-Only Z-Score Normalization",
      "description": "Computed per-channel RGB mean and standard deviation exclusively from the 45,000 training images and froze them for validation, test, and Streamlit inference.",
      "badge": "Zero Leakage"
    },
    {
      "step": "04",
      "title": "Dynamic Training Augmentation",
      "description": "Applied RandomCrop(32, padding=4, padding_mode='reflect') and RandomHorizontalFlip(p=0.5) strictly to training batches while keeping evaluation deterministic.",
      "badge": "Augmentation"
    }
  ],
  "mlDataPreparationNote": "All normalization statistics were calculated strictly from the 45,000-image training split, and test_batch remained completely isolated until single-pass final evaluation.",
  "mlModelArchitectureAndTraining": {
    "algorithmName": "ResNetCIFAR (Custom 3-Stage Residual CNN)",
    "architectureSummary": "Starts with a 3x3 stride-1 CIFAR stem (16 filters, no max-pooling), followed by three 2-block residual stages (16, 32, and 64 channels with stride-2 downsampling), Global Average Pooling, Dropout(0.2), and a Linear(64, 10) classifier totaling 175,258 parameters.",
    "configurationBadge": "175.3K Params // 10 Epochs",
    "hyperparameters": [
      {
        "name": "epochs",
        "value": "10",
        "description": "Full training schedule executed on 12-thread CPU (~103.6 s/epoch for ResNetCIFAR)."
      },
      {
        "name": "batch_size",
        "value": "128",
        "description": "Mini-batch size across training, validation, and test DataLoaders."
      },
      {
        "name": "learning_rate",
        "value": "0.001",
        "description": "Initial AdamW learning rate decayed smoothly via CosineAnnealingLR to eta_min=1e-5."
      },
      {
        "name": "weight_decay",
        "value": "0.0001",
        "description": "Decoupled L2 penalty applied by AdamW to prevent weight explosion."
      },
      {
        "name": "dropout",
        "value": "0.2",
        "description": "Regularization applied to the 64-dimensional Global Average Pooling feature vector."
      },
      {
        "name": "random_seed",
        "value": "42",
        "description": "Fixed seed across NumPy, PyTorch, and stratified splitting for deterministic runs."
      }
    ],
    "objectiveCriterion": "Multi-Class Cross-Entropy Loss (nn.CrossEntropyLoss)",
    "optimizationSolver": "AdamW Optimizer with CosineAnnealingLR Scheduler",
    "trainingProcess": "Both models trained for 10 epochs from scratch under deterministic seeding, saving weights solely when validation accuracy improved. ResNetCIFAR converged smoothly from 53.20% to 81.84% validation accuracy (0.5170 val loss), outperforming BaselineCNN's 77.80% peak validation accuracy."
  },
  "mlEvaluationStrategy": {
    "validationMethod": "90:10 Stratified Split + Isolated 10K Test Batch",
    "description": "Model selection relied strictly on 5,000 stratified validation samples per epoch, followed by a single-pass evaluation on the 10,000-image test_batch measuring accuracy, Cross-Entropy loss, per-class F1, confusion matrix, and 1,000-sample single-image CPU throughput.",
    "metrics": [
      {
        "label": "TEST ACCURACY",
        "value": "80.34%",
        "benchmark": "Baseline: 77.78%"
      },
      {
        "label": "MACRO F1-SCORE",
        "value": "0.8027",
        "benchmark": "Baseline: 0.7742"
      },
      {
        "label": "TEST LOSS",
        "value": "0.5621",
        "benchmark": "Baseline: 0.6399"
      },
      {
        "label": "MODEL PARAMS",
        "value": "175,258",
        "benchmark": "Baseline: 667,690"
      },
      {
        "label": "CPU THROUGHPUT",
        "value": "347.5 FPS",
        "benchmark": "Target > 60 FPS"
      }
    ]
  },
  "mlPerformanceVisualizations": [
    {
      "title": "Streamlit Inference UI — ResNetCIFAR Prediction Card",
      "badge": "Interactive App",
      "imageUrl": "/images/projects/cifar_10_image_classification/image01.webp",
      "telemetryNote": "Displays ResNetCIFAR model telemetry (80.34% Acc, 2.08 MB) and 99.37% confidence on a CIFAR-10 frog test sample."
    },
    {
      "title": "Streamlit Inference UI — Softmax Probability Breakdown",
      "badge": "Inference Output",
      "imageUrl": "/images/projects/cifar_10_image_classification/image02.webp",
      "telemetryNote": "Compares enlarged preview against 32x32 model input and plots 10-class Softmax probabilities via Altair."
    },
    {
      "title": "Streamlit Inference UI — Custom Image Upload Mode",
      "badge": "User Upload",
      "imageUrl": "/images/projects/cifar_10_image_classification/image03.webp",
      "telemetryNote": "Supports real-time classification on uploaded JPG, JPEG, PNG, and WEBP images resized to 32x32 pixels."
    },
    {
      "title": "Dataset Montage, Prediction Examples & Model Comparison",
      "badge": "Diagnostic Dashboard",
      "imageUrl": "/images/projects/cifar_10_image_classification/image04.webp",
      "telemetryNote": "Contrasts 80.34% vs 77.78% test accuracy and 175.3K vs 667.7K parameter efficiency alongside test prediction samples."
    },
    {
      "title": "Training & Validation Accuracy and Loss Curves (10 Epochs)",
      "badge": "Learning Curves",
      "imageUrl": "/images/projects/cifar_10_image_classification/image05.webp",
      "telemetryNote": "ResNetCIFAR lowers validation loss to 0.5170 and reaches 81.84% validation accuracy without overfitting divergence."
    },
    {
      "title": "Training-Split RGB Pixel Intensity Histogram & Boxplot",
      "badge": "EDA Statistics",
      "imageUrl": "/images/projects/cifar_10_image_classification/image06.webp",
      "telemetryNote": "Documents train-only channel distributions across [0, 255] prior to z-score normalization (Blue std=66.7 highest)."
    },
    {
      "title": "Per-Class Test Metrics & Split Class Balance",
      "badge": "Per-Class F1",
      "imageUrl": "/images/projects/cifar_10_image_classification/image07.webp",
      "telemetryNote": "Automobile (91.66% F1) and ship (89.58% F1) lead performance while cat (63.82% F1) trails across balanced splits."
    },
    {
      "title": "ResNetCIFAR 10-Class Confusion Matrix on Test Batch",
      "badge": "Confusion Matrix",
      "imageUrl": "/images/projects/cifar_10_image_classification/image08.webp",
      "telemetryNote": "Reveals strong vehicle diagonal recall (934 automobile, 909 truck) and primary cat-dog confusion (155 and 137 errors)."
    }
  ],
  "mlVisualizationTelemetryNote": "Across 10 epochs, ResNetCIFAR maintained a healthy generalization trajectory (79.55% train vs 81.84% val accuracy; 0.5873 train vs 0.5170 val loss) and transferred cleanly to 80.34% isolated test accuracy.",
  "mlModelInterpretationAndLimitations": {
    "interpretationSummary": "Confusion matrix and per-class F1 analysis show ResNetCIFAR excels on rigid vehicles with straight edges and distinct contexts (automobile 91.66% F1, ship 89.58% F1), whereas visually deformable quadrupeds sharing indoor or grassy scenes suffer higher error on 32x32 grids.",
    "keyDrivers": [
      {
        "factor": "Resolution-Preserving CIFAR Stem",
        "impact": "Retaining full 32x32 feature maps in Stage 1 instead of early 7x7/MaxPool downsampling yielded +2.56% test accuracy over BaselineCNN."
      },
      {
        "factor": "Global Average Pooling Head",
        "impact": "Replacing Flatten(2048)->Linear(2048,256) with spatial averaging reduced total parameters by 73.75% (667,690 to 175,258) and cut checkpoint size to 2.08 MB."
      },
      {
        "factor": "Object Rigidity vs Biological Texture",
        "impact": "Vehicle classes scored 81.96%-91.66% F1 from geometric silhouettes, whereas cat (63.82% F1) and dog (71.56% F1) exhibited 292 mutual misclassifications."
      }
    ],
    "limitations": [
      {
        "title": "10-Epoch CPU Training Budget",
        "description": "Training on CPU without CUDA acceleration constrained experiments to 10 epochs per model, leaving further accuracy gains from extended 50-100 epoch schedules unexplored."
      },
      {
        "title": "32x32 Spatial Resolution Bottleneck",
        "description": "Low pixel resolution discards fine facial and fur textures needed to disambiguate cropped cats, dogs, and birds against complex backgrounds."
      }
    ],
    "futureImprovements": [
      {
        "title": "MixUp, CutMix & Extended GPU Schedule",
        "target": "Train 50-100 epochs on GPU with label-mixing augmentations to reduce cat-dog confusion and push test accuracy above 90%."
      },
      {
        "title": "Squeeze-and-Excitation & INT8 Quantization",
        "target": "Add lightweight channel attention blocks and post-training INT8 quantization to shrink the 2.08 MB checkpoint under 1 MB."
      }
    ],
    "discoveryNote": "Error analysis of top-confidence mistakes confirms that cropped silhouettes and shared sky/water backgrounds drive the strongest false positives."
  },
  "mlModelArtifacts": [
    {
      "name": "resnet_best.pt",
      "format": "PT"
    },
    {
      "name": "baseline_best.pt",
      "format": "PT"
    },
    {
      "name": "normalization_stats.json",
      "format": "JSON"
    },
    {
      "name": "resnet_eval.json",
      "format": "JSON"
    },
    {
      "name": "resnet_history.csv",
      "format": "CSV"
    }
  ],
  "mlMethodologySpecs": {
    "learningParadigm": "Deep Learning",
    "modelArchitecture": "Custom ResNet CIFAR CNN",
    "coreFramework": "PyTorch",
    "validationMethod": "Stratified Split And Test"
  }
}
```
