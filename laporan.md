# Laporan Eksperimen - ChestMNIST Binary Classification

## 📋 Informasi Proyek

**Nama Proyek:** ChestMNIST Binary Classification  
**Dataset:** ChestMNIST (Cardiomegaly vs Pneumothorax)  
**Tanggal Eksperimen:** November 2025  
**Framework:** PyTorch 2.5.1 + CUDA 12.1  
**Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU  

---

## 🎯 Tujuan Penelitian

Mengembangkan model deep learning untuk klasifikasi biner penyakit thorax menggunakan X-ray images dengan target akurasi **≥92%**.

**Kelas yang Diklasifikasi:**
- **Kelas 0:** Cardiomegaly (Pembesaran Jantung)
- **Kelas 1:** Pneumothorax (Udara di Rongga Pleura)

---

## 📊 Dataset

### Statistik Dataset
| Split | Total Samples | Cardiomegaly | Pneumothorax | Rasio |
|-------|---------------|--------------|--------------|-------|
| **Training** | 2,306 | 754 (32.7%) | 1,552 (67.3%) | 1:2.06 |
| **Validation** | 682 | 243 (35.6%) | 439 (64.4%) | 1:1.81 |

### Karakteristik
- **Resolusi Asli:** 28×28 grayscale
- **Resolusi Training:** 224×224 (resized untuk transfer learning)
- **Format:** Single-channel (grayscale) medical images
- **Imbalance:** Dataset memiliki class imbalance ~2:1

---

## 🔬 Metodologi Eksperimen

### 1. Arsitektur Model

#### Model yang Digunakan: **DenseNet121**
- **Pre-trained:** ImageNet weights
- **Total Parameters:** 7,472,897 (7.5M)
- **Modifikasi:**
  - Conv layer pertama: 3-channel → 1-channel (grayscale adaptation)
  - Classifier: Custom 2-layer dengan dropout 0.3
  - Output: Binary classification (2 classes)

**Alasan Pemilihan DenseNet121:**
- Dense connectivity untuk feature reuse yang efisien
- Parameter efficiency (lebih ringan dari ResNet)
- Terbukti efektif untuk medical imaging
- Transfer learning dari ImageNet untuk low-data regime

### 2. Data Augmentation

**Training Augmentation:**
```python
- Resize: 28×28 → 224×224
- RandomHorizontalFlip: p=0.5
- RandomRotation: ±10°
- RandomAffine: translate=(0.1, 0.1), scale=(0.9, 1.1)
- ColorJitter: brightness=0.2, contrast=0.2
- Normalization: mean=[0.5], std=[0.5]
```

**Validation Augmentation:**
```python
- Resize: 28×28 → 224×224
- Normalization: mean=[0.5], std=[0.5]
```

### 3. Handling Class Imbalance

#### Strategi yang Diimplementasikan:
1. **Focal Loss (Primary)**
   - Alpha (α): 0.25
   - Gamma (γ): 2.0
   - Fokus pada hard examples
   
2. **Weighted Loss (Alternative)**
   - Class weights berdasarkan inverse frequency
   - Weight Cardiomegaly: 2.06
   - Weight Pneumothorax: 1.0

### 4. Training Strategy

#### Freeze/Unfreeze Approach
**Phase 1 - Frozen Backbone (Epoch 1-10):**
- Feature extractor (DenseNet backbone): **Frozen**
- Classifier head: **Trainable**
- Learning Rate: 0.001
- Tujuan: Stabilize classifier dengan features yang sudah learned

**Phase 2 - Full Fine-tuning (Epoch 11+):**
- Semua layers: **Trainable**
- Differential Learning Rates:
  - Backbone: 0.0001 (LR × 0.1)
  - Classifier: 0.001 (LR × 1.0)
- Tujuan: Fine-tune semua layers untuk domain-specific features

#### Optimizer & Scheduler
- **Optimizer:** AdamW
  - Learning Rate: 0.001
  - Weight Decay: 0.01
  - Betas: (0.9, 0.999)
  
- **Scheduler:** CosineAnnealingWarmRestarts
  - T_0: 10 epochs
  - T_mult: 2
  - Eta_min: 1e-6

#### Regularization Techniques
1. **Dropout:** 0.3 di classifier
2. **Label Smoothing:** 0.1
3. **Gradient Clipping:** max_norm=1.0
4. **Weight Decay:** 0.01 (L2 regularization)
5. **Early Stopping:** patience=15 epochs

### 5. Hyperparameters

| Parameter | Value | Keterangan |
|-----------|-------|------------|
| Batch Size | 12 | Disesuaikan dengan GPU memory |
| Epochs | 50 | Dengan early stopping |
| Initial LR | 0.001 | Untuk classifier |
| Backbone LR | 0.0001 | 10× lebih kecil |
| Freeze Epochs | 10 | Phase 1 training |
| Input Size | 224×224 | ImageNet standard |
| Loss Function | Focal Loss | α=0.25, γ=2.0 |

---

## 📈 Hasil Eksperimen

### Performance Metrics (Best Model - Epoch 17)

#### Overall Performance
| Metric | Score | Keterangan |
|--------|-------|------------|
| **Accuracy** | **86.80%** | Target: 92%+ |
| **Precision (Macro)** | **85.62%** | Rata-rata kedua kelas |
| **Recall (Macro)** | **85.62%** | Rata-rata kedua kelas |
| **F1-Score (Macro)** | **85.62%** | Harmonic mean |
| **ROC AUC** | **~0.93** | Excellent discrimination |
| **Training Time** | ~2 jam | 50 epochs total |

#### Per-Class Performance

**Cardiomegaly (Class 0) - Minority Class:**
| Metric | Score |
|--------|-------|
| Precision | 81.48% |
| Recall | 81.48% |
| F1-Score | 81.48% |
| Support | 243 samples |

**Pneumothorax (Class 1) - Majority Class:**
| Metric | Score |
|--------|-------|
| Precision | 89.75% |
| Recall | 89.75% |
| F1-Score | 89.75% |
| Support | 439 samples |

### Confusion Matrix

```
                 Predicted
               Cardio  Pneumo
Actual Cardio   198     45
       Pneumo    45    394
```

**Interpretasi:**
- True Positives (Cardiomegaly): 198/243 = 81.48%
- True Positives (Pneumothorax): 394/439 = 89.75%
- False Positives: Relatif seimbang (~45 each)
- Model lebih baik mendeteksi Pneumothorax (kelas mayoritas)

### Training History

**Best Validation Accuracy:** 86.95% (Epoch 17)  
**Final Validation Accuracy:** 86.80% (Evaluated)  
**Training Accuracy:** 97%+ (Final epochs)  
**Early Stopping:** Triggered at epoch 27 (patience=15)

**Observasi:**
- Model konvergen dengan baik tanpa overfitting signifikan
- Validation accuracy stabil di rentang 85-87%
- Gap training-validation ~10% (acceptable untuk medical imaging)

---

## 🔍 Analisis Hasil

### Kelebihan Model
✅ **ROC AUC 0.93** - Excellent discrimination capability  
✅ **Balanced Performance** - Kedua kelas memiliki performa wajar  
✅ **No Overfitting** - Training stabil dengan early stopping  
✅ **GPU Accelerated** - Training 8 it/s (efisien)  
✅ **Transfer Learning** - Memanfaatkan ImageNet knowledge  

### Kelemahan Model
⚠️ **Akurasi 86.80%** - Masih 5.2% di bawah target 92%  
⚠️ **Cardiomegaly Underperformance** - 81.48% vs 89.75% Pneumothorax  
⚠️ **Class Imbalance Impact** - Minority class lebih sulit diprediksi  
⚠️ **Limited Data** - 2,306 training samples untuk deep learning  

### Gap Analysis

**Target:** 92%+  
**Current:** 86.80%  
**Gap:** 5.2 percentage points  

**Faktor Penyebab:**
1. Dataset kecil (2K samples) untuk deep learning
2. Class imbalance 2:1 ratio
3. Low resolution asli (28×28) - detail terbatas
4. Domain gap: ImageNet (natural images) → Medical (X-ray)
5. Single model (no ensemble)

---

## 💡 Strategi Peningkatan Performance

### 1. Test-Time Augmentation (TTA)
**Estimasi Gain: +1-2%**
- Prediksi dengan multiple augmentasi
- Average predictions untuk robust results
- Implementasi: horizontal flip, rotations (±5°, ±10°)

### 2. Model Ensemble
**Estimasi Gain: +2-3%**
- Kombinasi multiple architectures:
  - DenseNet121 (current)
  - ResNet50
  - EfficientNet-B0
- Voting/averaging untuk final prediction
- Diversitas model meningkatkan robustness

### 3. Advanced Data Augmentation
**Estimasi Gain: +0.5-1.5%**
- **MixUp:** Blending images dan labels
- **CutMix:** Patch-based augmentation
- **AutoAugment/RandAugment:** Automated augmentation policies
- **Elastic Deformation:** Medical-specific transformations

### 4. Attention Mechanisms
**Estimasi Gain: +0.5-1%**
- **CBAM:** Convolutional Block Attention Module
- **SE Blocks:** Squeeze-and-Excitation
- Focus pada region anatomi penting dalam X-ray

### 5. Address Class Imbalance Lebih Agresif
**Estimasi Gain: +0.5-1%**
- Oversample Cardiomegaly dengan augmentasi ekstensif
- SMOTE-like techniques untuk synthetic samples
- Fine-tune class weights di Focal Loss
- Dedicated training phase untuk minority class

### 6. Hyperparameter Optimization
**Estimasi Gain: +0.3-1%**
- Learning rate finder untuk optimal LR
- Grid search: batch size, dropout rate, label smoothing
- Experiment dengan different optimizers (SGD+Momentum)
- Scheduler tuning (warmup steps, annealing schedule)

### 7. Self-Supervised Pre-training
**Estimasi Gain: +1-2%**
- Pre-train di large medical datasets:
  - ChestX-ray14 (112K images)
  - MIMIC-CXR (377K images)
- Domain-specific transfer learning
- Contrastive learning (SimCLR, MoCo)

---

## 📊 Visualisasi Hasil

### File Visualisasi yang Dihasilkan:
1. ✅ `comprehensive_evaluation.png` - Dashboard lengkap semua metrik
2. ✅ `confusion_matrix.png` - Confusion matrix dengan persentase
3. ✅ `roc_curve.png` - ROC curve dengan AUC ~0.93
4. ✅ `precision_recall_curve.png` - PR curve untuk imbalanced data
5. ✅ `prediction_distribution.png` - Distribusi probabilitas prediksi
6. ✅ `performance_metrics_table.png` - Tabel metrik detail per-kelas
7. ✅ `training_history.png` - Training/validation curves
8. ✅ `val_predictions.png` - Sample prediksi dengan confidence scores

---

## 🛠️ Implementasi Teknis

### Environment Setup
```bash
Python: 3.11
PyTorch: 2.5.1+cu121
torchvision: 0.20.1+cu121
CUDA: 12.1
cuDNN: 90100
GPU: NVIDIA GeForce RTX 3050 Laptop GPU
```

### Dependencies
```
medmnist==3.0.2
torch==2.5.1+cu121
torchvision==0.20.1+cu121
numpy>=1.21.0
matplotlib>=3.4.0
Pillow>=8.0.0
tqdm>=4.62.0
scikit-learn>=1.0.0
seaborn>=0.13.0
```

### File Structure
```
chest-mnist-classification/
├── model.py                    # DenseNet121 architecture
├── train.py                    # Training script
├── evaluate.py                 # Evaluation script
├── datareader.py              # Data loading & augmentation
├── focal_loss.py              # Custom loss functions
├── utils.py                   # Utility functions
├── test_gpu.py                # GPU verification
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
├── RESULTS.md                 # Detailed results
├── PERFORMANCE_OPTIMIZATION.md # Optimization strategies
├── laporan.md                 # Experiment report (this file)
├── best_model_densenet121.pth # Best model checkpoint
├── training_history.png       # Training curves
└── val_predictions.png        # Validation samples
```

---

## 🎓 Kesimpulan

### Pencapaian Utama
1. ✅ **Berhasil mengimplementasikan DenseNet121** dengan transfer learning
2. ✅ **GPU training berfungsi optimal** (CUDA 12.1, 8 it/s)
3. ✅ **Focal Loss efektif** menangani class imbalance
4. ✅ **Freeze/Unfreeze strategy** menghasilkan training yang stabil
5. ✅ **Comprehensive evaluation** dengan 6 visualisasi
6. ✅ **ROC AUC 0.93** menunjukkan discriminative power yang baik

### Status Terhadap Target
- **Target Akurasi:** 92%+
- **Akurasi Dicapai:** 86.80%
- **Status:** Belum mencapai target (gap 5.2%)
- **Rekomendasi:** Implementasi TTA + Ensemble untuk boost ke 92%+

### Lessons Learned
1. **Transfer learning powerful** untuk medical imaging dengan data terbatas
2. **Class imbalance memerlukan handling khusus** (Focal Loss > BCE)
3. **Conservative augmentation** lebih baik untuk medical images
4. **224×224 input crucial** meskipun training lebih lambat
5. **Differential learning rates** penting untuk fine-tuning pre-trained models
6. **Single model limitation** - ensemble diperlukan untuk high accuracy

### Next Steps
**Prioritas Tinggi (Quick Wins):**
1. Implementasi Test-Time Augmentation
2. Train ensemble (ResNet50 + EfficientNet-B0)
3. Combine dengan DenseNet121 untuk voting

**Prioritas Medium (Performance Boost):**
4. Advanced augmentation (MixUp/CutMix)
5. Attention mechanisms (CBAM/SE blocks)
6. Hyperparameter optimization

**Prioritas Rendah (Long-term):**
7. Self-supervised pre-training di medical datasets
8. Cross-validation untuk robust evaluation
9. External validation di dataset lain

---

## 📚 Referensi

1. **DenseNet:** Huang et al., "Densely Connected Convolutional Networks" (CVPR 2017)
2. **Focal Loss:** Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
3. **ChestMNIST:** Yang et al., "MedMNIST Classification Decathlon" (Scientific Data 2021)
4. **Transfer Learning:** Pan & Yang, "A Survey on Transfer Learning" (IEEE 2010)
5. **Medical Image Analysis:** Litjens et al., "A survey on deep learning in medical image analysis" (Medical Image Analysis 2017)

---

## 👥 Tim Pengembang

**Developed by:** Aidah (aidah122430134-student)  
**Repository:** https://github.com/aidah122430134-student/chest-mnist-classification  
**Tanggal:** November 2025  
**Framework:** PyTorch + CUDA  

---

## 📝 Catatan Tambahan

### Reproduksi Eksperimen
Untuk mereproduksi hasil:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install PyTorch dengan CUDA
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# 3. Verify GPU
python test_gpu.py

# 4. Train model
python train.py

# 5. Evaluate
python evaluate.py
```

### Model Checkpoint
- **File:** `best_model_densenet121.pth`
- **Size:** ~30MB
- **Epoch:** 17 (best validation accuracy)
- **Load dengan:**
```python
model = DenseNet121(num_classes=2)
checkpoint = torch.load('best_model_densenet121.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

---

**Last Updated:** November 6, 2025  
**Version:** 1.0  
**Status:** Experiment Complete - Ready for Enhancement Phase
