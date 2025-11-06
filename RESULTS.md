# ChestMNIST Binary Classification - Comprehensive Results

## 📊 Final Performance Metrics

### **Overall Performance**
| Metric | Score |
|--------|-------|
| **Accuracy** | **86.80%** |
| **Precision** | **89.75%** |
| **Recall** | **89.75%** |
| **F1-Score** | **89.75%** |
| **ROC AUC** | **~0.93** (from ROC curve) |

### **Per-Class Performance**

#### Cardiomegaly (Class 0)
- **Precision**: 81.48%
- **Recall**: 81.48%
- **F1-Score**: 81.48%
- **Support**: 243 samples

#### Pneumothorax (Class 1)
- **Precision**: 89.75%
- **Recall**: 89.75%
- **F1-Score**: 89.75%
- **Support**: 439 samples

---

## 📈 Generated Visualizations

### 1. **Confusion Matrix** (`confusion_matrix.png`)
- Shows true vs predicted classifications
- Displays both raw counts and percentages
- Helps identify misclassification patterns

### 2. **ROC Curve** (`roc_curve.png`)
- Area Under Curve (AUC): ~0.93
- Shows model's discrimination ability
- Compares against random classifier baseline

### 3. **Precision-Recall Curve** (`precision_recall_curve.png`)
- Average Precision (AP) score
- Important for imbalanced datasets
- Shows trade-off between precision and recall

### 4. **Prediction Distribution** (`prediction_distribution.png`)
- Histogram of predicted probabilities
- Separated by true class
- Shows model confidence and calibration

### 5. **Performance Metrics Table** (`performance_metrics_table.png`)
- Detailed metrics breakdown
- Per-class and overall statistics
- ROC AUC and Average Precision scores

### 6. **Comprehensive Evaluation** (`comprehensive_evaluation.png`)
- All plots in one view
- Complete overview of model performance

---

## 🎯 Key Findings

### **Strengths**
1. ✅ **High Precision (89.75%)**: Low false positive rate
2. ✅ **Balanced Performance**: Precision = Recall = F1
3. ✅ **Good AUC (~0.93)**: Excellent discrimination ability
4. ✅ **Better on Pneumothorax**: 89.75% vs 81.48% for Cardiomegaly

### **Areas for Improvement**
1. ⚠️ **Cardiomegaly Detection**: Lower performance (81.48%)
   - Minority class challenge (243 vs 439 samples)
   - Could benefit from more data or class balancing
2. ⚠️ **Gap to 92% Target**: Need +5.2% improvement
   - Suggest: Ensemble methods, TTA, advanced augmentation

---

## 🔬 Model Architecture

### **DenseNet121**
- **Total Parameters**: 7,472,897
- **Input Size**: 224×224 (resized from 28×28)
- **Pre-trained**: ImageNet weights
- **Loss Function**: Focal Loss (α=0.25, γ=2.0)
- **Optimizer**: AdamW (LR=0.0001, weight_decay=5e-5)
- **Training Strategy**: Freeze/Unfreeze (10 epochs frozen)

### **Data Augmentation**
- RandomHorizontalFlip (p=0.5)
- RandomRotation (±10°)
- RandomAffine (translate, scale, shear)
- ColorJitter (brightness, contrast)

---

## 📁 Project Structure

```
chest-mnist-classification/
│
├── model.py                    # DenseNet121 architecture
├── train.py                    # Training script with optimizations
├── evaluate.py                 # Comprehensive evaluation script
├── datareader.py               # Data loading and augmentation
├── focal_loss.py               # Custom loss functions
├── utils.py                    # Utility functions
│
├── best_model_densenet121.pth  # Saved best model
│
├── Results/
│   ├── comprehensive_evaluation.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── prediction_distribution.png
│   ├── performance_metrics_table.png
│   ├── training_history.png
│   └── val_predictions.png
│
└── PERFORMANCE_OPTIMIZATION.md # Optimization strategies
```

---

## 🚀 How to Use

### **Training**
```bash
python train.py
```

### **Evaluation**
```bash
python evaluate.py
```

### **Inference on New Data**
```python
import torch
from model import DenseNet121

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DenseNet121(in_channels=1, num_classes=2, pretrained=False)
checkpoint = torch.load('best_model_densenet121.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Predict
with torch.no_grad():
    output = model(image_tensor)
    probability = torch.sigmoid(output).item()
    prediction = int(probability > 0.5)
```

---

## 📊 Dataset Information

### **ChestMNIST Binary Classification**
- **Source**: MedMNIST dataset
- **Original Size**: 28×28 grayscale
- **Resized to**: 224×224 (for DenseNet)

### **Class Distribution**

#### Training Set (2,306 samples)
- Cardiomegaly (Label 0): 754 (32.7%)
- Pneumothorax (Label 1): 1,552 (67.3%)

#### Validation Set (682 samples)
- Cardiomegaly (Label 0): 243 (35.6%)
- Pneumothorax (Label 1): 439 (64.4%)

**Imbalance Ratio**: ~1:2 (handled via Focal Loss and weighted BCE)

---

## 🎓 Training Details

### **Optimization Strategies**
1. **Focal Loss**: Better for imbalanced medical imaging
2. **Freeze/Unfreeze**: Stabilize pre-trained features
3. **Differential LR**: Lower LR for backbone (0.1×)
4. **Label Smoothing**: Reduce overconfidence (0.1)
5. **Gradient Clipping**: Training stability (max_norm=1.0)
6. **CosineAnnealing**: Better LR scheduling

### **Best Epoch**: 17/50
- **Best Val Accuracy**: 86.95%
- **Training Accuracy**: ~95%+
- **Early Stopping**: Triggered at epoch 27

---

## 📈 Performance Comparison

| Model | Input Size | Accuracy | Precision | Recall | F1-Score |
|-------|-----------|----------|-----------|--------|----------|
| SimpleCNN (Baseline) | 28×28 | ~70% | - | - | - |
| DenseNet121 (v1) | 128×128 | 88.42% | - | - | - |
| DenseNet121 (v2) | 224×224 | 89.74% | - | - | - |
| **DenseNet121 (Final)** | **224×224** | **86.80%** | **89.75%** | **89.75%** | **89.75%** |

---

## 🎯 Recommendations for Further Improvement

### **To Reach 92%+ Accuracy**

#### 1. **Test-Time Augmentation (TTA)**
- Average predictions over augmented versions
- Expected gain: +1-2%

#### 2. **Model Ensemble**
- Combine DenseNet121 + ResNet50 + EfficientNet
- Expected gain: +2-3%

#### 3. **Advanced Augmentation**
- MixUp, CutMix, GridMask
- Expected gain: +0.5-1%

#### 4. **More Data**
- Collect more Cardiomegaly samples
- Use external datasets
- Expected gain: +1-2%

#### 5. **Attention Mechanisms**
- Add CBAM or SE blocks
- Expected gain: +0.5-1%

---

## 📝 Citation

```bibtex
@article{medmnist,
    title={MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis},
    author={Yang, Jiancheng and Shi, Rui and Ni, Bingbing},
    journal={arXiv preprint arXiv:2010.14925},
    year={2020}
}
```

---

## 🔧 Requirements

```
torch==2.5.1+cu121
torchvision==0.20.1+cu121
medmnist==3.0.2
numpy==2.2.6
matplotlib==3.10.6
scikit-learn==1.7.2
seaborn==0.13.2
tqdm==4.67.1
```

---

## 📞 Contact & Support

For questions or improvements, please open an issue or submit a pull request.

---

## ⚖️ License

This project is for educational and research purposes.

---

**Last Updated**: November 6, 2025  
**Best Model**: `best_model_densenet121.pth`  
**Accuracy**: 86.80%  
**Target**: 92%+ (Future work)
