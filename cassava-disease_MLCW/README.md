# 🌿 Cassava Leaf Disease Classification

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A complete machine learning pipeline for classifying cassava leaf diseases using deep learning feature extraction and traditional ML models.

## 📋 Project Overview

This project implements a multi-class image classification system to identify **5 types of cassava leaf conditions**:

| Class | Disease Name | Description |
|-------|--------------|-------------|
| `cbb` | Cassava Bacterial Blight | Bacterial infection causing leaf spots |
| `cbsd` | Cassava Brown Streak Disease | Viral disease affecting leaves and roots |
| `cgm` | Cassava Green Mite | Pest damage from green mites |
| `cmd` | Cassava Mosaic Disease | Most common viral disease |
| `healthy` | Healthy | No disease present |

### 🎯 Approach

We use a **hybrid approach** combining:
1. **Deep Learning**: VGG16 pre-trained CNN for feature extraction
2. **Traditional ML**: Logistic Regression, SVM, Random Forest for classification
3. **Dimensionality Reduction**: PCA for efficient feature processing

---

## 📁 Project Structure

```
cassava-disease_MLCW/
│
├── 📓 Notebooks
│   ├── 01_data_exploration.ipynb      # EDA and visualization
│   ├── 02_data_preprocessing.ipynb    # Data preprocessing & augmentation
│   ├── 03_feature_engineering.ipynb   # CNN feature extraction & PCA
│   ├── 04_model_training.ipynb        # Train all models
│   ├── 05_hyperparameter_tuning.ipynb # Optimize hyperparameters
│   ├── 06_model_evaluation.ipynb      # Evaluate & compare models
│   └── 07_deployment.ipynb            # Deployment interface
│
├── 📂 cassava-disease/                # Dataset folder
│   ├── train/train/{class_folders}/   # Training images
│   ├── test/test/                     # Test images
│   └── sample_submission_file.csv     # Test labels
│
├── 📂 outputs/                        # Generated outputs (auto-created)
│   ├── *.pkl                          # Saved models
│   ├── *.json                         # Configuration files
│   └── *.png                          # Visualizations
│
├── requirements.txt                   # Python dependencies
├── predict.py                         # Standalone prediction script
└── README.md                          # This file
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- NVIDIA GPU with CUDA (recommended for faster training)
- 8GB+ RAM

### Installation

1. **Clone or download the project**
   ```bash
   cd cassava-disease_MLCW
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify GPU (optional)**
   ```python
   import torch
   print(torch.cuda.is_available())
   if torch.cuda.is_available():
       print(torch.cuda.get_device_name(0))
   ```

### Running the Notebooks

Execute notebooks **in order** (01 → 07):

```bash
jupyter notebook
```

Or use VS Code / JupyterLab.

---

## 📊 Models Implemented

| Model | Type | Description |
|-------|------|-------------|
| **Logistic Regression** | Basic | Linear classifier with L2 regularization |
| **SVM (RBF Kernel)** | Distance-based | Non-linear classification using kernel trick |
| **Random Forest** | Ensemble | 300 decision trees with bagging |
| **CNN (VGG16)** | Deep Learning | Transfer learning with fine-tuning |

### Feature Pipeline

```
Image (224×224×3)
    ↓
VGG16 Feature Extraction (512 features)
    ↓
StandardScaler Normalization
    ↓
PCA Dimensionality Reduction (~95% variance)
    ↓
ML Classifier (LR / SVM / RF)
    ↓
Prediction (5 classes)
```

---

## 💻 Usage

### 1. Python API

```python
from predict import CassavaClassifier

# Initialize classifier
classifier = CassavaClassifier(...)

# Single prediction
result = classifier.predict('path/to/leaf_image.jpg')
print(result['class_full'])    # "Cassava Mosaic Disease"
print(result['confidence'])    # 0.92

# With visualization
result = classifier.predict_with_visualization('image.jpg')
```

### 2. Command Line

```bash
# Single image
python predict.py cassava_leaf.jpg

# Batch processing
python predict.py --batch ./test_images/

# Save results to JSON
python predict.py image.jpg -o result.json
```

### 3. Interactive (Jupyter)

Run `07_deployment.ipynb` and use the file upload widget.

---

## 📈 Results

### Model Comparison

| Model | Accuracy | F1-Score | AUC |
|-------|----------|----------|-----|
| Logistic Regression | ~XX% | ~XX% | ~X.XX |
| SVM (RBF) | ~XX% | ~XX% | ~X.XX |
| Random Forest | ~XX% | ~XX% | ~X.XX |
| CNN (VGG16) | ~XX% | ~XX% | ~X.XX |

*Note: Run the notebooks to generate actual results.*

### Sample Outputs

- **Confusion Matrix**: `outputs/confusion_matrices.png`
- **ROC Curves**: `outputs/roc_curves_comparison.png`
- **Feature Visualization**: `outputs/tsne_visualization.png`

---

## 📝 Report Structure

The project follows this report structure (2000-3000 words):

1. **Introduction** - Problem statement and objectives
2. **Literature Review** - Brief overview of approaches
3. **Data Collection** - Dataset description
4. **Data Exploration & Preprocessing** - EDA findings
5. **Feature Engineering** - CNN + PCA approach
6. **Model Training** - 4 models implemented
7. **Model Evaluation** - Comprehensive metrics
8. **Model Comparison** - Best model selection
9. **Deployment** - Simple prediction interface
10. **Interpretation & Insights** - Key findings
11. **Reflection** - Learnings and challenges
12. **Conclusion**
13. **References** (Harvard Format)
14. **Appendices** - Code and screenshots

---

## 👥 Team Members

| Name | Registration Number | Contribution |
|------|---------------------|--------------|
| [Student 1] | [Reg. Number] | 33.33% |
| [Student 2] | [Reg. Number] | 33.33% |
| [Student 3] | [Reg. Number] | 33.34% |

*Update this section with actual team details.*

---

## 📦 Dependencies

```
torch>=2.2.0
scikit-learn>=1.0.0
pandas>=1.4.0
numpy>=1.21.0
opencv-python>=4.5.0
matplotlib>=3.5.0
seaborn>=0.11.0
Pillow>=8.0.0
jupyter>=1.0.0
tqdm>=4.62.0
```

See `requirements.txt` for complete list.

---

## 🔧 Troubleshooting

### Common Issues

**1. GPU not detected**
```python
# Check CUDA installation
import torch
print(torch.cuda.is_available())
```
Install a CUDA-enabled PyTorch build that matches your NVIDIA driver and CUDA runtime.

**2. Out of memory error**
- Reduce `BATCH_SIZE` in notebooks
- Use smaller image subset for training

**3. Module not found**
```bash
pip install -r requirements.txt
```

**4. Slow training without GPU**
- Expected: CPU training is 10-50x slower
- Consider using Google Colab with GPU runtime

---

## 📚 References

- [Kaggle Cassava Disease Competition](https://www.kaggle.com/competitions/cassava-disease)
- [VGG16 Paper](https://arxiv.org/abs/1409.1556)
- [Transfer Learning for Image Classification](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)

---

## 📄 License

This project is for educational purposes as part of ML coursework.

---

## 🙏 Acknowledgments

- Dataset: [Kaggle Cassava Leaf Disease Classification](https://www.kaggle.com/competitions/cassava-disease)
- Pre-trained model: VGG16 (ImageNet weights)
- Course: Machine Learning Module

---

*Last updated: April 2026*
