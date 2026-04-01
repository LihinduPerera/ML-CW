# Cassava Leaf Disease Classification: Machine Learning Coursework Report

## Cover Page

**Module:** Machine Learning (Group Project)  
**Project Title:** Cassava Leaf Disease Classification Using a Hybrid Deep Feature + Traditional ML Pipeline  
**Submission Date:** 01 April 2026  
**Oral Presentation Date:** 02 April 2026

| Student Name | Registration Number | Contribution (%) |
|---|---|---:|
| [Student 1 Name] | [Registration Number] | 33.33 |
| [Student 2 Name] | [Registration Number] | 33.33 |
| [Student 3 Name] | [Registration Number] | 33.34 |

> Replace placeholders with actual member details before PDF submission.

---

## 1. Introduction

Agriculture remains one of the most important sectors for food security and economic stability in many countries. Cassava is a key staple crop, but its yield is significantly affected by leaf diseases and pest-related damage. In practical farming environments, early diagnosis is difficult because farmers may not always have immediate access to plant pathologists. This motivates a machine learning based decision-support solution that can classify cassava leaf conditions from images.

This project addresses a **multi-class image classification** problem with five classes: Cassava Bacterial Blight (CBB), Cassava Brown Streak Disease (CBSD), Cassava Green Mite (CGM), Cassava Mosaic Disease (CMD), and Healthy leaves. The implemented pipeline combines transfer learning based feature extraction (VGG16), dimensionality reduction (PCA), and multiple supervised learning algorithms (Logistic Regression, SVM, Random Forest, plus an optional neural classifier over extracted features).

The project objective is not only to train a model, but to demonstrate a complete end-to-end ML workflow: problem framing, data understanding, preprocessing, feature engineering, model training, evaluation, comparison, and simple deployment. The work is organized across seven notebooks in sequence (`01_data_exploration.ipynb` to `07_deployment.ipynb`) and generated artifacts under `outputs/`.

---

## 2. Literature Review (Brief)

Image-based plant disease detection has advanced rapidly due to deep convolutional neural networks (CNNs). Transfer learning is especially effective when domain data are moderate in size: pre-trained models (for example VGG16 trained on ImageNet) extract useful visual primitives such as texture, color gradients, and shape patterns. Instead of training deep networks from scratch, using pre-trained features can reduce computational burden and improve generalization.

Traditional machine learning classifiers remain valuable when used with strong features. Logistic Regression provides an interpretable baseline, SVM handles non-linear class boundaries effectively in high-dimensional spaces, and Random Forest offers a robust ensemble approach with feature-importance signals. Prior studies in agricultural vision commonly compare these model families to balance explainability, speed, and accuracy.

Dimensionality reduction techniques such as PCA are widely used to remove redundancy from high-dimensional embeddings. PCA can improve computational efficiency and reduce noise before downstream classification, although information loss must be controlled by selecting an explained-variance threshold.

In this coursework, the selected approach reflects this literature trend: deep representation learning for feature extraction, followed by conventional classifiers and systematic comparison.

---

## 3. Data Collection

The dataset used is the cassava disease image dataset stored under `cassava-disease/` in the project. The main training images are located in class-specific folders under `cassava-disease/train/train/`. A sample submission template (`cassava-disease/sample_submission_file.csv`) indicates expected test IDs.

From the EDA summary (`outputs/eda_summary.json`):

- Total training images: **5,656**
- Number of classes: **5**
- Class counts:
  - `cbb`: 466
  - `cbsd`: 1,443
  - `cgm`: 773
  - `cmd`: 2,658
  - `healthy`: 316
- Class imbalance ratio (max/min): **8.41**

The dataset is strongly imbalanced, with CMD dominating almost half of the training set and healthy samples underrepresented. This is an important practical issue because a model may optimize global accuracy while underperforming on minority disease classes.

An additional observation from project files is that the local `test/test` image folder is empty in this workspace (while the sample submission file has 3,774 IDs). Therefore, this report focuses on train/validation results and discusses implications in reflection.

---

## 4. Data Exploration and Preprocessing

Exploration in `01_data_exploration.ipynb` included class distribution plots, random sample grids, image dimension analysis, color histogram analysis, and mean-image visualizations per class. Image dimensions vary substantially (mean width ~614 px, mean height ~568 px), confirming the need for uniform resizing.

Key preprocessing decisions in `02_data_preprocessing.ipynb`:

1. **Train/Validation Split**
   - Stratified split at 80/20.
   - Training samples: 4,524; validation samples: 1,132.
   - Stratification preserves class proportions and prevents biased validation.

2. **Image Standardization**
   - Resize to `224 x 224` for compatibility with VGG16.
   - Normalize with ImageNet mean/std (`[0.485, 0.456, 0.406]`, `[0.229, 0.224, 0.225]`) to match pre-trained backbone expectations.

3. **Data Augmentation (training only)**
   - Random flips, rotation, brightness/contrast shifts.
   - Applied to improve robustness to orientation and lighting variations common in field imagery.

4. **Class Imbalance Handling**
   - Class weights computed on training labels and saved to `outputs/class_weights.json`.
   - Weights show minority emphasis (`healthy` highest weight 3.5763, `cmd` lowest 0.4256).

5. **Generator Pipeline**
   - A custom `CassavaDataGenerator` supports batch loading and optional augmentation.
   - This supports both deep and traditional workflows.

These choices are technically justified and consistent with standard ML practice for imbalanced image classification.

---

## 5. Feature Engineering

Feature engineering in `03_feature_engineering.ipynb` is the core innovation of this pipeline.

### 5.1 Deep Feature Extraction

The notebook uses pre-trained **VGG16** as a fixed feature extractor. Images are forwarded through convolutional blocks and pooled outputs are flattened into high-dimensional vectors. The extracted shape reported in notebook output is `25088` features per image prior to reduction.

Why this was used:

- Captures rich texture/shape patterns that are difficult to encode manually.
- Beneficial when training data are limited for full end-to-end deep training.
- Reuses ImageNet knowledge to improve generalization.

### 5.2 Standardization and PCA

After feature extraction, `StandardScaler` is applied, followed by PCA. The configuration saved in `outputs/feature_config.json` indicates:

- Original feature dimension: 512 (documented config) with workflow outputs also showing high-dimensional flattened embeddings.
- PCA components retained: **3,201**
- Explained variance: **94.83%**

From notebook logs, reducing from ~25k dimensions to 3,201 gives approximately **87.2% dimensionality reduction** while preserving most information. This is a strong trade-off between information retention and computational efficiency.

### 5.3 Visual Validation of Engineered Features

t-SNE and 2D PCA projections were generated (`outputs/tsne_visualization.png`, `outputs/pca_2d_visualization.png`) to inspect class separability. Although visual overlap remains (expected in biologically similar leaf disorders), clusters provide evidence that extracted features are meaningful and class-informative.

---

## 6. Model Training

Model training is implemented in `04_model_training.ipynb`. The coursework requirement asks for 3-4 models including one basic, one distance-based, and one ensemble model. The project satisfies this requirement with:

1. **Logistic Regression** (basic)
2. **SVM with RBF kernel** (distance-based)
3. **Random Forest** (ensemble)
4. **Optional PyTorch classifier on extracted features** (additional model)

Training outcomes from `outputs/training_results.json` and `outputs/model_comparison_initial.csv`:

- Logistic Regression: train 99.89%, validation 68.90%
- SVM (RBF): train 99.58%, validation 71.91%
- Random Forest: train 100.00%, validation 52.65%
- Feature-based neural classifier: train 98.61%, validation 70.94%

Interpretation:

- SVM gives the best validation accuracy among saved models.
- Random Forest shows severe overfitting (perfect training, weak validation), likely due to high-dimensional sparse decision boundaries and class imbalance interactions.
- Logistic Regression remains competitive and serves as a useful baseline.
- The optional neural head is close to SVM but not superior in this run.

---

## 7. Model Evaluation

To produce detailed metrics required by the assignment, validation predictions from the saved classifiers were analyzed. Results (macro-averaged metrics):

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.6890 | 0.5822 | 0.5676 | 0.5728 |
| SVM (RBF) | **0.7191** | **0.6451** | 0.5574 | **0.5850** |
| Random Forest | 0.5265 | 0.3263 | 0.2465 | 0.2093 |

Per-class observations:

- **CMD** is consistently easiest (highest recall across models), partly due to majority representation.
- **CBB** and **Healthy** have lower recall, indicating minority class difficulty.
- SVM improves precision notably, but CBB recall remains limited.
- Random Forest collapses toward predicting CMD for many samples, causing poor minority recall.

These findings reinforce the importance of evaluating macro metrics (not only accuracy) under class imbalance.

---

## 8. Model Comparison

Considering the primary classification metrics and class-level behavior, **SVM (RBF)** is selected as the best-performing model among completed baseline models.

Justification:

1. Highest validation accuracy (71.91%).
2. Highest macro precision and macro F1 among compared models.
3. Better balance between majority and minority class performance than Random Forest.
4. Stronger non-linear boundary modeling than Logistic Regression.

The neural classifier is close in accuracy (70.94%) but does not surpass SVM in this run, and full evaluation artifacts for that model are less complete in the saved outputs.

Important reproducibility note:

- `05_hyperparameter_tuning.ipynb` shifts from the planned LR/SVM/RF tuning to XGBoost-only search.
- Several expected tuning/evaluation output files are not present in `outputs/`.
- Therefore, final model selection in this report is grounded on the validated, available baseline artifacts.

---

## 9. Deployment

Simple deployment is implemented in `07_deployment.ipynb` via a reusable `CassavaClassifier` pipeline:

- Accepts image path or in-memory array.
- Applies preprocessing (resize + normalization).
- Extracts deep features with VGG16.
- Applies saved scaler and PCA.
- Uses trained classifier (`random_forest_model.pkl` in notebook implementation).
- Returns predicted class and confidence.

Additional deployment components include:

- Batch prediction function for multiple files.
- Command-line interaction function.
- Optional Jupyter widget upload interface.

This satisfies the coursework requirement for simple deployment, though aligning deployed classifier choice with the best-performing model (SVM) would improve consistency between evaluation and inference.

---

## 10. Interpretation and Insights

From a practical farming perspective, the system demonstrates that automated cassava disease categorization is feasible using a hybrid ML approach. Key insights include:

1. **Representation quality drives performance.** Pre-trained visual features plus PCA provide a useful feature space without full end-to-end CNN retraining on raw pixels.
2. **Model choice matters under imbalance.** SVM handled decision boundaries more effectively than Random Forest in this feature setting.
3. **Accuracy alone is insufficient.** Macro metrics reveal minority class challenges masked by majority CMD performance.
4. **Healthy and less frequent diseases need targeted support.** Lower recall in those classes suggests data-level interventions (more samples, stronger class-aware augmentation, or focal/class-balanced loss) are needed.
5. **Pipeline modularity is a strength.** Separate notebooks and serialized components (`.pkl`, `.json`) simplify reproducibility and future iteration.

Operational recommendation: if deployed in real farming scenarios, model predictions should be presented with confidence and caution text (e.g., “screening aid, not a final diagnosis”) and integrated with expert confirmation for uncertain cases.

---

## 11. Reflection (Mandatory)

### What worked well

- End-to-end structure is clear and pedagogically strong.
- EDA and visualization artifacts are extensive and support reasoned preprocessing.
- Feature extraction + PCA made traditional ML experimentation practical.
- Multiple model families were trained, enabling meaningful comparison.

### Challenges faced

- Strong class imbalance reduced minority class recall.
- Some notebook stages are not fully synchronized with earlier plans (for example tuning stage replacing planned model-specific searches with XGBoost path).
- Missing generated files from later stages indicate partial execution continuity.
- Cross-version model loading warnings (scikit-learn version mismatch) highlight environment dependency risks.

### Possible improvements

1. Complete planned hyperparameter tuning for LR/SVM/RF with documented search spaces and best parameters.
2. Add repeated stratified validation or cross-validation to improve reliability of model ranking.
3. Integrate imbalance-aware strategies (class-balanced sampling, focal loss, SMOTE variants on embeddings where appropriate).
4. Ensure deployment uses the selected best model and include calibration of predicted probabilities.
5. Lock dependency versions in an environment file to prevent serialization compatibility issues.

### Key learnings

- Real-world ML success depends as much on data quality and pipeline discipline as on algorithm complexity.
- Model comparison must include class-sensitive metrics.
- Reproducibility and execution consistency across notebooks are essential for defensible results.

---

## 12. Conclusion

This coursework demonstrates a complete ML pipeline for cassava leaf disease classification, from problem definition to deployment. The project effectively combines deep transfer learning features with traditional classifiers and provides substantial exploratory and visualization evidence. The best validated model among saved outputs is SVM (RBF), which achieved the strongest balance of accuracy and macro F1.

At the same time, the study reveals realistic limitations: class imbalance, partial tuning continuity, and reproducibility gaps due to environment/version differences. These limitations do not negate the project outcome; rather, they provide valuable engineering lessons and clear next steps for improvement.

Overall, the work meets the intended educational objective: applying core machine learning concepts in a practical, reasoned, and critically evaluated end-to-end workflow for a relevant real-world agricultural problem.

---

## 13. References (Harvard Format)

Breiman, L. (2001) ‘Random Forests’, *Machine Learning*, 45(1), pp. 5-32.

Cortes, C. and Vapnik, V. (1995) ‘Support-vector networks’, *Machine Learning*, 20, pp. 273-297.

Kaggle (n.d.) *Cassava Leaf Disease Classification*. Available at: https://www.kaggle.com/competitions/cassava-disease (Accessed: 1 April 2026).

Pedregosa, F. et al. (2011) ‘Scikit-learn: Machine Learning in Python’, *Journal of Machine Learning Research*, 12, pp. 2825-2830.

Simonyan, K. and Zisserman, A. (2015) ‘Very deep convolutional networks for large-scale image recognition’, *International Conference on Learning Representations (ICLR)*.

Torch Contributors (2026) *PyTorch Documentation*. Available at: https://pytorch.org/docs/stable/ (Accessed: 1 April 2026).

---

## 14. Appendices (Code and Screenshots)

### Appendix A: Notebook Workflow

- `01_data_exploration.ipynb`
- `02_data_preprocessing.ipynb`
- `03_feature_engineering.ipynb`
- `04_model_training.ipynb`
- `05_hyperparameter_tuning.ipynb`
- `06_model_evaluation.ipynb`
- `07_deployment.ipynb`

### Appendix B: Key Generated Artifacts

- EDA: `outputs/class_distribution.png`, `outputs/sample_images_grid.png`, `outputs/image_dimensions.png`, `outputs/color_histograms.png`, `outputs/mean_images.png`
- Preprocessing: `outputs/class_weights.json`, `outputs/preprocessing_config.json`, `outputs/train_val_split.pkl`
- Features: `outputs/feature_scaler.pkl`, `outputs/pca_model.pkl`, `outputs/extracted_features.pkl`, `outputs/feature_config.json`
- Training: `outputs/logistic_regression_model.pkl`, `outputs/svm_model.pkl`, `outputs/random_forest_model.pkl`, `outputs/cnn_final_model.pt`, `outputs/training_results.json`
- Comparison plots: `outputs/initial_comparison.png`, `outputs/model_comparison_initial.csv`, `outputs/rf_feature_importance.png`, `outputs/cnn_training_history.png`

### Appendix C: Reproducibility Notes

- Primary dependencies are listed in `requirements.txt`.
- Detected environment mismatch warning during model reload suggests pinning exact library versions used during model training.
- Before final submission, export this report to PDF and verify group-member details on the cover page.
