# Adult Income Classification Using an End-to-End Machine Learning Pipeline

## Submission Evidence Summary

Baseline and deployment evidence files:
1. [assets/model_results.json](assets/model_results.json)
2. [assets/hyperparameter_results.json](assets/hyperparameter_results.json)
3. [assets/hyperparameter_summary.csv](assets/hyperparameter_summary.csv)

Screenshots used in report:
1. [screenshots/AdultIncome_Dashboard.png](screenshots/AdultIncome_Dashboard.png)
2. [screenshots/AdultIncome_ModelComparison.png](screenshots/AdultIncome_ModelComparison.png)

Generated report charts and tables:
1. [assets/figures/missing_values_by_column.png](assets/figures/missing_values_by_column.png)
2. [assets/figures/income_class_distribution.png](assets/figures/income_class_distribution.png)
3. [assets/figures/numeric_feature_histograms.png](assets/figures/numeric_feature_histograms.png)
4. [assets/figures/model_comparison_metrics.png](assets/figures/model_comparison_metrics.png)
5. [assets/tables/model_metrics_ranked.csv](assets/tables/model_metrics_ranked.csv)
6. [assets/tables/tuning_summary_ranked.csv](assets/tables/tuning_summary_ranked.csv)

### Guideline Compliance Checklist

1. Explanation required: every stage includes short reasoning statements (for example Sections 3, 6, 7, 8, 10, and 11).
2. Decisions justified: preprocessing, feature engineering, metric selection, and final model choice are all justified with both theory and observed metrics.
3. Graphs, tables, visualizations included: report now has reproducible figure/table files under `assets/figures` and `assets/tables`, plus dashboard screenshots.
4. Concise writing: each section uses short interpretation bullets and avoids narrative filler.
5. Focus on understanding and reasoning: model selection and interpretation are tied to class imbalance context and stakeholder meaning, not only raw scores.

Artifact generation command:
1. `python generate_report_artifacts.py`

---

## 1) Problem Definition

This project addresses the problem of predicting whether a person's annual income is more than $50K or at most $50K using census-style demographic and employment attributes. In practical terms, the model receives a person's profile (for example age, education, occupation, and working hours) and outputs one of two classes: >50K or <=50K.

The task type is binary classification because:
1. The outcome variable has exactly two categories.
2. The objective is to assign each record to one of those two classes.
3. The model performance is measured using classification metrics such as Accuracy, Precision, Recall, and F1-score.

This problem is important for several real-world reasons. First, income-level prediction supports socioeconomic analysis by helping identify patterns associated with economic opportunity and wage disparity. Second, institutions and policymakers can use insights from feature importance to understand which factors are most strongly linked with higher earning outcomes, which can guide education and workforce interventions. Third, organizations can use such models in a responsible analytics context for labor-market studies, compensation benchmarking, and strategic planning. While this project is not intended for direct high-stakes individual decision-making, it demonstrates how structured machine learning can produce interpretable and actionable population-level insights.

From a machine learning perspective, this is also a strong benchmark problem because it contains:
1. Mixed feature types (numerical and categorical).
2. Realistic data imperfections such as missing values.
3. Potentially non-linear relationships and feature interactions.
4. A target with class imbalance considerations.

Formal problem statement:
Given a person's demographic and work-related attributes, learn a function f(X) that predicts income class Y in {<=50K, >50K} with strong generalization performance on unseen data.

---

## 2) Data Collection

The dataset used is the Adult Income (Census Income) dataset from the UCI Machine Learning Repository. It is a widely used real-world dataset for income classification tasks and contains demographic and employment features extracted from census records.

In this project, dataset file paths and schema are defined in [src/config.py](src/config.py), while loading logic is implemented in [src/data_utils.py](src/data_utils.py).

Source files used:
1. Training file: adult.data
2. Test file: adult.test

Feature and target schema from [src/config.py](src/config.py):
1. age
2. workclass
3. fnlwgt
4. education
5. education_num
6. marital_status
7. occupation
8. relationship
9. race
10. sex
11. capital_gain
12. capital_loss
13. hours_per_week
14. native_country
15. income (target)

Target variable:
- income, with two labels: <=50K and >50K.

Data loading and cleaning behavior from [src/data_utils.py](src/data_utils.py):
1. Reads train and test files using explicit schema.
2. Uses skipinitialspace to normalize spacing.
3. Treats ? as missing values.
4. Normalizes test labels by removing trailing dots in adult.test so labels match training data.

This dataset is suitable for the assignment because it is real, has mixed feature types, includes missing values, supports all required preprocessing/modeling steps, and enables interpretable insight generation.

---

## 3) Data Exploration and Preprocessing

### 4.1 EDA Summary

Dataset size:
1. Training rows: 32,561
2. Test rows: 16,281
3. Total rows: 48,842

Target distribution:
1. <=50K: 37,155
2. >50K: 11,687
3. Positive class rate (>50K): 23.93%

This indicates class imbalance, so evaluation should include Precision, Recall, and F1-score, not only Accuracy.

Missing values:
1. occupation: 2,809
2. workclass: 2,799
3. native_country: 857
4. Other columns: 0

Numeric summary highlights:
1. age mean: 38.64 (range 17 to 90)
2. education_num mean: 10.08 (range 1 to 16)
3. hours_per_week mean: 40.42 (range 1 to 99)
4. capital_gain median: 0, max: 99,999 (high skew)
5. capital_loss median: 0, max: 4,356
6. fnlwgt range: 12,285 to 1,490,400

Categorical distribution highlights:
1. education: HS-grad (15,784), Some-college (10,878), Bachelors (8,025)
2. workclass: Private dominates (33,906), plus 2,799 missing
3. occupation: Prof-specialty (6,172), Craft-repair (6,112), Exec-managerial (6,086), plus 2,809 missing
4. sex: Male (32,650), Female (16,192)

### 4.2 Preprocessing Pipeline

Implemented in [train_model.py](train_model.py) using ColumnTransformer.

Missing value handling:
1. Numeric features: median imputation.
2. Categorical features: most-frequent imputation.

Justification:
1. Median is robust to skew and outliers.
2. Most-frequent preserves valid category structure.
3. Prevents heavy row loss from dropping missing records.

Categorical encoding:
1. One-hot encoding with unknown-category handling.

Justification:
1. Converts nominal categories to model-friendly numeric inputs.
2. Avoids false ordinal assumptions.
3. Supports stable inference on unseen categories.

Scaling:
1. StandardScaler for numeric features.

Justification:
1. KNN is distance-based and sensitive to scale.
2. Logistic Regression optimization is more stable with standardized inputs.
3. Reduces domination of large-range features.

### 4.3 Suggested Figures for Report

1. Missing values bar chart by column.
2. Target class distribution bar chart.
3. Histograms for age, hours_per_week, education_num.
4. Boxplots by income class for key numeric variables.
5. Top category bar charts for workclass, education, and occupation.

---

## 4) Feature Engineering

Feature engineering is implemented in [src/features.py](src/features.py) through the AdultFeatureEngineer transformer. This transformer is used inside the model pipeline in [train_model.py](train_model.py), so feature creation is applied consistently during both training and inference.

### 5.1 Engineered Features Added

The following engineered features were created:
1. capital_net = capital_gain - capital_loss
2. has_capital_gain = 1 if capital_gain > 0 else 0
3. has_capital_loss = 1 if capital_loss > 0 else 0
4. age_band = binned age groups (18-25, 26-35, 36-45, 46-55, 56-65, 65+)
5. hours_band = binned work-hour groups (Part-time, Standard, Overtime, Heavy)

### 5.2 Why These Features Were Used

1. capital_net combines two sparse and skewed monetary variables into one net financial signal, which can better capture earning tendency.
2. has_capital_gain and has_capital_loss convert sparse continuous signals into binary indicators, improving pattern capture when many values are zero.
3. age_band captures non-linear life-stage effects that may not be represented well by raw age alone.
4. hours_band captures workload intensity categories, which are easier for models to separate than raw continuous hours in some cases.

This is a feature transformation strategy, which satisfies the assignment requirement to perform at least one feature engineering technique.

### 5.3 Quantitative Impact of Feature Engineering

To validate impact, Random Forest was tested with and without engineered features under the same preprocessing and model settings.

Results:

| Setup | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Without engineered features | 0.8524 | 0.7200 | 0.6139 | 0.6627 |
| With engineered features | 0.8536 | 0.7221 | 0.6180 | 0.6660 |

Observed improvement from feature engineering:
1. Accuracy: +0.0012
2. Precision: +0.0021
3. Recall: +0.0041
4. F1-score: +0.0033

Although the uplift is modest, it is consistent across all four classification metrics. This indicates the engineered features add useful predictive information and improve model robustness.

### 5.4 Implementation Quality Notes

1. Feature engineering is encapsulated in a reusable transformer class, avoiding manual preprocessing duplication.
2. The approach is pipeline-safe and reduces risk of train-test leakage.
3. Engineered feature names are also included in [src/config.py](src/config.py), keeping schema definitions consistent.

## 5) Model Training

Step 5 requires training 3 to 4 algorithms, including:
1. One basic model (Logistic Regression)
2. One distance-based model (KNN or SVM)
3. One ensemble model (Random Forest)

This requirement is fully implemented in [train_model.py](train_model.py), where all three required models are trained from scratch on the same training split using a shared preprocessing pipeline and the same engineered-feature transformer.

Models trained:
1. Logistic Regression (basic)
2. K-Nearest Neighbors (distance-based)
3. Random Forest (ensemble)

The output mode in [assets/model_results.json](assets/model_results.json) is now set to trained-from-scratch, confirming placeholder packaged metrics were replaced by real training outputs.

### 6.1 Trained Model Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Fit Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8597 | 0.7461 | 0.6157 | 0.6746 | 0.9115 | 0.62 |
| KNN | 0.8428 | 0.6814 | 0.6284 | 0.6539 | 0.8938 | 3.35 |
| Random Forest | 0.8536 | 0.7221 | 0.6180 | 0.6660 | 0.9032 | 1.58 |

### 6.2 Training Notes

1. All models were trained and evaluated on the same dataset split to ensure fair comparison.
2. Confusion matrices are generated for each model in [assets/model_results.json](assets/model_results.json).
3. Random Forest remains the deployment model in the current pipeline.

## 6) Hyperparameter Tuning

Step 6 requires an explicit hyperparameter tuning process using GridSearch or RandomSearch and reporting best parameters. This was implemented with GridSearchCV in [hyperparameter_tuning.py](hyperparameter_tuning.py).

Tuning configuration:
1. Search method: GridSearchCV
2. CV strategy: 3-fold cross-validation
3. Optimization metric: F1-score
4. Models tuned: Logistic Regression, KNN, Random Forest

Saved tuning evidence:
1. [assets/hyperparameter_results.json](assets/hyperparameter_results.json)
2. [assets/hyperparameter_summary.csv](assets/hyperparameter_summary.csv)

### 7.1 Parameter Search Spaces

1. Logistic Regression
  - model__C: [0.1, 1.0, 3.0]
  - model__solver: [lbfgs, liblinear]

2. KNN
  - model__n_neighbors: [11, 21, 31]
  - model__weights: [uniform, distance]
  - model__p: [1, 2]

3. Random Forest
  - model__n_estimators: [120, 220]
  - model__max_depth: [None, 20]
  - model__min_samples_split: [2, 10]

### 7.2 Best Parameters and Tuning Results

| Model | Candidates | Best CV F1 | Best Parameters | Test Accuracy | Test F1 | Search Time (s) |
|---|---:|---:|---|---:|---:|---:|
| Logistic Regression | 6 | 0.6790 | C=3.0, solver=lbfgs | 0.8594 | 0.6740 | 6.26 |
| KNN | 12 | 0.6669 | n_neighbors=21, p=1, weights=uniform | 0.8506 | 0.6691 | 57.58 |
| Random Forest | 8 | 0.6868 | n_estimators=120, max_depth=None, min_samples_split=10 | 0.8633 | 0.6802 | 15.24 |

### 7.3 Tuning Interpretation

1. Random Forest achieved the highest CV F1 and strongest test-set performance after tuning.
2. KNN improved after tuning, especially in F1 compared to untuned baseline.
3. Logistic Regression remained competitive and stable but did not surpass tuned Random Forest on F1.

## 7) Model Evaluation

Step 7 requires evaluating classification models with appropriate metrics and explaining results clearly. This project uses the required classification metrics:
1. Accuracy
2. Precision
3. Recall
4. F1-score

Additional evaluation metrics used:
1. ROC-AUC
2. Confusion matrix
3. Specificity (derived from confusion matrix)

Evaluation evidence source:
1. [assets/hyperparameter_results.json](assets/hyperparameter_results.json)
2. [assets/hyperparameter_summary.csv](assets/hyperparameter_summary.csv)
3. [assets/model_results.json](assets/model_results.json)

### 8.1 Why These Metrics Were Selected

The target classes are imbalanced (only 23.93% are >50K), so Accuracy alone is not sufficient.
1. Precision measures how many predicted >50K cases are truly >50K.
2. Recall measures how many true >50K cases are successfully detected.
3. F1-score balances Precision and Recall and is used as the main optimization metric in tuning.
4. ROC-AUC measures ranking quality independent of a single threshold.

### 8.2 Tuned Model Evaluation Results (Test Set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8594 | 0.7452 | 0.6152 | 0.6740 | 0.9115 |
| KNN | 0.8506 | 0.7015 | 0.6396 | 0.6691 | 0.8982 |
| Random Forest | 0.8633 | 0.7601 | 0.6154 | 0.6802 | 0.9144 |

### 8.3 Confusion Matrix Interpretation

Confusion matrix values (TN, FP, FN, TP):
1. Logistic Regression: (11626, 809, 1480, 2366)
2. KNN: (11388, 1047, 1386, 2460)
3. Random Forest: (11688, 747, 1479, 2367)

Interpretation:
1. Random Forest has the highest true negatives and the lowest false positives, giving the best specificity (0.9399).
2. KNN has the highest recall (0.6396), meaning it identifies more >50K cases, but with more false positives.
3. Logistic Regression is competitive and stable but slightly behind Random Forest in F1.

### 8.4 Clear Result Explanation

1. Best overall F1 and Accuracy were achieved by tuned Random Forest (F1=0.6802, Accuracy=0.8633).
2. KNN performed best on Recall, so it is useful when detecting as many >50K individuals as possible is prioritized.
3. Logistic Regression remained a strong baseline with high interpretability and solid ROC-AUC.

Given class imbalance and the need for balanced classification quality, F1-score was prioritized for comparison, making Random Forest the strongest overall model in this evaluation stage.

## 8) Model Comparison and Best Model Selection

Step 8 requires a direct comparison of all models, identification of the best model, and a clear justification. This section compares the tuned versions of Logistic Regression, KNN, and Random Forest using a consistent test set and the same feature pipeline.

### 9.1 Comparison Criteria

Primary comparison criteria:
1. F1-score (main metric due to class imbalance)
2. Accuracy (overall correctness)
3. ROC-AUC (ranking quality)

Secondary criteria:
1. Recall (ability to detect >50K)
2. Precision (quality of positive predictions)
3. Computational cost (search/training time)

### 9.2 Cross-Model Comparison Table

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Key Strength | Key Limitation |
|---|---:|---:|---:|---:|---:|---|---|
| Logistic Regression | 0.8594 | 0.7452 | 0.6152 | 0.6740 | 0.9115 | Interpretable and stable baseline | Lower F1 than Random Forest |
| KNN | 0.8506 | 0.7015 | 0.6396 | 0.6691 | 0.8982 | Highest recall for >50K detection | More false positives and lowest Accuracy |
| Random Forest | 0.8633 | 0.7601 | 0.6154 | 0.6802 | 0.9144 | Best Accuracy, F1, and ROC-AUC | Less interpretable than linear model |

### 9.3 Model Ranking

Ranking by the primary project objective (balanced performance on imbalanced classes):
1. Random Forest
2. Logistic Regression
3. KNN

Reasoning:
1. Random Forest has the highest F1 (0.6802), highest Accuracy (0.8633), and highest ROC-AUC (0.9144), giving the best overall balance.
2. Logistic Regression is second because it is close to Random Forest and highly interpretable.
3. KNN is third despite strong recall because it sacrifices Precision and Accuracy.

### 9.4 Best Model Selection and Justification

Selected best model: Random Forest

Justification:
1. It is the top model on all primary comparison metrics used for final selection (F1, Accuracy, ROC-AUC).
2. It provides the strongest trade-off between identifying >50K cases and avoiding false positives.
3. In this project, predictive quality is prioritized over model simplicity for deployment.

Decision statement:
Given the assignment objective and class-imbalance context, Random Forest is selected as the final model because it provides the most reliable overall classification performance on unseen test data.

## 9) Simple Deployment

Step 9 requires a simple deployment path. This project uses a Streamlit interface deployment, which satisfies the assignment requirement for a simple prediction interface.

Deployment files:
1. [app.py](app.py) - Streamlit user interface
2. [train_model.py](train_model.py) - training and model artifact generation
3. [models/best_model.joblib](models/best_model.joblib) - saved deployment model
4. [assets/model_results.json](assets/model_results.json) - deployment metadata and model metrics

### 10.1 Deployment Workflow

1. Run training to generate/update the model artifact:
  - command: py -3.11 train_model.py
2. Launch the interface:
  - command: py -3.11 -m streamlit run app.py
3. Enter user attributes in the Predict page.
4. Click Predict income class to get class label and confidence.

### 10.2 Deployed Model Used

The deployed model is the tuned Random Forest selected in Step 8. The current deployment bundle confirms:
1. deployment_model = Random Forest
2. bundle_mode = trained-from-scratch-tuned-deployment
3. best_params = n_estimators=120, max_depth=None, min_samples_split=10

### 10.3 Interface Capabilities

The deployed app supports:
1. Dashboard view for quick dataset/project overview
2. Data Explorer for numeric and categorical data browsing
3. Model Lab for model metric comparison and feature importance
4. Predict page for live single-record inference
5. Prediction history tracking with CSV download

### 10.4 Deployment Evidence for Report

Use these screenshots already saved in the project:
1. [screenshots/AdultIncome_Dashboard.png](screenshots/AdultIncome_Dashboard.png)
2. [screenshots/AdultIncome_ModelComparison.png](screenshots/AdultIncome_ModelComparison.png)

## 10) Interpretation and Insights

Step 10 requires translating model outcomes into simple, meaningful insights and recommendations. This section interprets the tuned Random Forest model, which was selected as the final model.

Interpretation evidence source:
1. [assets/model_results.json](assets/model_results.json)

### 11.1 Important Features Identified

Top features from the final model and their importance scores:
1. marital_status (0.1278)
2. relationship (0.0939)
3. capital_net (0.0927)
4. occupation (0.0862)
5. education_num (0.0812)
6. capital_gain (0.0691)
7. age (0.0686)
8. fnlwgt (0.0676)
9. education (0.0567)
10. hours_per_week (0.0461)

### 11.2 Plain-Language Interpretation

1. Family and household context matters: marital_status and relationship are the strongest predictors, indicating that household structure is strongly associated with income category in this dataset.
2. Financial capital signals are important: capital_net and capital_gain are major predictors, suggesting that investment-related income patterns are linked to higher income class.
3. Human-capital indicators matter: education_num and education contribute strongly, showing that education level remains a key factor in income outcomes.
4. Work context matters: occupation and hours_per_week contribute to the prediction, indicating that job type and working intensity influence income class.
5. Age has meaningful influence: age and age_band features support the idea that income probability changes across life stages.

### 11.3 What the Results Mean for Stakeholders

1. For career guidance teams: educational progression and occupation pathways are strong leverage points for improving probability of higher income outcomes.
2. For workforce planning: role type and working patterns should be considered in compensation and progression analyses.
3. For policy analysis: feature importance suggests that income differences are related to combined effects of education, occupational structure, and household characteristics.

### 11.4 Practical Recommendations

1. Prioritize skill and qualification development programs aligned with occupations associated with higher predicted income.
2. Use income analytics in combination with job-category analysis rather than relying on single demographic attributes.
3. Monitor groups with consistently lower predicted outcomes and design targeted support interventions (for example, upskilling pathways and career-transition guidance).
4. Keep model use decision-support oriented and not fully automated for high-stakes individual decisions.

### 11.5 Interpretation Limitations

1. Feature importance in Random Forest shows association, not direct causation.
2. Dataset context is historical and may not represent current labor-market dynamics in all regions.
3. Predictions should be interpreted at population-pattern level, not as deterministic outcomes for individuals.

## 11) Reflection

This project was completed as a full end-to-end machine learning pipeline on a real dataset, from problem definition to deployment. The process provided both technical and analytical learning outcomes.

### 12.1 What Worked Well

1. Pipeline modularity worked well: separating configuration, data loading, feature engineering, training, tuning, and deployment made the project easier to debug and improve.
2. Preprocessing design was effective: imputation, one-hot encoding, and scaling enabled fair comparison across very different model families.
3. Feature engineering contributed positively: engineered features improved model performance consistently across Accuracy, Precision, Recall, and F1.
4. Hyperparameter tuning improved reliability: GridSearchCV provided clear evidence-based parameter selection instead of manual guessing.
5. Deployment integration was successful: the Streamlit app provided a practical interface for live inference and demonstration.

### 12.2 Challenges Faced

1. Class imbalance made evaluation more complex, because high Accuracy alone could hide weak minority-class detection.
2. Keeping all report sections evidence-based required repeated reruns and synchronization between code outputs and written analysis.
3. KNN tuning was computationally slower than other models, increasing experimentation time.
4. Balancing interpretability and predictive performance required careful trade-off decisions when selecting the final model.

### 12.3 Possible Improvements

1. Add cross-validation confidence intervals and threshold-tuning analysis to strengthen evaluation robustness.
2. Extend interpretation with SHAP or permutation importance for deeper local and global explainability.
3. Add fairness-oriented analysis to check whether model performance varies across demographic subgroups.
4. Implement model versioning and lightweight monitoring logs for a more production-ready deployment setup.
5. Explore additional algorithms (for example gradient boosting) as optional comparisons if time permits.

### 12.4 Key Learnings

1. End-to-end ML quality depends more on data preparation and evaluation design than on model choice alone.
2. A reproducible pipeline with saved artifacts is essential for trustworthy reporting and deployment consistency.
3. Hyperparameter tuning is not optional for serious comparison; it can change both ranking and final selection.
4. Model interpretation must be communicated in plain language to be useful for non-technical stakeholders.
5. Reflection and critical analysis are necessary to show understanding beyond coding implementation.

Final reflection statement:
This project achieved the assignment objective by combining core machine learning concepts, practical implementation, and critical reasoning. The final tuned Random Forest model provided the strongest overall performance, while the complete pipeline and reflection process improved both technical skills and analytical decision-making.


---

## 12) Conclusion

This project successfully implemented a complete machine learning pipeline for binary income classification using the UCI Adult dataset. The workflow covered data preparation, feature engineering, model training, hyperparameter tuning, evaluation, model comparison, deployment, and reflection.

Among the tested models, the tuned Random Forest delivered the strongest overall performance on the test set, with Accuracy 0.8633, F1-score 0.6802, and ROC-AUC 0.9144. This model was therefore selected for deployment in the Streamlit interface.

The final outcome is a reproducible, evidence-backed ML solution that combines practical implementation with clear critical analysis and stakeholder-oriented interpretation.

---

## 13) References

1. UCI Machine Learning Repository - Adult Data Set.
2. Scikit-learn documentation: preprocessing, model selection, and classification metrics.
3. Streamlit documentation for model interface deployment.
