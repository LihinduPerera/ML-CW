import json
import time
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, ASSETS_DIR
from src.data_utils import load_adult_train_test
from src.features import AdultFeatureEngineer


train_df, test_df = load_adult_train_test()
X_train = train_df.drop(columns='income')
y_train = (train_df['income'] == '>50K').astype(int)
X_test = test_df.drop(columns='income')
y_test = (test_df['income'] == '>50K').astype(int)

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ]), NUMERIC_FEATURES),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ]), CATEGORICAL_FEATURES),
])


def evaluate(estimator, X_eval, y_eval):
    pred = estimator.predict(X_eval)
    proba = estimator.predict_proba(X_eval)[:, 1]
    return {
        'accuracy': round(float(accuracy_score(y_eval, pred)), 4),
        'precision': round(float(precision_score(y_eval, pred)), 4),
        'recall': round(float(recall_score(y_eval, pred)), 4),
        'f1': round(float(f1_score(y_eval, pred)), 4),
        'roc_auc': round(float(roc_auc_score(y_eval, proba)), 4),
        'confusion_matrix': confusion_matrix(y_eval, pred).tolist(),
    }


def run_search(name, model, param_grid):
    pipe = Pipeline([
        ('features', AdultFeatureEngineer()),
        ('preprocessor', preprocessor),
        ('model', model),
    ])

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring='f1',
        cv=3,
        n_jobs=-1,
        verbose=1,
    )

    start = time.time()
    grid.fit(X_train, y_train)
    elapsed = round(time.time() - start, 2)

    best_estimator = grid.best_estimator_
    test_metrics = evaluate(best_estimator, X_test, y_test)

    return {
        'search_type': 'GridSearchCV',
        'cv_folds': 3,
        'optimization_metric': 'f1',
        'candidate_count': len(grid.cv_results_['params']),
        'search_seconds': elapsed,
        'search_space': param_grid,
        'best_params': grid.best_params_,
        'best_cv_f1': round(float(grid.best_score_), 4),
        'test_metrics': test_metrics,
    }


search_results = {
    'dataset': 'UCI Adult Income',
    'target': 'income (>50K vs <=50K)',
    'generated_at': pd.Timestamp.utcnow().isoformat(),
    'models': {}
}

search_results['models']['Logistic Regression'] = run_search(
    'Logistic Regression',
    LogisticRegression(max_iter=1000),
    {
        'model__C': [0.1, 1.0, 3.0],
        'model__solver': ['lbfgs', 'liblinear'],
    }
)

search_results['models']['KNN'] = run_search(
    'KNN',
    KNeighborsClassifier(),
    {
        'model__n_neighbors': [11, 21, 31],
        'model__weights': ['uniform', 'distance'],
        'model__p': [1, 2],
    }
)

search_results['models']['Random Forest'] = run_search(
    'Random Forest',
    RandomForestClassifier(random_state=42, n_jobs=-1),
    {
        'model__n_estimators': [120, 220],
        'model__max_depth': [None, 20],
        'model__min_samples_split': [2, 10],
    }
)

ASSETS_DIR.mkdir(parents=True, exist_ok=True)
out_file = ASSETS_DIR / 'hyperparameter_results.json'
out_file.write_text(json.dumps(search_results, indent=2), encoding='utf-8')

summary = []
for model_name, details in search_results['models'].items():
    summary.append({
        'model': model_name,
        'best_cv_f1': details['best_cv_f1'],
        'test_accuracy': details['test_metrics']['accuracy'],
        'test_precision': details['test_metrics']['precision'],
        'test_recall': details['test_metrics']['recall'],
        'test_f1': details['test_metrics']['f1'],
        'test_roc_auc': details['test_metrics']['roc_auc'],
        'search_seconds': details['search_seconds'],
        'best_params': json.dumps(details['best_params']),
    })

summary_df = pd.DataFrame(summary)
summary_csv = ASSETS_DIR / 'hyperparameter_summary.csv'
summary_df.to_csv(summary_csv, index=False)

print(json.dumps(search_results, indent=2))
print('\nSaved tuning artifacts:')
print(out_file)
print(summary_csv)
