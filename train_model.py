import json
import time
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, BEST_MODEL_FILE, RESULTS_FILE
from src.data_utils import load_adult_train_test
from src.features import AdultFeatureEngineer, aggregate_feature_importance

train_df, test_df = load_adult_train_test()
X_train = train_df.drop(columns='income')
y_train = (train_df['income'] == '>50K').astype(int)
X_test = test_df.drop(columns='income')
y_test = (test_df['income'] == '>50K').astype(int)

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), NUMERIC_FEATURES),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), CATEGORICAL_FEATURES),
])


def evaluate(estimator, X_eval, y_eval):
    pred = estimator.predict(X_eval)
    proba = estimator.predict_proba(X_eval)[:,1]
    return {
        'accuracy': round(float(accuracy_score(y_eval, pred)), 4),
        'precision': round(float(precision_score(y_eval, pred)), 4),
        'recall': round(float(recall_score(y_eval, pred)), 4),
        'f1': round(float(f1_score(y_eval, pred)), 4),
        'roc_auc': round(float(roc_auc_score(y_eval, proba)), 4),
        'confusion_matrix': confusion_matrix(y_eval, pred).tolist(),
    }

# Deployment model: full random forest fit
lr_pipe = Pipeline([
    ('features', AdultFeatureEngineer()),
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(C=3.0, solver='lbfgs', max_iter=1000))
])
start = time.time()
lr_pipe.fit(X_train, y_train)
log_metrics = evaluate(lr_pipe, X_test, y_test)
log_metrics['fit_seconds'] = round(time.time()-start, 2)
log_metrics['best_params'] = {'C':3.0,'solver':'lbfgs','max_iter':1000}

knn_pipe = Pipeline([
    ('features', AdultFeatureEngineer()),
    ('preprocessor', preprocessor),
    ('model', KNeighborsClassifier(n_neighbors=21, weights='uniform', p=1))
])
start = time.time()
knn_pipe.fit(X_train, y_train)
knn_metrics = evaluate(knn_pipe, X_test, y_test)
knn_metrics['fit_seconds'] = round(time.time()-start, 2)
knn_metrics['best_params'] = {'n_neighbors':21,'weights':'uniform','p':1}

rf_pipe = Pipeline([
    ('features', AdultFeatureEngineer()),
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(n_estimators=120, max_depth=None, min_samples_split=10, n_jobs=-1, random_state=42))
])
start = time.time()
rf_pipe.fit(X_train, y_train)
rf_metrics = evaluate(rf_pipe, X_test, y_test)
rf_metrics['fit_seconds'] = round(time.time()-start, 2)
rf_metrics['best_params'] = {'n_estimators':120,'max_depth':None,'min_samples_split':10}

feature_names = rf_pipe.named_steps['preprocessor'].get_feature_names_out()
importances = rf_pipe.named_steps['model'].feature_importances_
feature_importance = [
    {'feature': feature, 'importance': round(float(score), 4)}
    for feature, score in aggregate_feature_importance(feature_names, importances)[:12]
]

results = {
    'bundle_mode': 'trained-from-scratch-tuned-deployment',
    'deployment_model': 'Random Forest',
    'models': {
        'Logistic Regression': log_metrics,
        'KNN': knn_metrics,
        'Random Forest': rf_metrics,
    },
    'feature_importance': feature_importance,
}

BEST_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(rf_pipe, BEST_MODEL_FILE)
RESULTS_FILE.write_text(json.dumps(results, indent=2), encoding='utf-8')
print(json.dumps(results, indent=2))
