import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from .config import CATEGORICAL_FEATURES

class AdultFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        for col in X.select_dtypes(include='object').columns:
            X[col] = X[col].astype(str).str.strip()
        X['capital_net'] = X['capital_gain'] - X['capital_loss']
        X['has_capital_gain'] = (X['capital_gain'] > 0).astype(int)
        X['has_capital_loss'] = (X['capital_loss'] > 0).astype(int)
        X['age_band'] = pd.cut(X['age'], bins=[0,25,35,45,55,65,100], labels=['18-25','26-35','36-45','46-55','56-65','65+'], include_lowest=True).astype(str)
        X['hours_band'] = pd.cut(X['hours_per_week'], bins=[0,25,40,50,100], labels=['Part-time','Standard','Overtime','Heavy'], include_lowest=True).astype(str)
        return X

def aggregate_feature_importance(feature_names, importances):
    grouped = {}
    for raw_name, importance in zip(feature_names, importances):
        base = raw_name
        if raw_name.startswith('num__'):
            base = raw_name.split('num__',1)[1]
        elif raw_name.startswith('cat__'):
            rem = raw_name.split('cat__',1)[1]
            if '__' in rem:
                rem = rem.split('__',1)[1]
            matched = None
            for col in CATEGORICAL_FEATURES:
                if rem == col or rem.startswith(col + '_'):
                    matched = col
                    break
            base = matched or rem
        grouped[base] = grouped.get(base, 0.0) + float(importance)
    return sorted(grouped.items(), key=lambda x: x[1], reverse=True)
