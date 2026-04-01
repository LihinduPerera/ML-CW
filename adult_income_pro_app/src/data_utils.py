import pandas as pd
from .config import COLUMNS, TRAIN_FILE, TEST_FILE

def load_adult_train_test():
    train_df = pd.read_csv(TRAIN_FILE, header=None, names=COLUMNS, skipinitialspace=True, na_values='?')
    test_df = pd.read_csv(TEST_FILE, header=None, names=COLUMNS, skipinitialspace=True, na_values='?', comment='|')
    test_df['income'] = test_df['income'].str.replace('.', '', regex=False)
    return train_df, test_df

def dataset_profile(train_df, test_df):
    full_df = pd.concat([train_df.assign(split='Train'), test_df.assign(split='Test')], ignore_index=True)
    missing = full_df.isna().sum().sort_values(ascending=False)
    return {
        'rows_total': int(len(full_df)),
        'train_rows': int(len(train_df)),
        'test_rows': int(len(test_df)),
        'feature_count': 14,
        'positive_rate': float((full_df['income'] == '>50K').mean()),
        'missing_by_column': {k:int(v) for k,v in missing.items() if int(v)>0},
    }
