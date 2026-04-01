from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
ASSETS_DIR = PROJECT_ROOT / 'assets'
TRAIN_FILE = DATA_DIR / 'adult.data'
TEST_FILE = DATA_DIR / 'adult.test'
BEST_MODEL_FILE = MODELS_DIR / 'best_model.joblib'
RESULTS_FILE = ASSETS_DIR / 'model_results.json'
COLUMNS = ['age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country','income']
NUMERIC_FEATURES = ['age','fnlwgt','education_num','capital_gain','capital_loss','hours_per_week','capital_net','has_capital_gain','has_capital_loss']
CATEGORICAL_FEATURES = ['workclass','education','marital_status','occupation','relationship','race','sex','native_country','age_band','hours_band']
TARGET_LABELS = {0:'<=50K',1:'>50K'}
