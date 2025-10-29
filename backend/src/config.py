'''
REAL CONFIGURATION LIKE A PRODUCTION READY ENVIRONMENT.
'''

DATA_PATH = 'data/raw_data.csv'

MLFLOW_URI = 'http://127.0.0.1:5000'

MODEL_PATH = ""

DATA_SOURCES = {
    'COLS_TO_DROP': ['name','city'],
    'TARGET': ['loan_approved'],
    'X_COLS_NAMES': ['income', 'credit_score', 'loan_amount', 'years_employed', 'points'],
    'Y_COL_NAME': ['loan_approved'],
    'DF_COL_NAMES': ['income', 'credit_score', 'loan_amount', 'years_employed', 'points', 'loan_approved']
}