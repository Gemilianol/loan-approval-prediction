''' REAL CONFIGURATION LIKE A PRODUCTION READY ENVIRONMENT. '''

DATA_PATH = 'data/raw_data.csv'

DATA_SOURCES = {
    'COLS_TO_DROP': ['name','city'],
    'TARGET': ['loan_approved'],
    'X_COLS_NAMES': ['income', 'credit_score', 'loan_amount', 'years_employed', 'points'],
    'Y_COL_NAME': ['loan_approved'],
    'DF_COL_NAMES': ['income', 'credit_score', 'loan_amount', 'years_employed', 'points', 'loan_approved'],
    'X_NUM_COLS': ['income', 'credit_score', 'loan_amount', 'years_employed', 'points'],
    'X_CAT_COLS': [] # No categorical features in this project. 
}

MODEL_PATH = ''

# ------------- MLFLOW ENVIRONMENT VARIABLES -------------- #

MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000' # Localhost (Default)

# NGINX: You can pass authentication headers to MLflow using 
# HTTP Basic authentication:

MLFLOW_TRACKING_USERNAME = ''

MLFLOW_TRACKING_PASSWORD = ''

# Or use with HTTP Bearer authentication. Basic authentication 
# takes precedence if set.

MLFLOW_TRACKING_TOKEN = ''


