''' REAL CONFIGURATION LIKE A PRODUCTION READY ENVIRONMENT. '''

DATA_PATH = 'data/raw_data.csv'

FAKE_DATA = {
            'name': ['city', 'city', 'city', 'city', 'city', 'city', 'city', 'city', 'city', 'city'],
            'city': ['city', 'city', 'city', 'city', 'city', 'city', 'city', 'city', 'city', 'city'],
            'income': [18258, 19359, 30151, 39150, 50999, 75486, 89542, 98753, 115259, 157829], 
            'credit_score': [650, 550, 700, 500, 330, 750, 650, 850, 777, 555],
            'loan_amount': [199598, 5555, 2999, 9874, 11999, 19874, 159842, 75423, 45123, 32145], 
            'years_employed': [20, 10, 15, 7, 19, 25, 35, 33, 5, 3],
            'points': [30, 25, 32, 20, 10, 30, 40, 100, 30, 26],
            'loan_approved': [False, False, True, True, False, True, True, True, False, False]
            }

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


