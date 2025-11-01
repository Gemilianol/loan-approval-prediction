import pandas as pd
import numpy as np
from src.config import DATA_PATH, MODEL_URI
from src.components.data_ingestion import load_data
from src.components.data_cleaning import data_cleaning
from src.components.feature_engineering import feature_engineering
from src.components.model_training import log_reg_train
import mlflow
import mlflow.sklearn
# For simplicity we start with a Logistic Regression
from sklearn.linear_model import LogisticRegression
# Base class that all estimators inherit from in scikit-learn
# For a Classifier: **ClassifierMixin** 
# For a Regressor:**RegressorMixin** 
# Any generic Estimator: **BaseEstimator**
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from mlflow.models import infer_signature
from src.utils.logger import logger


def load_mlflow_model(model_uri: str, data: pd.DataFrame) -> np.array:
    """ 
    This function will load the model from MLFLow and then predict the input data
    with that model. 

    Args:
        model_uri (str): Hardcoded model URI from config file
        data (pd.DataFrame): input data

    Raises:
        RuntimeError: Error with input data
        RuntimeError: Error with prediction

    Returns:
        np.array: Prediction
    """
    try:
        loaded_model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        logger.debug('Error loading the model from MLFlow =>%s', e)
        raise RuntimeError(f'Error loading the model from MLFlow => {e}') from e
    try:
        pred = loaded_model.predict(data)
        return pred
    except Exception as e:
        logger.debug('Error handling the data passed =>%s', e)
        raise RuntimeError(f'Error handling the data passed => {e}') from e
    
## If you want to try the function isolated of the project, then
## uncomment this snippet:

if __name__ == '__main__':
    
    # Should be in a 2D array format
    # pd.DataFrame({'feature1': [10], 'feature2': [20]})
    # Or np.array([[10, 20]])
    data = pd.DataFrame({
        'income': [0.8381808251212738], 
        'credit_score': [1.2919151379795572],
        'loan_amount': [1.082574354413527], 
        'years_employed': [1.5351379413785595],
        'points': [1.229885468202287]
    })
    
    pred = load_mlflow_model(MODEL_URI, data)
    
    print(pred)

# And run the script using the -m flag, treating the src folder as a package from backend:  
# python -m src.components.model_predict