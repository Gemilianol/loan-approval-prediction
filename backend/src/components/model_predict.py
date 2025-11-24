import pandas as pd
import numpy as np
from typing import Optional
from src.config import DATA_PATH, MODEL_URI
from src.components.data_ingestion import load_data
from src.components.data_cleaning import data_cleaning
from src.components.feature_engineering import split_train_and_test, feature_engineering
import mlflow
import mlflow.sklearn
from sklearn.base import ClassifierMixin
from src.utils.logger import logger


def load_mlflow_model(model_uri: str) -> ClassifierMixin:
    """ 
    This function will load the model from MLFLow.

    Args:
        model_uri (str): Hardcoded model URI from config file.

    Raises:
        RuntimeError: Error with model URI passed.

    Returns:
        MLFlow: MLFLOW Model.
    """
    try:
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        return loaded_model
    except Exception as e:
        logger.debug('Error loading the model from MLFlow => %s', e)
        raise RuntimeError(f'Error loading the model from MLFlow => {e}') from e
    
def predict_input( data: pd.DataFrame, model: Optional [ClassifierMixin] = None) -> np.ndarray | str:
    """ 
    This function will predict the input data with model loaded. 

    Args:
        loaded_model: Model loaded from MLFlow.
        data (pd.DataFrame): input data.

    Raises:
        RuntimeError: Error with prediction.

    Returns:
        np.array: Prediction.
    """
    try:
        pred = model.predict(data)
        return ['✅ Loan Approved' if pred[0] == 1 else '❌ Loan Rejected'][0]
    except Exception as e:
        logger.debug('Error predicting the data passed =>%s', e)
        raise RuntimeError(f'Error predicting the data passed => {e}') from e
    
## If you want to try the function isolated of the project, then
## uncomment this snippet:

# if __name__ == '__main__':
#     data = pd.DataFrame({
#         'income': [187258], 
#         'credit_score': [123],
#         'loan_amount': [19598], 
#         'years_employed': [11],
#         'points': [229]
#     })
    
#     model = load_mlflow_model(MODEL_URI)
    
#     pred = predict_input(data, model)
    
#     print(pred)

# And run the script using the -m flag, treating the src folder as a package from backend:  
# python -m src.components.model_predict