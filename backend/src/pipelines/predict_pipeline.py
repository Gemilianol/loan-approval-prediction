import pandas as pd
import numpy as np
from typing import Optional
from sklearn.base import ClassifierMixin
from src.components.model_predict import predict_input
from src.utils.logger import logger

def predict_pipeline( X: pd.DataFrame, model: Optional [ClassifierMixin] = None) -> np.ndarray:
    
    try:
        pred = predict_input(model, X)
        return pred
    except Exception as e:
        logger.debug('Something happened through the prediction process => %s', e)
        raise RuntimeError(f'Something happened through the prediction process => {e}') from e
    
## If you want to try the function isolated of the project, then
## uncomment this snippet:

if __name__ == '__main__':
    
    # data = pd.DataFrame({
    #     'income': np.int64([157200]), 
    #     'credit_score': np.int64([699]),
    #     'loan_amount': np.int64([15985]), 
    #     'years_employed': np.int64([10]),
    #     'points': np.float64([59])
    # })
    
    data = pd.DataFrame({
        'income': [187258], 
        'credit_score': [123],
        'loan_amount': [19598], 
        'years_employed': [11],
        'points': [229]
    })
    
    pred = predict_pipeline(data)
    
    print(pred)

# And run the script using the -m flag, treating the src folder as a package from backend:  
# python -m src.pipelines.predict_pipeline