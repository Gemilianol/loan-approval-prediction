import pandas as pd
import numpy as np
from src.components.model_predict import load_mlflow_model, predict_input
from src.config import MODEL_URI
from src.utils.logger import logger

# Cache the model loaded.
_MODEL = load_mlflow_model(MODEL_URI)
# _MODEL = None


def predict_pipeline(X: pd.DataFrame) -> np.ndarray:
    
    global _MODEL
    
    try:
        if _MODEL is None:
            print("Something happened. Trying to load the model again.")
            model = load_mlflow_model(MODEL_URI)
            pred = predict_input(model, X)
            return pred
        else:
            pred = predict_input(_MODEL, X)
            return pred
    except Exception as e:
        logger.debug('Something happened through the prediction process => %s', e)
        raise RuntimeError(f'Something happened through the prediction process => {e}') from e
    
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
    
    pred = predict_pipeline(data)
    
    print(pred)

# And run the script using the -m flag, treating the src folder as a package from backend:  
# python -m src.components.model_predict