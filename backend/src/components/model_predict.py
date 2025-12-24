import pandas as pd
import numpy as np
from typing import Optional
from src.config import DATA_PATH
from src.components.data_ingestion import load_data
from src.components.data_cleaning import data_cleaning
from src.components.feature_engineering import split_train_and_test, feature_engineering
import mlflow
import mlflow.sklearn
from sklearn.base import ClassifierMixin
from src.utils.logger import logger

def search_model_uri() -> str:
    '''
    Auxiliary function to get the 'best' model from MLFlow through its URI.
    
    :return: MLFlow model URI.
    :rtype: str
    
    '''
    try:
    #     # Basic search with pandas output (default) (Production Like)
    #     production_ready = mlflow.search_logged_models(
    #     experiment_ids=["0"],
    #     filter_string="""
    #         metrics.accuracy > 0.90
    #         AND metrics.roc_auc > 0.80
    #     """,
    #     order_by=[
    #     {"field_name": "metrics.accuracy", "ascending": False}  # Highest accuracy first
    # ],
    #     output_format='list' # Get results as a list instead of DataFrame
    # )
    
    # Basic search with pandas output (default) (Example Like)
        production_ready = mlflow.search_logged_models(
        experiment_ids=["0"],
        filter_string="""
            metrics.accuracy > 0.20
            AND metrics.roc_auc >= 0.50
        """,
        order_by=[
        {"field_name": "metrics.accuracy", "ascending": False}  # Highest accuracy first
    ],
        output_format='list' # Get results as a list instead of DataFrame
    )
        if production_ready != []:
            Logged_Model = production_ready[0].model_uri # Returns LoggedModel object.
        else:
            Logged_Model = ''
        # Then I need to catch the model URI to avoid hardcoded it. 
        return str(Logged_Model)
    
    # ------- Model Comparison ------- #
    
    # Get best model of each type
    # model_types = ["RandomForest", "LogisticRegression", "SVM"]
    # best_by_type = {}

    # for model_type in model_types:
    #     models = mlflow.search_logged_models(
    #         experiment_ids=["1"],
    #         filter_string=f"params.model_type = '{model_type}'",
    #         max_results=1,
    #         order_by=[{"field_name": "metrics.accuracy", "ascending": False}],
    #         output_format="list",
    #     )
    #     if models:
    #         best_by_type[model_type] = models[0]

    # # Compare results
    # for model_type, model in best_by_type.items():
    #     # Find accuracy in the metrics list
    #     accuracy = None
    #     for metric in model.metrics:
    #         if metric.key == "accuracy":
    #             accuracy = metric.value
    #             break

    #     accuracy_display = f"{accuracy:.4f}" if accuracy is not None else "N/A"
    #     print(
    #         f"{model_type}: Model ID = {model.model_id}, Run ID = {model.source_run_id}, Accuracy = {accuracy_display}"
    #     )
    except Exception as e:
        logger.debug('Error retrieving the MLFlow Model URI => %s', e)
        raise RuntimeError(f'Error retrieving the MLFlow Model URI => {e}') from e


def load_mlflow_model(model_uri: str) -> ClassifierMixin:
    """ 
    This function will load the model from MLFLow.

    Args:
        model_uri (str): Model URI gets from search_model_uri().

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
    
def predict_input( data: pd.DataFrame, model: ClassifierMixin) -> np.ndarray | str:
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

#     model_uri = search_model_uri()
    
#     model = load_mlflow_model(model_uri)
    
#     pred = predict_input(data, model)
    
#     print(pred)

# And run the script using the -m flag, treating the src folder as a package from backend:  
# python -m src.components.model_predict