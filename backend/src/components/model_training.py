import pandas as pd
import numpy as np
from src.config import DATA_PATH, MLFLOW_URI
from src.components.data_ingestion import load_data
from src.components.data_cleaning import data_cleaning
from src.components.feature_engineering import feature_engineering
import mlflow
import mlflow.sklearn
# For simplicity we start with a Logistic Regression
from sklearn.linear_model import LogisticRegression
# Base class that all estimators inherit from in scikit-learn
# For a Classifier: **ClassifierMixin** 
# For a Regressor: **RegressorMixin** 
# Any generic Estimator: **BaseEstimator**
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from mlflow.models import infer_signature
#from mlflow import MlflowClient
from src.utils.logger import logger

def log_reg_train(train: pd.DataFrame, test: pd.DataFrame) -> ClassifierMixin:
    """
    This function will be keep X and Y separately and then will run MLFlow
    experiment in order to train a Logistic Regression model. 

    Args:
        train (pd.DataFrame): The entire train dataset.
        test (pd.DataFrame): The entire test dataset.

    Raises:
        RuntimeError: Safeguard if something happens.

    Returns:
        ClassifierMixin: The trained Logistic Regression model URI. 
    """
    try:
        X_train, X_test = train.drop(columns=['loan_approved']), test.drop(columns=['loan_approved']) 
        y_train, y_test = train['loan_approved'], test['loan_approved']
    except Exception as e:
        logger.debug('Error handling the data passed => %s', e)
        raise RuntimeError(f'Error handling the data passed => {e}') from e
    try:
        # Manual logging approach
        with mlflow.start_run():
            # Define hyperparameters
            # params = {"C": 1.0, "max_iter": 1000, 
            # "solver": "lbfgs", "random_state": 42}

            # Log parameters
            #mlflow.log_params(params)
            
            # # Set the MLFlow Server - Localhost (Default)
            # mlflow.set_tracking_uri(MLFLOW_URI)
            
            # # Set the MLFlow Client
            # MlflowClient(tracking_uri=MLFLOW_URI)
            
            # Train a model as usual:
            model = LogisticRegression(random_state=0)
            model.fit(X_train, y_train)

            # Make predictions
            preds = model.predict(X_test)
            
            # Measures the proportion of correctly classified instances
            accuracy = accuracy_score(y_test, preds)
            print(f"Accuracy of Logistic Regression: {np.round(accuracy, 2)}")
            
            # Harmonic mean of precision and recall
            f1 = f1_score(y_test, preds) 
            print(f"F1 Score of Logistic Regression: {np.round(f1,2)}")
            
            # Evaluates the performance of a binary classifier across various classification thresholds
            roc_auc = roc_auc_score(y_test, preds) 
            print(f"AUC of Logistic Regression: {np.round(roc_auc,2)}")

            # Calculate and log metrics
            metrics = {
                "accuracy": accuracy,
                "f1_score": f1,
                "roc_auc": roc_auc,  
            }
            
            # Log the metris on MLFlow
            mlflow.log_metrics(metrics)

            # Infer model signature
            signature = infer_signature(X_train, model.predict(X_train))

            ## Log and register model in one step
            model_info=mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                signature=signature,
                registered_model_name="LoanPredictionModel",
                input_example=X_train[:5],  # Sample input for documentation
            )
            
            # Set a tag that we can use to remind ourselves what this model was for
            mlflow.set_logged_model_tags(
            model_info.model_id, {"Training Info": "Basic LR model for Loan Prediction Data"}
            )

        return model_info.model_uri
        
    # Here I want to catch any exception so:
    except Exception as e:
        logger.debug('Error training the model over the Dataset passed =>%s', e)
        raise RuntimeError(f'Error training the model over the Dataset passed  => {e}') from e

## If you want to try the function isolated of the project, then
## uncomment this snippet:

# if __name__ == '__main__':
#     df = load_data(DATA_PATH)
#     df = data_cleaning(df)
#     train, test = feature_engineering(df)
    
#     model_info = log_reg_train(train, test)
    
#     # I'll hold the model's URI at first on config.   
#     print(model_info)

# And run the script using the -m flag, treating the src folder as a package from backend:  
# python -m src.components.model_training