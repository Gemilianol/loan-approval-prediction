import pandas as pd
import numpy as np
from src.config import DATA_PATH, DATA_SOURCES
from src.components.data_ingestion import load_data
from src.components.data_cleaning import data_cleaning
from src.components.feature_engineering import feature_engineering
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

def log_reg_train(train: pd.DataFrame, test: pd.DataFrame) -> ClassifierMixin:
    """_summary_

    Args:
        train (pd.DataFrame): _description_
        test (pd.DataFrame): _description_

    Raises:
        RuntimeError: _description_

    Returns:
        ClassifierMixin: _description_
    """
    try:
        X_train, X_test = train.drop(columns=['loan_approved']), test.drop(columns=['loan_approved']) 
        y_train, y_test = train['loan_approved'], test['loan_approved']
    except Exception as e:
        logger.debug('Error handling the data passed =>%s', e)
        raise RuntimeError(f'Error handling the data passed => {e}') from e
    try:
        # Manual logging approach
        with mlflow.start_run():
            # Define hyperparameters
            # params = {"C": 1.0, "max_iter": 1000, "solver": "lbfgs", "random_state": 42}

            # Log parameters
            #mlflow.log_params(params)

            # Train model
            model = LogisticRegression(random_state=0)
            model.fit(X_train, y_train)

            # Make predictions
            preds = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, preds) # Measures the proportion of correctly classified instances
            print(f"Accuracy of Logistic Regression: {np.round(accuracy, 2)}")
            
            f1 = f1_score(y_test, preds) # Harmonic mean of precision and recall
            print(f"F1 Score of Logistic Regression: {np.round(f1,2)}")
            
            roc_auc = roc_auc_score(y_test, preds) # Evaluates the performance of a binary classifier across various classification thresholds
            print(f"AUC of Logistic Regression: {np.round(roc_auc,2)}")

            # Calculate and log metrics
            metrics = {
                "accuracy": accuracy,
                "f1_score": f1,
                "roc_auc": roc_auc,  
            }
            
            mlflow.log_metrics(metrics)

            # Infer model signature
            signature = infer_signature(X_train, model.predict(X_train))

            # Log the model
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                signature=signature,
                input_example=X_train[:5],  # Sample input for documentation
            )
        
        return model
        
    # Here I want to catch any exception so:
    except Exception as e:
        logger.debug('Error training the model over the Dataset passed =>%s', e)
        raise RuntimeError(f'Error training the model over the Dataset passed  => {e}') from e
    

## If you want to try the function isolated of the project, then
## uncomment this snippet:

if __name__ == '__main__':
    df = load_data(DATA_PATH)
    df = data_cleaning(df)
    train, test = feature_engineering(df)
    model = log_reg_train(train, test)
    # Should be in a 2D array format
    # pd.DataFrame({'feature1': [10], 'feature2': [20]})
    # Or np.array([[10, 20]])
    pred = model.predict(pd.DataFrame({
        'income': [0.8381808251212738], 
        'credit_score': [1.2919151379795572],
        'loan_amount': [1.082574354413527], 
        'years_employed': [1.5351379413785595],
        'points': [1.229885468202287]
    }))
    print(pred)

# And run the script using the -m flag, treating the src folder as a package from backend:  
# python -m src.components.model_training