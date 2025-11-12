import pandas as pd
import numpy as np
from src.config import DATA_PATH, MLFLOW_URI, DATA_SOURCES
from src.components.data_ingestion import load_data
from src.components.data_cleaning import data_cleaning
from src.components.feature_engineering import split_train_and_test, feature_engineering
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
#from mlflow import MlflowClient
# For simplicity we start with a Logistic Regression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# Base class that all estimators inherit from in scikit-learn
# For a Classifier: **ClassifierMixin** 
# For a Regressor: **RegressorMixin** 
# Any generic Estimator: **BaseEstimator**
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from src.utils.logger import logger

def log_reg_train(X_train, X_test, y_train, y_test) -> ClassifierMixin:
    """
    This function will run an MLFlow experiment in order to train a Logistic Regression model
    and then promote the model to production.

    Args:
        X_train (pd.DataFrame): X_train data.
        X_test (pd.DataFrame): X_test data.
        y_train (pd.DataFrame): Transformed y_train data.
        y_test (pd.DataFrame): Transformed y_test data.

    Raises:
        RuntimeError: Safeguard if something happens.

    Returns:
        ClassifierMixin: The trained Logistic Regression model URI (MLFlow). 
    """
    try:
        # Manual logging approach (Example from MLFlow)
        with mlflow.start_run():
            ## --- For more complex scenarios you can do this: --- ##
            
            # Define hyperparameters here (If they are needed)
            # params = {"C": 1.0, "max_iter": 1000, 
            # "solver": "lbfgs", "random_state": 42}

            # Log parameters into MLFlow
            #mlflow.log_params(params)
            
            # # Set the MLFlow Server - Localhost (Default)
            # mlflow.set_tracking_uri(MLFLOW_URI)
            
            # # Set the MLFlow Client
            # MlflowClient(tracking_uri=MLFLOW_URI)
            
            ## ---------------- And then you can: ---------------- ##
            
            # Train a model as usual:
            # model = LogisticRegression(random_state=0)
            # model.fit(X_train, y_train)

            # # Make predictions
            # preds = model.predict(X_test)
            
            ## ----------------------- OR: ----------------------- ##
            
            model = Pipeline(
                steps=[("scaler", StandardScaler()), 
                       ("classifier", LogisticRegression())]
                )
                
            model.fit(X_train, y_train)

            # # Make predictions
            preds = model.predict(X_test)
            
            # Measures the proportion of correctly classified instances
            accuracy = accuracy_score(y_test, preds)
            print(f"Accuracy of Logistic Regression: {np.round(accuracy, 2)}")
            
            # Harmonic mean of precision and recall
            f1 = f1_score(y_test, preds) 
            print(f"F1 Score of Logistic Regression: {np.round(f1,2)}")
            
            # Evaluates the performance of a binary classifier across 
            # various classification thresholds
            roc_auc = roc_auc_score(y_test, preds) 
            print(f"AUC of Logistic Regression: {np.round(roc_auc,2)}")

            # Keep the metrics calculated into a dict
            metrics = {
                "accuracy": accuracy,
                "f1_score": f1,
                "roc_auc": roc_auc,  
            }
            
            # Log the metris on MLFlow
            mlflow.log_metrics(metrics)
            
            ## --- More complex scenarios (Pipelines), you can use: --- ##
            
            # numeric_features = DATA_SOURCES['X_NUM_COLS']
            # numeric_transformer = Pipeline(
            #     steps=[("scaler", StandardScaler())]
            # )

            # categorical_features = DATA_SOURCES['X_CAT_COLS']
            # categorical_transformer = Pipeline(
            #     steps=[
            #         ("OHE", OneHotEncoder(sparse_output=False, drop='if_binary')),
            #     ]
            # )
            
            # A ColumnTransformer is designed to apply different preprocessing steps 
            # to different columns of the input data X. It receives X in the fit 
            # and transform methods and outputs a processed X.
            
            # preprocessor = ColumnTransformer(
            #     transformers=[
            #         ("num", numeric_transformer, numeric_features),
            #         ("cat", categorical_transformer, categorical_features)
            #     ], remainder='passthrough' # Important: what to do with columns not specified
            # )
            
            # model = Pipeline(
            #     steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression())]
            #     )
            
            ## --- Finally you have to infer the signature and log the model: --- ##
            
            # Infer model signature
            signature = infer_signature(X_train, model.predict(X_train))

            ## Log and register model in one step
            model = mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                signature=signature,
                registered_model_name="LoanPredictionModel",
                input_example=X_train[:5],  # Sample input for documentation
            )
            
            # OPTIONAL:
            # Set a tag that we can use to remind ourselves what this model was for:
            # mlflow.set_logged_model_tags(
            # model.model_id, {"Training Info": "Basic LR model for Loan Prediction Data"}
            # )
            
            # OPTIONAL:
            # mlflow.sklearn.save_model(model, f"../../models/Logistic_Regression_{model.model_id}")
            # print("Model saved locally at backend/models")

        return model.model_uri
        
    # Here I want to catch any exception so:
    except Exception as e:
        logger.debug('Error training the model over the X,y passed =>%s', e)
        raise RuntimeError(f'Error training the model over the X,y passed  => {e}') from e

## If you want to try the function isolated of the project, then
## uncomment this snippet:

if __name__ == '__main__':
    df = load_data(DATA_PATH)
    df = data_cleaning(df)
    print(df.head(5))
    X_train, X_test, y_train, y_test = split_train_and_test(df)
    X_train, X_test, y_train, y_test = feature_engineering(X_train, X_test, y_train, y_test)
    
    print(X_train.dtypes)
    
    model_info = log_reg_train(X_train, X_test, y_train, y_test)
    
    # I'll hold the model's URI at first on config.   
    print(model_info)

# And run the script using the -m flag, treating the src folder as a package from backend:  
# python -m src.components.model_training