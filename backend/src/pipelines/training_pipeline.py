import argparse
from typing import Optional
import pandas as pd 
import numpy as np
from src.components.data_ingestion import load_data
from src.components.data_cleaning import data_cleaning
from src.components.feature_engineering import split_train_and_test, feature_engineering
from src.components.model_training import log_reg_train
from src.components.model_predict import search_model_uri
from src.config import DATA_PATH, FAKE_DATA, DATA_SOURCES
from src.utils.logger import logger


def train_model_pipeline(force_retrain: Optional [bool] = False) -> str:
    """
    This function will be train a Logistic Regression from scratch if 
    you force the retrain manually.

    Args:
        force_retrain (bool, optional): Defaults to False.

    Raises:
        RuntimeError: Safeguard if something happen.

    Returns:
        
    """
    try:
        # First, I need to check if I've already had a trained model or
        # I'm forcing manually to retrain the model. So:
        if force_retrain or (search_model_uri() == ''):
            
            # HERE I've created (with some common sense) a fake data to simulate the real dataset:
            df = pd.DataFrame(FAKE_DATA)
            # To avoid MLFlow infer crash then:
            df[DATA_SOURCES['X_NUM_COLS']] = df[DATA_SOURCES['X_NUM_COLS']].astype(np.float64)
            
            # Use this for Production or an ETL pipeline instead:
            # df = load_data(DATA_PATH)
            df = data_cleaning(df)
            X_train, X_test, y_train, y_test = split_train_and_test(df)
            X_train, X_test, y_train, y_test = feature_engineering(X_train, X_test, y_train, y_test)
            
            log_reg_train(X_train, X_test, y_train, y_test)
            
            return
        
        else:
            print(f"🥇 You've already had a best model. Model URI: {search_model_uri()}")
            print("If you want to retrain manually, please run the script with '--force ratrain' instead.")
    
    except Exception as e:
        logger.debug('Error executing the Training Pipeline => %s', e)
        raise RuntimeError(f'Error executing the Training Pipeline => {e}') from e
    
if __name__ == '__main__':    
    # If we need to force the re-training model for some reason we can do it through:
    parser = argparse.ArgumentParser()
    parser.add_argument('--force_retrain', action='store_true', help='Force retraining even if model exists')
    args = parser.parse_args()
    
    # If the user force the re training then force_retrain comes True:
    train_model_pipeline(force_retrain=args.force_retrain)
    
    # On bash => from the root of the project => python backend/src/pipelines/training_pipeline.py --force_retrain
    # Or run as a package => python -m src.pipelines.training_pipeline