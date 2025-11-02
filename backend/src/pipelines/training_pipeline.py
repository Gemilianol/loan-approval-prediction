import argparse
import pandas as pd 
import numpy as np
from src.components.data_ingestion import load_data
from src.components.data_cleaning import data_cleaning
from src.components.feature_engineering import feature_engineering
from src.components.model_training import log_reg_train
from src.config import DATA_PATH, MODEL_URI
from src.utils.logger import logger


def train_model_pipeline(force_retrain=False) -> str:
    """
    This function will be train a Logistic Regression from scratch if no
    model trained is present or if you force the retrain manually.

    Args:
        force_retrain (bool, optional): Defaults to False.

    Raises:
        RuntimeError: Safeguard if something happen.

    Returns:
        str: New model URI to add on config file.
    """
    try:
        # First, I need to check if I've already had a trained model or
        # I'm forcing manually to retrain the model. So:
        
        if MODEL_URI == '' or force_retrain:
            
            df = load_data(DATA_PATH)
            df = data_cleaning(df)
            train, test = feature_engineering(df)
            
            model_info = log_reg_train(train, test)
            
            # I'll hold the model's URI at first on config.   
            print(model_info)
            return model_info
        
        else:
            print(f"You've already had a model. Model URI: {MODEL_URI}")
            print("If you want to retrain manually, please run the script with '--force ratrain' instead.")
    
    except Exception as e:
        logger.debug('Error handling the data passed =>%s', e)
        raise RuntimeError(f'Error handling the data passed => {e}') from e
    
if __name__ == '__main__':    
    # If we need to force the re-training model for some reason we can do it through:
    parser = argparse.ArgumentParser()
    parser.add_argument('--force_retrain', action='store_true', help='Force retraining even if model exists')
    args = parser.parse_args()
    
    # If the user force the re training then force_retrain comes True:
    train_model_pipeline(force_retrain=args.force_retrain)
    
    # On bash => from the root of the project => python backend/src/pipelines/training_pipeline.py --force_retrain