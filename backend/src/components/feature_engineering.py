import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from src.config import DATA_PATH, DATA_SOURCES
from src.components.data_ingestion import load_data
from src.components.data_cleaning import data_cleaning
from src.utils.logger import logger

def split_train_and_test(data: pd.DataFrame) -> pd.DataFrame | pd.Series:
    """ 
    Auxiliary function to do the train test split accordly.

    Args:
        data (pd.DataFrame): Get the dataset from the preview step (data_cleaning).

    Returns:
        X (pd.DataFrame): X_train, X_test 
        y (pd.Series): y_train, y_test
    """
    try:
        X = data.drop(columns=DATA_SOURCES['TARGET'])
        y = data[DATA_SOURCES['TARGET']]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) 
        
        return X_train, X_test, y_train, y_test
    
    # Here I want to catch any exception so:
    except Exception as e:
        logger.debug('Error splitting the Dataset passed => %s', e)
        raise RuntimeError(f'Error splitting the Dataset passed  => {e}') from e
    
def format_train_and_test(X_train: pd.DataFrame, 
                        X_test: pd.DataFrame,
                        y_train: pd.Series,
                        y_test: pd.Series) -> pd.DataFrame:
    """ 
    Another auxiliary function to get back the features' names and data format.

    Args:
        X (pd.DataFrame): X_train, X_test 
        y (pd.Series): y_train, y_test

    Returns:
        X,y (pd.DataFrame): X_train, X_test, y_train, y_test
    """
    try:        
        # OHE expected pd.Dataframe not pd.Series so:
        X_train = pd.DataFrame(X_train, columns=DATA_SOURCES['X_COLS_NAMES'], dtype='int')
        X_test = pd.DataFrame(X_test, columns=DATA_SOURCES['X_COLS_NAMES'], dtype='int')
        y_train = pd.DataFrame(y_train, columns=DATA_SOURCES['Y_COL_NAME'], dtype='int')
        y_test = pd.DataFrame(y_test, columns=DATA_SOURCES['Y_COL_NAME'], dtype='int')
        
        return X_train, X_test, y_train, y_test
    
    # Here I want to catch any exception so:
    except Exception as e:
        logger.debug('Error handling the train-test split passed => %s', e)
        raise RuntimeError(f'Error handling the train-test split passed  => {e}') from e

def feature_engineering(X_train: pd.DataFrame, 
                        X_test: pd.DataFrame,
                        y_train: pd.Series,
                        y_test: pd.Series) -> pd.DataFrame:
    """
    This function is in charge or apply transformations over the features.
    
    Since we will integrate MLFlow on the project, in general all the steps
    over the 'X' merges on one Pipeline object (SKlearn).
    
    Here I only will do the one hot encoding for the target.
    
    For this reason, I will leave commented a local version of it as an example
    and only will do the transformation needed over 'y'. 
    
    Args:
        X_train (pd.DataFrame): X_train data.
        X_test (pd.DataFrame): X_test data.
        y_train (pd.Series): y_train data.
        y_test (pd.Series): y_test data.

    Raises:
        RuntimeError: as a Safeguard.

    Returns:
        X_train (pd.DataFrame): Transformed X_train data.
        X_test (pd.DataFrame): Transformed X_test data.
        y_train (pd.DataFrame): Transformed y_train data.
        y_test (pd.DataFrame): Transformed y_test data.
    """
    try:
        #scaler = StandardScaler()
        onehot = OneHotEncoder(sparse_output=False, drop='if_binary') 
        # Avoid sparse matrix and get 2D array.
        
        # X_train = scaler.fit_transform(X_train) # Returns np.array
        # X_test = scaler.transform(X_test) # Returns np.array

        y_train = onehot.fit_transform(y_train) # Returns np.array
        y_test = onehot.transform(y_test) # Returns np.array
        
        X_train, X_test, y_train, y_test = format_train_and_test(X_train, X_test, y_train, y_test)
        
        ## --- Then if you want to get the transformed train-test sets do this --- ##
        
        # train = pd.concat([X_train, y_train], ignore_index=True, axis=1) => axis=1 means concat columns
        # test = pd.concat([X_test, y_test], ignore_index=True, axis=1) => axis=1 means concat columns
        
        # train.to_csv(../../data/train.csv)
        # test.to_csv(../../data/test.csv)
        
        # Also here you can dump the 'scaler' and the 'ohe' in order to load locally after.
        
        # import joblib
        # joblib.dump(scaler, '../../models/scaler.pkl')
        # joblib.dump(onehot, '../../models/ohe.pkl')
        
        # return train, test
        
        ## ----------------------------------------------------------------------- ##
        
        # Else returns the train-test split directly:
        return X_train, X_test, y_train, y_test
    
    # Here I want to catch any exception so:
    except Exception as e:
        logger.debug('Error transforming the Data passed => %s', e)
        raise RuntimeError(f'Error transforming the Data passed  => {e}') from e

## If you want to try the function isolated of the project, then
## uncomment this snippet:

# if __name__ == '__main__':
#     df = load_data(DATA_PATH)
#     df = data_cleaning(df)
#     X_train, X_test, y_train, y_test = split_train_and_test(df)
#     X_train, X_test, y_train, y_test = feature_engineering(X_train, X_test, y_train, y_test)
#     print(X_train.head(5))
#     print(y_test.head(5))

# And run the script using the -m flag, treating the src folder as a package from backend:  
# python -m src.components.feature_engineering