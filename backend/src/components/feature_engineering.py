import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from src.config import DATA_PATH, DATA_SOURCES
from src.components.data_ingestion import load_data
from src.components.data_cleaning import data_cleaning
from src.utils.logger import logger

def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """
    This function is in charge or apply transformations over the features 
    and also does the train test split accordly.
    
    HINT: You can also use isolated functions for Train/Test Split,
    StandardScaler and OneHotEncoder. As a result, some lines of code will repeated.
    In this case, for simplicity, I collapse everything here.
    
    Args:
        data (pd.DataFrame): Get the dataset from the preview step (data_cleaning).

    Returns:
        pd.DataFrame: Train + Test dataframes with the transformations needed.
    """
    try:
        X = data.drop(columns=DATA_SOURCES['TARGET'])
        y = data[DATA_SOURCES['TARGET']]
        # data['loan_approved'] returns pandas.core.series.Series
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) 
        # Returns pandas.core.frame.DataFrames & pandas.core.series.Series
        
        scaler = StandardScaler()
        onehot = OneHotEncoder(sparse_output=False, drop='if_binary') 
        # Avoid sparse matrix and get 2D array.
        
        # OHE expected pd.Dataframe not Series so:
        y_train = pd.DataFrame(y_train)
        y_test = pd.DataFrame(y_test)
        
        X_train = scaler.fit_transform(X_train) # Returns np.array
        X_test = scaler.transform(X_test) # Returns np.array

        y_train = onehot.fit_transform(y_train) # Returns np.array
        y_test = onehot.transform(y_test) # Returns np.array
        
        X_train = pd.DataFrame(X_train, columns=DATA_SOURCES['X_COLS_NAMES'])
        X_test = pd.DataFrame(X_test, columns=DATA_SOURCES['X_COLS_NAMES'])
        y_train = pd.DataFrame(y_train, columns=DATA_SOURCES['Y_COL_NAME'])
        y_test = pd.DataFrame(y_test, columns=DATA_SOURCES['Y_COL_NAME'])
        
        # axis=1 means concat columns
        train = pd.concat([X_train, y_train], ignore_index=True, axis=1)
        test = pd.concat([X_test, y_test], ignore_index=True, axis=1)
        
        # In order to keep the original columns' names then:
        train.columns = DATA_SOURCES['DF_COL_NAMES']
        test.columns = DATA_SOURCES['DF_COL_NAMES']
        
        return train, test
    
    # Here I want to catch any exception so:
    except Exception as e:
        logger.debug('Error cleaning the Dataset passed => %s', e)
        raise RuntimeError(f'Error cleaning the Dataset passed  => {e}') from e

## If you want to try the function isolated of the project, then
## uncomment this snippet:

# if __name__ == '__main__':
#     df = load_data(DATA_PATH)
#     df = data_cleaning(df)
#     train, test = feature_engineering(df)
#     print(train.head(5))
#     print(test.head(5))

# And run the script using the -m flag, treating the src folder as a package from backend:  
# python -m src.components.feature_engineering