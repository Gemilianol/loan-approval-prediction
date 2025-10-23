import pandas as pd
from src.utils.logger import logger
from src.components.data_ingestion import load_data
from src.config import DATA_PATH

# Pydantic, by default, does not know how to serialize or validate a Pandas DataFrame. 
# This is because DataFrames are complex objects and Pydantic requires explicit 
# instructions for handling such arbitrary types.

def data_cleaning(data: pd.DataFrame) -> pd.DataFrame:
    """
    This function will be in change of handle 
    the missing values and duplicates if they exists
    on the Dataset which is passed.
    
    Returns:
        pd.DataFrame: Cleaned Dataset.
    """
    try:
        data = data.dropna()
        data = data.drop_duplicates()
        return data
    
    # Here I want to catch any exception so:
    except Exception as e:
        logger.debug('Error cleaning the Dataset passed =>%s', e)
        raise RuntimeError(f'Error cleaning the Dataset passed  => {e}') from e

## If you want to try the function isolated of the project, then
## uncomment this snippet:
 
# if __name__ == '__main__':
#     df = load_data(DATA_PATH)
#     df = data_cleaning(df)
#     print(df.head(5))

# And run the script using the -m flag, treating the src folder as a package from backend:  
# python -m src.components.data_cleaning  
