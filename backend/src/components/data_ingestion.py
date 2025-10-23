import pandas as pd
from src.config import DATA_PATH
from src.utils.logger import logger
from pydantic import validate_call

@validate_call
def load_data(file_path: str) -> pd.DataFrame:
    """ A simple loader data function

    Args:
        file_path (str): which is located the data

    Returns:
        pd.DataFrame: Data as DataFrame
    """
    try:
        data = pd.read_csv(file_path)
        return data
        
    except FileNotFoundError as e:
        logger.debug('Error loading the CSV file %s', e)
        raise FileNotFoundError(f'Error loading the CSV file => {e}') from e
    
    # Here I want to catch any another exception so:
    except Exception as e:
        logger.debug('Error loading the CSV file =>%s', e)
        raise RuntimeError(f'Error loading the CSV file => {e}') from e

## If you want to try the function isolated of the project, then
## uncomment this snippet:
 
# if __name__ == '__main__':
#     df = load_data(DATA_PATH)
#     print(df.head(5))

# And run the script using the -m flag, treating the src folder as a package from backend:  
# python -m src.components.data_ingestion