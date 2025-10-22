import pandas as pd
from src.config import DATA_PATH
from src.utils.logger import logger
from pydantic import validate_call, ValidationError

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
    
    # Irrelevant because will raise before try bock
    # except ValidationError as e:
    #     logger.debug('Error validating the file path => %s', e)
    #     raise ValidationError(f'Error validating the file path => {e}') from e
        
    except FileNotFoundError as e:
        logger.debug('Error loading the CSV file %s', e)
        raise FileNotFoundError(f'Error loading the CSV file => {e}') from e
    
    # Here I want to catch any another exception so:
    except Exception as e:
        logger.debug('Error loading the CSV file =>%s', e)
        raise RuntimeError(f'Error loading the CSV file => {e}') from e

if __name__ == '__main__':
    df = load_data(DATA_PATH)
    print(df.head(5))

# Run the script using the -m flag, treating the src folder as a package:  
# python -m src.components.data_ingestion