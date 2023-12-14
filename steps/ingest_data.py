import logging
import pandas as pd
from zenml import step

class InjestData:
    def __init__(self, data_path:str):
        self.data_path = data_path
    
    def get_data(self):
        logging.info(f'Ingesting data from {self.data_path}')
        return pd.read_csv(self.data_path)
    
@step 
def ingest_df(data_path: str) -> pd.DataFrame:
    """Ingests data from a local csv file.
    Args:
        data_path: path to the local csv file.
    Returns:
        Pandas DataFrame of the ingested data.
    """
    try:
        ingest_data = InjestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f'Unable to ingest data from {data_path}')
        raise e