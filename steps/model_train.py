import logging
import pandas as pd
from zenml import step
from src.model_dev import MyLinearRegression
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

@step 
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: ModelNameConfig
    ) -> RegressorMixin:
    
    try:
        model = None
        if config.model_name == 'LinearRegression':
            model = MyLinearRegression()
            trained_model = model.train(X_train,y_train)
            return trained_model
        else:
            raise ValueError("Model name not supported")
    except Exception as e:
        logging.error("Model training failed")
        raise e