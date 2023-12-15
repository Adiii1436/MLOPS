import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

class Model(ABC):
    
    @abstractmethod
    def train(self, X_train, y_train):
        pass    

class MyLinearRegression(Model):
    
    def train(self, X_train, y_train, **kwargs):
        try:
            reg = SklearnLinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model trained")
            return reg 
        except Exception as e:
            logging.error("Model training failed")
            raise e