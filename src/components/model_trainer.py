import sys
from dataclasses import dataclass
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, create_catboost_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = 'artifacts/model.pkl'

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Only using CatBoost to keep it simple and avoid GridSearch issues
            model = create_catboost_model()
            model.fit(X_train, y_train)

            save_object(self.model_trainer_config.trained_model_file_path, model)
            logging.info("model.pkl created successfully")

            y_pred = model.predict(X_test)
            r2_square = r2_score(y_test, y_pred)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
