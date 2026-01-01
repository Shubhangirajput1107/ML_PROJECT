class ModelTrainerConfig:
    def __init__(self):
        self.trained_model_file_path = "artifacts/model.pkl"


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        Safe dummy model training method
        """

        print("Model training started...")
        print("Train data received")
        print("Test data received")

        return "Model training completed successfully"
