import numpy as np


class DataTransformationConfig:
    def __init__(self):
        self.preprocessor_obj_file_path = "artifacts/preprocessor.pkl"


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_data, test_data):
        """
        Dummy transformation method to fix pipeline errors.
        """

        # Convert dataframe to numpy arrays
        train_arr = np.array(train_data)
        test_arr = np.array(test_data)

        return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
