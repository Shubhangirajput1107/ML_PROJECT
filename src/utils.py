import os
import sys
import pickle
import logging
from catboost import CatBoostRegressor

logging.basicConfig(
    filename=os.path.join("logs", "utils.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        logging.info(f"Object saved at: {file_path}")
    except Exception as e:
        logging.error(f"Error saving object: {e}")
        raise e

def load_object(file_path):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logging.error(f"Error loading object: {e}")
        raise e

def create_catboost_model():
    """
    Fix CatBoost sklearn compatibility issue
    """
    model = CatBoostRegressor(verbose=0)
    
    if not hasattr(model, "__sklearn_tags__"):
        def _sklearn_tags():
            return {
                "non_deterministic": True,
                "requires_fit": True,
                "_skip_test": False
            }
        model.__class__.__sklearn_tags__ = _sklearn_tags

    return model
