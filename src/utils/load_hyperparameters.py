import json
from src.settings import HYPERPARAMETER_PATH, HYPERPARAMETER_GRID_PATH

def load_hyperparameters():
    with open(HYPERPARAMETER_PATH) as f:
        hyperparameters = json.load(f)
    return hyperparameters

def load_hyperparameter_grid():
    with open(HYPERPARAMETER_GRID_PATH) as f:
        hyperparameters = json.load(f)

    return hyperparameters