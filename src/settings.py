import os

USE_MONTE_CARLO = True
MONTE_CARLO_N_SAMPLES_PER_KERNEL = 5

PYTHONPATH = ".venv/Scripts/python.exe"
if not os.path.exists(PYTHONPATH):
    PYTHONPATH = "../venv/bin/python"


HYPERPARAMETER_PATH = "data/parameters/simulation/tuned_hyperparameters.json"
HYPERPARAMETER_GRID_PATH = "data/parameters/simulation/hyperparameter_grid.json"
