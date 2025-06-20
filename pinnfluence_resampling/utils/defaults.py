import platform
from pathlib import Path

from .. import datasets

DEFAULTS = {
    "lr": 0.001,
    "layers": [2] + [32] * 3 + [1],
    "n_iterations": 0,
    "n_iterations_lbfgs": 0,
    "num_domain": 1000,
    "k": 2,
    "c": 0,
    "optimizer": "adam",
    "device": "cpu",
    "seed": 42,
    "train_distribution": "Hammersley",
    "model_zoo_src": "./model_zoo",
    "model_zoo": "./model_zoo",
    "dataset_dir": datasets.__path__[0],
}
