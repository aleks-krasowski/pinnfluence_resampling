import json
import os
from pathlib import Path

import pandas as pd
import torch


def set_default_device(device: str = "cpu"):
    if device == "cpu":
        torch.set_default_device("cpu")
        print("Using CPU")
    elif device == "cuda":
        if torch.cuda.is_available():
            torch.set_default_device("cuda")
            print("Using CUDA")
        else:
            print("CUDA not available. Using CPU")
            torch.set_default_device("cpu")
    elif device == "mps":
        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            torch._dynamo.disable()
            torch.set_default_device("mps")
            torch._dynamo.reset()
            print("Using mps")
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            print("Using MPS")
        else:
            print("MPS not available. Using CPU")
            torch.set_default_device("cpu")
    else:
        print("Invalid device. Using CPU")
        torch.set_default_device("cpu")


def generate_experiment_df(experiment_path: str):
    experiment_path = Path(experiment_path)
    df = pd.DataFrame(
        columns=[
            "problem",
            "model_name",
            "strategy",
            "n_iterations_finetune",
            "n_iterations_lbfgs_finetune",
            "lr",
            "n_candidate_points",
            "n_samples",
            "distribution_k",
            "distribution_c",
            "scoring_method",
            "scoring_sign",
            "pertubation_strategy",
            "epoch",
            "criterion",
            "train_loss",
            "valid_loss",
            "test_loss",
            "l2re",
            "mse",
        ]
    )

    for dir in experiment_path.iterdir():
        config = dir / "config.json"

        if not config.exists():
            continue

        with open(config, "r") as f:
            config = json.load(f)

        model_name = config["model_name"]
        config["seed"] = int(model_name[model_name.rfind("_") + 1 :])

        csvs = list(dir.glob("*.csv"))
        if len(csvs) == 0:
            continue

        csv = csvs[0]
        df_ = pd.read_csv(csv)

        for criterion in ["train", "valid"]:

            best_row = df_.iloc[df_[f"{criterion}_loss"].idxmin()]

            config.update(
                {
                    "epoch": best_row["epoch"],
                    "criterion": criterion,
                    "train_loss": best_row["train_loss"],
                    "valid_loss": best_row["valid_loss"],
                    "test_loss": best_row["test_loss"],
                    "l2re": best_row["l2_relative_error"],
                    "mse": best_row["mse"],
                }
            )

            df = pd.concat([df, pd.DataFrame([config])])

    return df
