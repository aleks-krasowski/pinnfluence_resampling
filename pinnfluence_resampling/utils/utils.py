import json
import matplotlib.pyplot as plt
import os
import pandas as pd
from pathlib import Path
import seaborn as sns
import torch

colors = {
    "random": "blue",
    "Random": "blue",
    "RAR": "orange",
    "PINNfluence": "green",
    "PINNfluence (ours)": "green",
    "grad_dot": "red",
    "Grad-Dot": "red",
    "steepest_prediction_gradient": "brown",
    "$||\\nabla_{x} u||_2$": "brown",
    "steepest_loss_gradient": "purple",
    "$||\\nabla_{\\theta} \\mathcal{L}||_2$": "purple",
}


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
            torch.set_default_device("mps")
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
            "scoring_strategy",
            "scoring_sign",
            "training_strategy",
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


def plot_lineplots(
    df,
    problem=None,
    training_strategies=["add"],
    metrics=["test_loss", "l2re"],
    criterions=["train", "valid"],
    sampling_strategies=["distribution"],
    scoring_strategies=[
        "random",
        "RAR",
        "PINNfluence",
        "grad_dot",
        "steepest_prediction_gradient",
    ],
    colors=colors,
):

    _df = df.copy()

    if problem is not None:
        _df = _df.loc[df["problem"] == problem]

    for training_strategy in training_strategies:
        for criterion in criterions:
            for sampling_strategy in sampling_strategies:
                for metric in metrics:
                    plt.figure(figsize=(12, 6))
                    title = (
                        f"Strategy: {training_strategy}, "
                        f"Sampling: {sampling_strategy}, Criterion: {criterion}, Metric {metric}"
                    )
                    if problem is not None:
                        title = f"Problem: {problem}, {title}"
                    plt.title(title)

                    subset = _df.loc[
                        (_df["training_strategy"] == training_strategy)
                        & (_df["criterion"] == criterion)
                        & (_df["sampling_strategy"] == sampling_strategy)
                    ]

                    for scoring_strategy in scoring_strategies:
                        cur_df = subset.loc[
                            subset["scoring_strategy"] == scoring_strategy
                        ]

                        sns.lineplot(
                            data=cur_df,
                            x="n_samples",
                            y=metric,
                            label=scoring_strategy,
                            color=colors.get(scoring_strategy),
                            errorbar=("ci", 95),
                            estimator="mean",
                            err_style="bars",
                        )

                    plt.yscale("log")
                    plt.legend()
                    plt.xlabel("Number of Samples")
                    plt.xscale("log")
                    plt.show()
