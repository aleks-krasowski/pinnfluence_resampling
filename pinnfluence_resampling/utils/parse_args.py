import argparse

from .defaults import DEFAULTS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=DEFAULTS["seed"], help="Random seed"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULTS["lr"],
        help="Learning rate",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=DEFAULTS["layers"],
        help="Layer sizes",
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=DEFAULTS["n_iterations"],
        help="Number of iterations",
    )
    parser.add_argument(
        "--n_iterations_lbfgs",
        type=int,
        default=DEFAULTS["n_iterations_lbfgs"],
        help="Number of LBFGS iterations",
    )
    parser.add_argument(
        "--num_domain",
        type=int,
        default=DEFAULTS["num_domain"],
        help="Number of domain points",
    )
    parser.add_argument(
        "--num_boundary",
        type=int,
        default=DEFAULTS["num_boundary"],
        help="Number of boundary points",
    )
    parser.add_argument(
        "--num_initial",
        type=int,
        default=DEFAULTS["num_initial"],
        help="Number of initial points",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=DEFAULTS["model_zoo"],
        help="Path to save model",
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="burgers",
        help="Problem/PDE to run experiment for",
        choices=["allen_cahn", "burgers", "diffusion", "wave"],
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        help="Which optimizer to use for training",
        default=DEFAULTS["optimizer"],
    )
    parser.add_argument(
        "--float64",
        action="store_true",
        help="Use float64",
    )
    parser.add_argument(
        "--n_candidate_points",
        type=int,
        help="Number of candidate points to from for resampling.",
        default=10_000,
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        help="Number of samples to take from candidate points.",
        default=1_000,
    )
    parser.add_argument(
        "--distribution_k",
        type=int,
        help="K parameter for distribution.",
        default=1,
    )
    parser.add_argument(
        "--distribution_c",
        type=int,
        help="C parameter for distribution.",
        default=1,
    )
    parser.add_argument(
        "--scoring_strategy",
        type=str,
        help="Which scoring strategy to use.",
        default="PINNfluence",
        choices=[
            "PINNfluence",
            "RAR",
            "random",
            "grad_dot",
            "steepest_prediction_gradient",
            "steepest_loss_gradient",
        ],
    )
    parser.add_argument(
        "--scoring_sign",
        type=str,
        help="Which sign to use for scoring.",
        default="abs",
        choices=["abs", "pos", "neg"],
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        help="Which training strategy to use.",
        default="incremental",
        choices=["replace", "add", "incremental", "incremental_replace"],
    )
    parser.add_argument(
        "--n_iterations_finetune",
        type=int,
        help="Number of iterations to finetune.",
        default=1_000,
    )
    parser.add_argument(
        "--n_iterations_lbfgs_finetune",
        type=int,
        help="Number of LBFGS iterations to finetune.",
        default=0,
    )
    parser.add_argument(
        "--sampling_strategy",
        type=str,
        help="Which sampling strategy to use.",
        default="distribution",
        choices=["top_k", "distribution"],
    )
    parser.add_argument(
        "--model_version",
        type=str,
        help="Model version to load (chosen by best validation or train loss - or simply at final epoch)",
        default="full",
        choices=["full", "train", "valid"],
    )
    parser.add_argument(
        "--recover_run",
        action="store_true",
        help="Recover run (finetuned) and repeat finetuning.",
    )
    return parser.parse_args()
