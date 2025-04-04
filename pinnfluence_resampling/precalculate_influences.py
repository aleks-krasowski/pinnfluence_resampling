"""
This script precalculates the influence scores for a given model and saves them to a .npz file.

This is useful for evaluation of scoring and sampling strategies without the need to recalculate the influence scores
for each experiment.

Usage:
    python -m pinnfluence_resampling.precalculate_influences [options]

Options:
    ---- Model parameters ----
    These are used to load the pre-trained model checkpoint.
    And are equivalent to the ones found in pretrain.py

    --seed=<int>                Random seed [default: 42]
    --lr=<float>                Learning rate [default: 0.001]
    --layers=<list>             Layer sizes [default: [2] + [32] * 3 + [1]]
    --n_iterations=<int>        Number of iterations [default: 10000]
    --n_iterations_lbfgs=<int>  Number of LBFGS iterations [default: 0]
    --num_domain=<int>          Number of domain points [default: 1000]
    --num_boundary=<int>        Number of boundary points [default: 0]
    --num_initial=<int>         Number of initial points [default: 0]
    --problem=<str>             Problem name [default: burgers]
                                    choices=["allen_cahn", "burgers", "diffusion", "wave"]
    --float64                   Use float64 [default: False]

    
    ---- Influence parameters ----
    --scoring_strategy=<str>    Scoring strategy [default: PINNfluence]
                                    choices=["PINNfluence", "grad_dot"]  
    --save_path=<str>           Path to save model [default: ./model_zoo]
    
"""

import deepxde as dde
import numpy as np

from . import problem_factory
from .utils.parse_args import parse_args
from .utils.sampling import (
    sample_random_points,
    instantiate_IF,
    calculate_influence_scores,
    instantiate_grad_dot,
)


def main(
    n_candidate_points: int = 100_000,
    seed: int = 42,
    lr: float = 0.001,
    layers: list = [2] + [32] * 3 + [1],
    n_iterations: int = 10_000,
    n_iterations_lbfgs: int = 0,
    num_domain: int = 1_000,
    num_boundary: int = 0,
    num_initial: int = 0,
    save_path: str = "./model_zoo",
    problem_name: str = "burgers",
    optimizer: str = "adam",
    use_float64: bool = False,
    scoring_strategy: str = "PINNfluence",
):
    assert scoring_strategy in ["PINNfluence", "grad_dot"], "Can only precompute PINNfluence or grad_dot"
    if use_float64:
        dde.config.set_default_float("float64")
    dde.config.set_random_seed(seed)

    # Construct the problem and load the pretrained checkpoint
    model, data, model_name, checkpoint_loaded = problem_factory.construct_problem(
        problem_name=problem_name,
        lr=lr,
        layers=layers,
        n_iterations=n_iterations,
        n_iterations_lbfgs=n_iterations_lbfgs,
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_initial=num_initial,
        optimizer=optimizer,
        seed=seed,
        float64=use_float64,
    )

    assert (
        checkpoint_loaded
    ), "Could not load checkpoint. Influences shall be only calculated for already trained models"

    # reproduce sampling of Scorer class
    dde.config.set_random_seed(42)

    candidate_points = sample_random_points(
        geometry=model.data.geom, num_points=n_candidate_points
    )
    print(candidate_points.shape)

    if scoring_strategy == "grad_dot":
        graddot = instantiate_grad_dot(model)

        infl_scores = calculate_influence_scores(
            tda_instance=graddot, candidate_points=candidate_points
        )
        infl_scores_abs = np.abs(infl_scores).sum(axis=0)
        infl_scores_pos = infl_scores.sum(axis=0)
        infl_scores_neg = -infl_scores.sum(axis=0)

        np.savez_compressed(
            f"{save_path}/{model_name}_graddot_scores.npz",
            candidate_points=candidate_points,
            scores_abs=infl_scores_abs,
            scores_pos=infl_scores_pos,
            scores_neg=infl_scores_neg,
        )

    else:
        IF_instance = instantiate_IF(model)

        infl_scores = calculate_influence_scores(
            tda_instance=IF_instance, candidate_points=candidate_points
        )
        infl_scores_abs = np.abs(infl_scores).sum(axis=0)
        infl_scores_pos = infl_scores.sum(axis=0)
        infl_scores_neg = -infl_scores.sum(axis=0)

        np.savez_compressed(
            f"{save_path}/{model_name}_influence_scores.npz",
            candidate_points=candidate_points,
            scores_abs=infl_scores_abs,
            scores_pos=infl_scores_pos,
            scores_neg=infl_scores_neg,
        )


if __name__ == "__main__":
    args = parse_args()
    main(
        n_candidate_points=args.n_candidate_points,
        seed=args.seed,
        lr=args.lr,
        layers=args.layers,
        n_iterations=args.n_iterations,
        n_iterations_lbfgs=args.n_iterations_lbfgs,
        num_domain=args.num_domain,
        num_boundary=args.num_boundary,
        num_initial=args.num_initial,
        save_path=args.save_path,
        problem_name=args.problem,
        optimizer=args.optimizer,
        use_float64=args.float64,
        scoring_strategy=args.scoring_strategy,
    )
