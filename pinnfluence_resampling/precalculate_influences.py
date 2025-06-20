"""
This script precalculates the influence scores for a given model and saves them to a .npz file.

This is useful for evaluation of scoring and sampling strategies without the need to recalculate the influence scores
for each experiment.

Usage:
    python -m pinnfluence_resampling.precalculate_influences [options]

    Use --help to see all available options.
"""

import deepxde as dde
import numpy as np

from . import problem_factory
from .utils.parse_args import parse_precalculate_args as parse_args
from .utils.sampling import (calculate_influence_scores, instantiate_grad_dot,
                             instantiate_IF, sample_random_points)
from .utils.utils import set_default_device


def main(
    n_candidate_points: int = 100_000,
    seed: int = 42,
    lr: float = 0.001,
    layers: list = [2] + [32] * 3 + [1],
    n_iterations: int = 10_000,
    n_iterations_lbfgs: int = 0,
    num_domain: int = 1_000,
    save_path: str = "./model_zoo",
    problem_name: str = "burgers",
    optimizer: str = "adam",
    use_float64: bool = False,
    scoring_method: str = "PINNfluence",
    use_holdout_test: bool = False,
    precalc_infl_sample_uniformly: bool = False,
    device: str = "cpu",
):
    assert scoring_method in [
        "PINNfluence",
        "grad_dot",
    ], "Can only precompute PINNfluence or grad_dot"
    if use_float64:
        dde.config.set_default_float("float64")
    dde.config.set_random_seed(seed)

    set_default_device(device)

    # Construct the problem and load the pretrained checkpoint
    model, data, model_name, checkpoint_loaded = problem_factory.construct_problem(
        problem_name=problem_name,
        lr=lr,
        layers=layers,
        n_iterations=n_iterations,
        n_iterations_lbfgs=n_iterations_lbfgs,
        num_domain=num_domain,
        optimizer=optimizer,
        seed=seed,
        float64=use_float64,
    )

    assert (
        checkpoint_loaded
    ), "Could not load checkpoint. Influences shall be only calculated for already trained models"

    # reproduce sampling of Scorer class
    dde.config.set_random_seed(42)

    if not use_holdout_test:
        if precalc_infl_sample_uniformly:
            candidate_points = data.geom.uniform_points(n_candidate_points)
        else:
            candidate_points = sample_random_points(
                geometry=model.data.geom, num_points=n_candidate_points
            )
    else:
        # Use the holdout test set for candidate points
        # This is useful for evaluating the scoring strategy on the test set
        # without the need to recalculate the influence scores
        candidate_points = model.data.holdout_test_x

    print(candidate_points.shape)

    if scoring_method == "grad_dot":
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
        save_path=args.save_path,
        problem_name=args.problem,
        optimizer=args.optimizer,
        use_float64=args.float64,
        scoring_method=args.scoring_method,
        use_holdout_test=args.precalc_infl_use_holdout_test,
        precalc_infl_sample_uniformly=args.precalc_infl_sample_uniformly,
        device=args.device,
    )
