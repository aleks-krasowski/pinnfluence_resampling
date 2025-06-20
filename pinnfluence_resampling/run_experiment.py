"""
Run a finetuning experiment on a pretrained PINN model.

This script supports flexible sampling, scoring, and training
strategies to iteratively finetune a pretrained model.

Steps:
  1. (optional) Load a pretrained model checkpoint.
  2. Generate candidate points via the sampling strategy.
  3. Score candidates using the scoring method.
  4. Select points (distribution or top-k) for finetuning.
  5. Finetune the model with selected points.

Usage:
    python -m pinnfluence_resampling.run_experiment [options]

    Use --help to see all available options.
"""

import deepxde as dde

from . import problem_factory
from .utils.defaults import DEFAULTS
from .utils.finetuner import Finetuner
from .utils.parse_args import parse_run_experiment_args as parse_args
from .utils.sampling import Sampler, Scorer
from .utils.utils import set_default_device


def main(
    seed: int = 42,
    lr: float = 0.001,
    layers: list = [2] + [32] * 3 + [1],
    n_iterations: int = 10_000,
    n_iterations_lbfgs: int = 0,
    num_domain: int = 1_000,
    save_path: str = "/opt/model_zoo",
    problem_name: str = "burgers",
    optimizer: str = "adam",
    use_float64: bool = False,
    sampling_strategy: str = "distribution",
    n_candidate_points: int = 10_000,
    n_samples: int = 1_000,
    distribution_k: int = 1,
    distribution_c: int = 1,
    scoring_method: str = "RAR",
    scoring_sign: str = "abs",
    pertubation_strategy: str = "add",
    n_iterations_finetune: int = 1_000,
    n_iterations_lbfgs_finetune: int = 0,
    model_version: str = "full",
    recover_run: bool = False,
    device: str = "cpu",
    n_cycles_finetune: int = 100,
):
    set_default_device(device)

    dde.config.set_random_seed(seed)

    assert pertubation_strategy in [
        "add",
        "replace",
    ], "Invalid training strategy"

    # Construct problem and load pretrained checkpoint
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
        model_version=model_version,
    )

    # Verify that we have a loaded checkpoint to finetune
    if n_iterations > 0:
        assert (
            checkpoint_loaded
        ), f"You supplied n_iterations={n_iterations} - but could not load associated checkpoint."
    else:
        print("No checkpoint loaded. Finetuning on untrained model")

    # Check for pre-calculated influence scores to save computation
    potential_save_path = None
    if scoring_method == "PINNfluence":
        potential_save_path = (
            f"{DEFAULTS['model_zoo_src']}/{model_name}_influence_scores.npz"
        )
    elif scoring_method == "grad_dot":
        potential_save_path = (
            f"{DEFAULTS['model_zoo_src']}/{model_name}_graddot_scores.npz"
        )

    # Initialize sampling strategy for selecting points
    sampler = Sampler(
        strategy=sampling_strategy,
        num_samples=n_samples,
        k=distribution_k,
        c=distribution_c,
        data=data,
    )

    # Initialize scoring strategy (how to assign scores to candidate points)
    scorer = Scorer(
        strategy=scoring_method,
        model=model,
        n_candidate_points=n_candidate_points,
        summation_sign=scoring_sign,
        potential_precalculated=potential_save_path,
    )

    # Initialize finetuner to manage the training process
    finetuner = Finetuner(
        model=model,
        scorer=scorer,
        sampler=sampler,
        strategy=pertubation_strategy,
        n_iterations_finetune=n_iterations_finetune,
        n_iterations_lbfgs_finetune=n_iterations_lbfgs_finetune,
        model_name=model_name,
        save_path=save_path,
        recover_run=recover_run,
        n_cycles_finetune=n_cycles_finetune,
    )

    # Run the finetuning process
    finetuner()


if __name__ == "__main__":
    args = parse_args()
    main(
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
        sampling_strategy=args.sampling_strategy,
        n_candidate_points=args.n_candidate_points,
        n_samples=args.n_samples,
        distribution_k=args.distribution_k,
        distribution_c=args.distribution_c,
        scoring_method=args.scoring_method,
        scoring_sign=args.scoring_sign,
        pertubation_strategy=args.pertubation_strategy,
        n_iterations_finetune=args.n_iterations_finetune,
        n_iterations_lbfgs_finetune=args.n_iterations_lbfgs_finetune,
        model_version=args.model_version,
        recover_run=args.recover_run,
        device=args.device,
        n_cycles_finetune=args.n_cycles_finetune,
    )
