"""
Run a finetuning experiment on a pretrained model using Physics-Informed Neural Networks (PINNs).
This module provides functionality to finetune pre-trained PINN models using various
sampling and scoring strategies. It supports different training approaches including
replacing the training set, adding to it, or incrementally updating it.
The workflow consists of:

1. Loading a pre-trained model checkpoint
2. Generating new candidate points points based on specified sampling strategy
3. Scoring points using methods like Influence Functions (PINNfluence) 
        or alternatives like residual-based adaptive resampling (RAR)
4. Sampling using a distribution or Top k 
5. Finetuning the model with the selected points

Note: This script requires a pre-trained model checkpoint to perform finetuning.

Usage:
    python -m pinnfluence_resampling.run_experiment [options]

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

    ---- Scoring parameters ----
    These are used to score candidate points.

    --scoring_strategy=<str>    Scoring strategy [default: RAR]
                                    choices=[
                                        "PINNfluence", 
                                        "RAR", 
                                        "grad_dot", 
                                        "steepest_loss_gradient", 
                                        "steepest_prediction_gradient"
                                    ]
    --scoring_sign=<str>        Sign of the summation [default: abs]
                                    choices=["abs", "pos", "neg"]
                                    Note: only affects PINNfluence and grad_dot
    
    ---- Sampling parameters ----
    These are used to generate candidate points for scoring.

    --sampling_strategy=<str>   Sampling strategy [default: distribution]
                                    choices=["distribution", "top_k"]
    --n_candidate_points=<int>  Number of candidate points to sample [default: 10_000]
    --n_samples=<int>           Number of samples to accept from candidate points [default: 1_000]
    --distribution_k=<int>      Distribution parameter k [default: 1]
                                    high k -> more focus on high scoring points
    --distribution_c=<int>      Distribution parameter c [default: 1]
                                    high c -> more uniform
    

    ---- Training parameters ----
    These are used to control the training process.

    --save_path=<str>           Path to save model [default: ./model_zoo]
    --training_strategy=<str>   Training strategy [default: incremental]
                                    choices=[
                                        "replace",              # replace whole training set
                                        "add",                  # add to training set
                                        "incremental",          # incrementally add 
                                        "incremental_replace"   # repeatedly replace whole training set
                                    ]
    --n_iterations_finetune=<int> Number of finetuning iterations [default: 1_000]
                                    Note: for incremental strategies this will be repeated 100 times
    --n_iterations_lbfgs_finetune=<int> Number of LBFGS iterations during finetuning [default: 0]
    --model_version=<str>       Model version [default: full]
                                    choices=["full", "train", "test"]
                                    Chose pretrained model to load, either on full pretraining iterations
                                    or best performance on train or test loss
    --recover_run               Recover run from previous finetuning [default: False]
    
Example:
    python -m pinnfluence_resampling.run_experiment --seed 1 --layers 2 32 32 32 1 --problem=burgers --scoring_strategy=PINNfluence
    python -m pinnfluence_resampling.run_experiment --seed 1 --layers 2 32 32 32 1 --problem=burgers --scoring_strategy=RAR
    python -m pinnfluence_resampling.run_experiment --seed 1 --layers 2 32 32 32 1 --problem=burgers --scoring_strategy=PINNfluence --training_strategy=incremental_replace


"""

import deepxde as dde

from .utils.defaults import DEFAULTS
from . import problem_factory

from .utils.callbacks import BestModelCheckpoint, EvalMetricCallback
from .utils.parse_args import parse_args
from .utils.sampling import Sampler, Scorer
from .utils.finetuner import Finetuner


def main(
    seed: int = 42,
    lr: float = 0.001,
    layers: list = [2] + [32] * 3 + [1],
    n_iterations: int = 10_000,
    n_iterations_lbfgs: int = 0,
    num_domain: int = 1_000,
    num_boundary: int = 0,
    num_initial: int = 0,
    save_path: str = "/opt/model_zoo",
    problem_name: str = "burgers",
    optimizer: str = "adam",
    use_float64: bool = False,
    sampling_strategy: str = "distribution",
    n_candidate_points: int = 10_000,
    n_samples: int = 1_000,
    distribution_k: int = 1,
    distribution_c: int = 1,
    scoring_strategy: str = "RAR",
    scoring_sign: str = "abs",
    training_strategy: str = "incremental",
    n_iterations_finetune: int = 1_000,
    n_iterations_lbfgs_finetune: int = 0,
    model_version: str = "full",
    recover_run: bool = False,
):

    dde.config.set_random_seed(seed)

    assert training_strategy in [
        "replace",
        "add",
        "incremental",
        "incremental_replace",
    ], "Invalid training strategy"
    
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
        model_version=model_version,
    )

    # Verify that we have a loaded checkpoint to finetune
    assert (
        checkpoint_loaded
    ), "Could not load checkpoint. Finetuning shall be only performed on already trained models"
    print(scoring_strategy)

    # Check for pre-calculated influence scores to save computation
    potential_save_path = None
    if scoring_strategy == "PINNfluence":
        potential_save_path = (
            f"{DEFAULTS['model_zoo_src']}/{model_name}_influence_scores.npz"
        )
    elif scoring_strategy == "grad_dot":
        potential_save_path = (
            f"{DEFAULTS['model_zoo_src']}/{model_name}_graddot_scores.npz"
        )

    print(potential_save_path)

    # Initialize sampling strategy (how to select points from candidates based on scores)
    sampler = Sampler(
        strategy=sampling_strategy,
        num_samples=n_samples,
        k=distribution_k,
        c=distribution_c,
    )
    
    # Initialize scoring strategy (how to assign scores to candidate points)
    scorer = Scorer(
        strategy=scoring_strategy,
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
        strategy=training_strategy,
        n_iterations_finetune=n_iterations_finetune,
        n_iterations_lbfgs_finetune=n_iterations_lbfgs_finetune,
        model_name=model_name,
        save_path=save_path,
        recover_run=recover_run,
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
        num_boundary=args.num_boundary,
        num_initial=args.num_initial,
        save_path=args.save_path,
        problem_name=args.problem,
        optimizer=args.optimizer,
        use_float64=args.float64,
        sampling_strategy=args.sampling_strategy,
        n_candidate_points=args.n_candidate_points,
        n_samples=args.n_samples,
        distribution_k=args.distribution_k,
        distribution_c=args.distribution_c,
        scoring_strategy=args.scoring_strategy,
        scoring_sign=args.scoring_sign,
        training_strategy=args.training_strategy,
        n_iterations_finetune=args.n_iterations_finetune,
        n_iterations_lbfgs_finetune=args.n_iterations_lbfgs_finetune,
        model_version=args.model_version,
        recover_run=args.recover_run,
    )
