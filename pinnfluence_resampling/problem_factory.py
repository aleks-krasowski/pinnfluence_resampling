"""
shall provide a fairly generic way to instantiate the problems from this repo:

https://github.com/lu-group/pinn-sampling/

which implements this paper https://www.sciencedirect.com/science/article/pii/S0045782522006260

and more :-)
"""

from pathlib import Path
from typing import Callable

import deepxde as dde
import torch

from . import datasets
from .utils.defaults import DEFAULTS
from .utils.problems import problems

dataset_dir = datasets.__path__[0]


def create_net(
    layers=[2] + [20] * 3 + [1],
    output_transform=None,
):
    net = dde.maps.FNN(layers, "tanh", "Glorot normal")

    if output_transform is not None:
        net.apply_output_transform(output_transform)

    return net


def load_checkpoint(net, chkpt_path):
    chkpt = torch.load(chkpt_path, weights_only=False)
    print(f"Best epoch: {chkpt['epoch']}")
    net.load_state_dict(chkpt["model_state_dict"])
    return net


# all the geometries are 1D intervals over time
def create_geom(x_start=-1, x_end=1, t_start=0, t_end=1):
    geom = dde.geometry.Interval(x_start, x_end)
    timedomain = dde.geometry.TimeDomain(t_start, t_end)
    return dde.geometry.GeometryXTime(geom, timedomain)


def create_data(
    geom,
    equation,
    num_domain=1_000,
    num_test=10_000,
    solution=None,
):
    if isinstance(geom, dde.geometry.GeometryXTime):
        target_data = dde.data.TimePDE
    else:
        target_data = dde.data.PDE

    args = {
        "num_domain": num_domain,
        "num_test": num_test,
        "solution": solution,
        "train_distribution": "Hammersley",
    }

    data = target_data(geom, equation, [], **args)
    data.num_domain = num_domain

    return data


def resample_validation_and_test(
    data: dde.data.Data,
    load_testdata: Callable = None,
    solution: Callable = None,
    num_validation: int = 10_000,
    num_test: int = 10_000,
):
    """
    Note that DeepXDE per default only contains a single "test" set.
    As we choose our best checkpoint based on this test set it becomes rather a validation set.
    For problems where we don't have precomputed ground truth data we thus need to sample another
    test set that is not used for checkpoint selection for final evaluation.

    in this notation:
        - test_x and test_y correspond to validation
        - holdout_test_x and holdout_test_y correspond to test
    """
    assert (
        load_testdata is not None or solution is not None
    ), "Either load_testdata or solution must be provided"

    # resample validation - as per default uniformly sampled
    data.test_x = data.geom.random_points(num_validation, "pseudo")
    if solution is not None:
        data.test_y = solution(data.test_x)
    else:
        assert (
            data.test_y is None
        ), "target values were already set although no solution was provided"

    # sample holdout test
    if load_testdata is not None:
        X, y = load_testdata()
        data.holdout_test_x = X
        data.holdout_test_y = y
    else:
        data.holdout_test_x = data.geom.random_points(num_test, "pseudo")
        data.holdout_test_y = solution(data.holdout_test_x)


def compile_model(
    net,
    data,
    lr=0.001,
    model=None,
    optimizer="adam",
):
    if model is None:
        model = dde.Model(data, net)
    model.compile(optimizer, lr=lr)

    return model


def restore_data(data, chkpt_path):
    chkpt = torch.load(chkpt_path, weights_only=False)

    if "train_x_all" in chkpt.keys():
        data.train_x_all = chkpt["train_x_all"]
        data.train_x = chkpt["train_x"]
        data.train_x_bc = chkpt["train_x_bc"]
        data.test_x = chkpt["test_x"]
        data.test_y = chkpt["test_y"]
        data.holdout_test_x = chkpt["holdout_test_x"]
        data.holdout_test_y = chkpt["holdout_test_y"]
    return data


def construct_problem(
    problem_name: str,
    lr=0.001,
    layers=[2] + [32] * 3 + [1],
    num_domain=1_000,
    optimizer="adam",
    seed=42,
    n_iterations=10_000,
    n_iterations_lbfgs=0,
    checkpoint_path=None,
    force_reinit=False,
    float64=False,
    model_version=None,
    model_zoo_src=DEFAULTS["model_zoo_src"],
) -> tuple[dde.Model, dde.data.Data, str, str]:

    if float64:
        dde.config.set_default_float("float64")

    print(f"model_zoo_src: {model_zoo_src}")
    problem = problems[problem_name]

    equation = problem["equation"]
    output_transform = problem["output_transform"]
    load_testdata = problem["load_testdata"]
    solution = problem["solution"]

    model_name = get_model_name(
        problem_name=problem_name,
        optimizer=optimizer,
        n_iterations=n_iterations,
        n_iterations_lbfgs=n_iterations_lbfgs,
        num_domain=num_domain,
        layers=layers,
        seed=seed,
        float64=float64,
    )

    net = create_net(
        layers=layers,
        output_transform=output_transform,
    )

    geom = create_geom(
        x_start=problem.get("x_start", -1),
        x_end=problem.get("x_end", 1),
        t_start=problem.get("t_start", 0),
        t_end=problem.get("t_end", 1),
    )

    data = create_data(
        geom,
        equation,
        num_domain=num_domain,
        solution=solution,
    )

    resample_validation_and_test(
        data=data,
        load_testdata=load_testdata,
        solution=solution,
        num_validation=10_000,
        num_test=10_000,
    )

    if not force_reinit:
        if checkpoint_path is None:
            print(f"Loading checkpoint under {model_zoo_src} under name: {model_name}")
            if model_version is None or model_version == "full":
                chkpts = list(Path(model_zoo_src).rglob(f"{model_name}_full.pt"))
            elif model_version == "train":
                chkpts = list(Path(model_zoo_src).rglob(f"{model_name}_train.pt"))
            elif model_version == "valid":
                chkpts = list(Path(model_zoo_src).rglob(f"{model_name}.pt"))
            else:
                chkpts = list(Path(model_zoo_src).rglob(f"{model_name}.pt"))
            if len(chkpts) > 0:
                checkpoint_path = chkpts[0]
                print(f"Found checkpoint at {checkpoint_path}")

        if checkpoint_path is not None:
            net = load_checkpoint(net, checkpoint_path)
            data = restore_data(data, checkpoint_path)

    model = compile_model(
        net,
        data,
        lr=lr,
        optimizer=optimizer,
    )

    valid_set = set(map(tuple, data.test_x))
    test_set = set(map(tuple, data.holdout_test_x))

    assert len(valid_set & test_set) == 0, "Test and holdout test data overlap"
    return model, data, model_name, checkpoint_path


def get_model_name(
    problem_name: str,
    optimizer: str,
    n_iterations: int,
    n_iterations_lbfgs: int,
    num_domain: int,
    float64: bool,
    layers=[2] + [20] * 3 + [1],
    seed=42,
):
    model_name = f"{problem_name}_{optimizer}_{n_iterations}_adam_{n_iterations_lbfgs}_lbfgs_{num_domain}_domain_init_{len(layers)-2}_x_{layers[1]}_hidden_float64_{float64}_{seed}"
    return model_name
