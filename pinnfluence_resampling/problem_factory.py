"""
shall provide a fairly generic way to instantiate the problems from this repo:

https://github.com/lu-group/pinn-sampling/

which implements this paper https://www.sciencedirect.com/science/article/pii/S0045782522006260
"""

import deepxde as dde
import numpy as np
import os
from pathlib import Path
from scipy.io import loadmat
import torch

from typing import Callable

from .utils.defaults import DEFAULTS
from . import datasets

dataset_dir = datasets.__path__[0]
model_zoo_src = DEFAULTS["model_zoo_src"]


def create_net(layers=[2] + [20] * 3 + [1], output_transform=None):
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
def create_geomtime(x_start=-1, x_end=1, t_start=0, t_end=1):
    geom = dde.geometry.Interval(x_start, x_end)
    timedomain = dde.geometry.TimeDomain(t_start, t_end)
    return dde.geometry.GeometryXTime(geom, timedomain)


def create_data(
    geomtime,
    equation,
    num_domain=1_000,
    num_boundary=0,
    num_initial=0,
    num_test=10_000,
    solution=None,
):
    return dde.data.TimePDE(
        geomtime,
        equation,
        ic_bcs=[],
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_initial=num_initial,
        num_test=num_test,
        solution=solution,
        train_distribution="Hammersley",
    )


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
    num_boundary=0,
    num_initial=0,
    optimizer="adam",
    seed=42,
    n_iterations=10_000,
    n_iterations_lbfgs=0,
    checkpoint_path=None,
    force_reinit=False,
    float64=False,
    model_version=None,
):
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
        num_boundary=num_boundary,
        num_initial=num_initial,
        layers=layers,
        seed=seed,
        float64=float64,
    )

    geomtime = create_geomtime(x_start=problem.get("x_start", -1))

    net = create_net(layers=layers, output_transform=output_transform)

    data = create_data(
        geomtime,
        equation,
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_initial=num_initial,
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
    num_boundary: int,
    num_initial: int,
    float64: bool,
    layers=[2] + [20] * 3 + [1],
    seed=42,
):
    model_name = f"{problem_name}_{optimizer}_{n_iterations}_adam_{n_iterations_lbfgs}_lbfgs_{num_domain}_domain_{num_boundary}_bdry_{num_initial}_init_{len(layers)-2}_x_{layers[1]}_hidden_float64_{float64}_{seed}"
    return model_name


# ALLEN CAHN
# see https://github.com/lu-group/pinn-sampling/blob/main/src/allen_cahn/RAR_G.py


def allen_cahn_equation(x, y):
    u = y
    du_xx = dde.grad.hessian(y, x, i=0, j=0)
    du_t = dde.grad.jacobian(y, x, j=1)
    return du_t - 0.001 * du_xx + 5 * (u**3 - u)


def allen_cahn_output_transform(x, y):
    x_in = x[:, 0:1]
    t_in = x[:, 1:2]
    return t_in * (1 + x_in) * (1 - x_in) * y + torch.square(x_in) * torch.cos(
        np.pi * x_in
    )


def allen_cahn_load_testdata():
    default_float = dde.config.default_float()

    data = loadmat(f"{dataset_dir}/usol_D_0.001_k_5.mat")
    t = data["t"].astype(np.dtype(default_float))
    x = data["x"].astype(np.dtype(default_float))
    u = data["u"].astype(np.dtype(default_float))
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = u.flatten()[:, None]
    return X, y


# BURGERS
# see https://github.com/lu-group/pinn-sampling/blob/main/src/burgers/RAR_G.py


def burgers_equation(x, y):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t + y * dy_x - 0.01 / dde.backend.torch.pi * dy_xx


def burgers_output_transform(x, y):
    return -torch.sin(np.pi * x[:, 0:1]) + (1 - x[:, 0:1] ** 2) * (x[:, 1:]) * y


def burgers_load_testdata():
    default_float = dde.config.default_float()
    data = np.load(f"{dataset_dir}/Burgers.npz")
    t, x, exact = (
        data["t"].astype(np.dtype(default_float)),
        data["x"].astype(np.dtype(default_float)),
        data["usol"].astype(np.dtype(default_float)).T,
    )
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y


# DIFFUSION
# see: https://github.com/lu-group/pinn-sampling/blob/main/src/diffusion/RAR_G.py


def diffusion_equation(x, y):
    dy_t = dde.grad.jacobian(y, x, j=1)
    dy_xx = dde.grad.hessian(y, x, j=0)
    return (
        dy_t
        - dy_xx
        + torch.exp(-x[:, 1:])
        * (torch.sin(np.pi * x[:, 0:1]) - np.pi**2 * torch.sin(np.pi * x[:, 0:1]))
    )


def diffusion_output_transform(x, y):
    return torch.sin(np.pi * x[:, 0:1]) + (1 - x[:, 0:1] ** 2) * (x[:, 1:]) * y


def diffusion_solution(x):
    return np.sin(np.pi * x[:, 0:1]) * np.exp(-x[:, 1:])


# WAVE
# see https://github.com/lu-group/pinn-sampling/blob/main/src/wave/RAR_G.py


def wave_equation(x, y):
    dy_tt = dde.grad.hessian(y, x, i=1, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_tt - 4.0 * dy_xx


def wave_solution(x):
    return np.sin(np.pi * x[:, 0:1]) * np.cos(2 * np.pi * x[:, 1:2]) + 0.5 * np.sin(
        4 * np.pi * x[:, 0:1]
    ) * np.cos(8 * np.pi * x[:, 1:2])


def wave_output_transform(x, y):
    x_in = x[:, 0:1]
    t_in = x[:, 1:2]
    return (
        20 * y * x_in * (1 - x_in) * t_in**2
        + torch.sin(np.pi * x_in)
        + 0.5 * torch.sin(4 * np.pi * x_in)
    )


problems = {
    "allen_cahn": {
        "equation": allen_cahn_equation,
        "output_transform": allen_cahn_output_transform,
        "load_testdata": allen_cahn_load_testdata,
        "solution": None,
    },
    "burgers": {
        "equation": burgers_equation,
        "output_transform": burgers_output_transform,
        "load_testdata": burgers_load_testdata,
        "solution": None,
    },
    "diffusion": {
        "equation": diffusion_equation,
        "output_transform": diffusion_output_transform,
        "load_testdata": None,
        "solution": diffusion_solution,
    },
    "wave": {
        "equation": wave_equation,
        "output_transform": wave_output_transform,
        "load_testdata": None,
        "solution": wave_solution,
        "x_start": 0,  # NOTE the different x_start
    },
}
