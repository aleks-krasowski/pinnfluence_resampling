import deepxde as dde
import numpy as np
import torch
from scipy.io import loadmat

from .defaults import DEFAULTS

dataset_dir = DEFAULTS["dataset_dir"]

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


# DRIFT DIFFUSION
# see https://gitlab.fe.hhi.de/pinns/modelzoo/-/blob/main/deepxde/drift_diffusion_equation/train.py

# initial concentration
DRIFT_DIFFUSION_U_00 = 1.0
# frequency
DRIFT_DIFFUSION_S = 2
# phase shift
DRIFT_DIFFUSION_R = np.pi / 4
# diffusivity
DRIFT_DIFFUSION_ALPHA = 1
# velocity in x direction
DRIFT_DIFFUSION_BETA = 20
DRIFT_DIFFUSION_GAMMA = 1.0
DRIFT_DIFFUSION_X_START = 0.0
DRIFT_DIFFUSION_X_END = 2 * np.pi
DRIFT_DIFFUSION_T_START = 0.0
DRIFT_DIFFUSION_T_END = 1.0


def drift_diffusion_equation(x, y):
    du_x = dde.grad.jacobian(y, x, i=0, j=0)
    du_t = dde.grad.jacobian(y, x, i=0, j=1)
    du_xx = dde.grad.hessian(y, x, i=0, j=0)
    return du_t - DRIFT_DIFFUSION_ALPHA * du_xx + DRIFT_DIFFUSION_BETA * du_x


def drift_diffusion_solution(x):
    x_in = x[:, 0:1]
    t_in = x[:, 1:2]
    return (
        DRIFT_DIFFUSION_U_00
        * np.sin(
            DRIFT_DIFFUSION_R + DRIFT_DIFFUSION_S * (x_in - DRIFT_DIFFUSION_BETA * t_in)
        )
        * np.exp(-DRIFT_DIFFUSION_ALPHA * DRIFT_DIFFUSION_S**2 * t_in)
    )


def drift_diffusion_output_transform(x, y):
    ### applies both IC output transform and BCs (x_start, x_end)
    t = x[:, 1:2]
    x = x[:, 0:1]

    current_x_start_val = torch.tensor(DRIFT_DIFFUSION_X_START)
    current_x_end_val = torch.tensor(DRIFT_DIFFUSION_X_END)

    # IC term u(x, 0) = u_00 * sin(s * x + r)
    ic_at_x = DRIFT_DIFFUSION_U_00 * torch.sin(
        DRIFT_DIFFUSION_S * x + DRIFT_DIFFUSION_R
    )
    # IC at x_start
    ic_at_x_start = DRIFT_DIFFUSION_U_00 * torch.sin(
        DRIFT_DIFFUSION_S * current_x_start_val + DRIFT_DIFFUSION_R
    )
    # IC at x_end
    ic_at_x_end = DRIFT_DIFFUSION_U_00 * torch.sin(
        DRIFT_DIFFUSION_S * current_x_end_val + DRIFT_DIFFUSION_R
    )

    # BCs taken from solution of the PDE:
    g_start_t = (
        DRIFT_DIFFUSION_U_00
        * torch.sin(
            DRIFT_DIFFUSION_S * (current_x_start_val - DRIFT_DIFFUSION_BETA * t)
            + DRIFT_DIFFUSION_R
        )
        * torch.exp(-DRIFT_DIFFUSION_ALPHA * DRIFT_DIFFUSION_S**2 * t)
    )
    g_end_t = (
        DRIFT_DIFFUSION_U_00
        * torch.sin(
            DRIFT_DIFFUSION_S * (current_x_end_val - DRIFT_DIFFUSION_BETA * t)
            + DRIFT_DIFFUSION_R
        )
        * torch.exp(-DRIFT_DIFFUSION_ALPHA * DRIFT_DIFFUSION_S**2 * t)
    )

    # scalar terms for the spatial BC adjustment:
    domain_width = current_x_end_val - current_x_start_val
    term_g_minus_ic_start = g_start_t - ic_at_x_start
    term_g_minus_ic_end = g_end_t - ic_at_x_end

    weight_factor_start = (current_x_end_val - x) / domain_width
    weight_factor_end = (x - current_x_start_val) / domain_width

    spatial_bc_contribution = (
        weight_factor_start * term_g_minus_ic_start
        + weight_factor_end * term_g_minus_ic_end
    )

    # This factor ensures the neural network's contribution (u_nn) is zero:
    # - at t = 0 (due to multiplication by t)
    # - at x = x_start (due to (x - x_start_val))
    # - at x = x_end (due to (x_end_val - x))
    # So, u_nn does not interfere with the enforced IC and BCs.
    # (x - x_start_val) * (x_end_val - x) is a parabola-like term, zero at boundaries.
    nn_influence_factor = t * (x - current_x_start_val) * (current_x_end_val - x)

    # The final transformed output:
    # u_transformed = IC(x) + spatial_bc_terms + vanishing_nn_term * NN_output
    # This structure ensures:
    # 1. u(x,0) = IC(x) (because spatial_bc_contribution and nn_influence_factor are zero at t=0)
    # 2. u(x_start,t) = G_start(t)
    # 3. u(x_end,t) = G_end(t)
    transformed_u = ic_at_x + spatial_bc_contribution + nn_influence_factor * y

    return transformed_u


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
    "drift_diffusion": {
        "equation": drift_diffusion_equation,
        "output_transform": drift_diffusion_output_transform,
        "load_testdata": None,
        "solution": drift_diffusion_solution,
        "x_start": DRIFT_DIFFUSION_X_START,
        "x_end": DRIFT_DIFFUSION_X_END,
        "t_start": DRIFT_DIFFUSION_T_START,
        "t_end": DRIFT_DIFFUSION_T_END,
    },
    "wave": {
        "equation": wave_equation,
        "output_transform": wave_output_transform,
        "load_testdata": None,
        "solution": wave_solution,
        "x_start": 0,  # NOTE the different x_start
    },
}
