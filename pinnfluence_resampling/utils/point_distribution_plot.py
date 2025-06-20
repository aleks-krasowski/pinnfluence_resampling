import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pinnfluence_resampling.problem_factory import (construct_problem,
                                                    load_checkpoint)
from pinnfluence_resampling.utils.sampling import Scorer

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=2)

CHKPTS_TO_PLOT = [
    1_000,
    2_000,
    10_000,
    200_000,
]

SEEDS = range(1, 11)  # 1 to 10 inclusive

LAYERS = {
    "allen_cahn": [2] + [64] * 3 + [1],
    "burgers": [2] + [32] * 3 + [1],
    "diffusion": [2] + [32] * 3 + [1],
    "wave": [2] + [100] * 5 + [1],
    "drift_diffusion": [2, 64, 64, 64, 1],
}

NUM_POINTS = {
    "allen_cahn": 1000,
    "burgers": 1000,
    "diffusion": 30,
    "wave": 1000,
    "drift_diffusion": 1000,
}

METHOD_DIR_NAMES = {
    "PINNfluence": "pinn",
    "RAR": "rar",
    "Random": "rand",
    "grad_dot": "grad_dot",
    "steepest_prediction_gradient": "outgrad",
    "steepest_loss_gradient": "lossgrad",
}

DISTRIBUTION_PARMS = {
    "add": {
        "c": 0,
        "k": 2,
    },
    "replace": {
        "c": 1,
        "k": 1,
    },
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setting",
        type=str,
        default="add",
        help="Replace or Add points setting",
        choices=["replace", "add"],
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="allen_cahn",
        help="Problem to plot",
        choices=[
            "allen_cahn",
            "burgers",
            "diffusion",
            "drift_diffusion",
            "wave",
        ],
    )
    parser.add_argument(
        "--method",
        type=str,
        help="Which scoring strategy to use.",
        default="PINNfluence",
        choices=[
            "PINNfluence",
            "RAR",
            "RAR_with_BC",
            "Random",
            "grad_dot",
            "steepest_prediction_gradient",
            "steepest_loss_gradient",
        ],
    )
    parser.add_argument(
        "--save_path", type=str, default="/opt/out/figures", help="Path to save figures"
    )
    return parser.parse_args()


def apply_distribution(x, c=1, k=1):
    """Apply the distribution function."""
    x /= x.max()
    x = np.power(x, k) / np.power(x, k).mean() + c
    x /= x.sum()
    return x


def check_epoch(checkpoint, target_epoch):
    chkpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "epoch" in chkpt:
        return target_epoch == chkpt["epoch"]
    return False


def main(setting: str, problem: str, method: str, save_path: str):
    """Main function to plot point distribution."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    problem_path = problem
    if problem == "drift_diffusion":
        problem_path = "drift_diffusion_hard"

    model_zoo = (
        Path("/opt/model_zoo")
        / f"iterative_{setting}"
        / problem_path
        / METHOD_DIR_NAMES[method]
    )

    mean_preds = {chkpt: [] for chkpt in CHKPTS_TO_PLOT}
    mean_scores = {chkpt: [] for chkpt in CHKPTS_TO_PLOT}

    X = None

    for seed in SEEDS:
        model, data, model_name, _ = construct_problem(
            problem_name=problem,
            layers=LAYERS[problem],
            seed=seed,
            float64=True,
            n_iterations=0,
            num_domain=NUM_POINTS[problem],
        )
        model.net.eval()

        for chkpt_iter in CHKPTS_TO_PLOT:
            chkpts = []
            if chkpt_iter == 200_000:
                chkpts = list(model_zoo.rglob(f"{model_name}*full.pt"))
            else:
                chkpts = list(model_zoo.rglob(f"{model_name}*{chkpt_iter}.pt"))

            checkpoint = None
            for chkpt in chkpts:
                if check_epoch(chkpt, chkpt_iter):
                    checkpoint = chkpt
                    break

            if checkpoint is None:
                print(f"No checkpoint found for {chkpt_iter} in seed {seed}. Skipping.")
                continue

            print(
                f"Loading checkpoint {checkpoint} for seed {seed} and iteration {chkpt_iter}."
            )
            load_checkpoint(model.net, checkpoint)

            if X is None:
                if data.soln is not None:
                    X = data.geom.uniform_points(50_000)
                    y = data.soln(X)
                    marker_size = 24
                else:
                    X = data.holdout_test_x
                    y = data.holdout_test_y
                    marker_size = 14

            pred = model.predict(X)
            mean_preds[chkpt_iter].append(pred)

            scorer = Scorer(method, model=model, n_candidate_points=10_000, seed=42)

            candidate_points, scores = scorer()
            mean_scores[chkpt_iter].append(scores)

    mean_preds = {
        chkpt: np.mean(np.array(preds), axis=0) if len(preds) > 0 else None
        for chkpt, preds in mean_preds.items()
    }
    mean_scores = {
        chkpt: np.mean(np.array(scores), axis=0) if len(scores) > 0 else None
        for chkpt, scores in mean_scores.items()
    }

    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    fig, ax = plt.subplots(
        1,
        len(CHKPTS_TO_PLOT) + 1,
        figsize=(len(CHKPTS_TO_PLOT) * (3 + 1), 4),
        sharey=True,
    )

    for i, chkpt in enumerate(CHKPTS_TO_PLOT):
        pred = mean_preds[chkpt]
        # skip if checkpoints not found
        if pred is not None:
            sc = ax[i].scatter(
                X[:, 0],
                X[:, 1],
                c=pred,
                s=marker_size,
                cmap="coolwarm",
                norm=norm,
                rasterized=True,
            )
            ax[i].set_title(f"Iter {chkpt}")
            ax[i].set_xlabel("x")
            if i == 0:
                ax[i].set_ylabel("t")
                ax[i].set_yticks(np.linspace(X[1:2].min(), X[1:2].max(), 5))
            else:
                ax[i].set_yticklabels([])

            scores = mean_scores[chkpt]
            scores = apply_distribution(scores, **DISTRIBUTION_PARMS[setting])
            normalized_scores = (
                (scores - scores.min()) / (scores.max() - scores.min()) * 0.25
            )

            ax[i].scatter(
                candidate_points[:, 0],
                candidate_points[:, 1],
                s=marker_size,
                alpha=normalized_scores,
                c="black",
                rasterized=True,
            )

        ax[-1].scatter(
            X[:, 0],
            X[:, 1],
            c=y,
            s=marker_size,
            cmap="coolwarm",
            norm=norm,
            rasterized=True,
        )
        ax[-1].set_title("Target")
        ax[-1].set_xlabel("x")
        divider = make_axes_locatable(ax[-1])
        # Append a new axes to the right of the last plot.
        # size="5%" means the new axes width will be 5% of the parent axes width.
        # pad=0.1 creates a small gap between the plot and the colorbar.
        cax = divider.append_axes("right", size="5%", pad=0.1)
        # Create the colorbar in the newly created axes (`cax`)
        fig.colorbar(sc, cax=cax, orientation="vertical", label="Prediction")
        fig.savefig(
            save_path / f"{setting}_{problem}_{method}.pdf",
            bbox_inches="tight",
            dpi=300,
        )

        fig.savefig(
            save_path / f"{setting}_{problem}_{method}.png",
            bbox_inches="tight",
            dpi=300,
        )

        print("Saved figure to", save_path)


if __name__ == "__main__":
    params = parse_args()
    main(
        setting=params.setting,
        problem=params.problem,
        method=params.method,
        save_path=params.save_path,
    )
