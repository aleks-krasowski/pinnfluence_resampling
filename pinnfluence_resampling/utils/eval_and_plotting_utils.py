# NOTE: some variables may have changed names or been removed in the original code.
# please adjust accordingly if necessary.

import json
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

METHODS = [
    "PINNfluence",
    "RAR",
    "Grad-Dot",
    "Out-Grad",
    "Loss-Grad",
    "Random",
    "Static",
]

PROBLEMS = {
    "add": [
        ("diffusion", 1, "3_x_32"),
        ("allen_cahn", 10, "3_x_64"),
        ("burgers", 10, "3_x_32"),
        ("wave", 10, "5_x_100"),
        ("drift_diffusion_sine", 10, "3_x_64"),
    ],
    "replace": [
        ("diffusion", 30, "3_x_32"),
        ("allen_cahn", 1000, "3_x_64"),
        ("burgers", 1000, "3_x_32"),
        ("wave", 1000, "5_x_100"),
        ("drift_diffusion_sine", 1000, "3_x_64"),
    ],
}

PARAMS = {
    "add": {
        "distribution_k": 2,
        "distribution_c": 0,
        "sampling_strategy": "distribution",
        "n_iter_pretrain": 0,
    },
    "replace": {
        "distribution_k": 1,
        "distribution_c": 1,
        "sampling_strategy": "distribution",
        "n_iter_pretrain": 0,
    },
}

MARKERS = {
    "PINNfluence": "D",
    "RAR": "p",
    "Random": "*",
    "Grad-Dot": "^",
    "Out-Grad": "o",
    "Loss-Grad": "s",
    "Static": "x",
}

COLORS = {
    "PINNfluence": "blue",
    "RAR": "green",
    "Random": "black",
    "Grad-Dot": "red",
    "Out-Grad": "brown",
    "Loss-Grad": "purple",
    "Static": "grey",
}


def layers_to_list(layers_str):
    """
    Convert a string representation of layers to a list of integers.
    Example: "3_x_32" -> [3, 32]
    """
    parts = layers_str.split("_x_")
    return (
        [2]  # x and t inputs
        + int(parts[0])
        * [
            int(parts[1]),
        ]  # hidden layers
        + [1]  # output layer (u)
    )


def load_problem(problem, results):
    df = pd.DataFrame()

    # if results are already compiled, just load them
    if (results / "compiled" / f"{problem}.csv").exists():
        df = pd.read_csv(results / "compiled" / f"{problem}.csv")
        return df

    # else compile results into dataframe
    for csv in results.rglob(f"{problem}*.csv"):

        dir = csv.parent

        config = dir / "config.json"

        if not config.exists():
            continue

        # get checkpoint file
        chkpts = list(dir.glob("*full.pt"))
        if len(chkpts) == 0:
            chkpt = ""
        else:
            chkpt = chkpts[0]

        # load config
        # contains run info
        with open(config, "r") as f:
            config = json.load(f)

        # write config to df
        cur_df = pd.read_csv(csv)

        model_name = config["model_name"]

        cur_df["problem"] = problem
        cur_df["distribution_k"] = config["distribution_k"]
        cur_df["distribution_c"] = config["distribution_c"]
        cur_df["sampling_strategy"] = config["sampling_strategy"]
        cur_df["scoring_strategy"] = config["scoring_strategy"]
        cur_df["sign"] = config["scoring_sign"]
        cur_df["seed"] = model_name[model_name.rfind("_") + 1 :]
        cur_df["n_iter_pretrain"] = model_name[
            model_name.find("adam")
            + 5 : model_name.find("_", model_name.find("adam") + 6)
        ]
        cur_df["model_name"] = model_name
        cur_df["n_samples"] = config["n_samples"]
        cur_df["training_strategy"] = config["training_strategy"]
        cur_df["checkpoint"] = chkpt

        # append run
        df = pd.concat([df, cur_df])

    # pinnfluence.utils.callbacks.EvalMetricCallback considers each time data is resampled
    # as epoch 0 internally
    # results on test data are however equivalent to last epoch before resampling
    # so we can drop it to avoid duplicates in data and irregularities regarding the epoch number
    df = df.loc[df["epoch"] != 0]

    # rename columns etc for readability
    df.rename(
        columns={
            "train_loss": "Train Loss",
            "valid_loss": "Validation Loss",
            "test_loss": "Test Loss",
            "l2_relative_error": "L2 Relative Error",
            "mse": "Mean Squared Error",
            "epoch": "Iteration",
            "scoring_strategy": "Strategy",
        },
        inplace=True,
    )
    df = df.loc[(df["sign"] == "abs") & (df["seed"].astype(int) <= 10)]
    df.drop_duplicates(
        subset=[
            "Strategy",
            "Iteration",
            "seed",
            "n_iter_pretrain",
            "n_samples",
            "distribution_k",
            "distribution_c",
            "sampling_strategy",
            "model_name",
        ],
        inplace=True,
        keep="first",
    )

    df["Strategy"] = df["Strategy"].replace(
        {
            "PINNfluence": "PINNfluence",
            "random": "Random",
            "grad_dot": "Grad-Dot",
            "RAR": "RAR",
            "steepest_prediction_gradient": "Out-Grad",
            "steepest_loss_gradient": "Loss-Grad",
        }
    )

    order_dict = {
        "PINNfluence": 0,
        "RAR": 1,
        "Random": 2,
        "Grad-Dot": 3,
        "Out-Grad": 4,
        "Loss-Grad": 5,
    }

    df = df.sort_values(by="Strategy", key=lambda x: x.map(order_dict))

    df["seed"] = df["seed"].astype(int)

    # save compiled results
    if not (results / "compiled").exists():
        (results / "compiled").mkdir()
    df.to_csv(results / "compiled" / f"{problem}.csv", index=False)

    return df


# format x axis as thousands
def format_iterations(x, pos):
    return f"{int(x / 1000)}K"


def filter_runs(
    df,
    *,
    c,
    k,
    strategy,
    n_iter_pretrain,
    n_samples,
    layers,
):
    """
    Filter DataFrame by distribution, sampling strategy, and model settings.
    """
    mask = (
        (df["distribution_c"] == c)
        & (df["distribution_k"] == k)
        & (df["sampling_strategy"] == strategy)
        & (df["sign"] == "abs")
        & (df["n_iter_pretrain"] == n_iter_pretrain)
        & (df["n_samples"] == n_samples)
        & df["model_name"].str.contains(layers)
    )
    return df.loc[mask]


def plot_combined_l2_err(df, window=1):
    strategy_order = [
        "PINNfluence",
        "RAR",
        "Grad-Dot",
        "Out-Grad",
        "Loss-Grad",
        "Random",
        "Static",  # Added new baseline to the order
    ]

    df["Strategy"] = pd.Categorical(
        df["Strategy"], categories=strategy_order, ordered=True
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Iterate over strategies in the specified order
    for name in strategy_order:
        strategy_group = df[df["Strategy"] == name]
        if strategy_group.empty:
            continue

        # Group by iteration to handle the 10 seeds
        agg_data = (
            strategy_group.groupby("Iteration")["L2 Relative Error"]
            .agg(["mean", "std"])
            .reset_index()
        )

        # Sort by iteration for proper plotting
        agg_data = agg_data.sort_values("Iteration")

        # Calculate rolling statistics
        rolling_data = agg_data.set_index("Iteration")
        rolling_mean = rolling_data["mean"].rolling(window=window, min_periods=1).mean()

        # Calculate log-scale confidence intervals
        log_std = (
            rolling_data["std"]
            .div(rolling_data["mean"])
            .rolling(window=window, min_periods=1)
            .mean()
        )
        factor = np.exp(log_std)  # Multiplicative factor for log scale
        lower_bound = rolling_mean / factor
        upper_bound = rolling_mean * factor

        # Get the marker for the current strategy
        marker = MARKERS.get(name, "o")

        # Calculate the positions where markers should appear
        marker_positions = (rolling_mean.index % 50000 == 0) & (rolling_mean.index > 0)
        marker_indices = np.where(marker_positions)[0]

        # Plot the rolling mean line with markers
        (line,) = ax.plot(
            rolling_mean.index,
            rolling_mean.values,
            label=name,
            color=COLORS.get(name),
            marker=marker,
            markevery=marker_indices,
            markersize=8,
        )

        # Add confidence interval
        ax.fill_between(
            rolling_mean.index,
            lower_bound,
            upper_bound,
            color=COLORS.get(name),
            alpha=0.2,
        )

    # Style the plot
    ax.set_yscale("log")
    ax.set_ylabel("$L^2$ Relative Error")
    ax.set_xlabel("Iteration")

    # Style the ticks
    ax.tick_params(axis="both", which="major")
    ax.xaxis.set_major_formatter(FuncFormatter(format_iterations))

    fig.tight_layout()

    return fig, ax


def plot_combined_problems(
    problems_data,
    selected_methods,
    window=1,
    metric="L2 Relative Error",
    exclude_problem=None,
):
    """
    Create a single plot comparing selected methods across multiple problems,
    showing performance relative to Random sampling.
    Colors are assigned by problem and line styles by method.
    """
    # Create figure with a single plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    problem_colors = {
        "diffusion": "#1F77B4",  # Black
        "allen_cahn": "#FF7F0E",  # Blue
        "burgers": "#2CA02C",  # Green
        "wave": "#D62728",  # Red-orange
        "drift_diffusion_sine": "#9467BD",  # Purple
        "drift_diffusion_exp": "#8C564B",  # Brown
    }

    # Define line styles for different methods
    method_styles = {
        "PINNfluence": "-",
        "RAR": "--",
    }

    handles = []
    labels = []

    # Process each problem
    for problem_name, df in problems_data.items():

        # Skip if the problem is excluded
        if exclude_problem and problem_name == exclude_problem:
            continue

        # Make sure Random is in the dataframe
        if "Random" not in df["Strategy"].unique():
            print(f"Warning: 'Random' strategy not found in {problem_name} data")
            continue

        # Get only the methods we want to plot (excluding Random)
        plot_methods = [m for m in selected_methods if m != "Random"]

        # For each method, calculate relative performance against Random
        for method in plot_methods:
            # Group and aggregate data for Random strategy
            random_data = df[df["Strategy"] == "Random"].copy()
            if random_data.empty:
                continue

            random_agg = (
                random_data.groupby("Iteration")[metric].agg(["mean"]).reset_index()
            )
            random_agg.set_index("Iteration", inplace=True)

            # Group and aggregate data for the current method
            method_data = df[df["Strategy"] == method].copy()
            if method_data.empty:
                continue

            method_agg = (
                method_data.groupby("Iteration")[metric]
                .agg(["mean", "std"])
                .reset_index()
            )
            method_agg.set_index("Iteration", inplace=True)

            # Make sure indices match
            common_index = method_agg.index.intersection(random_agg.index)
            if len(common_index) == 0:
                continue

            method_agg = method_agg.loc[common_index]
            random_agg = random_agg.loc[common_index]

            # Calculate error ratio (method error / random error)
            ratio = method_agg["mean"] / random_agg["mean"]

            # Calculate rolling average of the ratio
            rolling_ratio = ratio.rolling(window=window, min_periods=1).mean()

            # Calculate confidence interval
            ratio_std = method_agg["std"]
            ratio_std = ratio_std.rolling(window=window, min_periods=1).mean()
            lower_bound = rolling_ratio - ratio_std
            upper_bound = rolling_ratio + ratio_std

            # Plot the rolling ratio with improved visibility
            label = f"{method} ({problem_name})"
            (line,) = ax.plot(
                rolling_ratio.index,
                rolling_ratio.values,
                label=label,
                color=problem_colors[problem_name],
                linestyle=method_styles[method],
                linewidth=2.5,  # Slightly thinner lines
                markerfacecolor="white",  # White fill for markers
                markeredgewidth=1.5,  # Bold marker edges
            )

            handles.append(line)
            labels.append(label)

            # Add light shading for confidence intervals
            ax.fill_between(
                rolling_ratio.index,
                lower_bound,
                upper_bound,
                color=problem_colors[problem_name],
                alpha=0.1,
            )

    # Add a horizontal line at y=1 (equal to random performance)
    ax.axhline(y=1, color="black", linestyle="-", alpha=1, linewidth=1)

    # Style the plot with improved formatting
    ax.set_ylabel(
        "Error Ratio (Method / Random)", rotation=270, labelpad=14
    )  # , fontsize=12)
    ax.set_xlabel("Iteration")  # , fontsize=12)

    # Set axis limits
    ax.set_ylim(0, 2)  # Adjusted based on your data range
    ax.set_xlim(0, 200000)  # Set to match your iteration range

    # Add grid for better readability
    ax.grid(True, linestyle="--", alpha=0.3)

    # Style the ticks
    ax.tick_params(axis="both", which="major")  # , labelsize=10)
    ax.xaxis.set_major_formatter(FuncFormatter(format_iterations))

    # Format the spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # Adjust layout with more padding
    plt.tight_layout(pad=2.0)

    return fig, ax


# Example usage:
def create_combined_plot(
    results,
    problems=PROBLEMS["add"],
    selected_methods=["PINNfluence", "RAR", "Random"],
    params=PARAMS["add"],
    exclude_problem=None,
):
    k = params["distribution_k"]
    c = params["distribution_c"]
    distribution = params["sampling_strategy"]
    n_iter_pretrain = params["n_iter_pretrain"]

    # Dictionary to store dataframes for each problem
    problems_data = {}

    for problem, n_samples, layers in problems:
        # Assume load_problem is defined elsewhere and returns a dataframe
        df = load_problem(problem, results)
        df = df.loc[df["Iteration"] % 2000 == 0]

        # Filter the DataFrame as in the plot function
        sub_df = df.loc[
            (df["distribution_c"] == c)
            & (df["distribution_k"] == k)
            & (df["sampling_strategy"] == distribution)
            & (df["sign"] == "abs")
            & (df["n_iter_pretrain"] == n_iter_pretrain)
            & (df["n_samples"] == n_samples)
            & (df["model_name"].str.contains(layers))
        ]

        # Store filtered dataframe
        problems_data[problem] = sub_df

    # Create the combined plot
    fig, ax = plot_combined_problems(
        problems_data, selected_methods, window=1, exclude_problem=exclude_problem
    )

    return fig, ax


def generate_combined_l2_err_plots(
    results: Path,
    results_no_resampling: Path,
    fig_dir: Path,
    window: int = 1,
    problems: dict = PROBLEMS["add"],
    params: dict = PARAMS["add"],
):
    fig_dir.mkdir(parents=True, exist_ok=True)

    k = params["distribution_k"]
    c = params["distribution_c"]
    distribution = params["sampling_strategy"]
    n_iter_pretrain = params["n_iter_pretrain"]

    for problem, n_samples, layers in problems:
        # load data and runs without resampling for comparison
        df_main = load_problem(problem, results)
        df_no_resample = load_problem(problem, results_no_resampling)

        for df in (df_main, df_no_resample):
            df.loc[df["Iteration"] == 1, "Iteration"] = 0

        df_main = df_main[df_main["Iteration"] % 2000 == 0]
        df_no_resample = df_no_resample[df_no_resample["Iteration"] % 2000 == 0]

        sub_df_main = filter_runs(
            df_main,
            c=c,
            k=k,
            strategy=distribution,
            n_iter_pretrain=n_iter_pretrain,
            n_samples=n_samples,
            layers=layers,
        )

        sub_df_no_resample = df_no_resample.loc[
            (df_no_resample["n_samples"] == 0)
            & (df_no_resample["model_name"].str.contains(layers))
        ].copy()
        sub_df_no_resample["Strategy"] = "No Resampling"

        common_cols = [
            "Iteration",
            "L2 Relative Error",
            "Test Loss",
            "Validation Loss",
            "Strategy",
            "seed",
            "model_name",
        ]
        combined_df = pd.concat(
            [sub_df_main[common_cols], sub_df_no_resample[common_cols]],
            ignore_index=True,
        )

        sanity_check = (
            combined_df.loc[combined_df["Iteration"] == combined_df["Iteration"].max()]
            .groupby("Strategy")["seed"]
            .count()
        )

        if sanity_check.max() < 10:
            print(
                f"Warning: Not enough runs for {problem} with n_samples={n_samples}, layers={layers}, c={c}, k={k}, distribution={distribution}, n_iter_pretrain={n_iter_pretrain}"
            )
            continue

        fig, ax = plot_combined_l2_err(
            combined_df.loc[combined_df["model_name"].str.contains(layers)],
            window=window,
        )
        fig.show()
        fig.savefig(
            fig_dir / f"{problem}_c{c}_k{k}_{distribution}_n{n_samples}_{layers}.pdf"
        )


def create_standalone_legend():
    # Define the strategy order (same as in your plot function)
    strategy_order = list(MARKERS.keys())

    # Create a figure for the legend only
    fig, ax = plt.subplots(figsize=(12, 1.5))

    # Create legend handles
    handles = []
    labels = []

    # Create a line for each strategy
    for i, name in enumerate(strategy_order):
        if name in COLORS:
            line = mlines.Line2D(
                [],
                [],
                color=COLORS[name],
                marker=MARKERS[name],
                markersize=10,
                label=name,
                linewidth=2,
            )
            handles.append(line)
            labels.append(name)

    # Create the legend
    legend = ax.legend(
        handles=handles,
        labels=labels,
        loc="center",
        ncol=len(handles),
        frameon=True,
        fontsize=12,
        handlelength=2,
        borderpad=1,
        columnspacing=2,
    )

    # Remove axis elements
    ax.set_frame_on(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    fig.tight_layout()

    return fig, ax


def create_separate_legend(problem_colors, method_styles):
    """
    Create a separate horizontal legend figure that can be used with multiple plots.
    """
    from matplotlib.lines import Line2D

    # Create a new figure for the legend.
    legend_fig = plt.figure(figsize=(12, 1.5))
    legend_ax = legend_fig.add_subplot(111)

    # Hide the axes, as we only want to display the legend.
    legend_ax.axis("off")

    # --- Create Legend Elements ---
    method_legend_elements = []
    for method, style in method_styles.items():
        method_legend_elements.append(
            Line2D([0], [0], color="black", linestyle="-", label=method, linewidth=0)
        )

    problem_legend_elements = []
    for problem, color in problem_colors.items():
        # Using a thicker line for color patches makes them more visible.
        problem_name = problem.replace("_", " ").title()
        if problem == "drift_diffusion_sine":
            problem_name = "Drift Diffusion"
        problem_legend_elements.append(
            Line2D(
                [0], [0], color=color, label=problem_name, linewidth=4, linestyle=style
            )
        )

    # Combine all legend elements
    all_elements = method_legend_elements + problem_legend_elements
    all_labels = [e.get_label() for e in all_elements]

    # Create a horizontal legend
    legend = legend_ax.legend(
        handles=all_elements,
        labels=all_labels,
        loc="center",
        ncol=len(all_elements),  # All elements in one row
        frameon=True,
        framealpha=1,
        edgecolor="gray",
        mode="expand",  # Expand horizontally
        bbox_to_anchor=(0, 0, 1, 1),  # Use the entire figure space
    )

    num_methods = len(method_styles)
    for i in range(num_methods):
        legend.get_texts()[i].set_fontweight("bold")

    legend_fig.tight_layout()

    return legend_fig


def plot_boxplots(
    df: pd.DataFrame,
    metric: str = "L2 Relative Error",
    fig_dir: Path = Path("./figures"),
):
    """
    Create boxplots with stripplots for each problem in the dataframe.

    Args:
        df (pd.DataFrame): DataFrame containing the results.
        metric (str): The metric to plot (e.g., 'L2 Relative Error', 'Test Loss').
        fig_dir (Path): Directory to save the figures.
    """
    if not fig_dir.exists():
        fig_dir.mkdir(parents=True)

    # Define the desired order for methods on the x-axis
    methods_order = [
        "PINNfluence",
        "RAR",
        "Grad-Dot",
        "Out-Grad",
        "Loss-Grad",
        "Random",
        "Static",
    ]

    method_colors = COLORS

    # Create the palette list in the correct order for Seaborn
    palette = [method_colors.get(m, "gray") for m in methods_order]

    # --- Plotting Loop ---
    # Iterate over each unique problem in the dataframe
    for problem in df["problem"].unique():
        problem_df = df[df["problem"] == problem]

        # --- Figure Setup ---
        fig, ax = plt.subplots(
            figsize=(16, 9)
        )  # Use a 16:9 aspect ratio for better spacing

        # --- Stripplot (Individual Data Points) ---
        # Plot individual points with controlled jitter and transparency.
        # This shows the distribution of raw data points.
        sns.stripplot(
            data=problem_df,
            x="Strategy",
            y=metric,
            ax=ax,
            order=methods_order,
            palette=palette,
            jitter=0.2,  # Increase jitter for better point separation
            size=20,  # Slightly larger points
            edgecolor="white",  # White edge makes points pop
            linewidth=0.7,
            alpha=0.7,  # Make points slightly transparent
        )

        # --- Boxplot (Statistical Summary) ---
        # Overlay a boxplot to show statistical summaries (quartiles, median).
        sns.boxplot(
            data=problem_df,
            x="Strategy",
            y=metric,
            ax=ax,
            order=methods_order,
            showfliers=False,  # Outliers are already shown by the stripplot
            palette=palette,
            boxprops={"alpha": 0.4, "edgecolor": "black", "linewidth": 1.5},
            whiskerprops={"color": "black", "linewidth": 1.5},
            capprops={"color": "black", "linewidth": 1.5},
            medianprops={"color": "black", "linewidth": 2.5, "linestyle": "-"},
        )

        # --- Aesthetics and Labels ---
        # Set a log scale for the y-axis to handle wide data ranges
        ax.set_yscale("log")

        # Improve axis labels
        y_label = ""
        if metric == "L2 Relative Error":
            y_label = "$L^2$ Relative Error"
        elif metric == "Test Loss":
            y_label = "Test Loss"
        ax.set_xlabel(None)
        ax.set_ylabel(y_label, labelpad=15)

        # # Customize tick parameters for better readability
        ax.tick_params(axis="x", labelrotation=22.5)
        ax.tick_params(
            axis="y",
        )

        # Ensure everything fits without overlapping
        plt.tight_layout()

        # Display the plot
        plt.savefig(
            f'./figures/replace_boxplot_{problem}_{metric.lower().replace(" ", "_")}.pdf',
            bbox_inches="tight",
            dpi=300,
        )
        plt.show()


def calculate_final_metrics(df, methods):
    """
    Calculates the mean and standard deviation for the final iteration results.

    Args:
        df (pd.DataFrame): The combined dataframe for a problem.
        methods (list): A list of method names to calculate metrics for.

    Returns:
        dict: A nested dictionary with the calculated metrics for each strategy.
              e.g., {'StrategyName': {'L2 Error': (mean, std), 'Loss': (mean, std)}}
    """
    max_iteration = df["Iteration"].max()
    final_df = df.loc[df["Iteration"] == max_iteration]

    problem_results = {}
    for strategy in methods:
        group = final_df[final_df["Strategy"] == strategy]

        if (
            group.empty
            or group[["L2 Relative Error", "Test Loss"]].isnull().values.any()
        ):
            problem_results[strategy] = {"L2 Error": "N/A", "Loss": "N/A"}
            continue

        l2_errors = group["L2 Relative Error"].values
        losses = group["Test Loss"].values

        problem_results[strategy] = {
            "L2 Error": (np.mean(l2_errors), np.std(l2_errors, ddof=1)),
            "Loss": (np.mean(losses), np.std(losses, ddof=1)),
        }
    return problem_results


def get_common_exponent(data_list):
    """Calculates the minimum common exponent from a list of numbers."""
    exponents = [
        int(np.floor(np.log10(val))) for val in data_list if val is not None and val > 0
    ]
    return int(np.min(exponents)) if exponents else 0


def format_latex_cell(value_tuple, common_exponent, is_bold):
    """
    Formats a (mean, std) tuple into a LaTeX string with mantissa and uncertainty.

    Args:
        value_tuple (tuple or str): The (mean, std) tuple or 'N/A'.
        common_exponent (int): The common power-of-10 exponent for the column.
        is_bold (bool): Whether to bold the cell content.

    Returns:
        str: The formatted LaTeX string for the table cell.
    """
    if not isinstance(value_tuple, tuple):
        return "N/A"

    mean, std = value_tuple
    if mean <= 0:
        return "N/A"

    mantissa = mean / (10**common_exponent)
    scaled_std = std / (10**common_exponent)

    # Format core string based on magnitude
    if mantissa >= 100:
        if scaled_std >= 10:
            content = f"{mantissa:.0f}{{\\scriptstyle \\pm {scaled_std:.0f}}}"
        else:
            content = f"{mantissa:.0f}{{\\scriptstyle \\pm {scaled_std:.1f}}}"
    else:
        content = f"{mantissa:.1f}{{\\scriptstyle \\pm {scaled_std:.1f}}}"

    # Add bolding if it's the minimum value
    return f"$\\mathbf{{{content}}}$" if is_bold else f"${content}$"


def generate_latex_section(metric_name, results_table, problems_list, methods_list):
    """
    Generates the full LaTeX block for a given metric (e.g., 'L2 Error' or 'Loss').

    Args:
        metric_name (str): The name of the metric to process ('L2 Error' or 'Loss').
        results_table (dict): The complete results data.
        problems_list (list): The list of problem tuples.
        methods_list (list): The list of method names.

    Returns:
        str: A string containing all the LaTeX rows for that metric section.
    """
    common_exponents = {}
    best_methods = {}

    # First pass: find the best method and common exponent for each problem column
    for problem_name, _, _ in problems_list:
        means_in_col = {
            method: results_table[method][problem_name][metric_name]
            for method in methods_list
        }
        valid_means = {
            m: v[0]
            for m, v in means_in_col.items()
            if isinstance(v, tuple) and v[0] > 0
        }

        if not valid_means:
            best_methods[problem_name] = None
            common_exponents[problem_name] = 0
            continue

        best_methods[problem_name] = min(valid_means, key=valid_means.get)
        common_exponents[problem_name] = get_common_exponent(valid_means.values())

    # Second pass: build the LaTeX table rows
    latex_rows = []

    # Build "Order of Magnitude" (OOM) row
    oom_row_parts = [r"{[OOM]}"]
    for problem_name, _, _ in problems_list:
        exp = common_exponents[problem_name]
        oom_row_parts.append(f"$[10^{{{exp}}}]$")
    latex_rows.append(" & ".join(oom_row_parts) + r" \\")

    # Build a row for each method
    for method in methods_list:
        row_parts = [f"        {method}"]
        for problem_name, _, _ in problems_list:
            value_tuple = results_table[method][problem_name][metric_name]
            common_exp = common_exponents[problem_name]
            is_best = method == best_methods[problem_name]
            cell_str = format_latex_cell(value_tuple, common_exp, is_best)
            row_parts.append(cell_str)
        latex_rows.append(" & ".join(row_parts) + r"\\")

    return "\n".join(latex_rows)


def generate_full_latex_table(results_table, problems_list, methods_list):
    """
    Assembles the complete LaTeX table string from all its components.

    Args:
        results_table (dict): The complete results data.
        problems_list (list): The list of problem tuples.
        methods_list (list): The list of method names.

    Returns:
        str: The final, complete LaTeX table as a single string.
    """
    # Generate sections for each metric
    l2_error_section = generate_latex_section(
        "L2 Error", results_table, problems_list, methods_list
    )
    loss_section = generate_latex_section(
        "Loss", results_table, problems_list, methods_list
    )

    # Create the header row
    problem_headers = " & ".join(
        [p[0].replace("_", " ").title() for p in problems_list]
    )

    # Assemble the final LaTeX string
    final_latex = f"""
\\begin{{table}}[h!]
    \\centering
    \\label{{tab:main_results}}
    \\begin{{tabular}}{{l|{"c|" * len(problems_list)}}}
        \\toprule
        Method & {problem_headers} \\\\
        \\midrule
{l2_error_section}
        \\midrule
{loss_section}
        \\bottomrule
    \\end{{tabular}}
\\end{{table}}
"""
    return final_latex.strip()
