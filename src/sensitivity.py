# =======================================================================
# SENSITIVITY ANALYSIS
# One-at-a-time sensitivity of the optimal network: each hyperparameter is
# changed on its own while the other four stay at their optimal values,
# and every setting is scored with 3-fold cross-validation.
# =======================================================================

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.functions as sf
from src.constants import *

# Output files (tables are saved in tables/, figures in figures/)
SENSITIVITY_FOLDS_TABLE = "sensitivity/table_sensitivity_folds.csv"
SENSITIVITY_SUMMARY_TABLE = "table_sensitivity_summary.csv"
SENSITIVITY_OVERVIEW_TABLE = "table_sensitivity_overview.csv"
SENSITIVITY_FIGURES_DIR = "sensitivity"

N_SPLITS_SENSITIVITY = 3

# Label used for the average of the three targets (the selection metric).
MEAN_TARGET = "Mean of three targets"

# Values tested for each hyperparameter, in plotting order. Each list holds
# the optimal value plus two values on each side where possible.
# - hidden_layers: depth kept at two layers and the widths halved or doubled,
#   so only the size of the network changes. (16, 8) was the smallest
#   architecture in the grid search, so smaller ones are tested here.
# - activation: all four functions the class supports.
# - learning_rate: steps of about half a decade (x3). 0.01 was the largest
#   value in the grid search, so larger ones are tested here.
# - batch_size: steps of x2.
# - weight_decay: the optimum is 0, which cannot be lowered, so four larger
#   values are tested, one decade apart.
SENSITIVITY_VALUES = {
    "hidden_layers": [(4, 2), (8, 4), (16, 8), (32, 16), (64, 32)],
    "activation": ["relu", "leakyrelu", "tanh", "sigmoid"],
    "learning_rate": [0.001, 0.003, 0.01, 0.03, 0.1],
    "batch_size": [128, 256, 512, 1024, 2048],
    "weight_decay": [0.0, 0.0001, 0.001, 0.01, 0.1],
}


def build_sensitivity_settings(optimal, values=SENSITIVITY_VALUES):
    """
    List every setting of the analysis as (hyperparameter, full set).

    The optimal set appears once under every hyperparameter, because it is
    the reference point of every curve. It is trained only once (see
    run_sensitivity).
    """
    settings = []

    for name, name_values in values.items():
        if optimal[name] not in name_values:
            raise ValueError(f"The optimal {name} ({optimal[name]}) is not in SENSITIVITY_VALUES.")

        for value in name_values:
            settings.append((name, {**optimal, name: value}))

    return settings


def _unique_sets(settings):
    """Hyperparameter sets of the analysis without repeats, in order."""
    unique = {}
    for name, hyperparameters in settings:
        unique.setdefault(sf._hyperparameter_key(hyperparameters), (name, hyperparameters))
    return list(unique.values())


def run_sensitivity(optimal, x, y, checkpoint_filename=SENSITIVITY_FOLDS_TABLE):
    """
    Run 3-fold cross-validation for every setting of the analysis.

    The results are saved to tables/<checkpoint_filename> after each
    setting. If the run stops (laptop closed, stop button, ...), running it
    again skips the settings already in that file and continues.

    Every setting uses the same three folds (fixed random_state), so the
    settings are compared on exactly the same data.

    Returns a DataFrame with one row per (setting, fold, target).
    """
    unique_sets = _unique_sets(build_sensitivity_settings(optimal))

    df_done = pd.DataFrame()
    done_keys = set()
    if os.path.exists(os.path.join(TABLES_DIR, checkpoint_filename)):
        df_done = sf.load_table(checkpoint_filename)
        done_keys = set(df_done.apply(sf._hyperparameter_key, axis=1))
        print(f"Resuming: {len(done_keys)} of {len(unique_sets)} settings already in {checkpoint_filename}")

    results = [df_done]

    for i, (name, hyperparameters) in enumerate(unique_sets, start=1):
        label = "optimal set" if hyperparameters == optimal else f"{name} = {hyperparameters[name]}"

        if sf._hyperparameter_key(hyperparameters) in done_keys:
            print(f"[{i:2d}/{len(unique_sets)}] {label}: already done")
            continue

        print(f"[{i:2d}/{len(unique_sets)}] {label}")
        df_folds, _ = sf.cross_validate(hyperparameters, x, y, n_splits=N_SPLITS_SENSITIVITY)
        results.append(df_folds)

        sf.save_table(pd.concat(results, ignore_index=True), checkpoint_filename)

    return pd.concat(results, ignore_index=True)


def summarize_sensitivity(df_folds, optimal):
    """
    Mean and standard deviation (across the 3 folds) of NRMSE for every
    setting and target, plus the mean of the three targets, which is the
    metric used to pick the optimal set.

    change_percent compares each setting with the optimal set on the same
    target: positive means a larger error than the optimal network.

    Returns the summary table (one row per hyperparameter, value and target).
    """
    df_folds = df_folds.copy()
    df_folds["key"] = df_folds.apply(sf._hyperparameter_key, axis=1)

    # Mean of the three targets in each fold, added as an extra "target"
    df_mean = (df_folds.groupby(["key", "fold"] + HYPERPARAMETER_COLUMNS, sort=False)
               .agg(NRMSE=("NRMSE", "mean"), best_epoch=("best_epoch", "first"), fit_time_s=("fit_time_s", "first"))
               .reset_index())
    df_mean["target"] = MEAN_TARGET
    df_all = pd.concat([df_folds, df_mean], ignore_index=True)

    stats = (df_all.groupby(["key", "target"], sort=False)
             .agg(NRMSE_mean=("NRMSE", "mean"), NRMSE_std=("NRMSE", "std"),
                  mean_best_epoch=("best_epoch", "mean"), fit_time_s=("fit_time_s", "mean"))
             .reset_index())

    optimal_key = sf._hyperparameter_key(optimal)
    reference = stats[stats["key"] == optimal_key].set_index("target")["NRMSE_mean"]

    records = []
    for name, hyperparameters in build_sensitivity_settings(optimal):
        key = sf._hyperparameter_key(hyperparameters)
        for _, row in stats[stats["key"] == key].iterrows():
            records.append({
                **{column: hyperparameters[column] for column in HYPERPARAMETER_COLUMNS},
                "varied_hyperparameter": name,
                "is_optimal": key == optimal_key,
                "target": row["target"],
                "NRMSE_mean": row["NRMSE_mean"],
                "NRMSE_std": row["NRMSE_std"],
                "change_percent": 100 * (row["NRMSE_mean"] / reference[row["target"]] - 1),
                "mean_best_epoch": row["mean_best_epoch"],
                "fit_time_s": row["fit_time_s"],
            })

    df_summary = pd.DataFrame.from_records(records)
    sf.save_table(df_summary, SENSITIVITY_SUMMARY_TABLE)

    return df_summary


def summarize_overview(df_summary):
    """
    One row per hyperparameter: the best and worst tested value (by the mean
    NRMSE of the three targets) and how much worse the worst value is than
    the optimal set. A large max_change_percent means the network is
    sensitive to that hyperparameter.
    """
    df_mean = df_summary[df_summary["target"] == MEAN_TARGET]

    records = []
    for name, df_name in df_mean.groupby("varied_hyperparameter", sort=False):
        best = df_name.loc[df_name["NRMSE_mean"].idxmin()]
        worst = df_name.loc[df_name["NRMSE_mean"].idxmax()]
        records.append({
            "hyperparameter": name,
            "optimal_value": df_name.loc[df_name["is_optimal"], name].iloc[0],
            "best_value": best[name],
            "best_NRMSE_mean": best["NRMSE_mean"],
            "worst_value": worst[name],
            "worst_NRMSE_mean": worst["NRMSE_mean"],
            "max_change_percent": worst["change_percent"],
        })

    df_overview = pd.DataFrame.from_records(records).sort_values("max_change_percent", ascending=False,
                                                                  ignore_index=True)
    sf.save_table(df_overview, SENSITIVITY_OVERVIEW_TABLE)

    return df_overview


def _format_value(value):
    """Tick label for a hyperparameter value."""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def plot_sensitivity(df_summary):
    """
    One figure per hyperparameter, 2 x 2 subplots (three targets and their
    mean). Each point is the mean NRMSE over the 3 folds and the error bar is
    the standard deviation. The optimal value is drawn as a star.

    The x positions are evenly spaced. The tested values are themselves
    evenly spaced on a log scale (x2 or x3 steps), so the axis is close to
    a log axis, but it also works for the activation names and for a weight
    decay of 0.
    """
    output_dir = os.path.join(FIGURES_DIR, SENSITIVITY_FIGURES_DIR)
    os.makedirs(output_dir, exist_ok=True)

    targets = TARGET_COLUMNS + [MEAN_TARGET]

    for name, df_name in df_summary.groupby("varied_hyperparameter", sort=False):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 7), layout="constrained")

        for ax, target in zip(axes.flat, targets):
            df_target = df_name[df_name["target"] == target].reset_index(drop=True)
            positions = np.arange(len(df_target))
            is_optimal = df_target["is_optimal"].to_numpy()

            # Connect the points only when the values have a natural order
            linestyle = "none" if name == "activation" else "-"

            ax.errorbar(positions, df_target["NRMSE_mean"], yerr=df_target["NRMSE_std"],
                        color=COLORS[0], linestyle=linestyle, linewidth=1.5, marker="o", markersize=7,
                        capsize=4, label="Mean ± SD (3 folds)")
            ax.plot(positions[is_optimal], df_target.loc[is_optimal, "NRMSE_mean"],
                    color=COLORS[1], linestyle="none", marker="*", markersize=16, zorder=3, label="Optimal value")

            ax.set_xticks(positions, [_format_value(value) for value in df_target[name]])
            ax.set_title(target.split(" [")[0])
            ax.set_xlabel(name)
            ax.set_ylabel("NRMSE")
            ax.grid(axis="y", alpha=0.3)

        handles, labels = axes.flat[0].get_legend_handles_labels()
        axes.flat[0].legend(handles[::-1], labels[::-1], fontsize=fontsize_legend)

        fig.savefig(os.path.join(output_dir, f"figure_sensitivity_{name}.png"), dpi=200)
        plt.close(fig)
