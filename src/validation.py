# =======================================================================
# VALIDATION OF THE OPTIMAL NETWORK
# 4-fold cross-validation of the optimal hyperparameter set, the summary
# table (mean and standard deviation of RMSE and NRMSE per target) and the
# 2 x 2 figure of training and validation loss for each fold.
# =======================================================================

import os

import matplotlib.pyplot as plt

import src.functions as sf
from src.constants import *

# Output files (tables are saved in tables/, figures in figures/)
CV_FOLDS_TABLE = "validation/table_cv_optimal_folds.csv"
CV_HISTORY_TABLE = "validation/table_cv_optimal_history.csv"
CV_SUMMARY_TABLE = "table_cv_optimal_summary.csv"
CV_LOSS_FIGURE = "figure_cv_optimal_loss_curves.png"

N_SPLITS_OPTIMAL = 4

# A fold whose lowest validation loss falls in the last 10% of the epochs
# may still have been improving when training stopped.
LATE_BEST_EPOCH_FRACTION = 0.9


def run_optimal_cv(hyperparameters, x, y):
    """
    Run 4-fold cross-validation of the optimal network and save three tables:
    one row per (fold, target), one row per (fold, epoch) with the loss
    curves, and the summary with one row per target.

    Returns the summary DataFrame.
    """
    df_folds, df_history = sf.cross_validate(hyperparameters, x, y, n_splits=N_SPLITS_OPTIMAL)
    df_summary = sf.summarize_cv(df_folds)

    sf.save_table(df_folds, CV_FOLDS_TABLE)
    sf.save_table(df_history, CV_HISTORY_TABLE)
    sf.save_table(df_summary, CV_SUMMARY_TABLE)

    check_convergence(df_folds)

    return df_summary


def check_convergence(df_folds):
    """
    Print the best epoch of each fold and warn if any fold reached its lowest
    validation loss near the end of training (more epochs may be needed).
    """
    df_epochs = df_folds.groupby("fold")[["best_epoch", "epochs"]].first()

    for fold, row in df_epochs.iterrows():
        late = row["best_epoch"] >= LATE_BEST_EPOCH_FRACTION * row["epochs"]
        note = "  <-- may still be improving, consider more epochs" if late else ""
        print(f"Fold {fold}: best epoch {row['best_epoch']} of {row['epochs']}{note}")


def plot_cv_loss_curves(log_scale=False):
    """
    2 x 2 figure (one subplot per fold) of training and validation loss
    against epoch, read from the saved history table so it can be redrawn
    without retraining. A dotted line marks the epoch with the lowest
    validation loss, whose weights the model keeps.
    """
    df_history = sf.load_table(CV_HISTORY_TABLE)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 7), layout="constrained")

    for ax, (fold, df_fold) in zip(axes.flat, df_history.groupby("fold")):
        best_row = df_fold.loc[df_fold["loss_valid"].idxmin()]

        ax.plot(df_fold["epoch"], df_fold["loss_train"], label="Training loss")
        ax.plot(df_fold["epoch"], df_fold["loss_valid"], label="Validation loss")
        ax.axvline(best_row["epoch"], color="gray", linestyle=":", label=f"Best epoch ({int(best_row['epoch'])})")

        ax.set_title(f"Fold {fold}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE)")
        if log_scale:
            ax.set_yscale("log")
        ax.legend(loc="upper right", fontsize=fontsize_legend)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, CV_LOSS_FIGURE), dpi=200)
    plt.close(fig)
