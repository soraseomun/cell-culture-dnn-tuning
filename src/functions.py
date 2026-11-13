# =======================================================================
# FUNCTIONS
# Shared helper functions used by every stage of the workflow.
# =======================================================================

import ast
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, ParameterGrid, train_test_split

from src.classes import FullyConnectedNeuralNetwork
from src.constants import *


# -----------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------

def set_seeds(seed=RANDOM_SEED):
    """Fix the Python, NumPy and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset(path=DATASET_PATH, n_samples=None):
    """
    Load the dataset and return (x, y) as NumPy arrays.

    n_samples : int or None
        If given, use a random subset of this many rows. Useful for quick
        test runs before running on the full dataset.
    """
    df = pd.read_csv(path)

    if n_samples is not None:
        df = df.sample(n=n_samples, random_state=RANDOM_SEED)

    x = df.loc[:, FEATURE_COLUMNS].to_numpy()
    y = df.loc[:, TARGET_COLUMNS].to_numpy()

    return x, y


# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------

def compute_metrics(y_true, y_pred):
    """
    Return one record per target with its RMSE and NRMSE.

    NRMSE = RMSE / IQR, where IQR is the interquartile range of the true
    values (same definition as the course example). Dividing by the IQR
    removes the units, so NRMSE can be compared and averaged across the
    three targets, which RMSE cannot.
    """
    iqr = np.quantile(y_true, q=0.75, axis=0) - np.quantile(y_true, q=0.25, axis=0)

    records = []
    for j, target in enumerate(TARGET_COLUMNS):
        rmse = root_mean_squared_error(y_true[:, j], y_pred[:, j])
        records.append({"target": target, "RMSE": rmse, "NRMSE": rmse / iqr[j]})

    return records


# -----------------------------------------------------------------------
# Training and evaluation
# -----------------------------------------------------------------------

def train_and_evaluate(hyperparameters, x_train, y_train, x_test, y_test, verbose=False):
    """
    Train one network and evaluate it on the test data.

    The seeds are reset right before the model is created. Without this,
    the initial weights would depend on how many models were trained
    earlier in the same run, so splitting the search across team members
    would give different results than running it all on one computer.

    Returns
    -------
    records : list[dict]
        One record per target: hyperparameters, target, RMSE, NRMSE,
        best_epoch, fit_time_s.
    model : FullyConnectedNeuralNetwork
        The trained model (model.history holds the loss curves).
    """
    set_seeds()

    model = FullyConnectedNeuralNetwork(random_state=RANDOM_SEED, verbose=verbose, **hyperparameters)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Read the values back from the model so every column is filled in,
    # including defaults that were not passed in `hyperparameters`.
    hp_record = {column: getattr(model, column) for column in HYPERPARAMETER_COLUMNS}

    records = []
    for metric_record in compute_metrics(y_test, y_pred):
        records.append({**hp_record, **metric_record,
                        "best_epoch": model.best_epoch, "fit_time_s": model.fit_time})

    return records, model


def split_grid(param_grid, n_parts, part):
    """
    Return the hyperparameter sets that one team member should run.

    The sets are dealt out in turn (1, 2, 3, 4, 1, 2, ...) rather than in
    blocks, so each member gets a similar mix of small and large networks
    and the run times stay roughly balanced.

    part : int
        1, 2, ..., n_parts
    """
    all_sets = list(ParameterGrid(param_grid))
    return all_sets[part - 1::n_parts]


def run_search(hyperparameter_sets, x, y):
    """
    Evaluate each hyperparameter set with a single train-test split.

    Returns a DataFrame with one row per (hyperparameter set, target).
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    records = []
    for i, hyperparameters in enumerate(hyperparameter_sets, start=1):
        set_records, _ = train_and_evaluate(hyperparameters, x_train, y_train, x_test, y_test)
        records.extend(set_records)
        progress_bar(i, len(hyperparameter_sets))
    print()

    return pd.DataFrame.from_records(records)


def cross_validate(hyperparameters, x, y, n_splits):
    """
    K-fold cross-validation of one hyperparameter set.

    Returns
    -------
    df_metrics : DataFrame
        One row per (fold, target): fold, hyperparameters, target, RMSE,
        NRMSE, best_epoch, fit_time_s.
    df_history : DataFrame
        One row per (fold, epoch): fold, epoch, loss_train, loss_valid.
        Use this for the 2 x 2 loss-curve figure.
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    metric_records = []
    history_records = []

    for fold, (idx_train, idx_test) in enumerate(kfold.split(x), start=1):
        records, model = train_and_evaluate(hyperparameters, x[idx_train], y[idx_train], x[idx_test], y[idx_test])

        metric_records.extend({"fold": fold, **record} for record in records)

        for epoch, loss_train, loss_valid in zip(model.history["epoch"],
                                                 model.history["loss_train"],
                                                 model.history["loss_valid"]):
            history_records.append({"fold": fold, "epoch": epoch, "loss_train": loss_train, "loss_valid": loss_valid})

        progress_bar(fold, n_splits)
    print()

    return pd.DataFrame.from_records(metric_records), pd.DataFrame.from_records(history_records)


# -----------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------

def rank_hyperparameters(df_results):
    """
    Rank hyperparameter sets by their mean NRMSE over the three targets
    (and over the folds, if the table has a fold column). Lowest first.
    """
    df_rank = (df_results
               .groupby(HYPERPARAMETER_COLUMNS, sort=False)["NRMSE"]
               .mean()
               .reset_index(name="mean_NRMSE")
               .sort_values("mean_NRMSE", ignore_index=True))
    return df_rank


def summarize_cv(df_metrics):
    """
    Mean and standard deviation of RMSE and NRMSE across folds, for each
    hyperparameter set and target. (pandas uses the sample standard
    deviation, ddof = 1.)
    """
    df_summary = (df_metrics
                  .groupby(HYPERPARAMETER_COLUMNS + ["target"], sort=False)[["RMSE", "NRMSE"]]
                  .agg(["mean", "std"]))
    df_summary.columns = [f"{metric}_{stat}" for metric, stat in df_summary.columns]
    return df_summary.reset_index()


# -----------------------------------------------------------------------
# Saving and loading tables
# -----------------------------------------------------------------------

def save_table(df, filename):
    """Save a DataFrame to tables/<filename>."""
    os.makedirs(TABLES_DIR, exist_ok=True)
    df.to_csv(os.path.join(TABLES_DIR, filename), index=False)


def load_table(filename):
    """
    Load tables/<filename>.

    hidden_layers is stored in the CSV as text such as "(16, 8)", so it is
    converted back to a tuple here.
    """
    df = pd.read_csv(os.path.join(TABLES_DIR, filename))

    if "hidden_layers" in df.columns:
        df["hidden_layers"] = df["hidden_layers"].apply(ast.literal_eval)

    return df


def combine_tables(filenames):
    """Load several tables (e.g. each member's search results) into one."""
    return pd.concat([load_table(filename) for filename in filenames], ignore_index=True)


# -----------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------

def progress_bar(c_iter, n_iters, n_updates=1000, length_bar=20):
    """Progress bar from the course repository (units/unit_06/src/functions.py)."""
    update_every = max(1, round(n_iters / n_updates))
    if c_iter % update_every == 0:
        progress = c_iter / n_iters
        length_fill = round(length_bar * progress)
        length_void = length_bar - length_fill
        sys.stdout.write(f"\r{progress:6.0%} [" + length_fill * "=" + length_void * " " + "]")
