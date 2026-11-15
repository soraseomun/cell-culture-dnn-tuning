# =======================================================================
# CONSTANTS
# Settings the whole team agreed on. If one of these changes, change it
# here only, so everyone's results stay comparable.
# =======================================================================

import os

# --- Reproducibility ---------------------------------------------------
# Same seed as the course examples. The seed is reset before every model
# is trained (see functions.train_and_evaluate), so a hyperparameter set
# gives the same result no matter who runs it or in what order.
RANDOM_SEED = 0

# --- Hyperparameter search ---------------------------------------------
# Fraction of the data held out as the test set during the search
# (train-test split is allowed at this stage instead of cross-validation).
TEST_SIZE = 0.2

# --- Paths -------------------------------------------------------------
DATASET_PATH = os.path.join("datasets", "dataset_cell_growth_7factor.csv")
TABLES_DIR = "tables"
FIGURES_DIR = "figures"

# --- Data columns ------------------------------------------------------
TARGET_COLUMNS = [
    "Specific Growth Rate [d^-1]",
    "Specific Death Rate [d^-1]",
    "Specific Production Rate [pg cell^-1 d^-1]",
]

FEATURE_COLUMNS = [
    "Temperature [C]",
    "pH",
    "Glucose [mM]",
    "Glutamine [mM]",
    "Lactate [mM]",
    "Ammonia [mM]",
    "Viable Cell Density [cells/mL]",
]

# --- Results tables ----------------------------------------------------
# Every results table starts with these columns, in this order, so tables
# from different team members can be combined with pandas.concat().
HYPERPARAMETER_COLUMNS = ["hidden_layers", "activation", "learning_rate", "batch_size", "weight_decay", "epochs"]

# --- Plot formatting ----------------------------------------------------
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
LINESTYLES = ["-", ":", "--", "-."]
fontsize_xy_label = 14
fontsize_ticks = 12
fontsize_legend = 12