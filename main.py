# =====================================================================
# SMOKE TEST (temporary main.py)
# Checks that the environment, data path, and shared code all work.
# Runs in under a minute: small data subset, few epochs.
# Replace this file with the real workflow later.
# =====================================================================

import src.functions as sf
from src.constants import *

# Small subset of the data so the test runs quickly
x, y = sf.load_dataset(n_samples=2000)
print(f"Loaded x: {x.shape}, y: {y.shape}")

# 1) Search: two hyperparameter sets, split as if there were 2 members
param_grid = dict(
    hidden_layers=[(16, 8)],
    activation=["relu", "tanh"],
    learning_rate=[0.001],
    batch_size=[256],
    weight_decay=[0.0],
    epochs=[5],
)
my_sets = sf.split_grid(param_grid, n_parts=2, part=1)
df_search = sf.run_search(my_sets, x, y)
sf.save_table(df_search, "table_smoke_search.csv")
print(df_search)

# 2) Cross-validation: 2 folds of one hyperparameter set
hyperparameters = dict(hidden_layers=(16, 8), activation="relu", learning_rate=0.001,
                       batch_size=256, weight_decay=0.0, epochs=5)
df_metrics, df_history = sf.cross_validate(hyperparameters, x, y, n_splits=2)
print(sf.summarize_cv(df_metrics))
print(df_history.head())

# 3) Save -> load round trip (hidden_layers should come back as a tuple)
sf.save_table(df_metrics, "table_smoke_cv.csv")
df_loaded = sf.load_table("table_smoke_cv.csv")
print("hidden_layers type after loading:", type(df_loaded.loc[0, "hidden_layers"]))
print(sf.rank_hyperparameters(df_loaded))

print("Smoke test finished without errors.")
