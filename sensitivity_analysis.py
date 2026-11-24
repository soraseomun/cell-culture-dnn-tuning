# =======================================================================
# Sensitivity analysis of the optimal network (section 5-2)
# Changes one hyperparameter at a time around the optimal set, scores each
# setting with 3-fold CV, saves the tables and draws one figure per
# hyperparameter. Safe to stop and rerun: finished settings are skipped.
# =======================================================================

import pandas as pd

import src.functions as sf
import src.sensitivity as ss
import src.validation as sv

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

x, y = sf.load_dataset()

optimal = sv.load_optimal_hyperparameters()
print("Optimal hyperparameters:", optimal)

df_folds = ss.run_sensitivity(optimal, x, y)

df_summary = ss.summarize_sensitivity(df_folds, optimal)
df_overview = ss.summarize_overview(df_summary)
print(df_overview)

ss.plot_sensitivity(df_summary)
