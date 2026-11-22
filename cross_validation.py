# =====================================================================
# 4-fold cross-validation of the optimal network (section 5-1)
# Reads the best hyperparameter set from the grid-search ranking table,
# runs 4-fold CV, saves the tables, draws the 2 x 2 loss figure and
# checks the epoch budget by repeating the CV with twice as many epochs.
# =====================================================================

import src.functions as sf
import src.validation as sv

x, y = sf.load_dataset()

best = sv.load_optimal_hyperparameters()
print("Optimal hyperparameters:", best)

df_summary = sv.run_optimal_cv(best, x, y)
print(df_summary[["target", "RMSE_mean", "RMSE_std", "NRMSE_mean", "NRMSE_std", "fit_time_s"]])

sv.plot_cv_loss_curves()

# Does training finish within the epoch budget? Repeat the CV with twice as many epochs.
df_check = sv.run_epoch_check(best, x, y)
print(df_check)
