# =====================================================================
# Polynomial regression comparison
# =====================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import root_mean_squared_error
from src.constants import *

#Load the dataset
df = pd.read_csv("datasets/dataset_cell_growth_7factor.csv")

#Separate the input features and the target variables
x = df.loc[:, FEATURE_COLUMNS].to_numpy()
y = df.loc[:, TARGET_COLUMNS].to_numpy()

#Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

#Polynomial degrees to evaluate
degrees = np.array ([1, 2, 3, 4, 5, 6])
records = []

#Train and evaluate a polynomial regression model for each degree
for degree in degrees:
    model = make_pipeline(StandardScaler(),
                      PolynomialFeatures(degree=degree, include_bias=False), LinearRegression())

    #Measure the training time
    #Train the model on the training set

    model.fit(x_train , y_train)

    #Predict the test set targets
    y_test_pred = model.predict(x_test)
    #Calculate the interquartile range of each target
    iqr = np.quantile(y_test, q=0.75, axis=0) - np.quantile(y_test, q=0.25, axis=0)

    #Calculate RMSE and NRMSE for each target
    for j, target in enumerate(TARGET_COLUMNS):
        rmse = root_mean_squared_error(y_test[:, j], y_test_pred[:, j])
        nrmse = rmse / iqr[j]
        records.append((degree, target, rmse, nrmse))

#Save the results of the degree search
columns = ["degree", "target", "RMSE", "NRMSE"]
df_results = pd.DataFrame.from_records(records, columns = columns)
df_results.to_csv(os.path.join(TABLES_DIR, "table_polynomial_search.csv"), index = False)

#Calculate the mean NRMSE across targets for each degree
mean_nrmse = []
for degree in degrees:
    mask = df_results.loc[:, "degree"] == degree
    mean_nrmse.append(np.mean(df_results.loc[mask, "NRMSE"]))
mean_nrmse = np.array(mean_nrmse)

#Find the degree with the lowest mean NRMSE
best_x = np.argmax(mean_nrmse == np.min(mean_nrmse))
best_degree = degrees[best_x]
print(f"Best degree: {best_degree}")

#Plot NRMSE against polynomial degree for each target
fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=200, layout="constrained")
for target in TARGET_COLUMNS:
    mask = df_results.loc[:, "target"] == target
    ax.plot(df_results.loc[mask, "degree"], df_results.loc[mask, "NRMSE"], marker = "o", label=target)

ax.set_xlabel("Polynomial degree")
ax.set_ylabel("NRMSE")
ax.legend()
fig.savefig(os.path.join(FIGURES_DIR, "figure_polynomial_degree.png"))
plt.close(fig)

#Build the model with the best polynomial degree
model = make_pipeline(StandardScaler(),
                      PolynomialFeatures(degree=best_degree, include_bias=False), LinearRegression())

#Run 4-fold cross-validation for each target
records_cv =[]
for j, target in enumerate(TARGET_COLUMNS):
    cv_results = cross_validate(model, x, y[:, j],cv=4, scoring="neg_root_mean_squared_error")

    #Convert the negative RMSE scores to positive values
    rmse = - cv_results["test_score"]

    #Normalize RMSE by the IQR of the target
    iqr = np.quantile(y[:,j], q=0.75) - np.quantile(y[:, j], q=0.25)
    nrmse = rmse/ iqr

    #Mean training time across the 4 folds
    fit_time_s = np.mean(cv_results["fit_time"])
    records_cv.append((best_degree, target,np.mean(rmse), np.std(rmse), np.mean(nrmse), np.std(nrmse),fit_time_s))

#Save the cross-validation results
columns_cv = ["degree", "target", "RMSE_mean","RMSE_std", "NRMSE_mean", "NRMSE_std", "fit_time_s"]
df_cv = pd.DataFrame.from_records(records_cv, columns=columns_cv)
df_cv.to_csv(os.path.join(TABLES_DIR, "table_polynomial_cv.csv"), index = False)
print(df_cv)

#Load the cross-validation results of both models
df_poly = pd.read_csv(os.path.join(TABLES_DIR, "table_polynomial_cv.csv"))
df_nn = pd.read_csv(os.path.join(TABLES_DIR, "table_cv_optimal_summary.csv"))

#Compare the NRMSE of both models for each target
records_comp =[]
for target in TARGET_COLUMNS:
    mask_poly = df_poly.loc[:, "target"] == target
    mask_nn = df_nn.loc[:, "target"] == target
    poly_mean = df_poly.loc[mask_poly, "NRMSE_mean"].to_numpy()[0]
    poly_std = df_poly.loc[mask_poly, "NRMSE_std"].to_numpy()[0]
    nn_mean = df_nn.loc[mask_nn, "NRMSE_mean"].to_numpy()[0]
    nn_std = df_nn.loc[mask_nn, "NRMSE_std"].to_numpy()[0]
    records_comp.append((target,poly_mean,poly_std,nn_mean,nn_std))

#Save the comparison table
columns_comp = ["target", "poly_NRMSE_mean", "poly_NRMSE_std", "nn_NRMSE_mean", "nn_NRMSE_std"]
df_comp = pd.DataFrame.from_records(records_comp, columns=columns_comp)
df_comp.to_csv(os.path.join(TABLES_DIR, "table_comparison_poly_vs_nn.csv"), index = False)
print(df_comp)

#Compare the mean training time of both models
poly_time = np.mean(df_poly.loc[:, "fit_time_s"])
nn_time = np.mean(df_nn.loc[:, "fit_time_s"])
print(f"Polynomial fit time: {poly_time:.2f} s")
print(f"Neural network fit time: {nn_time:.2f} s")





