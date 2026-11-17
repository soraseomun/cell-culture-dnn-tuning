import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import root_mean_squared_error
from src.constants import *


df = pd.read_csv("datasets/dataset_cell_growth_7factor.csv")
x = df.loc[:, FEATURE_COLUMNS].to_numpy()
y = df.loc[:, TARGET_COLUMNS].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

degrees = np.array ([1, 2, 3, 4, 5, 6])
records = []


for degree in degrees:
    model = make_pipeline(StandardScaler(),
                      PolynomialFeatures(degree=degree, include_bias=False), LinearRegression())
    t_start = time.perf_counter()

    model.fit(x_train , y_train)

    fit_time = time.perf_counter() - t_start

    y_test_pred = model.predict(x_test)

    iqr = np.quantile(y_test, q=0.75, axis=0) - np.quantile(y_test, q=0.25, axis=0)
    for j, target in enumerate(TARGET_COLUMNS):
        rmse = root_mean_squared_error(y_test[:, j], y_test_pred[:, j])
        nrmse = rmse / iqr[j]
        records.append((degree, target, rmse, nrmse, fit_time))

columns = ["degree", "target", "RMSE", "NRMSE", "fit_time_s"]
df_results = pd.DataFrame.from_records(records, columns = columns)
df_results.to_csv(os.path.join(TABLES_DIR, "table_polynomial_search.csv"), index = False)

mean_nrmse = []
for degree in degrees:
    mask = df_results.loc[:, "degree"] == degree
    mean_nrmse.append(np.mean(df_results.loc[mask, "NRMSE"]))
mean_nrmse = np.array(mean_nrmse)

best_x = np.argmax(mean_nrmse == np.min(mean_nrmse))
best_degree = degrees[best_x]
print(f"Best degree: {best_degree}")

fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=200, layout="constrained")
for target in TARGET_COLUMNS:
    mask = df_results.loc[:, "target"] == target
    ax.plot(df_results.loc[mask, "degree"], df_results.loc[mask, "NRMSE"], marker = "o", label=target)

ax.set_xlabel("Polynomial degree")
ax.set_ylabel("NRMSE")
ax.legend()
fig.savefig(os.path.join(FIGURES_DIR, "figure_polynomial_degree.png"))
plt.close(fig)

model = make_pipeline(StandardScaler(),
                      PolynomialFeatures(degree=best_degree, include_bias=False), LinearRegression())
records_cv =[]
for j, target in enumerate(TARGET_COLUMNS):
    cv_results = cross_validate(model, x, y[:, j],cv=4, scoring="neg_root_mean_squared_error")

    rmse = - cv_results["test_score"]
    iqr = np.quantile(y[:,j], q=0.75) - np.quantile(y[:, j], q=0.25)
    nrmse = rmse/ iqr
    fit_time_s = np.mean(cv_results["fit_time"])
    records_cv.append((best_degree, target,np.mean(rmse), np.std(rmse), np.mean(nrmse), np.std(nrmse),fit_time_s))

columns_cv = ["degree", "target", "RMSE_mean","RMSE_std", "NRMSE_mean", "NRMSE_std", "fit_time_s"]
df_cv = pd.DataFrame.from_records(records_cv, columns=columns_cv)
df_cv.to_csv(os.path.join(TABLES_DIR, "table_polynomial_cv.csv"), index = False)
print(df_cv)





