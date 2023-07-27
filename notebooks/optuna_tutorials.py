# %%
import optuna
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
# %% [markdown]
# # Optuna Tutorial
# ## 1. Basics 
# [Link](https://optuna.readthedocs.io/en/stable/tutorial/) to the tutorial. 
# * **Trial**: A single call of the objective function
# * **Study**: An optimization session, which is a set of trials
# * **Parameter**: A variable whose value is to be optimized, such as x in the above example
#
# When calling `.optimize()` for the same study object, we continue the same study and number of trials increases.
# %%

def generate_random_polynomial(max_val=-5, min_val=5, order=4):
    """
    Args:
        max_val, min_val: minimal and maximal coeficient value
        order: order of polynomial. If 1, then the output is constant.
    
    """
    coefs = np.random.uniform(min_val, max_val, size=order)

    def polynomial(x):
        return sum([a * x ** i for i, a in enumerate(coefs)])

    return polynomial
# %%
# These limits are used both for plotting and for search
min_x, max_x = -20, 20

polynomial = generate_random_polynomial(order=20)

def objective(trial):
    x = trial.suggest_float("x", min_x, max_x)
    return polynomial(x)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

best_params = study.best_params
found_x = best_params["x"]

x = np.arange(min_x, max_x)
plt.figure(figsize=(10, 5))
plt.scatter(x, polynomial(x), color="black")

x = np.arange(min_x, max_x, 3)
plt.plot(x, polynomial(x), color="black")

plt.axvline(found_x, color="turquoise", lw=3)
plt.xlabel("x")
plt.ylabel("f(x)")
# %%
study.trials
for trial in study.trials[:2]:  # Show first two trials
    print(trial)
# %% [markdown]
# ## 2. xgboost for [diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)
# 
# %%
from sklearn.datasets import load_diabetes
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice
# %%
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)
# %%
def objective(trial):
    max_depth = trial.suggest_int("max_depth", 2, 7)
    tree_method = trial.suggest_categorical("tree_method", ["auto", "exact", "approx", "hist"])
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.01, 1, step=0.01)
    alpha = trial.suggest_float("alpha", 0, 10, step=1)

    param = {
        'max_depth': max_depth,
        "tree_method": tree_method,
        "colsample_bytree": colsample_bytree,
        "alpha": alpha,
        # 'eta': 1,
        'objective': 'reg:squarederror',
        # "learning_rate": 0.1,
        # "n_estimators": 1000,
    }
    bst = xgb.train(param, dtrain)
    ypred = bst.predict(dtest)

    return mean_squared_error(y_test, ypred)
# %%
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=500)
# %%
study.best_value
# %%
plot_contour(study)
# %%
plot_optimization_history(study)
# %%
plot_parallel_coordinate(study)
# %%
