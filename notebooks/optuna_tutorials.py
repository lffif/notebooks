# %%
import optuna
import numpy as np
import matplotlib.pyplot as plt
# %% [markdown]
# # Optuna Tutorial
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
# %%