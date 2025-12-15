# import numpy as np
# import pandas as pd
# from sklearn.metrics import mean_squared_error, r2_score
# import time


# df = pd.read_excel("Clean_Salary_Dataset.xlsx")

# print("First 5 rows of the dataset:")
# print(df.head())
# print("\nColumns:", df.columns.tolist())


# df_encoded = pd.get_dummies(df, columns=["Education_Level"], drop_first=True)


# feature_cols = [col for col in df_encoded.columns
#                 if col not in ["Employee_ID", "Salary_$1000"]]

# X_raw = df_encoded[feature_cols].values  
# y = df_encoded["Salary_$1000"].values    

# # Add intercept term as first column (bias)
# n_samples, n_features = X_raw.shape
# X = np.hstack([np.ones((n_samples, 1)), X_raw])  

# feature_names = ["Intercept"] + feature_cols

# print("\nNumber of samples:", X.shape[0])
# print("Number of features (including intercept):", X.shape[1])



# def ols_solve_matrix(X, y, ridge=1e-6):
#     """
#     Solve OLS using the normal equation:
#         (X^T X + ridge * I) w = X^T y
#     Returns the weight vector w (including intercept, if intercept column is in X).
#     """
#     X = np.asarray(X, dtype=float)
#     y = np.asarray(y, dtype=float)
#     n_samples, n_features = X.shape

#     A = X.T @ X + ridge * np.eye(n_features)  # X^T X + λI
#     b = X.T @ y                               # X^T y

#     w = np.linalg.solve(A, b)
#     return w

# def evaluate_weights(X, y, w):
#     """
#     Compute predictions, MSE and R^2 given X, y and weights w.
#     """
#     y_hat = X @ w
#     mse = mean_squared_error(y, y_hat)
#     r2 = r2_score(y, y_hat)
#     return mse, r2



# t0 = time.perf_counter()
# w_ols = ols_solve_matrix(X, y)
# t_ols = time.perf_counter() - t0

# mse_ols, r2_ols = evaluate_weights(X, y, w_ols)

# print("\n=== OLS (Non-private) Baseline ===")
# print("MSE:", mse_ols)
# print("R^2:", r2_ols)
# print("Runtime (seconds):", t_ols)

# print("\nOLS Weights:")
# for name, val in zip(feature_names, w_ols):
#     print(f"  {name}: {val}")



# rng = np.random.default_rng(42)  

# def laplace_noise(scale, size):
#     """
#     Draw Laplace(0, scale) noise.
#     scale = sensitivity / epsilon (simplified for demo).
#     """
#     return rng.laplace(loc=0.0, scale=scale, size=size)



# def dp_functional_mechanism(X, y, epsilon, ridge=1e-3,
#                             delta_A=1.0, delta_b=1.0, delta_c=1.0):
#     """
#     Approximate Functional Mechanism for linear regression.
#     Noise is added to A = X^T X, b = X^T y and c = y^T y (though c is not used in solving).
#     Sensitivities delta_A, delta_b, delta_c are simplified for demonstration.
#     """
#     X = np.asarray(X, dtype=float)
#     y = np.asarray(y, dtype=float)
#     d = X.shape[1]

#     A = X.T @ X        
#     b = X.T @ y       
#     c = y.T @ y        

#     # Add Laplace noise to A, b, c
#     A_tilde = A + laplace_noise(delta_A / epsilon, size=A.shape)
#     b_tilde = b + laplace_noise(delta_b / epsilon, size=b.shape)
#     c_tilde = c + laplace_noise(delta_c / epsilon, size=())  

    
#     A_tilde = A_tilde + ridge * np.eye(d)

   
#     try:
#         w = np.linalg.solve(A_tilde, b_tilde)
#     except np.linalg.LinAlgError:
#         w, *_ = np.linalg.lstsq(A_tilde, b_tilde, rcond=None)
#     return w

# def dp_noisy_stats(X, y, epsilon, ridge=1e-3,
#                    delta_A=1.0, delta_b=1.0):
    
#     X = np.asarray(X, dtype=float)
#     y = np.asarray(y, dtype=float)
#     d = X.shape[1]

#     A = X.T @ X
#     b = X.T @ y

#     A_tilde = A + laplace_noise(delta_A / epsilon, size=A.shape)
#     b_tilde = b + laplace_noise(delta_b / epsilon, size=b.shape)

#     A_tilde = A_tilde + ridge * np.eye(d)

#     try:
#         w = np.linalg.solve(A_tilde, b_tilde)
#     except np.linalg.LinAlgError:
#         w, *_ = np.linalg.lstsq(A_tilde, b_tilde, rcond=None)
#     return w

# def dp_output_perturbation(X, y, epsilon, scale_coeff=0.5):
   
#     w = ols_solve_matrix(X, y)
#     noise = laplace_noise(scale_coeff / epsilon, size=w.shape)
#     w_noisy = w + noise
#     return w_noisy


# epsilons = [0.1, 0.5, 1, 2, 5, 10]

# results = []


# results.append({
#     "Algorithm": "OLS (non-private)",
#     "epsilon": None,
#     "MSE": mse_ols,
#     "R2": r2_ols,
#     "Runtime_sec": t_ols,
#     "Weights": w_ols.tolist()
# })

# for eps in epsilons:
   
#     t0 = time.perf_counter()
#     w_fm = dp_functional_mechanism(X, y, epsilon=eps)
#     t_fm = time.perf_counter() - t0
#     mse_fm, r2_fm = evaluate_weights(X, y, w_fm)

#     results.append({
#         "Algorithm": "DP-FunctionalMechanism",
#         "epsilon": eps,
#         "MSE": mse_fm,
#         "R2": r2_fm,
#         "Runtime_sec": t_fm,
#         "Weights": w_fm.tolist()
#     })

    
#     t0 = time.perf_counter()
#     w_ns = dp_noisy_stats(X, y, epsilon=eps)
#     t_ns = time.perf_counter() - t0
#     mse_ns, r2_ns = evaluate_weights(X, y, w_ns)

#     results.append({
#         "Algorithm": "DP-NoisyStats",
#         "epsilon": eps,
#         "MSE": mse_ns,
#         "R2": r2_ns,
#         "Runtime_sec": t_ns,
#         "Weights": w_ns.tolist()
#     })

    
#     t0 = time.perf_counter()
#     w_op = dp_output_perturbation(X, y, epsilon=eps, scale_coeff=0.5)
#     t_op = time.perf_counter() - t0
#     mse_op, r2_op = evaluate_weights(X, y, w_op)

#     results.append({
#         "Algorithm": "DP-OutputPerturbation",
#         "epsilon": eps,
#         "MSE": mse_op,
#         "R2": r2_op,
#         "Runtime_sec": t_op,
#         "Weights": w_op.tolist()
#     })


# results_df = pd.DataFrame(results)
# print("\n=== Summary of Results ===")
# print(results_df)

# results_df.to_csv("dp_lr_results.csv", index=False)
# print("\nResults saved to dp_lr_results.csv")


import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split  # NEW
import time

df = pd.read_excel("Clean_Salary_Dataset.xlsx")

print("First 5 rows of the dataset:")
print(df.head())
print("\nColumns:", df.columns.tolist())

df_encoded = pd.get_dummies(df, columns=["Education_Level"], drop_first=True)

feature_cols = [col for col in df_encoded.columns
                if col not in ["Employee_ID", "Salary_$1000"]]

X_raw = df_encoded[feature_cols].values
y = df_encoded["Salary_$1000"].values

# Add intercept term as first column (bias)
n_samples, n_features = X_raw.shape
X = np.hstack([np.ones((n_samples, 1)), X_raw])

feature_names = ["Intercept"] + feature_cols

print("\nNumber of samples:", X.shape[0])
print("Number of features (including intercept):", X.shape[1])

# ========= NEW: Train–Test Split =========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)
print("\nTrain samples:", X_train.shape[0])
print("Test samples :", X_test.shape[0])
# =========================================

def ols_solve_matrix(X, y, ridge=1e-6):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n_samples, n_features = X.shape

    A = X.T @ X + ridge * np.eye(n_features)  # X^T X + λI
    b = X.T @ y                               # X^T y

    w = np.linalg.solve(A, b)
    return w

def evaluate_weights(X, y, w):
    y_hat = X @ w
    mse = mean_squared_error(y, y_hat)
    r2 = r2_score(y, y_hat)
    return mse, r2

# ===== Baseline OLS: train on TRAIN, test on TEST =====
t0 = time.perf_counter()
w_ols = ols_solve_matrix(X_train, y_train)
t_ols = time.perf_counter() - t0

mse_ols, r2_ols = evaluate_weights(X_test, y_test, w_ols)

print("\n=== OLS (Non-private) Baseline ===")
print("MSE (test):", mse_ols)
print("R^2 (test):", r2_ols)
print("Runtime (seconds):", t_ols)

print("\nOLS Weights:")
for name, val in zip(feature_names, w_ols):
    print(f"  {name}: {val}")

rng = np.random.default_rng(42)

def laplace_noise(scale, size):
    return rng.laplace(loc=0.0, scale=scale, size=size)

def dp_functional_mechanism(X, y, epsilon, ridge=1e-3,
                            delta_A=1.0, delta_b=1.0, delta_c=1.0):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    d = X.shape[1]

    A = X.T @ X
    b = X.T @ y
    c = y.T @ y

    A_tilde = A + laplace_noise(delta_A / epsilon, size=A.shape)
    b_tilde = b + laplace_noise(delta_b / epsilon, size=b.shape)
    c_tilde = c + laplace_noise(delta_c / epsilon, size=())  # not used further

    A_tilde = A_tilde + ridge * np.eye(d)

    try:
        w = np.linalg.solve(A_tilde, b_tilde)
    except np.linalg.LinAlgError:
        w, *_ = np.linalg.lstsq(A_tilde, b_tilde, rcond=None)
    return w

def dp_noisy_stats(X, y, epsilon, ridge=1e-3,
                   delta_A=1.0, delta_b=1.0):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    d = X.shape[1]

    A = X.T @ X
    b = X.T @ y

    A_tilde = A + laplace_noise(delta_A / epsilon, size=A.shape)
    b_tilde = b + laplace_noise(delta_b / epsilon, size=b.shape)

    A_tilde = A_tilde + ridge * np.eye(d)

    try:
        w = np.linalg.solve(A_tilde, b_tilde)
    except np.linalg.LinAlgError:
        w, *_ = np.linalg.lstsq(A_tilde, b_tilde, rcond=None)
    return w

def dp_output_perturbation(X, y, epsilon, scale_coeff=0.5):
    w = ols_solve_matrix(X, y)
    noise = laplace_noise(scale_coeff / epsilon, size=w.shape)
    w_noisy = w + noise
    return w_noisy

epsilons = [0.1, 0.5, 1, 2, 5, 10]
results = []

# Baseline row (on TEST set)
results.append({
    "Algorithm": "OLS (non-private)",
    "epsilon": None,
    "MSE_test": mse_ols,
    "R2_test": r2_ols,
    "Runtime_sec": t_ols,
    "Weights": w_ols.tolist()
})

for eps in epsilons:
    # Functional Mechanism: train on TRAIN, evaluate on TEST
    t0 = time.perf_counter()
    w_fm = dp_functional_mechanism(X_train, y_train, epsilon=eps)
    t_fm = time.perf_counter() - t0
    mse_fm, r2_fm = evaluate_weights(X_test, y_test, w_fm)

    results.append({
        "Algorithm": "DP-FunctionalMechanism",
        "epsilon": eps,
        "MSE_test": mse_fm,
        "R2_test": r2_fm,
        "Runtime_sec": t_fm,
        "Weights": w_fm.tolist()
    })

    # Noisy Sufficient Statistics
    t0 = time.perf_counter()
    w_ns = dp_noisy_stats(X_train, y_train, epsilon=eps)
    t_ns = time.perf_counter() - t0
    mse_ns, r2_ns = evaluate_weights(X_test, y_test, w_ns)

    results.append({
        "Algorithm": "DP-NoisyStats",
        "epsilon": eps,
        "MSE_test": mse_ns,
        "R2_test": r2_ns,
        "Runtime_sec": t_ns,
        "Weights": w_ns.tolist()
    })

    # Output Perturbation
    t0 = time.perf_counter()
    w_op = dp_output_perturbation(X_train, y_train, epsilon=eps, scale_coeff=0.5)
    t_op = time.perf_counter() - t0
    mse_op, r2_op = evaluate_weights(X_test, y_test, w_op)

    results.append({
        "Algorithm": "DP-OutputPerturbation",
        "epsilon": eps,
        "MSE_test": mse_op,
        "R2_test": r2_op,
        "Runtime_sec": t_op,
        "Weights": w_op.tolist()
    })

results_df = pd.DataFrame(results)
print("\n=== Summary of Results (Test Set) ===")
print(results_df)

results_df.to_csv("dp_lr_results.csv", index=False)
print("\nResults saved to dp_lr_results.csv")