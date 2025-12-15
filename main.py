import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # NEW: For data scaling/normalization
import time
import warnings
# Ignore warnings about np.linalg.lstsq rcond=None usage
warnings.filterwarnings("ignore") 

# --- FUNCTIONS (Fully defined, including the previously missing dp_noisy_stats) ---

def ols_solve_matrix(X, y, ridge=1e-6):
    """Solve OLS using the normal equation."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n_samples, n_features = X.shape
    A = X.T @ X + ridge * np.eye(n_features)
    b = X.T @ y
    w = np.linalg.solve(A, b)
    return w

def evaluate_weights(X, y, w):
    """Compute predictions, MSE and R^2."""
    y_hat = X @ w
    mse = mean_squared_error(y, y_hat)
    r2 = r2_score(y, y_hat)
    return mse, r2

# Removed the fixed seed line (e.g., rng = np.random.default_rng(42))
# This allows the noise to be truly random in each iteration of the outer loop for Expected Utility.
rng = np.random.default_rng() 

def laplace_noise(scale, size):
    """Draw Laplace(0, scale) noise."""
    return rng.laplace(loc=0.0, scale=scale, size=size)

def dp_functional_mechanism(X, y, epsilon, ridge=1e-3, delta_A=1.0, delta_b=1.0, delta_c=1.0):
    """Differential Private Functional Mechanism."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    d = X.shape[1]
    A = X.T @ X
    b = X.T @ y
    c = y.T @ y
    A_tilde = A + laplace_noise(delta_A / epsilon, size=A.shape)
    b_tilde = b + laplace_noise(delta_b / epsilon, size=b.shape)
    c_tilde = c + laplace_noise(delta_c / epsilon, size=())
    A_tilde = A_tilde + ridge * np.eye(d)
    try:
        w = np.linalg.solve(A_tilde, b_tilde)
    except np.linalg.LinAlgError:
        w, *_ = np.linalg.lstsq(A_tilde, b_tilde, rcond=None)
    return w

def dp_noisy_stats(X, y, epsilon, ridge=1e-3, delta_A=1.0, delta_b=1.0):
    """Differential Private Noisy Sufficient Statistics (NSS) - RE-ADDED."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    d = X.shape[1]

    A = X.T @ X
    b = X.T @ y

    # Add Laplace noise to the sufficient statistics
    A_tilde = A + laplace_noise(delta_A / epsilon, size=A.shape)
    b_tilde = b + laplace_noise(delta_b / epsilon, size=b.shape)

    A_tilde = A_tilde + ridge * np.eye(d)

    try:
        w = np.linalg.solve(A_tilde, b_tilde)
    except np.linalg.LinAlgError:
        w, *_ = np.linalg.lstsq(A_tilde, b_tilde, rcond=None)
    return w


def dp_output_perturbation(X, y, epsilon, scale_coeff=0.5):
    """Differential Private Output Perturbation."""
    w = ols_solve_matrix(X, y)
    noise = laplace_noise(scale_coeff / epsilon, size=w.shape)
    w_noisy = w + noise
    return w_noisy

# --- DATA LOADING AND PREPARATION (MODIFIED FOR SCALING) ---

df = pd.read_excel("Clean_Salary_Dataset.xlsx")
df_encoded = pd.get_dummies(df, columns=["Education_Level"], drop_first=True)

feature_cols = [col for col in df_encoded.columns
                if col not in ["Employee_ID", "Salary_$1000"]]

X_raw = df_encoded[feature_cols].values
y = df_encoded["Salary_$1000"].values

# Add intercept term as first column (bias)
n_samples, n_features = X_raw.shape
X = np.hstack([np.ones((n_samples, 1)), X_raw])

feature_names = ["Intercept"] + feature_cols

# Train–Test Split (keeping the split random_state for data consistency)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# =========================================================
# NEW: Data Scaling for Numerical Stability
# =========================================================

# Separate intercept column (index 0) as it should not be scaled
X_train_features = X_train[:, 1:] 
X_test_features = X_test[:, 1:]

scaler_X = StandardScaler()
X_train_scaled_features = scaler_X.fit_transform(X_train_features)
X_test_scaled_features = scaler_X.transform(X_test_features)

# Recombine with the unscaled intercept column
X_train_scaled = np.hstack([X_train[:, :1], X_train_scaled_features])
X_test_scaled = np.hstack([X_test[:, :1], X_test_scaled_features])

# Scale the target variable y (crucial for stabilizing the noise application)
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

print("\n--- Running with Scaled Data ---")

# ===== Baseline OLS (Using Scaled Data) =====
t0 = time.perf_counter()
# Train on SCALED training data
w_ols = ols_solve_matrix(X_train_scaled, y_train_scaled)
t_ols = time.perf_counter() - t0
# Evaluate on SCALED test data
mse_ols, r2_ols = evaluate_weights(X_test_scaled, y_test_scaled, w_ols)

print("\n=== OLS (Non-private) Baseline (Scaled) ===")
print("MSE (test):", mse_ols)
print("R^2 (test):", r2_ols)


# --- AVERAGING LOOP (EXPECTED UTILITY CALCULATION) ---

epsilons = [0.1, 0.5, 1, 2, 5, 10]
NUM_TRIALS = 20 # Number of runs to average
all_trial_results = []

print(f"\n--- Running {NUM_TRIALS} Trials for Expected Utility ---")

# Outer loop to calculate average performance (Expected Utility)
for trial in range(NUM_TRIALS): 
    for eps in epsilons:
        
        # DP-FunctionalMechanism (using scaled data)
        w_fm = dp_functional_mechanism(X_train_scaled, y_train_scaled, epsilon=eps)
        mse_fm, r2_fm = evaluate_weights(X_test_scaled, y_test_scaled, w_fm)

        all_trial_results.append({
            "Algorithm": "DP-FunctionalMechanism",
            "epsilon": eps,
            "MSE": mse_fm,
            "R2": r2_fm,
            "Trial": trial
        })
        
        # DP-NoisyStats (using scaled data) - RE-ADDED
        w_ns = dp_noisy_stats(X_train_scaled, y_train_scaled, epsilon=eps)
        mse_ns, r2_ns = evaluate_weights(X_test_scaled, y_test_scaled, w_ns)

        all_trial_results.append({
            "Algorithm": "DP-NoisyStats",
            "epsilon": eps,
            "MSE": mse_ns,
            "R2": r2_ns,
            "Trial": trial
        })

        # DP-OutputPerturbation (using scaled data)
        w_op = dp_output_perturbation(X_train_scaled, y_train_scaled, epsilon=eps, scale_coeff=0.5)
        mse_op, r2_op = evaluate_weights(X_test_scaled, y_test_scaled, w_op)

        all_trial_results.append({
            "Algorithm": "DP-OutputPerturbation",
            "epsilon": eps,
            "MSE": mse_op,
            "R2": r2_op,
            "Trial": trial
        })

# --- FINAL SUMMARY (Averaging) ---

all_results_df = pd.DataFrame(all_trial_results)

# Calculate the average R2 and MSE (Expected Utility)
summary_df = all_results_df.groupby(['Algorithm', 'epsilon']).agg(
    Avg_MSE=('MSE', 'mean'),
    Avg_R2=('R2', 'mean'),
    StdDev_R2=('R2', 'std') # Standard deviation measures the instability/variance
).reset_index()

# Add the OLS baseline to the summary
summary_df = pd.concat([
    pd.DataFrame([{
        "Algorithm": "OLS (non-private)",
        "epsilon": None,
        "Avg_MSE": mse_ols,
        "Avg_R2": r2_ols,
        "StdDev_R2": 0.0
    }]),
    summary_df
]).sort_values(by=['Algorithm', 'epsilon'], ascending=[True, True])


print("\n=== SUMMARY OF EXPECTED UTILITY (Average of 20 Runs) ===")
print(summary_df)

summary_df.to_csv("dp_lr_expected_utility.csv", index=False)
print("\nAverage results saved to dp_lr_expected_utility.csv")