import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import seaborn as sns 
import sys # Import sys for exiting on critical error

# Set up Matplotlib style
plt.style.use('ggplot')

# Load the data
try:
    df = pd.read_csv('dp_lr_results.csv')
except FileNotFoundError:
    print("Error: 'dp_lr_results.csv' not found. Check the file path.")
    sys.exit(1)

# --- DIAGNOSTIC STEP ---
print("--- Diagnostic Output ---")
print(f"Available Columns: {df.columns.tolist()}")
print("-------------------------")

# --- 1. Shared Data Prep ---
try:
    # --- FIX: Set names directly based on the 'Available Columns' output ---
    MSE_COL_NAME = 'MSE_test'
    R2_COL_NAME = 'R2_test'
    # ---------------------------------------------------------------------

    ols_data = df[df['Algorithm'] == 'OLS (non-private)'].iloc[0]
    OLS_MSE = ols_data[MSE_COL_NAME]
    OLS_R2 = ols_data[R2_COL_NAME]

except KeyError as e:
    print(f"\nFATAL ERROR: Failed to find column {e}. Please check the 'Available Columns' output above and manually update 'MSE_COL_NAME' and 'R2_COL_NAME' in the code.")
    sys.exit(1)

df_plot = df[df['Algorithm'] != 'OLS (non-private)'].copy()
df_runtime = df.groupby('Algorithm')['Runtime_sec'].mean().reset_index()
order_runtime = df_runtime.sort_values('Runtime_sec', ascending=False)['Algorithm'].tolist()

# The weights data prep is kept for code stability but figures based on it are removed.
def parse_weights(weights_str):
    """Safely parses the string representation of a list of weights."""
    try:
        cleaned_str = weights_str.strip().strip('"')
        return ast.literal_eval(cleaned_str)
    except Exception:
        return [np.nan] * 6 

df['Weights_List'] = df['Weights'].apply(parse_weights)
features = ['Intercept', 'Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5']
df[features] = pd.DataFrame(df['Weights_List'].tolist(), index=df.index)

# Define the two data subsets for Figure 7.4 and 7.5
df_op = df_plot[df_plot['Algorithm'] == 'DP-OutputPerturbation'].copy()
df_other_dp = df_plot[df_plot['Algorithm'] != 'DP-OutputPerturbation'].copy()

# --- 3. Generate Matplotlib Figures (Sequential Naming) ---

# Figure 7.1: MSE vs. Privacy Budget (Epsilon)
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_plot, x='epsilon', y=MSE_COL_NAME, hue='Algorithm', marker='o')
plt.axhline(y=OLS_MSE, color='r', linestyle='--', label=f'OLS Baseline ({OLS_MSE:.2f})')
plt.xscale('log')
plt.title('Figure 7.1: Mean Squared Error (MSE) vs. Privacy Budget (Epsilon)', fontsize=14)
plt.xlabel('Privacy Budget (Epsilon) [Log Scale]', fontsize=12)
plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
plt.legend(title='DP Mechanism')
plt.grid(True, which="both", ls="--", c='0.7')
plt.tight_layout()
plt.savefig('figure_7_1_mse.png')
plt.close()

# Figure 7.2: R^2 Score vs. Privacy Budget (Epsilon)
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_plot, x='epsilon', y=R2_COL_NAME, hue='Algorithm', marker='o')
plt.axhline(y=OLS_R2, color='r', linestyle='--', label=f'OLS Baseline ({OLS_R2:.4f})')
plt.xscale('log')
plt.title('Figure 7.2: R^2 Score vs. Privacy Budget (Epsilon)', fontsize=14)
plt.xlabel('Privacy Budget (Epsilon) [Log Scale]', fontsize=12)
plt.ylabel('R^2 Score', fontsize=12)
plt.legend(title='DP Mechanism')
plt.grid(True, which="both", ls="--", c='0.7')
plt.tight_layout()
plt.savefig('figure_7_2_r2.png')
plt.close()

# Figure 7.3: Comparison of Average Execution Time (Efficiency)
plt.figure(figsize=(8, 5))
sns.barplot(data=df_runtime, x='Algorithm', y='Runtime_sec', order=order_runtime, palette='viridis')
plt.title('Figure 7.3: Comparison of Average Execution Time (Efficiency)', fontsize=14)
plt.xlabel('Algorithm', fontsize=12)
plt.ylabel('Average Execution Time (s)', fontsize=12)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('figure_7_3_runtime.png')
plt.close()

# Figure 7.4 : MSE: DP Mechanisms (Excluding Output Perturbation) vs. OLS
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_other_dp, x='epsilon', y=MSE_COL_NAME, hue='Algorithm', marker='o')
plt.axhline(y=OLS_MSE, color='r', linestyle='--', label=f'OLS Baseline ({OLS_MSE:.2f})')
plt.xscale('log')
plt.title('Figure 7.4: MSE: DP Mechanisms (Excluding Output Perturbation) vs. OLS', fontsize=14)
plt.xlabel('Privacy Budget (Epsilon) [Log Scale]', fontsize=12)
plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
plt.legend(title='DP Mechanism')
plt.grid(True, which="both", ls="--", c='0.7')
plt.tight_layout()
plt.savefig('figure_7_4_other_dp_mse.png')
plt.close()

# Figure 7.5 : R^2 Score: DP Mechanisms (Excluding Output Perturbation) vs. OLS
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_other_dp, x='epsilon', y=R2_COL_NAME, hue='Algorithm', marker='o')
plt.axhline(y=OLS_R2, color='r', linestyle='--', label=f'OLS Baseline ({OLS_R2:.4f})')
plt.xscale('log')
plt.title('Figure 7.5: R^2 Score: DP Mechanisms (Excluding Output Perturbation) vs. OLS', fontsize=14)
plt.xlabel('Privacy Budget (Epsilon) [Log Scale]', fontsize=12)
plt.ylabel('R^2 Score', fontsize=12)
plt.legend(title='DP Mechanism', loc='lower right')
plt.grid(True, which="both", ls="--", c='0.7')
plt.tight_layout()
plt.savefig('figure_7_5_other_dp_r2.png')
plt.close()