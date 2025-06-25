import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

# === Config ===
input_cols = [
    'aspirate_speed', 'dispense_speed', 'aspirate_wait_time', 'dispense_wait_time',
    'retract_speed', 'pre_asp_air_vol', 'post_asp_air_vol', 'blowout_vol'
]
output_targets = ['time', 'deviation', 'variability']
save_folder = r"C:\Users\owenm\OneDrive\Desktop\North Robotics\utoronto_demo\utoronto_demo\output\experiment_calibration_20250624_210105"

os.makedirs(save_folder, exist_ok=True)

# === Load CSV ===
df = pd.read_csv(
    r'C:\Users\owenm\OneDrive\Desktop\North Robotics\utoronto_demo\utoronto_demo\output\experiment_calibration_20250624_210105\experiment_summary.csv'
)

# === Standardize inputs ===
X = df[input_cols]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === SHAP Analysis for All Targets ===
shap_dfs = []

for target in output_targets:
    y = df[target]

    # Train model
    model = xgb.XGBRegressor()
    model.fit(X_scaled, y)

    # Use TreeExplainer to avoid GPU dependency
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    # Save summary plot
    shap.summary_plot(shap_values, X, show=False)
    plt.title(f'SHAP Summary for {target}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f'shap_summary_{target}.png'))
    plt.clf()

    # Store mean SHAP values with Feature as index
    mean_shap = pd.DataFrame({
        'Feature': input_cols,
        target: np.abs(shap_values).mean(axis=0)
    }).set_index("Feature")

    shap_dfs.append(mean_shap)

# === Combine and Normalize SHAP Importances ===
combined = pd.concat(shap_dfs, axis=1)
normalized = combined / combined.max()

# === Plot Normalized Comparison ===
normalized.plot(kind='barh', figsize=(10, 6))
plt.xlabel('Normalized Mean SHAP Value (0–1)')
plt.title('Normalized Feature Importance Comparison Across Targets')
plt.grid(True, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(save_folder, 'shap_comparison_all_targets_normalized.png'))
plt.clf()

# Define weights (adjust if needed)
weight_time = 1.0
weight_deviation = 0.5
weight_variability = 2.0

top_trials = []

for vol, group in df.groupby('volume'):
    group = group.copy()
    
    # Min-max normalization of each objective
    group['norm_time'] = (group['time'] - group['time'].min()) / (group['time'].max() - group['time'].min())
    group['norm_deviation'] = (group['deviation'] - group['deviation'].min()) / (group['deviation'].max() - group['deviation'].min())
    group['norm_variability'] = (group['variability'] - group['variability'].min()) / (group['variability'].max() - group['variability'].min())
    
    # Weighted total score
    group['total_score'] = (
        weight_time * group['norm_time'] +
        weight_deviation * group['norm_deviation'] +
        weight_variability * group['norm_variability']
    )
    
    # Get top 3 trials with lowest total_score
    top3 = group.nsmallest(3, 'total_score')
    top_trials.append(top3)

# Combine into one DataFrame
best_trials = pd.concat(top_trials, ignore_index=True)

# Optional: Save to file
best_trials.to_csv(os.path.join(save_folder, 'top_3_trials_per_volume_normalized.csv'), index=False)

import seaborn as sns

# Plot histograms per parameter
for param in input_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(data=best_trials, x=param, hue='volume', multiple='stack', bins=10, palette='tab10')
    plt.title(f'Distribution of Top Trials by Volume: {param}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f'top_trials_histogram_{param}.png'))
    plt.clf()
