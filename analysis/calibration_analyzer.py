import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import seaborn as sns

input_cols = [
    'aspirate_speed', 'dispense_speed', 'aspirate_wait_time', 'dispense_wait_time',
    'retract_speed', 'pre_asp_air_vol', 'post_asp_air_vol', 'blowout_vol'
]

output_targets = ['time', 'deviation', 'variability']

def run_shap_analysis(df, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    X = df[input_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    shap_dfs = []

    for target in output_targets:
        y = df[target]
        model = xgb.XGBRegressor()
        model.fit(X_scaled, y)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)

        shap.summary_plot(shap_values, X, show=False)
        plt.title(f'SHAP Summary for {target}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f'shap_summary_{target}.png'))
        plt.clf()

        mean_shap = pd.DataFrame({
            'Feature': input_cols,
            target: np.abs(shap_values).mean(axis=0)
        }).set_index("Feature")

        shap_dfs.append(mean_shap)

    combined = pd.concat(shap_dfs, axis=1)
    normalized = combined / combined.max()

    normalized.plot(kind='barh', figsize=(10, 6))
    plt.xlabel('Normalized Mean SHAP Value (0–1)')
    plt.title('Normalized Feature Importance Comparison Across Targets')
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'shap_comparison_all_targets_normalized.png'))
    plt.clf()

def get_top_trials(df, save_folder, weight_time=1.0, weight_deviation=1.0, weight_variability=1.0, top_n=3):
    top_trials = []

    for vol, group in df.groupby('volume'):
        group = group.copy()
        group['norm_time'] = (group['time'] - group['time'].min()) / (group['time'].max() - group['time'].min())
        group['norm_deviation'] = (group['deviation'] - group['deviation'].min()) / (group['deviation'].max() - group['deviation'].min())
        group['norm_variability'] = (group['variability'] - group['variability'].min()) / (group['variability'].max() - group['variability'].min())

        group['total_score'] = (
            weight_time * group['norm_time'] +
            weight_deviation * group['norm_deviation'] +
            weight_variability * group['norm_variability']
        )

        top_n_rows = group.nsmallest(top_n, 'total_score')
        top_trials.append(top_n_rows)

    best_trials = pd.concat(top_trials, ignore_index=True)
    best_trials.to_csv(os.path.join(save_folder, 'top_3_trials_per_volume_normalized.csv'), index=False)
    return best_trials

def plot_top_trial_histograms(best_trials, save_folder):
    for param in input_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(data=best_trials, x=param, hue='volume', multiple='stack', bins=10, palette='tab10')
        plt.title(f'Distribution of Top Trials by Volume: {param}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f'top_trials_histogram_{param}.png'))
        plt.clf()
