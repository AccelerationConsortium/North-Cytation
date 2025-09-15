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
    'retract_speed', 'pre_asp_air_vol', 'post_asp_air_vol', 'overaspirate_vol'
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

def plot_time_vs_deviation(results_df, save_folder):
    """Scatter plot of Time vs. Deviation with variability > 1 marked and volumes color-coded."""
    plt.figure(figsize=(10, 6))
    
    for vol in sorted(results_df['volume'].unique()):
        df_sub = results_df[results_df['volume'] == vol]
        label = f"{vol} mL"

        # Normal points
        plt.scatter(
            df_sub["time"],
            df_sub["deviation"],
            label=label,
            alpha=0.7,
            edgecolors="k"
        )

        # X markers for variability > 1.0
        high_var = df_sub[df_sub["variability"] > 2.0]
        plt.scatter(
            high_var["time"],
            high_var["deviation"],
            marker="x",
            color="black",
            s=100,
            label=None
        )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Deviation From Target (%)")
    plt.title("Time vs Deviation with High-Variability Markers")
    plt.legend(title="Volume")
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(save_folder, "time_vs_deviation_scatter.png")
    plt.savefig(save_path)
    plt.clf()
    print("Saved time vs deviation plot to:", save_path)

def plot_boxplots(df, save_folder):
    for target in ['deviation', 'variability', 'time']:
        if target in df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x='volume', y=target)
            plt.title(f'{target.capitalize()} by Volume')
            plt.tight_layout()
            plt.savefig(os.path.join(save_folder, f'boxplot_{target}_by_volume.png'), dpi=300)
            plt.close()

def plot_pairplot(df, save_folder):
    key_params = ['deviation', 'time', 'variability', 'aspirate_speed', 'dispense_speed']
    available = [p for p in key_params if p in df.columns]
    if len(available) >= 3:
        sns.pairplot(df[available])
        plt.savefig(os.path.join(save_folder, 'parameter_pairplot.png'), dpi=300)
        plt.close()

def plot_learning_curves(df, save_folder):
    df = df.copy()
    df['trial_index'] = pd.to_numeric(df['trial_index'], errors='coerce')
    df = df.dropna(subset=['trial_index']).sort_values(['volume', 'trial_index'])

    metrics = ['deviation', 'time', 'variability']
    for metric in metrics:
        if metric not in df.columns:
            continue
        volumes = sorted(df['volume'].unique())
        plt.figure(figsize=(14, 5 * len(volumes)))
        for i, vol in enumerate(volumes):
            ax = plt.subplot(len(volumes), 1, i+1)
            vol_df = df[df['volume'] == vol].sort_values('trial_index')
            ax.scatter(vol_df['trial_index'], vol_df[metric], alpha=0.6, s=40)
            if len(vol_df) >= 3:
                ma = vol_df[metric].rolling(window=3, center=True).mean()
                ax.plot(vol_df['trial_index'], ma, label='Moving Average', color='orange')

                vol_df[metric] = pd.to_numeric(vol_df[metric], errors='coerce')
                cleaned = vol_df.dropna(subset=['trial_index', metric])
                z = np.polyfit(cleaned['trial_index'], cleaned[metric], 1)

                ax.plot(vol_df['trial_index'], np.poly1d(z)(vol_df['trial_index']), 'r--', label='Trend')
            ax.set_title(f'{metric.capitalize()} Learning Curve - {vol} mL')
            ax.set_xlabel('Trial Index')
            ax.set_ylabel(metric.capitalize())
            ax.legend()
            ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f'learning_curves_{metric}.png'), dpi=300)
        plt.close()

def plot_improvement_summary(df, save_folder):
    metrics = ['deviation', 'time', 'variability']
    summary = []
    for vol in sorted(df['volume'].unique()):
        sub = df[df['volume'] == vol].sort_values('trial_index')
        n = len(sub)
        if n < 4:
            continue
        for metric in metrics:
            if metric not in sub.columns:
                continue
            first = sub.iloc[:n//2][metric].mean()
            second = sub.iloc[n//2:][metric].mean()
            if first != 0:
                improvement = (first - second) / first * 100
            else:
                improvement = 0
            summary.append({
                'volume': vol,
                'metric': metric,
                'first_half': first,
                'second_half': second,
                'improvement_pct': improvement
            })
    if summary:
        df_summary = pd.DataFrame(summary)
        df_summary.to_csv(os.path.join(save_folder, 'improvement_summary.csv'), index=False)

        plt.figure(figsize=(12, 8))
        for i, metric in enumerate(metrics):
            plt.subplot(3, 1, i+1)
            sub = df_summary[df_summary['metric'] == metric]
            plt.bar(sub['volume'].astype(str), sub['improvement_pct'], color='teal')
            plt.title(f'{metric.capitalize()} Improvement (% reduction)')
            plt.ylabel('Improvement (%)')
            plt.xlabel('Volume (mL)')
            plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, 'improvement_summary.png'), dpi=300)
        plt.close()

