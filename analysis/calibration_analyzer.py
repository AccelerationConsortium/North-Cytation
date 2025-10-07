import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

# Temporarily skip xgboost/shap imports to avoid version conflict
try:
    import shap
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    # Test if XGBRegressor is actually available
    xgb.XGBRegressor()
    SHAP_AVAILABLE = True
except (ImportError, AttributeError) as e:
    print(f"SHAP/XGBoost not available: {e}")
    SHAP_AVAILABLE = False

input_cols = [
    'aspirate_speed', 'dispense_speed', 'aspirate_wait_time', 'dispense_wait_time',
    'retract_speed', 'blowout_vol', 'post_asp_air_vol', 'overaspirate_vol'
]

output_targets = ['time', 'deviation']  # Removed 'variability' - it doesn't exist

def run_shap_analysis(df, save_folder):
    if not SHAP_AVAILABLE:
        print("Skipping SHAP analysis - xgboost/shap not available due to version conflict")
        return
    
    os.makedirs(save_folder, exist_ok=True)
    
    # Handle backward compatibility: if old data has pre_asp_air_vol but not blowout_vol
    if 'pre_asp_air_vol' in df.columns and 'blowout_vol' not in df.columns:
        print("Info: Converting pre_asp_air_vol to blowout_vol for backward compatibility")
        df['blowout_vol'] = df['pre_asp_air_vol']
    
    # Filter input columns to only those present in the dataframe
    available_cols = [col for col in input_cols if col in df.columns]
    missing_cols = [col for col in input_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Missing columns for SHAP analysis: {missing_cols}")
        print(f"Using available columns: {available_cols}")
    
    if len(available_cols) < 2:
        print("Error: Not enough columns for SHAP analysis")
        return
    
    X = df[available_cols]
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
            'Feature': available_cols,
            target: np.abs(shap_values).mean(axis=0)
        }).set_index("Feature")

        shap_dfs.append(mean_shap)

    # Skip normalized feature importance comparison - not needed
    print("SHAP analysis complete - individual plots saved")

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
        # Use matplotlib instead of seaborn
        volumes = sorted(best_trials['volume'].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(volumes)))
        for i, vol in enumerate(volumes):
            vol_data = best_trials[best_trials['volume'] == vol][param]
            plt.hist(vol_data, bins=10, alpha=0.7, label=f'{vol*1000:.0f}μL', color=colors[i])
        plt.legend()
        plt.title(f'Distribution of Top Trials by Volume: {param}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f'top_trials_histogram_{param}.png'))
        plt.clf()

def plot_measured_volume_over_time(raw_df, save_folder):
    """Plot measured volume over measurements with target volume dashed lines."""
    os.makedirs(save_folder, exist_ok=True)
    
    if 'mass' not in raw_df.columns:
        print("Warning: No 'mass' column found - cannot create measured volume plot")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Convert mass to volume (assuming water density = 1.0 g/mL)
    raw_df = raw_df.copy()
    raw_df['measured_volume_ul'] = raw_df['mass'] * 1000  # Convert g to μL
    
    if 'volume' in raw_df.columns:
        # Plot by target volume
        volumes = sorted(raw_df['volume'].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(volumes)))
        
        for i, vol in enumerate(volumes):
            vol_data = raw_df[raw_df['volume'] == vol].copy()
            vol_data = vol_data.reset_index(drop=True)
            
            target_ul = vol * 1000  # Convert mL to μL
            
            plt.scatter(
                range(len(vol_data)), 
                vol_data['measured_volume_ul'],
                color=colors[i],
                alpha=0.7,
                label=f'{target_ul:.0f}μL target',
                s=50
            )
            
            # Add target volume dashed line
            plt.axhline(y=target_ul, 
                       color=colors[i], 
                       linestyle='--', 
                       alpha=0.8,
                       linewidth=2)
    else:
        # Plot all measurements
        plt.scatter(range(len(raw_df)), raw_df['measured_volume_ul'], alpha=0.7, s=50)
    
    plt.xlabel('Measurement Number')
    plt.ylabel('Measured Volume (μL)')
    plt.title('Measured Volume Over Time with Target Volume Lines')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(save_folder, 'measured_volume_over_time.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved measured volume plot to: {save_path}")

def plot_time_vs_deviation(results_df, save_folder, optimal_conditions=None):
    """Scatter plot of Time vs. Deviation with precision-test winning conditions highlighted."""
    
    # Check required columns
    required_cols = ['time', 'deviation', 'volume']
    missing_cols = [col for col in required_cols if col not in results_df.columns]
    if missing_cols:
        print(f"Warning: Cannot create scatter plot - missing columns: {missing_cols}")
        return
    
    plt.figure(figsize=(10, 6))
    
    volumes = sorted(results_df['volume'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(volumes)))
    
    print(f"Creating scatter plot for {len(volumes)} volumes: {[v*1000 for v in volumes]} μL")
    
    for i, vol in enumerate(volumes):
        df_sub = results_df[results_df['volume'] == vol]
        label = f"{vol*1000:.0f}μL"
        
        print(f"  Volume {vol*1000:.0f}μL: {len(df_sub)} points")

        # Normal optimization points
        plt.scatter(
            df_sub["time"],
            df_sub["deviation"],
            color=colors[i],
            label=label,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
            s=60
        )

        # Highlight precision-test winning conditions with stars
        if optimal_conditions:
            optimal_for_vol = [opt for opt in optimal_conditions if opt.get('target_volume_mL') == vol]
            for opt in optimal_for_vol:
                # Find the corresponding trial in results_df
                matching_trials = df_sub[
                    (abs(df_sub.get('aspirate_speed', 0) - opt.get('aspirate_speed', -999)) < 0.1) &
                    (abs(df_sub.get('dispense_speed', 0) - opt.get('dispense_speed', -999)) < 0.1)
                ]
                if not matching_trials.empty:
                    plt.scatter(
                        matching_trials["time"],
                        matching_trials["deviation"],
                        marker="★",
                        color="gold",
                        s=200,
                        edgecolors="black",
                        linewidth=2,
                        label="Precision Test Winner" if i == 0 else None
                    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Deviation (μL)")  # Fixed: should be μL not %
    plt.title("Time vs Deviation Scatter Plot")
    plt.legend(title="Volume")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(save_folder, "time_vs_deviation_scatter.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved scatter plot to: {save_path}")

def plot_boxplots(df, save_folder):
    for target in ['deviation', 'variability', 'time']:
        if target in df.columns:
            plt.figure(figsize=(10, 6))
            # Use matplotlib boxplot instead
            volumes = sorted(df['volume'].unique())
            data_by_volume = [df[df['volume'] == vol][target].dropna() for vol in volumes]
            plt.boxplot(data_by_volume, labels=[f'{v*1000:.0f}μL' for v in volumes])
            plt.title(f'{target.capitalize()} by Volume')
            plt.tight_layout()
            plt.savefig(os.path.join(save_folder, f'boxplot_{target}_by_volume.png'), dpi=300)
            plt.close()

def plot_pairplot(df, save_folder):
    key_params = ['deviation', 'time', 'variability', 'aspirate_speed', 'dispense_speed']
    available = [p for p in key_params if p in df.columns]
    if len(available) >= 3:
        # Skip pairplot - requires seaborn, not essential
        print("Skipping pairplot - seaborn not used")
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

