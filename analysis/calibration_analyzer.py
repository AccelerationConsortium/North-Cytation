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
    """Plot measured volume over time using calculated_volume if available (preferred).
    Fallback sequence:
      1. If 'calculated_volume' column exists (mL) -> use *1000.
      2. Else if 'calculated_volume_ul' exists -> use directly.
      3. Else if 'mass' exists -> assume density 1.0 g/mL (legacy) and mass*1000.
      4. Otherwise abort.
    """
    os.makedirs(save_folder, exist_ok=True)

    if raw_df is None or len(raw_df) == 0:
        print("Warning: Empty raw_df - skipping measured volume plot")
        return

    raw_df = raw_df.copy()

    if 'calculated_volume' in raw_df.columns:
        raw_df['measured_volume_ul'] = raw_df['calculated_volume'] * 1000
        source = 'calculated_volume'
    elif 'calculated_volume_ul' in raw_df.columns:
        raw_df['measured_volume_ul'] = raw_df['calculated_volume_ul']
        source = 'calculated_volume_ul'
    elif 'mass' in raw_df.columns:
        raw_df['measured_volume_ul'] = raw_df['mass'] * 1000
        source = 'mass (density=1.0 assumption)'
    else:
        print("Warning: No volume or mass column available for measured volume plot")
        return

    plt.figure(figsize=(12, 8))

    if 'volume' in raw_df.columns:
        volumes = sorted(raw_df['volume'].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(volumes)))
        for i, vol in enumerate(volumes):
            vol_data = raw_df[raw_df['volume'] == vol].reset_index(drop=True)
            target_ul = vol * 1000
            plt.scatter(
                range(len(vol_data)),
                vol_data['measured_volume_ul'],
                color=colors[i],
                alpha=0.7,
                label=f'{target_ul:.0f}μL target',
                s=50
            )
            plt.axhline(y=target_ul,
                        color=colors[i],
                        linestyle='--',
                        alpha=0.8,
                        linewidth=2)
    else:
        plt.scatter(range(len(raw_df)), raw_df['measured_volume_ul'], alpha=0.7, s=50)

    plt.xlabel('Measurement Number')
    plt.ylabel('Measured Volume (μL)')
    plt.title('Measured Volume Over Time (source: {0})'.format(source))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(save_folder, 'measured_volume_over_time.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved measured volume plot to: {save_path} (source={source})")

def plot_measured_time_over_measurements(raw_df, save_folder, optimal_conditions=None):
    """Plot measured time over measurements with precision test winner time reference."""
    os.makedirs(save_folder, exist_ok=True)
    
    if raw_df is None or len(raw_df) == 0:
        print("Warning: Empty raw_df - skipping measured time plot")
        return
    
    if 'time' not in raw_df.columns:
        # Silently skip - time column is optional for raw measurements
        return
    
    plt.figure(figsize=(12, 8))
    
    raw_df = raw_df.copy()
    
    if 'volume' in raw_df.columns:
        # Plot by target volume
        volumes = sorted(raw_df['volume'].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(volumes)))
        
        for i, vol in enumerate(volumes):
            vol_data = raw_df[raw_df['volume'] == vol].reset_index(drop=True)
            
            target_ul = vol * 1000  # Convert mL to μL for display
            
            plt.scatter(
                range(len(vol_data)), 
                vol_data['time'],
                color=colors[i],
                alpha=0.7,
                label=f'{target_ul:.0f}μL target',
                s=50
            )
            
            # Add precision test winner time line if available
            if optimal_conditions:
                optimal_for_vol = [opt for opt in optimal_conditions if opt.get('target_volume_mL') == vol]
                if optimal_for_vol:
                    winner_time = optimal_for_vol[0].get('time', None)
                    if winner_time:
                        plt.axhline(y=winner_time, 
                                   color=colors[i], 
                                   linestyle='--', 
                                   alpha=0.8,
                                   linewidth=2,
                                   label=f'{target_ul:.0f}μL winner time' if i < 3 else "")  # Only label first few to avoid clutter
    else:
        # Plot all measurements
        plt.scatter(range(len(raw_df)), raw_df['time'], alpha=0.7, s=50)
    
    plt.xlabel('Measurement Number')
    plt.ylabel('Measured Time (seconds)')
    plt.title('Measured Time Over Measurements with Precision Test Winner Times')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(save_folder, 'measured_time_over_measurements.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved measured time plot to: {save_path}")

def plot_time_vs_deviation(results_df, save_folder, optimal_conditions=None, show_absolute=False):
    """Scatter plot of Time vs. Deviation (% or absolute) color-coded by volume.
    Precision winner highlighting removed for visual simplicity."""
    
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
        df_sub = results_df[results_df['volume'] == vol].copy()
        if show_absolute:
            # Convert percent deviation to absolute deviation in µL for this volume
            df_sub['deviation_abs_ul'] = (df_sub['deviation'] / 100.0) * (vol * 1000.0)
        label = f"{vol*1000:.0f}μL"
        
        print(f"  Volume {vol*1000:.0f}μL: {len(df_sub)} points")

        # Normal optimization points
        y_vals = df_sub["deviation_abs_ul"] if show_absolute else df_sub["deviation"]
        plt.scatter(
            df_sub["time"],
            y_vals,
            color=colors[i],
            label=label,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
            s=60
        )

        # (Previously precision winners highlighted; removed per simplification request)

    plt.xlabel("Time (seconds)")
    plt.ylabel("Absolute Deviation (μL)" if show_absolute else "Deviation (%)")
    plt.title("Time vs Deviation (absolute)" if show_absolute else "Time vs Deviation (%)")
    plt.legend(title="Volume")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(save_folder, f"time_vs_deviation_scatter{'_abs' if show_absolute else ''}.png")
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

def plot_learning_curves(df, save_folder, metrics=None):
    df = df.copy()
    df['trial_index'] = pd.to_numeric(df['trial_index'], errors='coerce')
    df = df.dropna(subset=['trial_index']).sort_values(['volume', 'trial_index'])
    if metrics is None:
        metrics = ['deviation', 'time', 'variability']
    metrics = [m for m in metrics if m in df.columns]
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

