#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# CLEAR OUTPUT LOCATION
TEST_OUTPUT_DIR = r"C:\Users\Imaging Controller\Desktop\Calibration_SDL_Output\New_Method\20251007_122521_water"

def load_real_calibration_data():
    """Load data from the actual calibration output folder."""
    # The actual calibration data location
    calibration_base = r"C:\Users\Imaging Controller\Desktop\Calibration_SDL_Output\New_Method\20251007_122521_water"
    
    print(f"Looking for calibration data in: {calibration_base}")
    
    # Common file patterns to look for
    possible_files = [
        "raw_data.csv",
        "summary.csv", 
        "results.csv",
        "optimization_results.csv",
        "raw_replicate_data.csv"
    ]
    
    for filename in possible_files:
        filepath = os.path.join(calibration_base, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                print(f"✅ Loaded {len(df)} rows from {filename}")
                print("Columns:", df.columns.tolist())
                print("\nFirst few rows:")
                print(df.head())
                return df, filename
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
    
    print("❌ No data files found in calibration folder")
    return None, None



def plot_measured_volume_over_time(df, save_folder, target_volumes=None):
    """Plot measured volume over measurements with target volume dashed lines."""
    os.makedirs(save_folder, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    # Convert mass to volume (assuming water density = 1.0 g/mL)
    if 'mass' in df.columns:
        df['measured_volume_ul'] = df['mass'] * 1000  # Convert g to μL
    else:
        print("No 'mass' column found in data")
        return
    
    if 'volume' in df.columns:
        # Plot by target volume
        volumes = sorted(df['volume'].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(volumes)))
        
        for i, vol in enumerate(volumes):
            vol_data = df[df['volume'] == vol].copy()
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
        plt.scatter(range(len(df)), df['measured_volume_ul'], alpha=0.7, s=50)
        
        # Add target lines if provided
        if target_volumes:
            for target_ml in target_volumes:
                target_ul = target_ml * 1000
                plt.axhline(y=target_ul, 
                           linestyle='--', 
                           alpha=0.8,
                           linewidth=2,
                           label=f'{target_ul:.0f}μL target')
    
    plt.xlabel('Measurement Number')
    plt.ylabel('Measured Volume (μL)')
    plt.title('Measured Volume Over Time with Target Volume Lines')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(save_folder, 'measured_volume_over_time.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved measured volume plot to: {save_path}")
    return save_path

def test_scatter_plot_debug(df, save_folder):
    """Debug the scatter plot issues."""
    os.makedirs(save_folder, exist_ok=True)
    
    print("=== SCATTER PLOT DEBUG ===")
    print("DataFrame shape:", df.shape)
    print("Columns:", df.columns.tolist())
    
    # Check required columns
    required_cols = ['time', 'deviation', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        return
    
    print("Unique volumes:", sorted(df['volume'].unique()))
    print("Time range:", df['time'].min(), "to", df['time'].max())
    print("Deviation range:", df['deviation'].min(), "to", df['deviation'].max())
    
    # Create scatter plot with debug info
    plt.figure(figsize=(10, 6))
    
    volumes = sorted(df['volume'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(volumes)))
    
    for i, vol in enumerate(volumes):
        df_vol = df[df['volume'] == vol]
        print(f"\nVolume {vol*1000:.0f}μL: {len(df_vol)} points")
        
        plt.scatter(
            df_vol['time'],
            df_vol['deviation'], 
            color=colors[i],
            alpha=0.7,
            label=f'{vol*1000:.0f}μL',
            s=60,
            edgecolors='black',
            linewidth=0.5
        )
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Deviation (μL)')
    plt.title('Time vs Deviation Debug Plot')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(save_folder, 'scatter_plot_debug.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved debug scatter plot to: {save_path}")
    print("\n✅ Debug scatter plot created successfully!")

def run_shap_analysis_fixed(df, save_folder, skip_variation=True):
    """Fixed SHAP analysis that skips variation if it doesn't exist."""
    try:
        import shap
        import xgboost as xgb
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        print(f"Cannot run SHAP analysis - missing packages: {e}")
        return
    
    os.makedirs(save_folder, exist_ok=True)
    
    input_cols = [
        'aspirate_speed', 'dispense_speed', 'aspirate_wait_time', 'dispense_wait_time',
        'retract_speed', 'blowout_vol', 'post_asp_air_vol', 'overaspirate_vol'
    ]
    
    # Skip variation if requested or if column doesn't exist
    output_targets = ['time', 'deviation']
    if not skip_variation and 'variability' in df.columns:
        output_targets.append('variability')
    
    print(f"Running SHAP analysis for targets: {output_targets}")
    
    # Handle backward compatibility
    if 'pre_asp_air_vol' in df.columns and 'blowout_vol' not in df.columns:
        print("Converting pre_asp_air_vol to blowout_vol for backward compatibility")
        df['blowout_vol'] = df['pre_asp_air_vol']
    
    # Filter input columns to only those present
    available_cols = [col for col in input_cols if col in df.columns]
    missing_cols = [col for col in input_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Missing columns for SHAP analysis: {missing_cols}")
    print(f"Using available columns: {available_cols}")
    
    if len(available_cols) < 2:
        print("Not enough columns for SHAP analysis")
        return
    
    X = df[available_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    for target in output_targets:
        if target not in df.columns:
            print(f"Skipping {target} - not in DataFrame")
            continue
            
        print(f"Analyzing target: {target}")
        y = df[target]
        model = xgb.XGBRegressor(random_state=42)
        model.fit(X_scaled, y)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
        
        # Create SHAP summary plot
        shap.summary_plot(shap_values, X, show=False)
        plt.title(f'SHAP Summary for {target}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f'shap_summary_{target}.png'), dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved SHAP plot for {target}")

def main():
    """Main test function."""
    print("=== CALIBRATION ANALYZER TEST ===")
    print(f"📁 ALL OUTPUT WILL BE SAVED TO: {TEST_OUTPUT_DIR}")
    print("="*60)
    
    # Create clear output directory
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    
    # Try to load real calibration data
    print("\n🔍 LOOKING FOR REAL CALIBRATION DATA...")
    real_data, filename = load_real_calibration_data()
    
    if real_data is not None and len(real_data) > 0:
        print(f"\n✅ USING REAL DATA FROM: {filename}")
        
        # Test measured volume plot if we have mass data
        if 'mass' in real_data.columns:
            print("\n📊 Creating measured volume over time plot...")
            target_volumes = real_data['volume'].unique() if 'volume' in real_data.columns else [0.01, 0.05, 0.1]
            plot_measured_volume_over_time(real_data, TEST_OUTPUT_DIR, target_volumes)
        else:
            print("❌ No 'mass' column found - cannot create volume plot")
        
        # Test scatter plot if we have the right columns
        if all(col in real_data.columns for col in ['time', 'deviation', 'volume']):
            print("\n📊 Testing scatter plot with real data...")
            test_scatter_plot_debug(real_data, TEST_OUTPUT_DIR)
        else:
            missing = [col for col in ['time', 'deviation', 'volume'] if col not in real_data.columns]
            print(f"❌ Missing columns for scatter plot: {missing}")
    
    # Create synthetic data as backup
    print("\n🔧 CREATING SYNTHETIC DATA FOR TESTING...")
    synthetic_data = create_synthetic_optimize_results()
    print(f"✅ Created synthetic data with {len(synthetic_data)} rows")
    
    # Test scatter plot with synthetic data
    print("\n📊 Testing scatter plot with synthetic data...")
    test_scatter_plot_debug(synthetic_data, TEST_OUTPUT_DIR)
    
    print(f"\n🎉 ALL TESTS COMPLETE!")
    print(f"📁 Check all results in: {TEST_OUTPUT_DIR}")

def create_synthetic_optimize_results():
    """Create synthetic optimization results that match what the calibration produces."""
    np.random.seed(42)
    
    volumes = [0.05, 0.1, 0.2, 0.3, 0.5]  # mL
    n_trials_per_volume = 12
    
    results = []
    
    for vol in volumes:
        for trial in range(n_trials_per_volume):
            # Calculate realistic time based on volume (from your calibration formula)
            base_time = vol * 10.146 + 9.5813 + np.random.normal(0, 2)
            time_seconds = max(8, base_time)
            
            # Calculate realistic deviation in μL (absolute, not percentage)
            base_deviation_ul = np.random.uniform(0.5, 3.0)  # 0.5-3μL deviation
            
            result = {
                'volume': vol,
                'trial_index': trial,
                'time': time_seconds,
                'deviation': base_deviation_ul,  # in μL
                'variability': abs(np.random.normal(1.2, 0.5)),  # replicate variability in μL
                'aspirate_speed': np.random.uniform(5, 15),
                'dispense_speed': np.random.uniform(5, 15),
                'aspirate_wait_time': np.random.uniform(1, 30),
                'dispense_wait_time': np.random.uniform(1, 15),
                'retract_speed': np.random.uniform(5, 12),
                'blowout_vol': np.random.uniform(0.01, 0.1),
                'post_asp_air_vol': np.random.uniform(0.01, 0.08),
                'overaspirate_vol': np.random.uniform(0.01, 0.05)
            }
            
            results.append(result)
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    main()
