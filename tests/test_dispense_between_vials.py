from os import remove
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
from pipetting_data.pipetting_parameters import PipettingParameters

# Test continuous mass monitoring stability with 10 repetitions

input_vial_status_file="../utoronto_demo/status/sample_input_vials.csv"

lash_e = Lash_E(input_vial_status_file, initialize_biotek=False, simulate=False)

volume = 0.010
n_runs = 10

print("="*60)
print(f"TESTING CONTINUOUS MASS MONITORING - {n_runs} REPETITIONS")
print("="*60)

lash_e.nr_robot.move_vial_to_location(2, 'clamp', 0)

# Store results for analysis
mass_results = []
baseline_stability_data = []

for run in range(1, n_runs + 1):
    print(f"\n--- RUN {run}/{n_runs} ---")
    
    # Aspirate fresh volume for each run
    lash_e.nr_robot.aspirate_from_vial(0, volume, liquid='water')
    
    # Continuous mass monitoring with data saving
    mass_diff = lash_e.nr_robot.dispense_into_vial(2, volume, 
                                                   continuous_mass_monitoring=True, 
                                                   save_mass_data=True, 
                                                   liquid='water')
    mass_results.append(mass_diff)
    
    # Analyze latest mass data file for baseline stability
    import os
    import glob
    mass_files = glob.glob(os.path.join("..", "output", "mass_measurements", "mass_data_2_*.csv"))
    if mass_files:
        latest_file = max(mass_files, key=os.path.getctime)
        df = pd.read_csv(latest_file)
        
        # Calculate baseline statistics
        pre_baseline_data = df[df['phase'] == 'baseline_pre']['mass_g']
        post_baseline_data = df[df['phase'] == 'baseline_post']['mass_g']
        
        if len(pre_baseline_data) > 0 and len(post_baseline_data) > 0:
            baseline_stability_data.append({
                'run': run,
                'mass_diff': mass_diff,
                'pre_baseline_mean': pre_baseline_data.mean(),
                'pre_baseline_std': pre_baseline_data.std(),
                'pre_baseline_range': pre_baseline_data.max() - pre_baseline_data.min(),
                'pre_baseline_count': len(pre_baseline_data),
                'post_baseline_mean': post_baseline_data.mean(),
                'post_baseline_std': post_baseline_data.std(), 
                'post_baseline_range': post_baseline_data.max() - post_baseline_data.min(),
                'post_baseline_count': len(post_baseline_data)
            })
    
    print(f"Run {run} result: {mass_diff:.6f} g")

# Convert to DataFrame for analysis
stability_df = pd.DataFrame(baseline_stability_data)

print("\n" + "="*60)
print("BASELINE STABILITY ANALYSIS")
print("="*60)

if len(stability_df) > 0:
    # Mass measurement statistics
    print(f"\nMASS MEASUREMENT STATISTICS ({len(mass_results)} runs):")
    print(f"Mean:        {np.mean(mass_results):.6f} g")
    print(f"Std Dev:     {np.std(mass_results):.6f} g")
    print(f"Min:         {np.min(mass_results):.6f} g") 
    print(f"Max:         {np.max(mass_results):.6f} g")
    print(f"Range:       {np.max(mass_results) - np.min(mass_results):.6f} g")
    print(f"CV%:         {(np.std(mass_results)/np.mean(mass_results)*100):.3f}%")
    
    # Pre-baseline stability
    print(f"\nPRE-BASELINE STABILITY:")
    print(f"Mean std dev:    {stability_df['pre_baseline_std'].mean():.6f} g")
    print(f"Mean range:      {stability_df['pre_baseline_range'].mean():.6f} g") 
    print(f"Avg readings:    {stability_df['pre_baseline_count'].mean():.1f}")
    
    # Post-baseline stability  
    print(f"\nPOST-BASELINE STABILITY:")
    print(f"Mean std dev:    {stability_df['post_baseline_std'].mean():.6f} g")
    print(f"Mean range:      {stability_df['post_baseline_range'].mean():.6f} g")
    print(f"Avg readings:    {stability_df['post_baseline_count'].mean():.1f}")
    
    # Save stability analysis
    timestamp_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    stability_filename = f"baseline_stability_analysis_{timestamp_str}.csv"
    stability_filepath = os.path.join("..", "output", "mass_measurements", stability_filename)
    stability_df.to_csv(stability_filepath, index=False)
    print(f"\nSaved stability analysis to: {stability_filename}")
    
    # Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Mass measurements over runs
    axes[0,0].plot(range(1, len(mass_results)+1), mass_results, 'bo-')
    axes[0,0].axhline(y=np.mean(mass_results), color='r', linestyle='--', label=f'Mean: {np.mean(mass_results):.6f}g')
    axes[0,0].set_xlabel('Run Number')
    axes[0,0].set_ylabel('Mass Difference (g)')
    axes[0,0].set_title('Mass Measurement Repeatability')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend()
    
    # Pre-baseline stability
    axes[0,1].plot(stability_df['run'], stability_df['pre_baseline_std'], 'go-', label='Std Dev')
    axes[0,1].set_xlabel('Run Number')
    axes[0,1].set_ylabel('Pre-baseline Std Dev (g)')
    axes[0,1].set_title('Pre-baseline Stability')
    axes[0,1].grid(True, alpha=0.3)
    
    # Post-baseline stability
    axes[1,0].plot(stability_df['run'], stability_df['post_baseline_std'], 'ro-', label='Std Dev')
    axes[1,0].set_xlabel('Run Number')  
    axes[1,0].set_ylabel('Post-baseline Std Dev (g)')
    axes[1,0].set_title('Post-baseline Stability')
    axes[1,0].grid(True, alpha=0.3)
    
    # Overall distribution
    axes[1,1].hist(mass_results, bins=min(8, len(mass_results)), alpha=0.7, edgecolor='black')
    axes[1,1].axvline(x=np.mean(mass_results), color='r', linestyle='--', label=f'Mean: {np.mean(mass_results):.6f}g')
    axes[1,1].set_xlabel('Mass Difference (g)')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Mass Measurement Distribution')
    axes[1,1].legend()
    
    plt.tight_layout()
    plot_filename = f"baseline_stability_summary_{timestamp_str}.png"
    plot_filepath = os.path.join("..", "output", "mass_measurements", plot_filename)
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved summary plot to: {plot_filename}")

else:
    print("No baseline data collected for analysis")

print(f"\nTarget volume: {volume:.3f} mL (~{volume*1.0:.3f} g for water)")
print("="*60)

lash_e.nr_robot.remove_pipet()
lash_e.nr_robot.return_vial_home(2)
lash_e.nr_robot.move_home()
