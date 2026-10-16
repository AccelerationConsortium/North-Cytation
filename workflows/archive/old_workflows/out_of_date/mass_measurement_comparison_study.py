"""
Mass Measurement Method Comparison Study

Compares continuous monitoring vs traditional before/after mass measurement
across different volumes to assess precision, accuracy, and systematic differences.

Study Design:
- 2 volumes: 10 µL and 25 µL
- 2 methods: continuous monitoring and traditional before/after
- 50 trials each = 200 total measurements
- Statistical analysis with histograms and comparative metrics
"""

import sys
import os
from networkx import volume
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
from pipetting_data.pipetting_parameters import PipettingParameters

def main():
    # Experimental setup
    input_vial_status_file = "../utoronto_demo/status/sample_input_vials.csv"
    simulate_mode = False  # Set to False for hardware testing
    lash_e = Lash_E(input_vial_status_file, initialize_biotek=False, simulate=simulate_mode)
    
    # Test parameters
    volumes_uL = [15]  # Test volumes in µL
    volumes_mL = [v/1000 for v in volumes_uL]  # Convert to mL
    n_trials = 50
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("output", "mass_comparison_study", f"study_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("MASS MEASUREMENT METHOD COMPARISON STUDY")
    print("="*80)
    print(f"Mode: {'SIMULATION' if simulate_mode else 'HARDWARE'}")
    print(f"Volumes: {volumes_uL} µL")
    
    if simulate_mode:
        print("NOTE: In simulation mode, continuous monitoring is not functional.")
        print("      Only traditional before/after method will be tested for reference.")
        print(f"Trials: {n_trials} (traditional method only)")
        print(f"Total measurements: {len(volumes_uL) * n_trials}")
    else:
        print(f"Trials per method per volume: {n_trials}")
        print(f"Total measurements: {len(volumes_uL) * 2 * n_trials}")
    
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # Initialize data collection
    all_results = []
    
    # Move target vial to clamp
    lash_e.nr_robot.move_vial_to_location(2, 'clamp', 0)
    
    

    # Loop through each volume
    for vol_idx, (vol_uL, vol_mL) in enumerate(zip(volumes_uL, volumes_mL)):
        #Precondition tip with 5 aspirate/dispense cycles
        for i in range (0, 5):
            lash_e.nr_robot.aspirate_from_vial(0, vol_mL, liquid='water')
            lash_e.nr_robot.dispense_into_vial(0, vol_mL, liquid='water', initial_move=False)

        print(f"\n{'='*60}")
        print(f"TESTING VOLUME: {vol_uL} µL ({vol_mL:.3f} mL)")
        print(f"{'='*60}")
        
        # Test continuous monitoring method (only if not in simulation mode)
        if not simulate_mode:
            print(f"\n--- CONTINUOUS MONITORING METHOD ({n_trials} trials) ---")
            continuous_masses = []
            
            for trial in range(1, n_trials + 1):
                print(f"Continuous trial {trial}/{n_trials}...", end=" ")
                
                # Aspirate fresh volume
                lash_e.nr_robot.aspirate_from_vial(0, vol_mL, liquid='water')
                
                # Dispense with continuous monitoring (save individual data files)
                mass_diff = lash_e.nr_robot.dispense_into_vial(2, vol_mL, 
                                                              continuous_mass_monitoring=True, 
                                                              save_mass_data=True,  # Save individual files for analysis
                                                              liquid='water')
                continuous_masses.append(mass_diff)
                print(f"{mass_diff:.6f} g")
                
                # Record result
                all_results.append({
                    'volume_uL': vol_uL,
                    'volume_mL': vol_mL,
                    'method': 'continuous',
                    'trial': trial,
                    'mass_diff_g': mass_diff
                })
        else:
            print(f"\n--- CONTINUOUS MONITORING SKIPPED (simulation mode) ---")
            continuous_masses = []
        
        # Test traditional before/after method
        print(f"\n--- TRADITIONAL BEFORE/AFTER METHOD ({n_trials} trials) ---")
        traditional_masses = []
        
        for trial in range(1, n_trials + 1):
            print(f"Traditional trial {trial}/{n_trials}...", end=" ")
            
            # Aspirate fresh volume
            lash_e.nr_robot.aspirate_from_vial(0, vol_mL, liquid='water')
            
            # Dispense with traditional measurement
            mass_diff = lash_e.nr_robot.dispense_into_vial(2, vol_mL, 
                                                          measure_weight=True,
                                                          continuous_mass_monitoring=False,
                                                          liquid='water')
            traditional_masses.append(mass_diff)
            print(f"{mass_diff:.6f} g")
            
            # Record result  
            all_results.append({
                'volume_uL': vol_uL,
                'volume_mL': vol_mL,
                'method': 'traditional',
                'trial': trial,
                'mass_diff_g': mass_diff
            })
        
        # Volume-specific analysis
        print(f"\n--- VOLUME {vol_uL} µL SUMMARY ---")
        
        if not simulate_mode and continuous_masses:
            cont_mean = np.mean(continuous_masses)
            cont_std = np.std(continuous_masses)
            trad_mean = np.mean(traditional_masses)
            trad_std = np.std(traditional_masses)
            
            print(f"Continuous:  {cont_mean:.6f} ± {cont_std:.6f} g  (CV: {cont_std/cont_mean*100:.3f}%)")
            print(f"Traditional: {trad_mean:.6f} ± {trad_std:.6f} g  (CV: {trad_std/trad_mean*100:.3f}%)")
            print(f"Difference:  {cont_mean - trad_mean:.6f} g")
            print(f"Expected:    ~{vol_mL:.6f} g (water density ≈ 1.0 g/mL)")
        else:
            trad_mean = np.mean(traditional_masses)
            trad_std = np.std(traditional_masses)
            print(f"Traditional: {trad_mean:.6f} ± {trad_std:.6f} g  (CV: {trad_std/trad_mean*100:.3f}%)")
            if simulate_mode:
                print(f"Expected:    ~{vol_mL:.6f} g (water density ≈ 1.0 g/mL)")
                print("NOTE: In simulation mode, mass values are simulated and do not reflect real measurements.")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save raw data
    raw_data_file = os.path.join(output_dir, "raw_measurement_data.csv")
    results_df.to_csv(raw_data_file, index=False)
    print(f"\nSaved raw data to: {raw_data_file}")
    
    # Statistical analysis and visualization
    print(f"\n{'='*80}")
    print("COMPREHENSIVE STATISTICAL ANALYSIS")
    print(f"{'='*80}")
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = {'continuous': 'blue', 'traditional': 'red'}
    
    for vol_idx, vol_uL in enumerate(volumes_uL):
        vol_data = results_df[results_df['volume_uL'] == vol_uL]
        
        # Histogram comparison for this volume
        ax = axes[vol_idx, 0]
        for method in ['continuous', 'traditional']:
            method_data = vol_data[vol_data['method'] == method]['mass_diff_g']
            ax.hist(method_data, bins=15, alpha=0.6, label=f'{method.title()}', 
                   color=colors[method], edgecolor='black', density=True)
            
            # Add statistics
            mean_val = method_data.mean()
            std_val = method_data.std()
            ax.axvline(mean_val, color=colors[method], linestyle='--', 
                      label=f'{method.title()} mean: {mean_val:.6f}g')
        
        ax.set_xlabel('Mass Difference (g)')
        ax.set_ylabel('Density')
        ax.set_title(f'{vol_uL} µL Volume - Method Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Box plot comparison for this volume
        ax = axes[vol_idx, 1]
        method_groups = [vol_data[vol_data['method'] == m]['mass_diff_g'] for m in ['continuous', 'traditional']]
        bp = ax.boxplot(method_groups, labels=['Continuous', 'Traditional'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        
        ax.set_ylabel('Mass Difference (g)')
        ax.set_title(f'{vol_uL} µL Volume - Distribution Comparison')
        ax.grid(True, alpha=0.3)
        
        # Add expected value line
        expected_mass = volumes_mL[vol_idx]
        for ax_row in [axes[vol_idx, 0], axes[vol_idx, 1]]:
            ax_row.axhline(expected_mass, color='green', linestyle=':', 
                          label=f'Expected: {expected_mass:.6f}g', alpha=0.8)
            ax_row.legend()
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, "method_comparison_plots.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plots to: {plot_file}")
    
    # Detailed statistical summary
    summary_stats = []
    
    print(f"\nDETAILED STATISTICAL SUMMARY:")
    print("-" * 80)
    
    for vol_uL in volumes_uL:
        vol_mL = vol_uL / 1000
        expected_mass = vol_mL
        
        print(f"\n{vol_uL} µL VOLUME (Expected: {expected_mass:.6f} g):")
        print("-" * 40)
        
        for method in ['continuous', 'traditional']:
            data = results_df[(results_df['volume_uL'] == vol_uL) & 
                            (results_df['method'] == method)]['mass_diff_g']
            
            stats = {
                'volume_uL': vol_uL,
                'method': method,
                'n_trials': len(data),
                'mean_g': data.mean(),
                'std_g': data.std(),
                'cv_percent': data.std()/data.mean()*100,
                'min_g': data.min(),
                'max_g': data.max(),
                'range_g': data.max() - data.min(),
                'median_g': data.median(),
                'bias_g': data.mean() - expected_mass,
                'bias_percent': (data.mean() - expected_mass) / expected_mass * 100
            }
            summary_stats.append(stats)
            
            print(f"{method.upper():>12}: "
                  f"μ={stats['mean_g']:.6f}g, "
                  f"σ={stats['std_g']:.6f}g, "
                  f"CV={stats['cv_percent']:.3f}%, "
                  f"bias={stats['bias_g']:.6f}g ({stats['bias_percent']:.2f}%)")
    
    # Method comparison analysis (only if both methods have data)
    print(f"\nMETHOD COMPARISON ANALYSIS:")
    print("-" * 80)
    
    if simulate_mode:
        print("Skipping method comparison - only traditional method tested in simulation mode.")
        print("For full comparison analysis, set simulate_mode = False and run on hardware.")
    else:
        for vol_uL in volumes_uL:
            cont_data = results_df[(results_df['volume_uL'] == vol_uL) & 
                                  (results_df['method'] == 'continuous')]['mass_diff_g']
            trad_data = results_df[(results_df['volume_uL'] == vol_uL) & 
                                  (results_df['method'] == 'traditional')]['mass_diff_g']
            
            # Check if we have data for both methods
            if len(cont_data) == 0 or len(trad_data) == 0:
                print(f"\n{vol_uL} µL Volume: Insufficient data for comparison")
                continue
            
            # Statistical tests
            from scipy import stats as scipy_stats
            
            # F-test for equal variances
            f_stat = cont_data.var() / trad_data.var()
            f_pvalue = 2 * min(scipy_stats.f.cdf(f_stat, len(cont_data)-1, len(trad_data)-1),
                              1 - scipy_stats.f.cdf(f_stat, len(cont_data)-1, len(trad_data)-1))
            
            # t-test for equal means
            t_stat, t_pvalue = scipy_stats.ttest_ind(cont_data, trad_data)
            
            print(f"\n{vol_uL} µL Volume:")
            print(f"  Precision comparison (σ): Continuous={cont_data.std():.6f}g vs Traditional={trad_data.std():.6f}g")
            print(f"  Precision ratio: {cont_data.std()/trad_data.std():.3f} (< 1.0 = continuous better)")
            print(f"  Mean difference: {cont_data.mean() - trad_data.mean():.6f}g")
            print(f"  F-test p-value: {f_pvalue:.6f} {'(significantly different variances)' if f_pvalue < 0.05 else '(similar variances)'}")
            print(f"  t-test p-value: {t_pvalue:.6f} {'(significantly different means)' if t_pvalue < 0.05 else '(similar means)'}")
    
    # Save summary statistics
    summary_df = pd.DataFrame(summary_stats)
    summary_file = os.path.join(output_dir, "statistical_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSaved statistical summary to: {summary_file}")
    
    # Final cleanup
    print(f"\n{'='*80}")
    print("STUDY COMPLETE")
    print(f"{'='*80}")
    print(f"Total measurements: {len(results_df)}")
    print(f"Results saved in: {output_dir}")
    
    # Cleanup robot
    lash_e.nr_robot.remove_pipet()
    lash_e.nr_robot.return_vial_home(2)
    lash_e.nr_robot.move_home()

if __name__ == "__main__":
    main()