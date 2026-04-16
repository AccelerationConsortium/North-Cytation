#!/usr/bin/env python3
"""
Safe analysis of active glycerol experiment results.
Uses snapshot approach to avoid file I/O conflicts with running workflow.
"""

import pandas as pd
import shutil
import os
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from datetime import datetime

def analyze_active_results():
    """Analyze current experiment results without interfering with active workflow."""
    
    # File paths
    source_file = 'output/glycerol_sobol_exp1_20260415_122943/200uL/incremental_results.csv'
    snapshot_file = 'temp_analysis_snapshot.csv'
    
    print('=== SAFE SNAPSHOT ANALYSIS ===')
    print(f'Source: {source_file}')
    print(f'Snapshot: {snapshot_file}')
    print()
    
    try:
        # Create safe snapshot copy
        shutil.copy2(source_file, snapshot_file)
        print('✅ Snapshot created successfully')
        
        # Read snapshot (no interference with active workflow)
        df = pd.read_csv(snapshot_file)
        total_rows = len(df)
        print(f'📊 Rows in snapshot: {total_rows}')
        
        if total_rows == 0:
            print('❌ No data to analyze')
            return
        
        # Time distribution analysis
        print('\n--- ELAPSED TIME ANALYSIS ---')
        print(f'Mean time: {df["elapsed_s"].mean():.1f} seconds')
        print(f'Median time: {df["elapsed_s"].median():.1f} seconds')
        print(f'Min time: {df["elapsed_s"].min():.1f} seconds')
        print(f'Max time: {df["elapsed_s"].max():.1f} seconds')
        print(f'Std dev: {df["elapsed_s"].std():.1f} seconds')
        
        # Accuracy distribution analysis
        print('\n--- ACCURACY ANALYSIS ---')
        print(f'Mean accuracy: {df["accuracy_pct"].mean():.2f}%')
        print(f'Median accuracy: {df["accuracy_pct"].median():.2f}%')
        print(f'Min accuracy: {df["accuracy_pct"].min():.2f}%')
        print(f'Max accuracy: {df["accuracy_pct"].max():.2f}%')
        print(f'Std dev: {df["accuracy_pct"].std():.2f}%')
        
        # Accuracy ranges
        good_acc = ((df["accuracy_pct"] >= 95) & (df["accuracy_pct"] <= 105)).sum()
        acceptable_acc = ((df["accuracy_pct"] >= 90) & (df["accuracy_pct"] <= 110)).sum()
        poor_acc = ((df["accuracy_pct"] < 90) | (df["accuracy_pct"] > 110)).sum()
        print(f'Good accuracy (95-105%): {good_acc} rows ({good_acc/total_rows*100:.1f}%)')
        print(f'Acceptable accuracy (90-110%): {acceptable_acc} rows ({acceptable_acc/total_rows*100:.1f}%)')
        print(f'Poor accuracy (<90% or >110%): {poor_acc} rows ({poor_acc/total_rows*100:.1f}%)')
        
        # Volume target distribution
        print('\n--- VOLUME TARGET ANALYSIS ---')
        print(f'Mean target: {df["volume_ul_target"].mean():.1f} uL')
        print(f'Min target: {df["volume_ul_target"].min():.1f} uL')
        print(f'Max target: {df["volume_ul_target"].max():.1f} uL')
        print(f'Std dev: {df["volume_ul_target"].std():.1f} uL')
        
        # Environmental data (if available)
        if 'temp_c' in df.columns:
            print('\n--- ENVIRONMENTAL CONDITIONS ---')
            print(f'Mean temp: {df["temp_c"].mean():.1f}°C')
            print(f'Mean humidity: {df["humidity_pct"].mean():.1f}%')
            print(f'Mean pressure: {df["pressure_pa"].mean():.0f} Pa')
        
        # Bottle age tracking (if available)
        if 'days_bottle_opened' in df.columns:
            print('\n--- GLYCEROL BOTTLE STATUS ---')
            print(f'Bottle opened: {df["glycerol_opened_date"].iloc[0]}')
            print(f'Days since opened: {df["days_bottle_opened"].iloc[0]} days')
        
        # Progress tracking
        print('\n--- EXPERIMENT PROGRESS ---')
        print(f'Experiments completed: {total_rows}')
        # Estimate based on Sobol file (5120 total experiments)
        progress_pct = (total_rows / 5120) * 100
        print(f'Progress: {progress_pct:.1f}% (assuming 5120 total experiments)')
        
        # Identify slow experiments (>300s)
        print('\n--- SLOW EXPERIMENT ANALYSIS ---')
        slow_threshold = 300  # seconds
        slow_experiments = df[df["elapsed_s"] > slow_threshold].copy()
        
        if len(slow_experiments) > 0:
            print(f'⚠️ Found {len(slow_experiments)} experiments > {slow_threshold}s:')
            print(f'Slowest experiment: {slow_experiments["elapsed_s"].max():.1f}s')
            
            # Show parameter patterns for slow experiments
            slow_params = slow_experiments[["row_index", "elapsed_s", "volume_ul_target", 
                                          "aspirate_speed", "dispense_speed", 
                                          "aspirate_wait_time", "dispense_wait_time",
                                          "accuracy_pct"]].copy()
            
            # Sort by time (slowest first)
            slow_params = slow_params.sort_values("elapsed_s", ascending=False)
            
            print('\nSLOWEST EXPERIMENTS (top 10):')
            print(slow_params.head(10).to_string(index=False))
            
            # Analyze parameter correlations with slow runs
            print('\nPARAMETER ANALYSIS FOR SLOW RUNS:')
            print(f'Volume range: {slow_experiments["volume_ul_target"].min():.1f} - {slow_experiments["volume_ul_target"].max():.1f} uL')
            print(f'Aspirate speed range: {slow_experiments["aspirate_speed"].min()} - {slow_experiments["aspirate_speed"].max()}')
            print(f'Dispense speed range: {slow_experiments["dispense_speed"].min()} - {slow_experiments["dispense_speed"].max()}')
            print(f'Aspirate wait time range: {slow_experiments["aspirate_wait_time"].min():.1f} - {slow_experiments["aspirate_wait_time"].max():.1f}s')
            print(f'Dispense wait time range: {slow_experiments["dispense_wait_time"].min():.1f} - {slow_experiments["dispense_wait_time"].max():.1f}s')
            
            # Save detailed slow experiment data
            slow_output_path = os.path.join(os.path.dirname(source_file), 'slow_experiments_analysis.csv')
            slow_experiments.to_csv(slow_output_path, index=False)
            print(f'\n📄 Detailed slow experiment data saved: {os.path.basename(slow_output_path)}')
            
        else:
            print(f'✅ No experiments found > {slow_threshold}s (all experiments running efficiently)')
        
        # Generate histograms
        print('\n--- GENERATING HISTOGRAMS ---')
        try:
            output_dir = os.path.dirname(source_file)  # Same folder as CSV
            
            # Time histogram
            plt.figure(figsize=(10, 6))
            plt.hist(df["elapsed_s"], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            plt.xlabel('Elapsed Time (seconds)')
            plt.ylabel('Frequency')
            plt.title(f'Distribution of Experiment Times (n={total_rows})')
            plt.grid(True, alpha=0.3)
            time_plot_path = os.path.join(output_dir, 'time_distribution_histogram.png')
            plt.savefig(time_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f'📊 Time histogram saved: {os.path.basename(time_plot_path)}')
            
            # Accuracy histogram
            plt.figure(figsize=(10, 6))
            plt.hist(df["accuracy_pct"], bins=30, alpha=0.7, color='forestgreen', edgecolor='black')
            plt.xlabel('Accuracy (%)')
            plt.ylabel('Frequency')
            plt.title(f'Distribution of Pipetting Accuracy (n={total_rows})')
            plt.grid(True, alpha=0.3)
            # Add vertical lines for reference
            plt.axvline(x=100, color='green', linestyle='-', alpha=0.8, label='Target (100%)')
            plt.axvline(x=95, color='orange', linestyle='--', alpha=0.7, label='Good Range (95-105%)')
            plt.axvline(x=105, color='orange', linestyle='--', alpha=0.7, label='')
            plt.axvline(x=90, color='red', linestyle='--', alpha=0.7, label='Acceptable Range (90-110%)')
            plt.axvline(x=110, color='red', linestyle='--', alpha=0.7, label='')
            plt.legend()
            acc_plot_path = os.path.join(output_dir, 'accuracy_distribution_histogram.png')
            plt.savefig(acc_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f'📊 Accuracy histogram saved: {os.path.basename(acc_plot_path)}')
            
            # Time-Accuracy Scatter Plot (clipped at 300s)
            plt.figure(figsize=(12, 8))
            
            # Clip time at 300s for better visualization
            time_clipped = df["elapsed_s"].clip(upper=300)  
            
            # Create scatter plot with color-coded points
            scatter = plt.scatter(time_clipped, df["accuracy_pct"], 
                                alpha=0.6, s=50, c=df["volume_ul_target"], 
                                cmap='viridis', edgecolors='black', linewidth=0.5)
            
            # Add colorbar for volume
            cbar = plt.colorbar(scatter)
            cbar.set_label('Target Volume (µL)', rotation=270, labelpad=20)
            
            # Reference lines
            plt.axhline(y=100, color='green', linestyle='-', alpha=0.8, label='Target (100%)')
            plt.axhline(y=95, color='orange', linestyle='--', alpha=0.7, label='Good Range (95-105%)')
            plt.axhline(y=105, color='orange', linestyle='--', alpha=0.7, label='')
            plt.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='Acceptable Range (90-110%)')
            plt.axhline(y=110, color='red', linestyle='--', alpha=0.7, label='')
            plt.axvline(x=300, color='purple', linestyle='--', alpha=0.7, label='Time Clip (300s)')
            
            plt.xlabel('Elapsed Time (seconds, clipped at 300s)')
            plt.ylabel('Accuracy (%)')
            plt.title(f'Time vs Accuracy Analysis (n={total_rows})')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Set reasonable axis limits - don't artificially clip the low end
            plt.xlim(df["elapsed_s"].min() - 5, 305)
            plt.ylim(df["accuracy_pct"].min() - 2, df["accuracy_pct"].max() + 2)
            
            pareto_plot_path = os.path.join(output_dir, 'time_accuracy_plot.png')
            plt.savefig(pareto_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f'📊 Time-Accuracy plot saved: {os.path.basename(pareto_plot_path)}')
            
            print(f'📁 Plots saved to: {output_dir}')
            
        except Exception as plot_error:
            print(f'⚠️ Could not generate histograms: {plot_error}')
            print('Analysis data is still available above.')
        
        # Clean up snapshot
        os.remove(snapshot_file)
        print(f'\n🗑️ Cleanup: Removed {snapshot_file}')
        
    except FileNotFoundError:
        print(f'❌ Error: Could not find {source_file}')
        print('Make sure the path is correct and the experiment is running.')
        
    except Exception as e:
        print(f'❌ Error: {e}')
        
        # Clean up on error
        if os.path.exists(snapshot_file):
            os.remove(snapshot_file)
            print(f'🗑️ Error cleanup: Removed {snapshot_file}')

if __name__ == "__main__":
    analyze_active_results()