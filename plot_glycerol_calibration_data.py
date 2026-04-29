#!/usr/bin/env python3
"""
Plot glycerol calibration data showing pipetting accuracy and precision.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns

# Load the raw replicate data
data_file = r"C:\Users\owenm\OneDrive\Desktop\Calibration_SDL\glycerol_8params_largerspace_2ndround\20250929_140922_glycerol_LLM\raw_replicate_data.csv"
df = pd.read_csv(data_file)

# Calculate accuracy metrics
df['accuracy_percent'] = ((df['mass'] - df['volume']) / df['volume']) * 100
df['absolute_error'] = abs(df['mass'] - df['volume'])

# Create experiment groups (each set of 3 replicates)
df['experiment_id'] = df.index // 3

# Calculate summary statistics per experiment
summary_stats = df.groupby('experiment_id').agg({
    'mass': ['mean', 'std', 'count'],
    'accuracy_percent': ['mean', 'std'],
    'absolute_error': 'mean',
    'volume': 'first',  # Target volume 
    'aspirate_speed': 'first',
    'dispense_speed': 'first',
    'aspirate_wait_time': 'first',
    'dispense_wait_time': 'first',
    'overaspirate_vol': 'first'
}).reset_index()

# Flatten column names
summary_stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in summary_stats.columns.values]

# Create comprehensive plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Glycerol Pipetting Calibration Analysis', fontsize=16, fontweight='bold')

# Plot 1: Measured vs Target Volume
axes[0,0].scatter(df['volume'], df['mass'], alpha=0.6, s=30)
axes[0,0].plot([0.045, 0.055], [0.045, 0.055], 'r-', linewidth=2, label='Perfect accuracy')
axes[0,0].set_xlabel('Target Volume (mL)')
axes[0,0].set_ylabel('Measured Volume (mL)')
axes[0,0].set_title('Measured vs Target Volume')
axes[0,0].grid(True, alpha=0.3)
axes[0,0].legend()

# Add accuracy bands
target = df['volume'].iloc[0]
axes[0,0].axhspan(target*0.95, target*1.05, alpha=0.2, color='green', label='±5%')
axes[0,0].axhspan(target*0.90, target*1.10, alpha=0.1, color='yellow', label='±10%')

# Plot 2: Accuracy over time/experiments
axes[0,1].scatter(summary_stats['experiment_id'], summary_stats['accuracy_percent_mean'], 
                 alpha=0.7, s=40, c=summary_stats['accuracy_percent_std'], cmap='viridis')
axes[0,1].axhline(0, color='red', linestyle='--', alpha=0.7, label='Perfect accuracy')
axes[0,1].axhline(5, color='orange', linestyle='--', alpha=0.5, label='±5%')
axes[0,1].axhline(-5, color='orange', linestyle='--', alpha=0.5)
axes[0,1].set_xlabel('Experiment Number')
axes[0,1].set_ylabel('Accuracy (% error)')
axes[0,1].set_title('Accuracy Over Experiments')
axes[0,1].grid(True, alpha=0.3)
axes[0,1].legend()

# Add colorbar for standard deviation
cbar = plt.colorbar(axes[0,1].collections[0], ax=axes[0,1])
cbar.set_label('Precision (Std Dev %)')

# Plot 3: Precision (reproducibility) over experiments
axes[0,2].plot(summary_stats['experiment_id'], summary_stats['mass_std']*1000, 'o-', markersize=4)
axes[0,2].axhline(1.0, color='red', linestyle='--', alpha=0.7, label='1 µL std')
axes[0,2].axhline(2.0, color='orange', linestyle='--', alpha=0.7, label='2 µL std')
axes[0,2].set_xlabel('Experiment Number')  
axes[0,2].set_ylabel('Precision (µL std dev)')
axes[0,2].set_title('Precision Over Experiments')
axes[0,2].grid(True, alpha=0.3)
axes[0,2].legend()

# Plot 4: Accuracy vs Aspirate Speed
axes[1,0].scatter(summary_stats['aspirate_speed'], summary_stats['accuracy_percent_mean'], 
                 alpha=0.7, s=60, c=summary_stats['accuracy_percent_std'], cmap='plasma')
axes[1,0].axhline(0, color='red', linestyle='--', alpha=0.7)
axes[1,0].set_xlabel('Aspirate Speed')
axes[1,0].set_ylabel('Accuracy (% error)')
axes[1,0].set_title('Accuracy vs Aspirate Speed')
axes[1,0].grid(True, alpha=0.3)

# Plot 5: Accuracy vs Dispense Speed  
axes[1,1].scatter(summary_stats['dispense_speed'], summary_stats['accuracy_percent_mean'],
                 alpha=0.7, s=60, c=summary_stats['accuracy_percent_std'], cmap='plasma')
axes[1,1].axhline(0, color='red', linestyle='--', alpha=0.7)
axes[1,1].set_xlabel('Dispense Speed')
axes[1,1].set_ylabel('Accuracy (% error)')
axes[1,1].set_title('Accuracy vs Dispense Speed')
axes[1,1].grid(True, alpha=0.3)

# Plot 6: Distribution of measurements
axes[1,2].hist(df['mass']*1000, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[1,2].axvline(df['volume'].iloc[0]*1000, color='red', linestyle='--', linewidth=2, label='Target (50 µL)')
axes[1,2].set_xlabel('Measured Volume (µL)')
axes[1,2].set_ylabel('Frequency')
axes[1,2].set_title('Distribution of Measured Volumes')
axes[1,2].grid(True, alpha=0.3)
axes[1,2].legend()

plt.tight_layout()
plt.savefig('glycerol_calibration_analysis.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'glycerol_calibration_analysis.png'")
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("GLYCEROL PIPETTING CALIBRATION SUMMARY")
print("="*60)

print(f"\nTarget Volume: {df['volume'].iloc[0]*1000:.1f} µL")
print(f"Number of Experiments: {len(summary_stats)}")
print(f"Replicates per Experiment: 3")
print(f"Total Measurements: {len(df)}")

print(f"\nOverall Accuracy:")
mean_accuracy = df['accuracy_percent'].mean()
std_accuracy = df['accuracy_percent'].std()
print(f"  Mean Error: {mean_accuracy:+.2f}% (±{std_accuracy:.2f}%)")
print(f"  Absolute Error: {df['absolute_error'].mean()*1000:.2f} ± {df['absolute_error'].std()*1000:.2f} µL")

print(f"\nOverall Precision:")
mean_precision = summary_stats['mass_std'].mean() * 1000
print(f"  Average Std Dev: {mean_precision:.2f} µL")
print(f"  CV: {(summary_stats['mass_std'].mean() / summary_stats['mass_mean'].mean() * 100):.2f}%")

print(f"\nBest Performance:")
best_exp = summary_stats.loc[summary_stats['accuracy_percent_mean'].abs().idxmin()]
print(f"  Experiment {best_exp['experiment_id']:.0f}: {best_exp['accuracy_percent_mean']:+.2f}% error, {best_exp['mass_std']*1000:.2f} µL precision")

print(f"\nWorst Performance:")  
worst_exp = summary_stats.loc[summary_stats['accuracy_percent_mean'].abs().idxmax()]
print(f"  Experiment {worst_exp['experiment_id']:.0f}: {worst_exp['accuracy_percent_mean']:+.2f}% error, {worst_exp['mass_std']*1000:.2f} µL precision")

# Parameter ranges
print(f"\nParameter Ranges:")
print(f"  Aspirate Speed: {df['aspirate_speed'].min()}-{df['aspirate_speed'].max()}")
print(f"  Dispense Speed: {df['dispense_speed'].min()}-{df['dispense_speed'].max()}")  
print(f"  Wait Times: {df['aspirate_wait_time'].min()}-{df['aspirate_wait_time'].max()}, {df['dispense_wait_time'].min()}-{df['dispense_wait_time'].max()}")
print(f"  Overaspirate: {df['overaspirate_vol'].min()}-{df['overaspirate_vol'].max()} mL")