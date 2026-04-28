#!/usr/bin/env python3
"""
Simple glycerol calibration data analysis - debugging version.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Load the raw replicate data
data_file = r"C:\Users\owenm\OneDrive\Desktop\Calibration_SDL\glycerol_8params_largerspace\20250929_113019_glycerol_LLM\raw_replicate_data.csv"
df = pd.read_csv(data_file)

# Glycerol density at room temperature (~1.26 g/mL)
GLYCEROL_DENSITY = 1.26  # g/mL

print("Column names:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())
print(f"\nData shape: {df.shape}")
print(f"Target volume: {df['volume'].iloc[0]*1000} µL")

# Convert mass to volume using glycerol density
df['measured_volume'] = df['mass'] / GLYCEROL_DENSITY

# Calculate accuracy metrics using actual volume
df['accuracy_percent'] = ((df['measured_volume'] - df['volume']) / df['volume']) * 100
df['absolute_error'] = abs(df['measured_volume'] - df['volume'])

# Convert timestamps to datetime for time-based plotting
df['start_datetime'] = pd.to_datetime(df['start_time'])
df['end_datetime'] = pd.to_datetime(df['end_time'])

# Create experiment groups (each set of 3 replicates)
df['experiment_id'] = df.index // 3

print(f"\nNumber of experiments: {df['experiment_id'].max() + 1}")
print(f"Accuracy range: {df['accuracy_percent'].min():.1f}% to {df['accuracy_percent'].max():.1f}%")
print(f"Mean accuracy: {df['accuracy_percent'].mean():.2f} ± {df['accuracy_percent'].std():.2f}%")

# Create simple plots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Glycerol Pipetting Calibration Analysis', fontsize=14)

# Plot 1: Measured vs Target Volume
axes[0,0].scatter(df['volume']*1000, df['measured_volume']*1000, alpha=0.6, s=20)
axes[0,0].plot([49, 51], [49, 51], 'r-', linewidth=2, label='Perfect accuracy')
axes[0,0].set_xlabel('Target Volume (µL)')
axes[0,0].set_ylabel('Measured Volume (µL)')
axes[0,0].set_title('Measured vs Target Volume')
axes[0,0].grid(True, alpha=0.3)
axes[0,0].legend()

# Plot 2: Accuracy over experiments
exp_accuracy = df.groupby('experiment_id')['accuracy_percent'].mean()
axes[0,1].plot(exp_accuracy.index, exp_accuracy.values, 'o-', markersize=4)
axes[0,1].axhline(0, color='red', linestyle='--', alpha=0.7, label='Perfect accuracy')
axes[0,1].axhline(5, color='orange', linestyle='--', alpha=0.5, label='±5%')
axes[0,1].axhline(-5, color='orange', linestyle='--', alpha=0.5)
axes[0,1].set_xlabel('Experiment Number')
axes[0,1].set_ylabel('Accuracy (% error)')
axes[0,1].set_title('Accuracy Over Time')
axes[0,1].grid(True, alpha=0.3)
axes[0,1].legend()

# Plot 3: Measured Volume Over Experiments
axes[1,0].scatter(df['experiment_id'], df['measured_volume']*1000, alpha=0.6, s=20, c=df['replicate'], cmap='Set1')
axes[1,0].axhline(df['volume'].iloc[0]*1000, color='red', linestyle='--', linewidth=2, label='Target (50 µL)')
axes[1,0].set_xlabel('Experiment Number')
axes[1,0].set_ylabel('Measured Volume (µL)')
axes[1,0].set_title('Measured Volume Over Experiments')
axes[1,0].grid(True, alpha=0.3)
axes[1,0].legend()

# Plot 4: Precision over experiments
exp_precision = df.groupby('experiment_id')['measured_volume'].std() * 1000
axes[1,1].plot(exp_precision.index, exp_precision.values, 'o-', markersize=4, color='green')
axes[1,1].axhline(1.0, color='red', linestyle='--', alpha=0.7, label='1 µL std')
axes[1,1].axhline(2.0, color='orange', linestyle='--', alpha=0.7, label='2 µL std')
axes[1,1].set_xlabel('Experiment Number')
axes[1,1].set_ylabel('Precision (µL std dev)')
axes[1,1].set_title('Precision Over Experiments')
axes[1,1].grid(True, alpha=0.3)
axes[1,1].legend()

# Plot 5: Distribution of measurements
axes[1,2].hist(df['measured_volume']*1000, bins=25, alpha=0.7, color='skyblue', edgecolor='black')
axes[1,2].axvline(df['volume'].iloc[0]*1000, color='red', linestyle='--', linewidth=2, 
                 label=f'Target ({df["volume"].iloc[0]*1000:.0f} µL)')
axes[1,2].set_xlabel('Measured Volume (µL)')
axes[1,2].set_ylabel('Frequency')
axes[1,2].set_title('Distribution of Measurements')
axes[1,2].grid(True, alpha=0.3)
axes[1,2].legend()

plt.tight_layout()
plt.savefig('glycerol_calibration_simple.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'glycerol_calibration_simple.png'")

# Summary statistics
print("\n" + "="*50)
print("CALIBRATION SUMMARY")
print("="*50)
print(f"Target Volume: {df['volume'].iloc[0]*1000:.1f} µL")
print(f"Measurements: {len(df)} total ({len(df)//3} experiments × 3 replicates)")
print(f"Mean Error: {df['accuracy_percent'].mean():+.2f}% ± {df['accuracy_percent'].std():.2f}%")
print(f"Absolute Error: {df['absolute_error'].mean()*1000:.2f} ± {df['absolute_error'].std()*1000:.2f} µL")
print(f"Range: {df['measured_volume'].min()*1000:.1f} - {df['measured_volume'].max()*1000:.1f} µL")
print(f"Density correction applied: {GLYCEROL_DENSITY} g/mL")

# Best and worst experiments
exp_stats = df.groupby('experiment_id').agg({
    'accuracy_percent': 'mean',
    'measured_volume': 'std'
}).reset_index()

best_idx = exp_stats['accuracy_percent'].abs().idxmin()
worst_idx = exp_stats['accuracy_percent'].abs().idxmax()

print(f"\nBest Experiment (#{best_idx}): {exp_stats.loc[best_idx, 'accuracy_percent']:+.2f}% error")
print(f"Worst Experiment (#{worst_idx}): {exp_stats.loc[worst_idx, 'accuracy_percent']:+.2f}% error")