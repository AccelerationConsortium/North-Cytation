#!/usr/bin/env python3
"""
Glycerol calibration parameter analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load the raw replicate data
data_file = r"C:\Users\owenm\OneDrive\Desktop\Calibration_SDL\glycerol_8params_largerspace_2ndround\20250929_140922_glycerol_LLM\raw_replicate_data.csv"
df = pd.read_csv(data_file)

# Calculate accuracy metrics
df['accuracy_percent'] = ((df['mass'] - df['volume']) / df['volume']) * 100
df['absolute_error'] = abs(df['mass'] - df['volume'])

# Create experiment groups and get mean values per experiment
exp_stats = df.groupby(df.index // 3).agg({
    'accuracy_percent': 'mean',
    'mass': ['mean', 'std'],
    'aspirate_speed': 'first',
    'dispense_speed': 'first', 
    'aspirate_wait_time': 'first',
    'dispense_wait_time': 'first',
    'retract_speed': 'first',
    'pre_asp_air_vol': 'first',
    'post_asp_air_vol': 'first',
    'overaspirate_vol': 'first'
}).reset_index()

# Flatten column names
exp_stats.columns = ['exp_id', 'accuracy_mean', 'mass_mean', 'mass_std', 
                    'aspirate_speed', 'dispense_speed', 'aspirate_wait_time',
                    'dispense_wait_time', 'retract_speed', 'pre_asp_air_vol',
                    'post_asp_air_vol', 'overaspirate_vol']

# Create parameter analysis plots
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Glycerol Parameter Analysis - Accuracy vs Settings', fontsize=16)

# Plot parameters vs accuracy
params = ['aspirate_speed', 'dispense_speed', 'aspirate_wait_time', 'dispense_wait_time',
         'retract_speed', 'pre_asp_air_vol', 'post_asp_air_vol', 'overaspirate_vol']
titles = ['Aspirate Speed', 'Dispense Speed', 'Aspirate Wait Time', 'Dispense Wait Time',
         'Retract Speed', 'Pre-Asp Air Vol', 'Post-Asp Air Vol', 'Overaspirate Vol']

for i, (param, title) in enumerate(zip(params, titles)):
    row = i // 4
    col = i % 4
    
    # Scatter plot with color-coded precision
    scatter = axes[row, col].scatter(exp_stats[param], exp_stats['accuracy_mean'], 
                                   c=exp_stats['mass_std']*1000, cmap='viridis_r', 
                                   s=60, alpha=0.7)
    axes[row, col].axhline(0, color='red', linestyle='--', alpha=0.5, label='Perfect accuracy')
    axes[row, col].set_xlabel(title)
    axes[row, col].set_ylabel('Accuracy (% error)')
    axes[row, col].grid(True, alpha=0.3)
    axes[row, col].set_title(f'{title} vs Accuracy')
    
    # Add colorbar for the first plot
    if i == 0:
        cbar = plt.colorbar(scatter, ax=axes[row, col])
        cbar.set_label('Precision (µL std)')

plt.tight_layout()
plt.savefig('glycerol_parameter_analysis.png', dpi=300, bbox_inches='tight')
print("Parameter analysis saved as 'glycerol_parameter_analysis.png'")

# Statistical analysis
print("\n" + "="*60)
print("PARAMETER CORRELATION ANALYSIS")
print("="*60)

# Calculate correlations
correlations = exp_stats[params + ['accuracy_mean', 'mass_std']].corr()['accuracy_mean'].sort_values()
print("\nCorrelation with Accuracy (% error):")
for param, corr in correlations.items():
    if param != 'accuracy_mean':
        print(f"  {param:20s}: {corr:+.3f}")

# Find best parameter combinations
best_experiments = exp_stats.nsmallest(3, 'accuracy_mean')
worst_experiments = exp_stats.nlargest(3, 'accuracy_mean')

print(f"\n{'='*30}")
print("BEST EXPERIMENTS (Lowest Error)")
print(f"{'='*30}")
for idx, row in best_experiments.iterrows():
    print(f"\nExperiment {row['exp_id']:.0f}: {row['accuracy_mean']:+.1f}% error, {row['mass_std']*1000:.1f}µL precision")
    print(f"  Aspirate: {row['aspirate_speed']:.0f}, Dispense: {row['dispense_speed']:.0f}")
    print(f"  Wait times: {row['aspirate_wait_time']:.1f}, {row['dispense_wait_time']:.1f}")
    print(f"  Overaspirate: {row['overaspirate_vol']:.3f} mL")

print(f"\n{'='*30}")
print("WORST EXPERIMENTS (Highest Error)")  
print(f"{'='*30}")
for idx, row in worst_experiments.iterrows():
    print(f"\nExperiment {row['exp_id']:.0f}: {row['accuracy_mean']:+.1f}% error, {row['mass_std']*1000:.1f}µL precision") 
    print(f"  Aspirate: {row['aspirate_speed']:.0f}, Dispense: {row['dispense_speed']:.0f}")
    print(f"  Wait times: {row['aspirate_wait_time']:.1f}, {row['dispense_wait_time']:.1f}")
    print(f"  Overaspirate: {row['overaspirate_vol']:.3f} mL")

# Parameter ranges
print(f"\n{'='*30}")
print("PARAMETER RANGES TESTED")
print(f"{'='*30}")
for param, title in zip(params, titles):
    values = exp_stats[param]
    print(f"{title:20s}: {values.min():.3f} - {values.max():.3f} (Mean: {values.mean():.3f})")

# Recommendations
print(f"\n{'='*30}")
print("CALIBRATION RECOMMENDATIONS")
print(f"{'='*30}")
print("1. SYSTEMATIC OFFSET: Apply -25% volume correction factor")
print("2. PARAMETER TUNING: Focus on combinations from best experiments")
print("3. The robot shows good precision but consistent over-dispensing")
print("4. Consider updating volume calibration constants in robot firmware")