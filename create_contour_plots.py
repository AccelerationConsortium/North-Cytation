#!/usr/bin/env python3
"""
Create contour plots for surfactant experimental data.
Generates colored contour maps for turbidity (log scale) and ratio (linear scale).
"""

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    import matplotlib.colors as colors
    print("All imports successful")
except ImportError as e:
    print(f"Import error: {e}")
    exit(1)

# Read the experimental data
data_file = r"C:\Users\Imaging Controller\Desktop\utoronto_demo\output\surfactant_grid_SDS_TTAB_20260212_205209_HARDWARE\iterative_experiment_results.csv"
df = pd.read_csv(data_file)

# Filter to experimental wells only
exp_data = df[df['well_type'] == 'experiment'].copy()

# Extract concentrations and measurements
x = np.log10(exp_data['surf_A_conc_mm'].values)  # Log scale SDS concentrations  
y = np.log10(exp_data['surf_B_conc_mm'].values)  # Log scale TTAB concentrations
turbidity = exp_data['turbidity_600'].values
ratio_values = exp_data['ratio'].values

print(f"Data points: {len(exp_data)}")
print(f"SDS concentration range: {exp_data['surf_A_conc_mm'].min():.2e} to {exp_data['surf_A_conc_mm'].max():.2e} mM")
print(f"TTAB concentration range: {exp_data['surf_B_conc_mm'].min():.2e} to {exp_data['surf_B_conc_mm'].max():.2e} mM")
print(f"Turbidity range: {turbidity.min():.4f} to {turbidity.max():.4f}")
print(f"Ratio range: {ratio_values.min():.3f} to {ratio_values.max():.3f}")

# Create interpolation grids (log space)
xi = np.linspace(x.min(), x.max(), 50)
yi = np.linspace(y.min(), y.max(), 50)
xi_grid, yi_grid = np.meshgrid(xi, yi)

# Interpolate data onto grid
turbidity_grid = griddata((x, y), turbidity, (xi_grid, yi_grid), method='cubic')
ratio_grid = griddata((x, y), ratio_values, (xi_grid, yi_grid), method='cubic')

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Turbidity with logarithmic contour levels
turbidity_min, turbidity_max = turbidity.min(), turbidity.max()
# Create log-spaced contour levels (fewer levels since only 96 data points)
turb_levels = np.logspace(np.log10(max(turbidity_min, 0.01)), np.log10(turbidity_max), 8)

cs1 = ax1.contourf(xi_grid, yi_grid, turbidity_grid, levels=turb_levels, 
                   cmap='viridis', norm=colors.LogNorm(vmin=turbidity_min, vmax=turbidity_max))
ax1.contour(xi_grid, yi_grid, turbidity_grid, levels=turb_levels, colors='white', alpha=0.3, linewidths=0.5)

# Add data points
scatter1 = ax1.scatter(x, y, c=turbidity, s=40, cmap='viridis', edgecolors='white', linewidth=0.5,
                       norm=colors.LogNorm(vmin=turbidity_min, vmax=turbidity_max))

# Format axes for turbidity plot
ax1.set_xlabel('log₁₀(SDS Concentration [mM])', fontsize=12)
ax1.set_ylabel('log₁₀(TTAB Concentration [mM])', fontsize=12)
ax1.set_title('Turbidity at 600 nm (Log Scale)', fontsize=14, fontweight='bold')

# Add colorbar for turbidity
cbar1 = plt.colorbar(cs1, ax=ax1)
cbar1.set_label('Turbidity (600 nm)', fontsize=11)

# Add concentration labels on top and right axes
ax1_top = ax1.twiny()
ax1_right = ax1.twinx()

# Convert log ticks back to actual concentrations
x_ticks = ax1.get_xticks()
y_ticks = ax1.get_yticks()

# Create concentration labels
x_conc_labels = [f'{10**x:.0e}' for x in x_ticks if x_ticks[0] <= x <= x_ticks[-1]]
y_conc_labels = [f'{10**y:.0e}' for y in y_ticks if y_ticks[0] <= y <= y_ticks[-1]]

ax1_top.set_xlim(ax1.get_xlim())
ax1_top.set_xticks(x_ticks)
ax1_top.set_xticklabels(x_conc_labels, fontsize=9)
ax1_top.set_xlabel('SDS Concentration [mM]', fontsize=10)

ax1_right.set_ylim(ax1.get_ylim())
ax1_right.set_yticks(y_ticks)
ax1_right.set_yticklabels(y_conc_labels, fontsize=9)
ax1_right.set_ylabel('TTAB Concentration [mM]', fontsize=10)

# Plot 2: Ratio with linear contour levels
ratio_min, ratio_max = ratio_values.min(), ratio_values.max()
# Create linear-spaced contour levels (fewer levels since only 96 data points)
ratio_levels = np.linspace(ratio_min, ratio_max, 8)

cs2 = ax2.contourf(xi_grid, yi_grid, ratio_grid, levels=ratio_levels, cmap='plasma')
ax2.contour(xi_grid, yi_grid, ratio_grid, levels=ratio_levels, colors='white', alpha=0.3, linewidths=0.5)

# Add data points
scatter2 = ax2.scatter(x, y, c=ratio_values, s=40, cmap='plasma', edgecolors='white', linewidth=0.5,
                       vmin=ratio_min, vmax=ratio_max)

# Format axes for ratio plot
ax2.set_xlabel('log₁₀(SDS Concentration [mM])', fontsize=12)
ax2.set_ylabel('log₁₀(TTAB Concentration [mM])', fontsize=12)
ax2.set_title('Fluorescence Ratio (F373/F384)', fontsize=14, fontweight='bold')

# Add colorbar for ratio
cbar2 = plt.colorbar(cs2, ax=ax2)
cbar2.set_label('Ratio (F373/F384)', fontsize=11)

# Add concentration labels on top and right axes for ratio plot
ax2_top = ax2.twiny()
ax2_right = ax2.twinx()

ax2_top.set_xlim(ax2.get_xlim())
ax2_top.set_xticks(x_ticks)
ax2_top.set_xticklabels(x_conc_labels, fontsize=9)
ax2_top.set_xlabel('SDS Concentration [mM]', fontsize=10)

ax2_right.set_ylim(ax2.get_ylim())
ax2_right.set_yticks(y_ticks)
ax2_right.set_yticklabels(y_conc_labels, fontsize=9)
ax2_right.set_ylabel('TTAB Concentration [mM]', fontsize=10)

# Adjust layout and save
plt.tight_layout()

# Save the plot
output_file = r"C:\Users\Imaging Controller\Desktop\utoronto_demo\output\surfactant_grid_SDS_TTAB_20260212_205209_HARDWARE\contour_plots.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.show()

print(f"Contour plots saved to: {output_file}")