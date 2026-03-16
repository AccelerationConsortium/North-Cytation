"""
Quick sketch of synthetic turbidity and ratio functions for feedback
"""
import numpy as np
import matplotlib.pyplot as plt

def synthetic_turbidity_ratio(norm_a, norm_b):
    """
    Apply the synthetic functions to normalized concentrations
    """
    # ===== TURBIDITY SIMULATION: Narrow band from upper-right corner toward middle =====
    # High turbidity starts from upper-right corner, extends in narrow band toward center
    
    # Distance from upper-right corner (1,1)
    corner_distance = np.sqrt((1.0 - norm_a)**2 + (1.0 - norm_b)**2)
    
    # Create narrow band extending from corner toward middle (0.5, 0.5)
    # Direction vector from corner to center
    direction_to_center_a = 0.5 - 1.0  # -0.5
    direction_to_center_b = 0.5 - 1.0  # -0.5
    
    # Project current position onto the corner-to-center line
    relative_a = norm_a - 1.0
    relative_b = norm_b - 1.0
    projection_length = (relative_a * direction_to_center_a + relative_b * direction_to_center_b) / 0.707  # Normalize
    
    # Distance from the corner-to-center line (perpendicular distance)
    perp_distance = abs(relative_a * direction_to_center_b - relative_b * direction_to_center_a) / 0.707
    
    # Narrow band: high turbidity only within narrow strip
    band_width = 0.15  # Narrow band
    band_length = 0.7   # Extends 70% toward middle
    
    in_band = (perp_distance < band_width) and (projection_length > 0) and (projection_length < band_length)
    
    if in_band:
        # Strong effect near corner, weaking toward middle
        length_factor = 1.0 - (projection_length / band_length)
        width_factor = 1.0 - (perp_distance / band_width)
        turb_factor = 0.8 * length_factor * width_factor
    else:
        # Outside band - minimal turbidity
        turb_factor = 0.05
    
    turb_factor = max(0, min(1, turb_factor))
    
    # Scale to realistic range
    turbidity_baseline = 0.04
    turbidity_elevated = 3.0
    simulated_turbidity = turbidity_baseline + (turbidity_elevated - turbidity_baseline) * turb_factor
    
    # ===== RATIO SIMULATION: Inverted edge-based transitions =====
    # Ratio HIGH everywhere, transitions to LOW along edges
    
    # Edge effects: transitions when near boundaries (same as before)
    edge_a_effect = np.tanh(8.0 * (norm_a - 0.7))  # High when A > 0.7
    edge_b_effect = np.tanh(8.0 * (norm_b - 0.8))  # High when B > 0.8 (slightly different)
    
    # Combine edge effects (either edge can trigger transition)
    edge_combined = np.maximum(edge_a_effect, edge_b_effect)
    
    # Extension toward middle: weaker transition in central regions
    center_distance = np.sqrt((norm_a - 0.5)**2 + (norm_b - 0.5)**2)
    middle_extension = 0.4 * np.exp(-2.0 * center_distance)  # Weaker effect toward center
    
    # INVERT: Start high, go low at edges (opposite of before)
    ratio_factor = 0.9 - 0.4 * (edge_combined + middle_extension)  # HIGH baseline, LOW at edges
    ratio_factor = max(0, min(1, ratio_factor))
    
    # Scale to realistic range
    ratio_baseline = 0.70
    ratio_elevated = 0.85
    simulated_ratio = ratio_baseline + (ratio_elevated - ratio_baseline) * ratio_factor
    
    return simulated_turbidity, simulated_ratio

# Create grid of normalized concentrations
n_points = 100
norm_a_grid = np.linspace(0, 1, n_points)
norm_b_grid = np.linspace(0, 1, n_points)
A_mesh, B_mesh = np.meshgrid(norm_a_grid, norm_b_grid)

# Apply synthetic functions
turbidity_grid = np.zeros_like(A_mesh)
ratio_grid = np.zeros_like(A_mesh)

for i in range(n_points):
    for j in range(n_points):
        turb, ratio = synthetic_turbidity_ratio(A_mesh[i,j], B_mesh[i,j])
        turbidity_grid[i,j] = turb
        ratio_grid[i,j] = ratio

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Turbidity plot
im1 = ax1.contourf(A_mesh, B_mesh, turbidity_grid, levels=20, cmap='viridis')
ax1.set_xlabel('Normalized Surf A Concentration')
ax1.set_ylabel('Normalized Surf B Concentration')
ax1.set_title('Synthetic Turbidity Pattern\n(Diagonal truncated at 60%, corner peak)')
plt.colorbar(im1, ax=ax1, label='Turbidity')

# Ratio plot  
im2 = ax2.contourf(A_mesh, B_mesh, ratio_grid, levels=20, cmap='plasma')
ax2.set_xlabel('Normalized Surf A Concentration')
ax2.set_ylabel('Normalized Surf B Concentration') 
ax2.set_title('Synthetic Ratio Pattern\n(Edge transitions extending to middle)')
plt.colorbar(im2, ax=ax2, label='Ratio')

plt.tight_layout()
plt.show()

# Print value ranges for verification
print(f"Turbidity range: {np.min(turbidity_grid):.3f} - {np.max(turbidity_grid):.3f}")
print(f"Ratio range: {np.min(ratio_grid):.3f} - {np.max(ratio_grid):.3f}")