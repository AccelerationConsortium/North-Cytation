"""
Rectangle Identification with Directional Gradients
Uses the same rectangle method as surfactant_grid_adaptive_concentrations,
then calculates max-min gradients in SDS and TTAB directions for each rectangle.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
import os

def load_experiment_data(csv_file_path):
    """Load and classify the experimental data."""
    df = pd.read_csv(csv_file_path)
    experiment_data = df[df['well_type'] == 'experiment'].copy()
    
    # Classification criteria (same as adaptive concentrations)
    turbidity_threshold = 0.15  # Above this = likely interaction
    ratio_baseline = experiment_data['ratio'].median()  # Use median as baseline reference
    ratio_std = experiment_data['ratio'].std()
    ratio_threshold = 2.0 * ratio_std  # 2 standard deviations from baseline
    
    print(f"Classification thresholds: turbidity > {turbidity_threshold:.3f}, ratio deviation > {ratio_threshold:.3f}")
    
    # Classify each well
    experiment_data['is_baseline'] = (
        (experiment_data['turbidity_600'] <= turbidity_threshold) & 
        (np.abs(experiment_data['ratio'] - ratio_baseline) <= ratio_threshold)
    )
    
    baseline_count = experiment_data['is_baseline'].sum()
    non_baseline_count = len(experiment_data) - baseline_count
    print(f"Classification results: {baseline_count} baseline, {non_baseline_count} non-baseline wells")
    
    # Add log concentrations
    experiment_data['log_sds'] = np.log10(experiment_data['surf_A_conc_mm'])
    experiment_data['log_ttab'] = np.log10(experiment_data['surf_B_conc_mm'])
    
    return experiment_data

def largest_rectangle_in_histogram(heights):
    """Find the area of the largest rectangle in a histogram."""
    stack = []
    max_area = 0
    best_rect = (0, 0, 0, 0)  # (area, left, right, height)
    
    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            area = height * width
            if area > max_area:
                max_area = area
                left = 0 if not stack else stack[-1] + 1
                best_rect = (area, left, i - 1, height)
        stack.append(i)
    
    while stack:
        height = heights[stack.pop()]
        width = len(heights) if not stack else len(heights) - stack[-1] - 1
        area = height * width
        if area > max_area:
            max_area = area
            left = 0 if not stack else stack[-1] + 1
            best_rect = (area, left, len(heights) - 1, height)
    
    return best_rect

def find_baseline_rectangle(experiment_data):
    """Find the largest baseline rectangle using histogram method."""
    
    # Get sorted unique concentrations
    surf_A_concs = sorted(experiment_data['surf_A_conc_mm'].unique())
    surf_B_concs = sorted(experiment_data['surf_B_conc_mm'].unique())
    
    print(f"Grid dimensions: {len(surf_A_concs)} x {len(surf_B_concs)}")
    
    # Create binary matrix (1 = baseline, 0 = non-baseline)
    matrix = np.zeros((len(surf_B_concs), len(surf_A_concs)), dtype=int)
    
    for i, surf_B_conc in enumerate(surf_B_concs):
        for j, surf_A_conc in enumerate(surf_A_concs):
            well = experiment_data[
                (experiment_data['surf_A_conc_mm'] == surf_A_conc) & 
                (experiment_data['surf_B_conc_mm'] == surf_B_conc)
            ]
            if len(well) > 0:
                matrix[i, j] = 1 if well.iloc[0]['is_baseline'] else 0
    
    # Find largest rectangle using histogram method
    max_area = 0
    best_rectangle = None
    heights = np.zeros(len(surf_A_concs), dtype=int)
    
    for row in range(len(surf_B_concs)):
        # Update heights for this row
        for col in range(len(surf_A_concs)):
            if matrix[row, col] == 1:
                heights[col] += 1
            else:
                heights[col] = 0
        
        # Find largest rectangle in this histogram
        area, left_col, right_col, height = largest_rectangle_in_histogram(heights)
        
        if area > max_area:
            max_area = area
            bottom_row = row - height + 1
            top_row = row
            best_rectangle = {
                'area': area,
                'bottom_row': bottom_row,
                'top_row': top_row,
                'left_col': left_col,
                'right_col': right_col,
                'surf_A_min': surf_A_concs[left_col],
                'surf_A_max': surf_A_concs[right_col],
                'surf_B_min': surf_B_concs[bottom_row],
                'surf_B_max': surf_B_concs[top_row]
            }
    
    print(f"\\nLargest baseline rectangle found:")
    print(f"  Area: {best_rectangle['area']} wells")
    print(f"  SDS range: {best_rectangle['surf_A_min']:.6f} - {best_rectangle['surf_A_max']:.6f} mM")
    print(f"  TTAB range: {best_rectangle['surf_B_min']:.6f} - {best_rectangle['surf_B_max']:.6f} mM")
    
    return best_rectangle, surf_A_concs, surf_B_concs

def define_rectangle_regions(best_rectangle, surf_A_concs, surf_B_concs, experiment_data):
    """
    Define the non-baseline rectangular regions based on the baseline rectangle.
    """
    
    # Extract baseline rectangle bounds
    baseline_sds_max = best_rectangle['surf_A_max'] 
    baseline_ttab_max = best_rectangle['surf_B_max']
    
    # Define concentration ranges
    sds_min = min(surf_A_concs)
    sds_max = max(surf_A_concs)
    ttab_min = min(surf_B_concs)
    ttab_max = max(surf_B_concs)
    
    # Define the 3 non-baseline rectangles 
    rectangles = [
        {
            'name': 'High_SDS_Region',
            'description': 'High SDS, Low TTAB (above baseline)',
            'sds_bounds': (baseline_sds_max, sds_max),
            'ttab_bounds': (ttab_min, baseline_ttab_max),
            'color': 'orange'
        },
        {
            'name': 'High_TTAB_Region', 
            'description': 'Low SDS, High TTAB (right of baseline)',
            'sds_bounds': (sds_min, baseline_sds_max),
            'ttab_bounds': (baseline_ttab_max, ttab_max),
            'color': 'green'
        },
        {
            'name': 'High_Both_Region',
            'description': 'High SDS + High TTAB (top-right corner)',
            'sds_bounds': (baseline_sds_max, sds_max),
            'ttab_bounds': (baseline_ttab_max, ttab_max),
            'color': 'purple'
        }
    ]
    
    return rectangles

def train_gradient_gp(experiment_data):
    """Train GP for gradient calculations."""
    
    # Prepare training data
    X_train = experiment_data[['log_sds', 'log_ttab']].values
    y_train = experiment_data['turbidity_600'].values
    
    # Standardize
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    scaler_y = StandardScaler() 
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    
    # Train GP
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.01)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=5, normalize_y=False)
    gp.fit(X_train_scaled, y_train_scaled)
    
    print(f"\\n✓ GP trained for gradient calculation")
    print(f"✓ Kernel: {gp.kernel_}")
    
    return gp, scaler_X, scaler_y

def calculate_directional_gradients_in_rectangles(rectangles, experiment_data, gp, scaler_X, scaler_y):
    """
    Calculate SDS and TTAB directional gradients in LOG SPACE for each rectangle.
    This accounts for the logarithmic nature of concentration relationships.
    """
    
    results = []
    
    for rect in rectangles:
        print(f"\\n=== {rect['name']} ===")
        
        # Find wells in this rectangle
        sds_min, sds_max = rect['sds_bounds']
        ttab_min, ttab_max = rect['ttab_bounds']
        
        wells_in_rect = experiment_data[
            (experiment_data['surf_A_conc_mm'] >= sds_min) &
            (experiment_data['surf_A_conc_mm'] <= sds_max) &
            (experiment_data['surf_B_conc_mm'] >= ttab_min) &
            (experiment_data['surf_B_conc_mm'] <= ttab_max)
        ].copy()
        
        print(f"Wells in rectangle: {len(wells_in_rect)}")
        
        if len(wells_in_rect) == 0:
            print("No wells in rectangle - skipping")
            continue
        
        # Calculate gradients in LOG SPACE at each point in the rectangle
        sds_log_gradients = []
        ttab_log_gradients = []
        
        for _, well in wells_in_rect.iterrows():
            log_sds = well['log_sds']  # log10(SDS concentration)
            log_ttab = well['log_ttab']  # log10(TTAB concentration)
            
            # Small perturbations in LOG SPACE (this is key!)
            delta_log = 0.05  # Small step in log10 space (≈ 12% concentration change)
            
            # Create perturbation points in log space
            points = np.array([
                [log_sds + delta_log, log_ttab],   # Increase SDS by ~12%
                [log_sds - delta_log, log_ttab],   # Decrease SDS by ~12%
                [log_sds, log_ttab + delta_log],   # Increase TTAB by ~12%
                [log_sds, log_ttab - delta_log]    # Decrease TTAB by ~12%
            ])
            
            # Scale and predict
            points_scaled = scaler_X.transform(points)
            preds_scaled = gp.predict(points_scaled)
            preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
            
            # Calculate directional gradients in log space
            # ∂(turbidity)/∂(log10_SDS) - how turbidity changes per unit log SDS change
            grad_log_sds = (preds[0] - preds[1]) / (2 * delta_log)
            
            # ∂(turbidity)/∂(log10_TTAB) - how turbidity changes per unit log TTAB change  
            grad_log_ttab = (preds[2] - preds[3]) / (2 * delta_log)
            
            sds_log_gradients.append(grad_log_sds)
            ttab_log_gradients.append(grad_log_ttab)
        
        # Calculate statistics for log-space gradients
        sds_log_gradients = np.array(sds_log_gradients)
        ttab_log_gradients = np.array(ttab_log_gradients)
        
        rect_results = {
            'name': rect['name'],
            'description': rect['description'],
            'n_wells': len(wells_in_rect),
            'sds_bounds': rect['sds_bounds'],
            'ttab_bounds': rect['ttab_bounds'],
            'log_sds_bounds': (np.log10(sds_min), np.log10(sds_max)),
            'log_ttab_bounds': (np.log10(ttab_min), np.log10(ttab_max)),
            'sds_log_gradient_stats': {
                'min': sds_log_gradients.min(),
                'max': sds_log_gradients.max(),
                'range': sds_log_gradients.max() - sds_log_gradients.min(),
                'mean': sds_log_gradients.mean(),
                'std': sds_log_gradients.std()
            },
            'ttab_log_gradient_stats': {
                'min': ttab_log_gradients.min(), 
                'max': ttab_log_gradients.max(),
                'range': ttab_log_gradients.max() - ttab_log_gradients.min(),
                'mean': ttab_log_gradients.mean(),
                'std': ttab_log_gradients.std()
            },
            'color': rect['color']
        }
        
        results.append(rect_results)
        
        # Print results with log space notation
        print(f"SDS direction gradients (∂turbidity/∂log10_SDS):")
        print(f"  Min: {rect_results['sds_log_gradient_stats']['min']:.4f}")
        print(f"  Max: {rect_results['sds_log_gradient_stats']['max']:.4f}")
        print(f"  Range (max-min): {rect_results['sds_log_gradient_stats']['range']:.4f}")
        print(f"  Mean: {rect_results['sds_log_gradient_stats']['mean']:.4f}")
        
        print(f"TTAB direction gradients (∂turbidity/∂log10_TTAB):")
        print(f"  Min: {rect_results['ttab_log_gradient_stats']['min']:.4f}")
        print(f"  Max: {rect_results['ttab_log_gradient_stats']['max']:.4f}")
        print(f"  Range (max-min): {rect_results['ttab_log_gradient_stats']['range']:.4f}")
        print(f"  Mean: {rect_results['ttab_log_gradient_stats']['mean']:.4f}")
        
        # Additional log space interpretation
        log_sds_span = np.log10(sds_max) - np.log10(sds_min)
        log_ttab_span = np.log10(ttab_max) - np.log10(ttab_min)
        print(f"\\nLog space spans:")
        print(f"  SDS: {log_sds_span:.2f} log10 units ({sds_min:.4f} - {sds_max:.3f} mM)")
        print(f"  TTAB: {log_ttab_span:.2f} log10 units ({ttab_min:.4f} - {ttab_max:.3f} mM)")
    
    return results

def visualize_rectangles_with_gradients(experiment_data, best_rectangle, rectangles, gradient_results,
                                      output_dir='output/rectangle_gradients'):
    """
    Visualize the rectangles with gradient statistics.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Separate baseline and non-baseline wells
    baseline_wells = experiment_data[experiment_data['is_baseline']]
    nonbaseline_wells = experiment_data[~experiment_data['is_baseline']]
    
    # Plot 1: Rectangle identification
    ax1.scatter(baseline_wells['surf_A_conc_mm'], baseline_wells['surf_B_conc_mm'],
               c='lightblue', s=50, alpha=0.7, label=f'Baseline wells (n={len(baseline_wells)})', 
               marker='o', edgecolors='blue', linewidth=1)
    ax1.scatter(nonbaseline_wells['surf_A_conc_mm'], nonbaseline_wells['surf_B_conc_mm'],
               c='red', s=60, alpha=0.8, label=f'Non-baseline wells (n={len(nonbaseline_wells)})', 
               marker='s', edgecolors='darkred', linewidth=1)
    
    # Draw baseline rectangle (green box)
    from matplotlib.patches import Rectangle as MPLRectangle
    baseline_rect_patch = MPLRectangle(
        (best_rectangle['surf_A_min'], best_rectangle['surf_B_min']),
        best_rectangle['surf_A_max'] - best_rectangle['surf_A_min'],
        best_rectangle['surf_B_max'] - best_rectangle['surf_B_min'],
        linewidth=3, edgecolor='green', facecolor='green', alpha=0.3,
        label=f"Baseline rectangle ({best_rectangle['area']} wells)"
    )
    ax1.add_patch(baseline_rect_patch)
    
    # Draw non-baseline rectangles
    for rect_result in gradient_results:
        sds_min, sds_max = rect_result['sds_bounds']
        ttab_min, ttab_max = rect_result['ttab_bounds']
        
        rect_patch = MPLRectangle(
            (sds_min, ttab_min), sds_max - sds_min, ttab_max - ttab_min,
            linewidth=3, edgecolor=rect_result['color'], facecolor=rect_result['color'], alpha=0.3,
            label=f"{rect_result['name']} ({rect_result['n_wells']} wells)"
        )
        ax1.add_patch(rect_patch)
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('SDS Concentration (mM)', fontsize=12)
    ax1.set_ylabel('TTAB Concentration (mM)', fontsize=12)
    ax1.set_title('Rectangle Identification (Adaptive Concentration Method)', fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: SDS gradient ranges (LOG SPACE)
    rect_names = [r['name'].replace('_', '\\n') for r in gradient_results]
    sds_ranges = [r['sds_log_gradient_stats']['range'] for r in gradient_results]
    colors = [r['color'] for r in gradient_results]
    
    bars = ax2.bar(rect_names, sds_ranges, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('SDS Log Gradient Range\\n(∂turbidity/∂log10_SDS)', fontsize=12)
    ax2.set_title('SDS Direction Log Gradient Ranges by Rectangle', fontweight='bold', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, sds_ranges):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sds_ranges)*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: TTAB gradient ranges (LOG SPACE)
    ttab_ranges = [r['ttab_log_gradient_stats']['range'] for r in gradient_results]
    
    bars = ax3.bar(rect_names, ttab_ranges, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax3.set_ylabel('TTAB Log Gradient Range\\n(∂turbidity/∂log10_TTAB)', fontsize=12)
    ax3.set_title('TTAB Direction Log Gradient Ranges by Rectangle', fontweight='bold', fontsize=14)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, ttab_ranges):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(ttab_ranges)*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Summary table (LOG SPACE)
    table_data = []
    for r in gradient_results:
        table_data.append([
            r['name'].replace('_', ' '),
            f"{r['n_wells']}",
            f"{r['sds_log_gradient_stats']['range']:.3f}",
            f"{r['ttab_log_gradient_stats']['range']:.3f}",
            f"{r['sds_log_gradient_stats']['mean']:.3f}",
            f"{r['ttab_log_gradient_stats']['mean']:.3f}"
        ])
    
    ax4.axis('tight')
    ax4.axis('off')
    table = ax4.table(cellText=table_data, 
                     colLabels=['Rectangle', 'Wells', 'SDS Log Range', 'TTAB Log Range', 'SDS Log Mean', 'TTAB Log Mean'],
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax4.set_title('Log Gradient Statistics Summary\\n(∂turbidity/∂log10_concentration)', fontweight='bold', fontsize=14, pad=20)
    
    plt.tight_layout()
    
    output_file = f'{output_dir}/rectangle_gradients.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\\nVisualization saved: {output_file}")
    
    plt.show()

def main():
    """Main function."""
    
    data_file = r"C:\\Users\\owenm\\OneDrive\\Desktop\\North Robotics\\utoronto_demo\\utoronto_demo\\surfactant_grid_SDS_TTAB_20260209_164920\\surfactant_grid_SDS_TTAB_20260209_164920\\complete_experiment_results.csv"
    
    if not os.path.exists(data_file):
        print(f"File not found: {data_file}")
        return
    
    print("="*70)
    print("RECTANGLE IDENTIFICATION WITH DIRECTIONAL GRADIENTS")
    print("="*70)
    
    print("\\n1. Loading and classifying experimental data...")
    experiment_data = load_experiment_data(data_file)
    
    print("\\n2. Finding largest baseline rectangle...")
    best_rectangle, surf_A_concs, surf_B_concs = find_baseline_rectangle(experiment_data)
    
    print("\\n3. Defining non-baseline rectangular regions...")
    rectangles = define_rectangle_regions(best_rectangle, surf_A_concs, surf_B_concs, experiment_data)
    
    print(f"\\nDefined {len(rectangles)} non-baseline rectangles:")
    for rect in rectangles:
        print(f"  - {rect['name']}: {rect['description']}")
    
    print("\\n4. Training GP for gradient calculations...")
    gp, scaler_X, scaler_y = train_gradient_gp(experiment_data)
    
    print("\\n5. Calculating directional gradients in each rectangle...")
    gradient_results = calculate_directional_gradients_in_rectangles(rectangles, experiment_data, gp, scaler_X, scaler_y)
    
    print("\\n6. Creating visualization...")
    visualize_rectangles_with_gradients(experiment_data, best_rectangle, rectangles, gradient_results)
    
    print("\\n" + "="*70)
    print("SUMMARY: DIRECTIONAL LOG GRADIENT RANGES")
    print("="*70)
    print("All gradients calculated as ∂(turbidity)/∂(log10_concentration)")
    
    for r in gradient_results:
        print(f"\\n{r['name']} ({r['n_wells']} wells):")
        print(f"  SDS direction:  range = {r['sds_log_gradient_stats']['range']:.4f} (min: {r['sds_log_gradient_stats']['min']:.4f}, max: {r['sds_log_gradient_stats']['max']:.4f})")
        print(f"  TTAB direction: range = {r['ttab_log_gradient_stats']['range']:.4f} (min: {r['ttab_log_gradient_stats']['min']:.4f}, max: {r['ttab_log_gradient_stats']['max']:.4f})")
        print(f"  Log space spans: SDS = {r['log_sds_bounds'][1] - r['log_sds_bounds'][0]:.2f}, TTAB = {r['log_ttab_bounds'][1] - r['log_ttab_bounds'][0]:.2f}")

if __name__ == "__main__":
    main()