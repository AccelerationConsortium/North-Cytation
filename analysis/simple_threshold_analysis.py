"""
Simple Threshold Analysis for Surfactant Grid Data
- Turbidity: fixed threshold at 0.01
- Ratio: k-means clustering to find natural bifurcation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

def simple_classification_analysis(csv_file_path):
    """
    Simple classification using turbidity=0.01 and k-means for ratio.
    """
    # Load data
    df = pd.read_csv(csv_file_path)
    experiment_data = df[df['well_type'] == 'experiment'].copy()
    
    print(f"Analyzing {len(experiment_data)} experimental wells")
    print(f"Turbidity range: {experiment_data['turbidity_600'].min():.3f} - {experiment_data['turbidity_600'].max():.3f}")
    print(f"Ratio range: {experiment_data['ratio'].min():.3f} - {experiment_data['ratio'].max():.3f}")
    
    # TURBIDITY: Fixed threshold at 0.1
    turbidity_threshold = 0.1
    turbidity_baseline = experiment_data['turbidity_600'] <= turbidity_threshold
    
    print(f"\\nTurbidity classification (≤ {turbidity_threshold}):")
    print(f"  Baseline: {turbidity_baseline.sum()}")
    print(f"  Non-baseline: {(~turbidity_baseline).sum()}")
    
    # RATIO: K-means clustering
    ratio_values = experiment_data['ratio'].values.reshape(-1, 1)
    scaler = StandardScaler()
    ratio_scaled = scaler.fit_transform(ratio_values)
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    ratio_clusters = kmeans.fit_predict(ratio_scaled)
    
    # Determine which cluster is baseline (higher ratio typically = baseline)
    cluster_0_mean = experiment_data[ratio_clusters == 0]['ratio'].mean()
    cluster_1_mean = experiment_data[ratio_clusters == 1]['ratio'].mean()
    baseline_cluster = 0 if cluster_0_mean > cluster_1_mean else 1
    
    ratio_baseline = (ratio_clusters == baseline_cluster)
    
    # Get the boundary
    centers_scaled = kmeans.cluster_centers_
    centers_original = scaler.inverse_transform(centers_scaled).flatten()
    ratio_boundary = np.mean(centers_original)
    
    print(f"\\nRatio k-means classification:")
    print(f"  Cluster 0 (n={np.sum(ratio_clusters == 0)}): mean ratio = {cluster_0_mean:.3f}")
    print(f"  Cluster 1 (n={np.sum(ratio_clusters == 1)}): mean ratio = {cluster_1_mean:.3f}")
    print(f"  Baseline cluster: {baseline_cluster}")
    print(f"  Boundary at: {ratio_boundary:.3f}")
    print(f"  Baseline: {ratio_baseline.sum()}")
    print(f"  Non-baseline: {(~ratio_baseline).sum()}")
    
    # Store results
    experiment_data['turbidity_baseline'] = turbidity_baseline
    experiment_data['ratio_baseline'] = ratio_baseline
    experiment_data['ratio_boundary'] = ratio_boundary
    
    # COMBINED CLASSIFICATION: Both methods must agree it's baseline
    combined_baseline_and = turbidity_baseline & ratio_baseline
    # Alternative: Either method says baseline
    combined_baseline_or = turbidity_baseline | ratio_baseline
    
    experiment_data['combined_baseline_and'] = combined_baseline_and
    experiment_data['combined_baseline_or'] = combined_baseline_or
    
    print(f"\\nCombined classification:")
    print(f"  AND logic (both must be baseline): {combined_baseline_and.sum()} baseline, {(~combined_baseline_and).sum()} non-baseline")
    print(f"  OR logic (either can be baseline): {combined_baseline_or.sum()} baseline, {(~combined_baseline_or).sum()} non-baseline")
    
    # Create simple visualization
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(12, 15))
    
    # Plot 1: Turbidity classification in concentration space
    colors_turb = ['red' if not x else 'lightblue' for x in turbidity_baseline]
    ax1.scatter(experiment_data['surf_A_conc_mm'], experiment_data['surf_B_conc_mm'],
               c=colors_turb, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('SDS Concentration (mM)')
    ax1.set_ylabel('TTAB Concentration (mM)')
    ax1.set_title(f'Turbidity Classification\\n(Threshold ≤ {turbidity_threshold})')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Ratio classification in concentration space
    colors_ratio = ['red' if not x else 'lightgreen' for x in ratio_baseline]
    ax2.scatter(experiment_data['surf_A_conc_mm'], experiment_data['surf_B_conc_mm'],
               c=colors_ratio, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('SDS Concentration (mM)')
    ax2.set_ylabel('TTAB Concentration (mM)')
    ax2.set_title(f'Ratio K-Means Classification\\n(Boundary = {ratio_boundary:.3f})')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Turbidity histogram
    turb_base = experiment_data[turbidity_baseline]['turbidity_600']
    turb_nonbase = experiment_data[~turbidity_baseline]['turbidity_600']
    
    ax3.hist(turb_base, bins=20, alpha=0.6, color='lightblue', 
             label=f'Baseline (n={len(turb_base)})', density=True)
    ax3.hist(turb_nonbase, bins=20, alpha=0.6, color='red', 
             label=f'Non-baseline (n={len(turb_nonbase)})', density=True)
    ax3.axvline(x=turbidity_threshold, color='black', linestyle='--', linewidth=2, 
                label=f'Threshold ({turbidity_threshold})')
    ax3.set_xlabel('Turbidity (600nm)')
    ax3.set_ylabel('Density')
    ax3.set_title('Turbidity Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Ratio histogram with k-means
    ratio_base = experiment_data[ratio_baseline]['ratio']
    ratio_nonbase = experiment_data[~ratio_baseline]['ratio']
    
    ax4.hist(ratio_base, bins=20, alpha=0.6, color='lightgreen',
             label=f'Baseline (n={len(ratio_base)})', density=True)
    ax4.hist(ratio_nonbase, bins=20, alpha=0.6, color='red',
             label=f'Non-baseline (n={len(ratio_nonbase)})', density=True)
    ax4.axvline(x=ratio_boundary, color='black', linestyle='--', linewidth=2, 
                label=f'K-means boundary ({ratio_boundary:.3f})')
    ax4.axvline(x=centers_original[0], color='blue', linestyle=':', alpha=0.7, label='Cluster centers')
    ax4.axvline(x=centers_original[1], color='blue', linestyle=':', alpha=0.7)
    ax4.set_xlabel('Fluorescence Ratio')
    ax4.set_ylabel('Density')
    ax4.set_title('Ratio Distribution (K-Means)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Combined AND classification 
    colors_and = ['red' if not x else 'gold' for x in combined_baseline_and]
    ax5.scatter(experiment_data['surf_A_conc_mm'], experiment_data['surf_B_conc_mm'],
               c=colors_and, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax5.set_xscale('log')
    ax5.set_yscale('log')
    ax5.set_xlabel('SDS Concentration (mM)')
    ax5.set_ylabel('TTAB Concentration (mM)')
    ax5.set_title(f'Combined AND Logic\\n(Both turbidity AND ratio baseline)\\nBaseline: {combined_baseline_and.sum()}/{len(experiment_data)}')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Combined OR classification
    colors_or = ['red' if not x else 'orange' for x in combined_baseline_or]
    ax6.scatter(experiment_data['surf_A_conc_mm'], experiment_data['surf_B_conc_mm'],
               c=colors_or, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax6.set_xscale('log')
    ax6.set_yscale('log')
    ax6.set_xlabel('SDS Concentration (mM)')
    ax6.set_ylabel('TTAB Concentration (mM)')
    ax6.set_title(f'Combined OR Logic\\n(Either turbidity OR ratio baseline)\\nBaseline: {combined_baseline_or.sum()}/{len(experiment_data)}')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    os.makedirs('output/simple_classification', exist_ok=True)
    output_file = 'output/simple_classification/turbidity_and_kmeans_ratio.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\\nVisualization saved: {output_file}")
    
    # Summary
    print(f"\\n" + "="*50)
    print("CLASSIFICATION SUMMARY")
    print("="*50)
    print(f"Turbidity threshold: ≤ {turbidity_threshold}")
    print(f"  Baseline wells: {turbidity_baseline.sum()}")
    print(f"\\nRatio k-means boundary: {ratio_boundary:.3f}")
    print(f"  Baseline wells: {ratio_baseline.sum()}")
    print(f"\\nCombined AND (both methods agree baseline): {combined_baseline_and.sum()}")
    print(f"Combined OR (either method says baseline): {combined_baseline_or.sum()}")
    print(f"\\nRecommended: Use AND logic for strict baseline classification")
    
    plt.show()
    
    return {
        'turbidity_threshold': turbidity_threshold,
        'ratio_boundary': ratio_boundary,
        'combined_baseline_and': combined_baseline_and,
        'combined_baseline_or': combined_baseline_or,
        'data': experiment_data
    }

if __name__ == "__main__":
    data_file = r"C:\Users\owenm\OneDrive\Desktop\North Robotics\utoronto_demo\utoronto_demo\surfactant_grid_SDS_TTAB_20260209_164920\surfactant_grid_SDS_TTAB_20260209_164920\complete_experiment_results.csv"
    
    if os.path.exists(data_file):
        results = simple_classification_analysis(data_file)
        print(f"\\nRecommended thresholds:")
        print(f"  Turbidity: ≤ {results['turbidity_threshold']}")
        print(f"  Ratio: boundary at {results['ratio_boundary']:.3f}")
        print(f"\\nFor overall baseline classification:")
        print(f"  AND logic (strict): {results['combined_baseline_and'].sum()} baseline wells")
        print(f"  OR logic (lenient): {results['combined_baseline_or'].sum()} baseline wells")
    else:
        print(f"File not found: {data_file}")