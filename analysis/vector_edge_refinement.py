"""
Vector Edge Refinement for Boundary Detection
Treats [turbidity, ratio] as a 2D vector field and finds strongest boundaries
by measuring vector changes across edges between neighboring grid points.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import os

def load_and_prepare_data(csv_file_path):
    """Load experimental data and prepare for vector field analysis."""
    df = pd.read_csv(csv_file_path)
    experiment_data = df[df['well_type'] == 'experiment'].copy()
    
    print(f"Loaded {len(experiment_data)} experimental data points")
    print(f"Turbidity range: {experiment_data['turbidity_600'].min():.4f} - {experiment_data['turbidity_600'].max():.3f}")
    print(f"Ratio range: {experiment_data['ratio'].min():.4f} - {experiment_data['ratio'].max():.4f}")
    
    return experiment_data

def normalize_vector_field(experiment_data):
    """
    Normalize turbidity and ratio to comparable scales using Option B:
    - Log turbidity + z-score
    - Z-score ratio
    """
    
    # Option B: log turbidity + z-score both
    epsilon = 1e-6  # Small value to handle any zeros
    log_turbidity = np.log10(experiment_data['turbidity_600'] + epsilon)
    
    # Z-score the log turbidity
    scaler_turb = StandardScaler()
    turb_normalized = scaler_turb.fit_transform(log_turbidity.values.reshape(-1, 1)).flatten()
    
    # Z-score the ratio
    scaler_ratio = StandardScaler()
    ratio_normalized = scaler_ratio.fit_transform(experiment_data['ratio'].values.reshape(-1, 1)).flatten()
    
    print(f"\\nNormalization completed:")
    print(f"Log turbidity: {log_turbidity.min():.3f} - {log_turbidity.max():.3f} → z-score: {turb_normalized.min():.3f} - {turb_normalized.max():.3f}")
    print(f"Ratio: {experiment_data['ratio'].min():.4f} - {experiment_data['ratio'].max():.4f} → z-score: {ratio_normalized.min():.3f} - {ratio_normalized.max():.3f}")
    
    # Add normalized values to dataframe
    experiment_data = experiment_data.copy()
    experiment_data['log_turbidity'] = log_turbidity
    experiment_data['turb_normalized'] = turb_normalized
    experiment_data['ratio_normalized'] = ratio_normalized
    experiment_data['log_sds'] = np.log10(experiment_data['surf_A_conc_mm'])
    experiment_data['log_ttab'] = np.log10(experiment_data['surf_B_conc_mm'])
    
    return experiment_data, scaler_turb, scaler_ratio

def create_grid_structure(experiment_data):
    """Create grid structure for neighbor identification."""
    
    # Get unique concentrations
    sds_concs = sorted(experiment_data['surf_A_conc_mm'].unique())
    ttab_concs = sorted(experiment_data['surf_B_conc_mm'].unique())
    
    print(f"\\nGrid structure: {len(sds_concs)} SDS × {len(ttab_concs)} TTAB = {len(sds_concs) * len(ttab_concs)} grid positions")
    
    # Create grid mapping
    grid_data = {}
    for _, row in experiment_data.iterrows():
        sds_idx = sds_concs.index(row['surf_A_conc_mm'])
        ttab_idx = ttab_concs.index(row['surf_B_conc_mm'])
        grid_data[(sds_idx, ttab_idx)] = {
            'sds_conc': row['surf_A_conc_mm'],
            'ttab_conc': row['surf_B_conc_mm'],
            'log_sds': row['log_sds'],
            'log_ttab': row['log_ttab'],
            'turb_norm': row['turb_normalized'],
            'ratio_norm': row['ratio_normalized'],
            'turbidity': row['turbidity_600'],
            'ratio': row['ratio']
        }
    
    return grid_data, sds_concs, ttab_concs

def calculate_edge_scores(grid_data, sds_concs, ttab_concs):
    """
    Calculate vector difference scores for each edge between neighboring points.
    Score = sqrt((T2'-T1')² + (R2'-R1')²) where T',R' are normalized values.
    """
    
    edges = []
    
    n_sds = len(sds_concs)
    n_ttab = len(ttab_concs)
    
    print(f"Calculating edge scores for grid...")
    
    # Check horizontal edges (SDS direction)
    for ttab_idx in range(n_ttab):
        for sds_idx in range(n_sds - 1):
            pos1 = (sds_idx, ttab_idx)
            pos2 = (sds_idx + 1, ttab_idx)
            
            if pos1 in grid_data and pos2 in grid_data:
                data1 = grid_data[pos1]
                data2 = grid_data[pos2]
                
                # Vector difference in normalized space
                turb_diff = data2['turb_norm'] - data1['turb_norm']
                ratio_diff = data2['ratio_norm'] - data1['ratio_norm']
                score = np.sqrt(turb_diff**2 + ratio_diff**2)
                
                # Midpoint in log space
                log_sds_mid = (data1['log_sds'] + data2['log_sds']) / 2
                log_ttab_mid = (data1['log_ttab'] + data2['log_ttab']) / 2
                
                edges.append({
                    'pos1': pos1,
                    'pos2': pos2,
                    'direction': 'horizontal',
                    'score': score,
                    'log_sds_mid': log_sds_mid,
                    'log_ttab_mid': log_ttab_mid,
                    'sds_mid': 10**log_sds_mid,
                    'ttab_mid': 10**log_ttab_mid,
                    'turb_diff': turb_diff,
                    'ratio_diff': ratio_diff
                })
    
    # Check vertical edges (TTAB direction)
    for sds_idx in range(n_sds):
        for ttab_idx in range(n_ttab - 1):
            pos1 = (sds_idx, ttab_idx)
            pos2 = (sds_idx, ttab_idx + 1)
            
            if pos1 in grid_data and pos2 in grid_data:
                data1 = grid_data[pos1]
                data2 = grid_data[pos2]
                
                # Vector difference in normalized space  
                turb_diff = data2['turb_norm'] - data1['turb_norm']
                ratio_diff = data2['ratio_norm'] - data1['ratio_norm']
                score = np.sqrt(turb_diff**2 + ratio_diff**2)
                
                # Midpoint in log space
                log_sds_mid = (data1['log_sds'] + data2['log_sds']) / 2
                log_ttab_mid = (data1['log_ttab'] + data2['log_ttab']) / 2
                
                edges.append({
                    'pos1': pos1,
                    'pos2': pos2,
                    'direction': 'vertical',
                    'score': score,
                    'log_sds_mid': log_sds_mid,
                    'log_ttab_mid': log_ttab_mid,
                    'sds_mid': 10**log_sds_mid,
                    'ttab_mid': 10**log_ttab_mid,
                    'turb_diff': turb_diff,
                    'ratio_diff': ratio_diff
                })
    
    edges_df = pd.DataFrame(edges)
    
    print(f"Calculated {len(edges_df)} edge scores")
    print(f"Score range: {edges_df['score'].min():.4f} - {edges_df['score'].max():.4f}")
    print(f"Mean score: {edges_df['score'].mean():.4f}")
    
    return edges_df

def select_boundary_points(edges_df, sds_concs, ttab_concs, n_points=32):
    """
    Select high-scoring edge midpoints with minimum spacing enforcement.
    """
    
    # Sort edges by score (highest first)
    edges_sorted = edges_df.sort_values('score', ascending=False).reset_index(drop=True)
    
    # Calculate minimum spacing in log space
    log_sds_step = np.log10(sds_concs[1]) - np.log10(sds_concs[0]) if len(sds_concs) > 1 else 0.1
    log_ttab_step = np.log10(ttab_concs[1]) - np.log10(ttab_concs[0]) if len(ttab_concs) > 1 else 0.1
    d_min = 0.5 * min(log_sds_step, log_ttab_step)
    
    print(f"\\nSelecting {n_points} boundary points...")
    print(f"Log space grid steps: SDS = {log_sds_step:.3f}, TTAB = {log_ttab_step:.3f}")
    print(f"Minimum spacing: {d_min:.3f} log units")
    
    selected_points = []
    
    for _, edge in edges_sorted.iterrows():
        candidate_log_pos = np.array([edge['log_sds_mid'], edge['log_ttab_mid']])
        
        # Check if this candidate is too close to already selected points
        if len(selected_points) > 0:
            selected_log_positions = np.array([[p['log_sds_mid'], p['log_ttab_mid']] for p in selected_points])
            distances = cdist([candidate_log_pos], selected_log_positions)[0]
            
            if np.min(distances) < d_min:
                continue  # Skip this candidate - too close to existing selection
        
        # Add this point to selection
        selected_points.append({
            'log_sds_mid': edge['log_sds_mid'],
            'log_ttab_mid': edge['log_ttab_mid'], 
            'sds_mid': edge['sds_mid'],
            'ttab_mid': edge['ttab_mid'],
            'score': edge['score'],
            'direction': edge['direction'],
            'turb_diff': edge['turb_diff'],
            'ratio_diff': edge['ratio_diff']
        })
        
        if len(selected_points) >= n_points:
            break
    
    selected_df = pd.DataFrame(selected_points)
    
    print(f"Selected {len(selected_df)} points (requested {n_points})")
    print(f"Score range of selected points: {selected_df['score'].min():.4f} - {selected_df['score'].max():.4f}")
    
    return selected_df

def visualize_boundary_refinement(experiment_data, edges_df, selected_df, 
                                output_dir='output/vector_edge_refinement'):
    """
    Visualize the vector field analysis and selected boundary points.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: Original data with selected boundary points
    sc1 = ax1.scatter(experiment_data['surf_A_conc_mm'], experiment_data['surf_B_conc_mm'],
                     c=experiment_data['turbidity_600'], s=60, cmap='viridis', alpha=0.8,
                     edgecolors='black', linewidth=1, label='Existing data')
    
    # Overlay selected boundary points
    ax1.scatter(selected_df['sds_mid'], selected_df['ttab_mid'],
               c='red', s=100, alpha=1.0, marker='^', edgecolors='darkred', linewidth=2,
               label=f'Selected boundary points (n={len(selected_df)})')
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('SDS Concentration (mM)', fontsize=12)
    ax1.set_ylabel('TTAB Concentration (mM)', fontsize=12)
    ax1.set_title('Vector Edge Refinement - Selected Boundary Points', fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    cbar1 = plt.colorbar(sc1, ax=ax1)
    cbar1.set_label('Turbidity', fontsize=10)
    
    # Plot 2: Edge scores visualization
    sc2 = ax2.scatter(edges_df['sds_mid'], edges_df['ttab_mid'],
                     c=edges_df['score'], s=30, cmap='Reds', alpha=0.7,
                     edgecolors='black', linewidth=0.5)
    
    # Highlight selected edges
    ax2.scatter(selected_df['sds_mid'], selected_df['ttab_mid'],
               c='blue', s=80, alpha=1.0, marker='s', edgecolors='darkblue', linewidth=2,
               label='Selected edges')
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('SDS Concentration (mM)', fontsize=12)
    ax2.set_ylabel('TTAB Concentration (mM)', fontsize=12)
    ax2.set_title('Edge Scores (Vector Change Magnitude)', fontweight='bold', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    cbar2 = plt.colorbar(sc2, ax=ax2)
    cbar2.set_label('Vector Change Score', fontsize=10)
    
    # Plot 3: Score distribution and selection threshold
    ax3.hist(edges_df['score'], bins=50, alpha=0.7, color='gray', label=f'All edges (n={len(edges_df)})')
    ax3.axvline(selected_df['score'].min(), color='red', linestyle='--', linewidth=2,
               label=f'Selection threshold: {selected_df["score"].min():.3f}')
    ax3.axvline(selected_df['score'].mean(), color='blue', linestyle='-', linewidth=2,
               label=f'Selected avg: {selected_df["score"].mean():.3f}')
    
    ax3.set_xlabel('Vector Change Score', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Edge Score Distribution', fontweight='bold', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Direction analysis
    horizontal_edges = selected_df[selected_df['direction'] == 'horizontal']
    vertical_edges = selected_df[selected_df['direction'] == 'vertical']
    
    categories = ['Horizontal\\n(SDS direction)', 'Vertical\\n(TTAB direction)']
    counts = [len(horizontal_edges), len(vertical_edges)]
    avg_scores = [
        horizontal_edges['score'].mean() if len(horizontal_edges) > 0 else 0,
        vertical_edges['score'].mean() if len(vertical_edges) > 0 else 0
    ]
    
    ax4_twin = ax4.twinx()
    bars1 = ax4.bar([x - 0.2 for x in range(len(categories))], counts, 0.4,
                   label='Count', color='blue', alpha=0.7)
    bars2 = ax4_twin.bar([x + 0.2 for x in range(len(categories))], avg_scores, 0.4,
                        label='Avg Score', color='red', alpha=0.7)
    
    ax4.set_xlabel('Edge Direction', fontsize=12)
    ax4.set_ylabel('Number of Selected Edges', fontsize=12, color='blue')
    ax4_twin.set_ylabel('Average Score', fontsize=12, color='red')
    ax4.set_title('Selected Edges by Direction', fontweight='bold', fontsize=14)
    ax4.set_xticks(range(len(categories)))
    ax4.set_xticklabels(categories)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, count in zip(bars1, counts):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                f'{count}', ha='center', va='bottom', color='blue', fontweight='bold')
    
    for bar, score in zip(bars2, avg_scores):
        ax4_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_scores)*0.01,
                     f'{score:.3f}', ha='center', va='bottom', color='red', fontweight='bold')
    
    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    
    output_file = f'{output_dir}/vector_edge_refinement.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\\nVisualization saved: {output_file}")
    
    plt.show()

def save_recommendations(selected_df, output_dir='output/vector_edge_refinement'):
    """Save the recommended sampling points to CSV."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create recommendations dataframe
    recommendations = selected_df.copy()
    recommendations['rank'] = range(1, len(selected_df) + 1)
    recommendations = recommendations[['rank', 'sds_mid', 'ttab_mid', 'score', 'direction', 
                                    'turb_diff', 'ratio_diff']].round(4)
    
    recommendations.rename(columns={
        'sds_mid': 'sds_conc_mm',
        'ttab_mid': 'ttab_conc_mm',
        'score': 'boundary_score',
        'turb_diff': 'turbidity_change',
        'ratio_diff': 'ratio_change'
    }, inplace=True)
    
    output_file = f'{output_dir}/boundary_refinement_recommendations.csv'
    recommendations.to_csv(output_file, index=False)
    
    print(f"\\nRecommendations saved: {output_file}")
    print("\\nTop 10 recommended sampling points:")
    print(recommendations.head(10).to_string(index=False))
    
    return recommendations

def main():
    """Main function for vector edge refinement."""
    
    data_file = r"C:\\Users\\owenm\\OneDrive\\Desktop\\North Robotics\\utoronto_demo\\utoronto_demo\\surfactant_grid_SDS_TTAB_20260209_164920\\surfactant_grid_SDS_TTAB_20260209_164920\\complete_experiment_results.csv"
    
    if not os.path.exists(data_file):
        print(f"File not found: {data_file}")
        return
    
    print("="*70)
    print("VECTOR EDGE REFINEMENT FOR BOUNDARY DETECTION")
    print("="*70)
    
    print("\\n1. Loading experimental data...")
    experiment_data = load_and_prepare_data(data_file)
    
    print("\\n2. Normalizing vector field [turbidity, ratio]...")
    experiment_data, scaler_turb, scaler_ratio = normalize_vector_field(experiment_data)
    
    print("\\n3. Creating grid structure...")
    grid_data, sds_concs, ttab_concs = create_grid_structure(experiment_data)
    
    print("\\n4. Calculating edge scores (vector changes)...")
    edges_df = calculate_edge_scores(grid_data, sds_concs, ttab_concs)
    
    print("\\n5. Selecting boundary refinement points...")
    selected_df = select_boundary_points(edges_df, sds_concs, ttab_concs, n_points=32)
    
    print("\\n6. Creating visualization...")
    visualize_boundary_refinement(experiment_data, edges_df, selected_df)
    
    print("\\n7. Saving recommendations...")
    recommendations = save_recommendations(selected_df)
    
    print("\\n" + "="*70)
    print("VECTOR EDGE REFINEMENT RESULTS")  
    print("="*70)
    print(f"✓ Analyzed {len(edges_df)} edges between neighboring grid points")
    print(f"✓ Vector field normalization: log(turbidity) + z-score both outputs")
    print(f"✓ Edge scoring: sqrt((ΔT')² + (ΔR')²) - Euclidean distance in normalized space")
    print(f"✓ Selected {len(selected_df)} boundary points with spacing enforcement")
    print(f"✓ Score range: {selected_df['score'].min():.4f} - {selected_df['score'].max():.4f}")
    
    horizontal_count = len(selected_df[selected_df['direction'] == 'horizontal'])
    vertical_count = len(selected_df[selected_df['direction'] == 'vertical'])
    print(f"\\n✓ Edge directions: {horizontal_count} horizontal (SDS), {vertical_count} vertical (TTAB)")
    print("✓ Strategy: Sample at midpoints of strongest vector field boundaries")
    print("✓ Benefit: Automatically finds transitions in turbidity, ratio, or both")

if __name__ == "__main__":
    main()