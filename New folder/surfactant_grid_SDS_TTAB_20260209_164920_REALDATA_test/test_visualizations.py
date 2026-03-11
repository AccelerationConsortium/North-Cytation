#!/usr/bin/env python3
"""
Test script to verify visualization functions work with existing data.
"""

import sys
import os
# Add the workflows directory to the path
workflows_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'workflows')
sys.path.insert(0, workflows_path)

import pandas as pd
import logging
from pathlib import Path

# Import the functions we want to test
from surfactant_grid_adaptive_concentrations import generate_surfactant_grid_heatmaps, calculate_adaptive_concentration_bounds

def test_visualizations():
    """Test both visualization functions with existing data."""
    
    # Setup
    current_dir = Path(__file__).parent
    csv_file = current_dir / "complete_experiment_results.csv"
    output_dir = str(current_dir)
    
    # Setup logger
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(__name__)
    
    if not csv_file.exists():
        print(f"Error: CSV file not found: {csv_file}")
        return
    
    print("Testing visualization functions with existing data...")
    print("="*60)
    
    # Test 1: Heatmap generation
    print("\\n1. Testing heatmap generation...")
    try:
        test_heatmap_dir = current_dir / "test_heatmaps"
        test_heatmap_dir.mkdir(exist_ok=True)
        
        generate_surfactant_grid_heatmaps(
            csv_file_path=str(csv_file),
            output_dir=str(test_heatmap_dir), 
            logger=logger,
            surfactant_a_name="SDS",
            surfactant_b_name="TTAB"
        )
        print("✅ Heatmap generation PASSED")
        
    except Exception as e:
        print(f"❌ Heatmap generation FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Adaptive concentration bounds
    print("\\n2. Testing adaptive concentration bounds calculation...")
    try:
        # Load experimental data
        df = pd.read_csv(csv_file)
        experiment_data = df[df['well_type'] == 'experiment'].copy()
        
        new_bounds = calculate_adaptive_concentration_bounds(
            experiment_df=experiment_data,
            surfactant_a_name="SDS", 
            surfactant_b_name="TTAB",
            output_dir=output_dir,
            logger=logger
        )
        
        print("✅ Adaptive bounds calculation PASSED")
        print(f"   New bounds calculated: {new_bounds}")
        
    except Exception as e:
        print(f"❌ Adaptive bounds calculation FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    print("\\n" + "="*60)
    print("Testing complete!")
    print("\\nGenerated files:")
    
    # List generated files
    heatmap_dir = current_dir / "test_heatmaps"
    if heatmap_dir.exists():
        for file in heatmap_dir.glob("*.png"):
            print(f"  📊 {file.relative_to(current_dir)}")
    
    threshold_dir = current_dir / "concentration_thresholds"
    if threshold_dir.exists():
        for file in threshold_dir.glob("*.png"):
            print(f"  📈 {file.relative_to(current_dir)}")

if __name__ == "__main__":
    test_visualizations()