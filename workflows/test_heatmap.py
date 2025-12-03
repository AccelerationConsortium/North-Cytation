"""
Quick test of the heatmap visualization function
"""
import sys
sys.path.append("../utoronto_demo")
import pandas as pd
import numpy as np
import os

# Import the heatmap function
from surfactant_grid_turbidity_screening import create_turbidity_heatmap, MATPLOTLIB_AVAILABLE

def test_heatmap_function():
    """Test the heatmap function with mock data"""
    print("Testing heatmap visualization...")
    print(f"Matplotlib available: {MATPLOTLIB_AVAILABLE}")
    
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available - heatmap test skipped")
        return
    
    # Create mock data that matches the workflow output format
    concentrations = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]  # 6x6 grid
    
    # Create mock results DataFrame
    results_data = []
    for i, conc_a in enumerate(concentrations):
        for j, conc_b in enumerate(concentrations):
            # Simulate turbidity with some pattern (higher when concentrations are similar)
            turbidity = 0.1 + 0.5 * np.exp(-abs(np.log10(conc_a) - np.log10(conc_b)))
            results_data.append({
                'conc_a_mm': conc_a,
                'conc_b_mm': conc_b, 
                'turbidity': turbidity
            })
    
    results_df = pd.DataFrame(results_data)
    print(f"Mock data created: {len(results_df)} data points")
    
    # Create test output folder
    output_folder = "test_heatmap_output"
    os.makedirs(output_folder, exist_ok=True)
    
    # Test heatmap creation
    heatmap_file = create_turbidity_heatmap(
        results_df=results_df,
        concentrations=concentrations,
        surfactant_a_name="TestA", 
        surfactant_b_name="TestB",
        output_folder=output_folder,
        logger=None
    )
    
    if heatmap_file and os.path.exists(heatmap_file):
        print(f"✅ Heatmap test successful: {heatmap_file}")
        print(f"File size: {os.path.getsize(heatmap_file)} bytes")
    else:
        print("❌ Heatmap test failed")

if __name__ == "__main__":
    test_heatmap_function()