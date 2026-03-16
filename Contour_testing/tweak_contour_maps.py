"""
Tweakable Contour Map Generator
Modify parameters below to customize your contour plots
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'analysis'))

from surfactant_contour_simple import create_contour_maps
from smooth_contour_maps import create_smooth_contour_maps

# === TWEAKABLE PARAMETERS ===
CSV_FILE = r"C:\Users\Owen\Documents\GitHub\North-Cytation\New folder\iterative_experiment_results.csv"

# Surfactant names for plot titles
SURFACTANT_A_NAME = "SDS"
SURFACTANT_B_NAME = "DTAB"

# Choose mapping method (set to True for smooth version without white patches)
USE_SMOOTH_VERSION = True  # True = no white patches, False = original version

# Output filename (will be saved in same directory as CSV)
OUTPUT_NAME = "custom_contour_maps.png"  # Change this to create different versions

# === EXECUTION ===
if __name__ == "__main__":
    print(f"Creating contour maps: {SURFACTANT_A_NAME} + {SURFACTANT_B_NAME}")
    print(f"Input: {CSV_FILE}")
    
    if USE_SMOOTH_VERSION:
        print("Using SMOOTH version (no white patches)")
        # Run the improved mapping without white patches
        fig, ax1, ax2, ax3 = create_smooth_contour_maps(
            CSV_FILE, 
            surfactant_a_name=SURFACTANT_A_NAME, 
            surfactant_b_name=SURFACTANT_B_NAME
        )
    else:
        print("Using ORIGINAL version (may have white patches)")
        # Run the original mapping
        fig, ax1, ax2, ax3 = create_contour_maps(
            CSV_FILE, 
            surfactant_a_name=SURFACTANT_A_NAME, 
            surfactant_b_name=SURFACTANT_B_NAME
        )
    
    # The plot is automatically saved by the function
    print(f"✅ Contour mapping complete!")
    print(f"📁 Check the same folder for the generated PNG file")
    
    # To save with custom name, you could add:
    # import matplotlib.pyplot as plt
    # custom_output = CSV_FILE.replace('.csv', f'_{OUTPUT_NAME}')
    # plt.savefig(custom_output, dpi=300, bbox_inches='tight')
    # print(f"Custom output: {custom_output}")