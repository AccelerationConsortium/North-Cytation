"""
Generate contour maps for surfactant experiment data 
Uses the existing surfactant_contour_simple.py functionality
"""
import sys
import os

# Add the analysis directory to path to import contour mapping function
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'analysis'))

from surfactant_contour_simple import create_contour_maps

if __name__ == "__main__":
    # Path to your CSV data
    csv_file_path = r"C:\Users\Owen\Documents\GitHub\North-Cytation\New folder\iterative_experiment_results.csv"
    
    print("Creating contour maps for SDS + DTAB surfactant data...")
    print(f"Input CSV: {csv_file_path}")
    
    # Create the contour maps - same exact mapping as existing system
    fig, ax1, ax2, ax3 = create_contour_maps(
        csv_file_path, 
        surfactant_a_name="SDS", 
        surfactant_b_name="DTAB"
    )
    
    print("Contour mapping complete!")