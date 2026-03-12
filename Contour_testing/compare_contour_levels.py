"""
Quick Contour Level Comparison Tool
Generate multiple versions with different detail levels to compare
"""
import os
import sys
sys.path.append('.')

# Import our adjustable contour function
from adjustable_contour_levels import create_adjustable_contour_maps

CSV_FILE = r"C:\Users\Owen\Documents\GitHub\North-Cytation\New folder\iterative_experiment_results.csv"

# Test different level combinations
level_tests = [
    {"name": "Ultra_Clean", "turb": 4, "ratio": 4, "fluor": 4},     # Very clean
    {"name": "Clean", "turb": 5, "ratio": 6, "fluor": 5},          # Clean (current)
    {"name": "Moderate", "turb": 6, "ratio": 7, "fluor": 6},       # Moderate detail
    {"name": "Detailed", "turb": 8, "ratio": 10, "fluor": 8},      # Original high detail
]

print("🎯 GENERATING CONTOUR LEVEL COMPARISONS...")
print("This will create 4 versions with different detail levels")
print("="*60)

for test in level_tests:
    print(f"\\n📊 Creating {test['name']} version...")
    print(f"   Levels: Turbidity={test['turb']}, Ratio={test['ratio']}, Fluorescence={test['fluor']}")
    
    try:
        output_path = create_adjustable_contour_maps(
            CSV_FILE,
            surfactant_a_name="SDS", 
            surfactant_b_name="DTAB",
            turb_levels=test['turb'],
            ratio_levels=test['ratio'], 
            fluor_levels=test['fluor'],
            grid_res=80
        )
        
        # Rename with descriptive name
        new_name = CSV_FILE.replace('.csv', f'_COMPARISON_{test["name"]}.png')
        os.rename(output_path, new_name)
        print(f"   ✅ Saved: {os.path.basename(new_name)}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\\n" + "="*60)
print("🔍 COMPARISON COMPLETE!")
print("Check your folder for 4 different versions:")
print("   • Ultra_Clean (4,4,4) - Minimal detail, cleanest")
print("   • Clean (5,6,5) - Recommended for your data")  
print("   • Moderate (6,7,6) - More detail")
print("   • Detailed (8,10,8) - Original high detail")
print("\\n💡 TIP: Compare them to see which level of detail works best!")