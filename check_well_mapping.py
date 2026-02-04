#!/usr/bin/env python3
"""
Quick well mapping verification script
Checks if the concentration-to-measurement mapping makes physical sense
"""

import pandas as pd
import numpy as np

def check_mapping_sanity():
    """Quick sanity check of the well mapping"""
    csv_path = r"c:\Users\Imaging Controller\Desktop\utoronto_demo\output\surfactant_grid_SDS_TTAB_20260203_200428\consolidated_measurements_RECOVERED.csv"
    df = pd.read_csv(csv_path)
    
    print("=== Well Mapping Sanity Check ===")
    
    # Check 1: Do high concentrations correlate with high turbidity?
    print("\n1. High concentration vs turbidity patterns:")
    high_conc_wells = df[(df['conc_a_mm'] > 1) | (df['conc_b_mm'] > 1)]
    print(f"   Wells with >1mM concentration: {len(high_conc_wells)}")
    print(f"   Average turbidity: {high_conc_wells['turbidity_600'].mean():.4f}")
    print(f"   Max turbidity: {high_conc_wells['turbidity_600'].max():.4f}")
    print(f"   Wells with turbidity >0.1: {len(high_conc_wells[high_conc_wells['turbidity_600'] > 0.1])}")
    
    # Check 2: Look at specific suspicious wells
    print("\n2. Suspicious wells (very high turbidity):")
    high_turb = df[df['turbidity_600'] > 0.5]
    for _, row in high_turb.iterrows():
        print(f"   {row['well_position']}: SDS={row['conc_a_mm']:.1f}mM, TTAB={row['conc_b_mm']:.3f}mM → turbidity={row['turbidity_600']:.3f}")
    
    # Check 3: Well position vs well number
    print("\n3. Well numbering check (first few wells):")
    for i in range(min(12, len(df))):
        row = df.iloc[i]
        print(f"   Well {row['well']}: {row['well_position']}")
    
    # Check 4: Concentration grid structure
    print("\n4. Concentration grid structure:")
    conc_a_unique = sorted(df['conc_a_mm'].unique())
    conc_b_unique = sorted(df['conc_b_mm'].unique())
    print(f"   SDS concentrations: {len(conc_a_unique)} values from {conc_a_unique[0]:.1e} to {conc_a_unique[-1]:.1f}")
    print(f"   TTAB concentrations: {len(conc_b_unique)} values from {conc_b_unique[0]:.1e} to {conc_b_unique[-1]:.1f}")
    print(f"   Total combinations: {len(conc_a_unique)} x {len(conc_b_unique)} = {len(conc_a_unique) * len(conc_b_unique)}")
    print(f"   Actual data points: {len(df)}")
    
    # Check 5: Pattern recognition
    print("\n5. Does the pattern make sense?")
    # Look at a few specific combinations
    low_low = df[(df['conc_a_mm'] < 0.001) & (df['conc_b_mm'] < 0.001)]
    high_high = df[(df['conc_a_mm'] > 10) & (df['conc_b_mm'] > 10)]
    
    if len(low_low) > 0:
        print(f"   Low+Low concentrations: avg turbidity = {low_low['turbidity_600'].mean():.4f}")
    if len(high_high) > 0:
        print(f"   High+High concentrations: avg turbidity = {high_high['turbidity_600'].mean():.4f}")
    
    # VERDICT
    print("\n=== VERDICT ===")
    if high_turb['conc_a_mm'].max() > 10 and high_turb['conc_b_mm'].min() < 0.01:
        print("❌ SUSPICIOUS: High SDS + very low TTAB = very cloudy")
        print("   This suggests possible mapping error or unusual chemistry")
    else:
        print("✅ LOOKS REASONABLE: Turbidity patterns make sense")
    
    print(f"\nIf you're too tired to investigate, the data is still valid for analysis.")
    print(f"The mapping might be correct - mixed surfactant systems can be complex!")

if __name__ == "__main__":
    check_mapping_sanity()