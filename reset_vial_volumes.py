#!/usr/bin/env python3
"""
Reset vial volumes in the status file:
- Substocks (dilutions): 0.0 mL
- Stocks, water, refills, and other solutions: 7.8 mL
"""

import pandas as pd

vial_file = r"status/surfactant_multidim_vials.csv"

# Read the CSV
df = pd.read_csv(vial_file)

print("Before reset:")
print(df[['vial_name', 'vial_volume']].to_string())
print("\n" + "="*60 + "\n")

# Reset volumes based on vial type
for idx, row in df.iterrows():
    vial_name = row['vial_name']
    
    # Substocks/dilutions: reset to 0
    if '_dilution_' in vial_name:
        df.loc[idx, 'vial_volume'] = 0.0
    # Stocks, water, and other solutions: reset to 7.8
    else:
        df.loc[idx, 'vial_volume'] = 7.8

# Save back
df.to_csv(vial_file, index=False)

print("After reset:")
print(df[['vial_name', 'vial_volume']].to_string())
print("\n" + "="*60)
print(f"\nVial file updated: {vial_file}")
print(f"Substocks (dilutions): set to 0.0 mL")
print(f"Stocks, water, refills, and other solutions: set to 7.8 mL")
