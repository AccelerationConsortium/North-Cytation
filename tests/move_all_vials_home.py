#!/usr/bin/env python3
"""
Simple program to move all vials in input file back to their home locations
Usage: python move_all_vials_home.py
"""

import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import pandas as pd

# Input file
INPUT_VIAL_FILE = "../utoronto_demo/status/ilya_input_vials.csv"

def main():
    print("Moving all vials home...")
    
    # Initialize robot
    lash_e = Lash_E(INPUT_VIAL_FILE, simulate=False)
    
    # Read vial data
    vials = pd.read_csv(INPUT_VIAL_FILE)
    
    # Move each vial home
    for _, vial in vials.iterrows():
        vial_name = vial['vial_name']
        try:
            print(f"Moving {vial_name} home...")
            lash_e.nr_robot.return_vial_home(vial_name)
        except Exception as e:
            print(f"Error with {vial_name}: {e}")
    
    lash_e.nr_robot.move_home()
    print("Done!")

if __name__ == "__main__":
    main()