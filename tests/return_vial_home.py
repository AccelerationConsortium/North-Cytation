#!/usr/bin/env python3
"""
Emergency Vial Return Utility

This utility program safely returns any vial from the clamp back to its home position.
Use this when workflows are interrupted and you need to remotely clear the clamp.

Usage:
    python return_vial_home.py [vial_name]
    
If no vial name is provided, it will attempt to return the most recently used vial.
"""

import sys
import os
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import required modules
try:
    from master_usdl_coordinator import Lash_E
    print("[SUCCESS] Imported required modules")
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

# Configuration
SIMULATE = False  # Set to True for testing without hardware
INPUT_VIAL_STATUS_FILE = "status/calibration_vials_short.csv"

def return_vial_to_home(vial_name=None):
    """
    Safely return vial from clamp to home position.
    
    Args:
        vial_name (str): Name of vial to return. If None, attempts to find active vial.
    """
    
    print("=" * 60)
    print("EMERGENCY VIAL RETURN UTILITY")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Simulation mode: {SIMULATE}")
    print("")
    
    try:
        # Initialize Lash_E coordinator
        print("Initializing Lash_E coordinator...")
        lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=SIMULATE)
        
        
        # Determine which vial to return
        if vial_name is None:
            # Try to find the most common vials that might be in clamp
            common_vials = ["liquid_source_0", "measurement_vial", "liquid_source", "water_stock"]
            
            print("No vial name specified. Checking common vial positions...")
            for test_vial in common_vials:
                try:
                    # Check if this vial exists in the status file
                    print(f"Checking for vial: {test_vial}")
                    # This will raise an exception if vial doesn't exist
                    lash_e.nr_robot.return_vial_home(test_vial)
                    print(f"[SUCCESS] Returned {test_vial} from clamp to home position")
                    return True
                except Exception as e:
                    print(f"[INFO] {test_vial} not found or already at home: {str(e)}")
                    continue
            
            print("[WARNING] No common vials found to return. Try specifying vial name manually.")
            return False
            
        else:
            # Return specific vial
            print(f"Returning specified vial: {vial_name}")
            lash_e.nr_robot.return_vial_home(vial_name)
            print(f"[SUCCESS] Returned {vial_name} from clamp to home position") 
            return True
            
    except Exception as e:
        print(f"[ERROR] Failed to return vial: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to handle command line arguments."""
    
    # Check if vial name provided as command line argument
    vial_name = None
    if len(sys.argv) > 1:
        vial_name = sys.argv[1]
        print(f"Target vial: {vial_name}")
    else:
        print("No vial specified - will attempt to return common vials")
    
    print("")
    
    # Attempt to return vial
    success = return_vial_to_home(vial_name)
    
    if success:
        print("")
        print("[SUCCESS] Vial return operation completed successfully!")
        print("The clamp should now be clear.")
    else:
        print("")
        print("[ERROR] Vial return operation failed.")
        print("You may need to specify the exact vial name or check the robot status.")
        sys.exit(1)


if __name__ == "__main__":
    main()