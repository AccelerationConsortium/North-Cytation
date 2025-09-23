"""
Example of using the new dispense_from_vials_into_wellplate method with strategy pattern.

This example shows:
1. New DataFrame-based interface (recommended)
2. Strategy selection (serial vs batched vs auto)
3. Parameter system integration
4. Backwards compatibility with old interface
"""

import pandas as pd
from pipetting_data.pipetting_parameters import PipettingParameters

def example_new_interface(robot):
    """Example using the new clean interface"""
    
    # Create wellplate layout with vial names as columns
    wellplate_df = pd.DataFrame({
        'vial_A': [0.1, 0.2, 0.0, 0.15],  # Well 0-3 volumes from vial_A
        'vial_B': [0.0, 0.1, 0.3, 0.0],   # Well 0-3 volumes from vial_B  
        'water':  [0.2, 0.0, 0.1, 0.2]    # Well 0-3 volumes from water vial
    })
    
    # Example 1: Auto strategy (recommended for most cases)
    basic_params = PipettingParameters()
    robot.dispense_from_vials_into_wellplate(
        wellplate_df, 
        parameters=basic_params,
        strategy="auto"  # Will choose serial or batched based on parameters
    )
    
    # Example 2: Serial strategy for precision (full parameter support)
    precision_params = PipettingParameters(
        pre_asp_air_vol=0.05,      # Air before aspiration
        post_asp_air_vol=0.02,     # Air after aspiration
        blowout_vol=0.1,           # Blowout for precision
        asp_disp_cycles=2,         # Extra mixing cycles
        aspirate_wait_time=2.0     # Longer wait times
    )
    robot.dispense_from_vials_into_wellplate(
        wellplate_df,
        parameters=precision_params, 
        strategy="serial",          # Force serial for full parameter support
        well_plate_type="48 WELL PLATE"
    )
    
    # Example 3: Batched strategy for speed (limited parameters)
    speed_params = PipettingParameters(
        aspirate_speed=25,          # Fast aspiration
        dispense_speed=20,          # Fast dispensing
        aspirate_wait_time=0.5      # Minimal wait times
    )
    robot.dispense_from_vials_into_wellplate(
        wellplate_df,
        parameters=speed_params,
        strategy="batched",         # Force batched for speed optimization
        low_volume_cutoff=0.1
    )

def example_backwards_compatibility(robot):
    """Example showing backwards compatibility with old interface"""
    
    # Old style DataFrame (without proper column names)
    old_wellplate_df = pd.DataFrame([
        [0.1, 0.0, 0.2],  # Well 0: volumes from vial_A, vial_B, water
        [0.2, 0.1, 0.0],  # Well 1: volumes from vial_A, vial_B, water  
        [0.0, 0.3, 0.1],  # Well 2: volumes from vial_A, vial_B, water
        [0.15, 0.0, 0.2]  # Well 3: volumes from vial_A, vial_B, water
    ])
    
    # Set proper column names (recommended migration step)
    old_wellplate_df.columns = ['vial_A', 'vial_B', 'water']
    
    # Old interface still works (with deprecation warning)
    robot.dispense_from_vials_into_wellplate(
        old_wellplate_df, 
        vial_names=['vial_A', 'vial_B', 'water'],  # DEPRECATED
        well_plate_type="48 WELL PLATE",
        dispense_speed=20,                         # Legacy parameter
        wait_time=2,                              # Legacy parameter  
        asp_cycles=1,                             # Legacy parameter
        blowout_vol=0.1,                          # Legacy parameter
        pipet_back_and_forth=True                 # Legacy parameter (forces serial)
    )

def migration_guide():
    """
    Migration Guide from Old to New Interface:
    
    OLD INTERFACE:
    robot.dispense_from_vials_into_wellplate(
        wellplate_df, 
        vial_names=['vial_A', 'vial_B'], 
        dispense_speed=20,
        wait_time=2,
        asp_cycles=1,
        blowout_vol=0.1,
        pipet_back_and_forth=True
    )
    
    NEW INTERFACE:
    # Step 1: Ensure DataFrame has proper column names
    wellplate_df.columns = ['vial_A', 'vial_B']  # Set vial names as columns
    
    # Step 2: Create PipettingParameters object
    params = PipettingParameters(
        dispense_speed=20,
        aspirate_wait_time=2,
        asp_disp_cycles=1,
        blowout_vol=0.1
    )
    
    # Step 3: Use new interface
    robot.dispense_from_vials_into_wellplate(
        wellplate_df,
        parameters=params,
        strategy="serial"  # serial for precision (was pipet_back_and_forth=True)
    )
    
    BENEFITS:
    - ✅ Cleaner interface (vial names are column names)
    - ✅ Strategy pattern (serial vs batched vs auto)
    - ✅ Full PipettingParameters support in serial mode
    - ✅ Speed optimization in batched mode
    - ✅ Backwards compatibility during transition
    """
    pass

if __name__ == "__main__":
    print("This is an example file showing the new wellplate dispensing interface.")
    print("Import this module and call the example functions with a robot instance.")
    print("See migration_guide() docstring for migration instructions.")