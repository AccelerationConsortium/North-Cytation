#Test the capping and decapping of vials

import sys
import pandas as pd
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E

def pipet_into_wp():
    lash_e = Lash_E("../utoronto_demo/status/sample_capped_vials.csv", simulate = False, initialize_biotek=False)

    #lash_e.nr_robot.check_input_file()
    #lash_e.nr_track.check_input_file()

    #lash_e.grab_new_wellplate()
    # lash_e.nr_robot.aspirate_from_vial('Sample_A',0.3)
    # lash_e.nr_robot.dispense_into_wellplate([0,1,2], [0.1,0.1,0.1])
    # lash_e.nr_robot.remove_pipet()
    #lash_e.discard_used_wellplate()

def test_dispense_from_vials_into_wellplate():
    lash_e = Lash_E("../utoronto_demo/status/sample_capped_vials.csv", simulate = False, initialize_biotek=False)
    
    # Check input files
    #lash_e.nr_robot.check_input_file()
    #lash_e.nr_track.check_input_file()
    
   
    # Create wellplate DataFrame with Sample_A volumes for different wells
    # Columns are vial names, rows are wells (index corresponds to well number)
    well_plate_df = pd.DataFrame({
        "Sample_A": [0.05, 0.1, 0.05, 0.0, 0.0, 0.0],  # dispense 0.1 mL into wells 0, 1, 2
        "Sample_B": [0.0, 0.0, 0.0, 0.15, 0.15, 0.0]  # dispense 0.15 mL into wells 3, 4
    })
    
    # Use serial strategy for dispensing
    lash_e.nr_robot.dispense_from_vials_into_wellplate(
        well_plate_df=well_plate_df, 
        strategy="serial"
    )
    lash_e.nr_robot.move_home()
    
    # Remove pipet and discard wellplate

#pipet_into_wp()
test_dispense_from_vials_into_wellplate()