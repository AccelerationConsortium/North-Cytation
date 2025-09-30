#!/usr/bin/env python3
"""
Step-by-step diagnostic to find exactly where the failure occurs
"""

from calibration_sdl_base import *
import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E

# Exact same setup as calibration_sdl_short.py
LIQUID = "glycerol"
SIMULATE = False
INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/calibration_vials_short.csv"

lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=SIMULATE, initialize_biotek=False)
lash_e.nr_robot.check_input_file()
lash_e.nr_robot.move_vial_to_location("measurement_vial_0", "clamp", 0)

# Test the exact parameters that are failing - convert speeds to integers
params = {
    'aspirate_speed': int(23.0),
    'aspirate_wait_time': 25.0, 
    'dispense_speed': int(9.0),
    'dispense_wait_time': 3.0,
    'pre_asp_air_vol': 0.1,
    'post_asp_air_vol': 0.0,
    'retract_speed': int(1.0),
    'overaspirate_vol': 0.025
}

print("Testing each step individually...")

try:
    print("Step 1: Creating PipettingParameters...")
    from pipetting_data.pipetting_parameters import PipettingParameters
    
    aspirate_params = PipettingParameters(
        aspirate_speed=params["aspirate_speed"],
        aspirate_wait_time=params["aspirate_wait_time"],
        retract_speed=params["retract_speed"],
        pre_asp_air_vol=params["pre_asp_air_vol"],
        post_asp_air_vol=params["post_asp_air_vol"],
    )
    print("✅ Created aspirate parameters successfully")
    
    dispense_params = PipettingParameters(
        dispense_speed=params["dispense_speed"],
        dispense_wait_time=params["dispense_wait_time"],
        air_vol=params["pre_asp_air_vol"] + params["post_asp_air_vol"],
    )
    print("✅ Created dispense parameters successfully")
    
    print("Step 2: Checking pump state before aspiration...")
    print(f"Current pump speeds: {lash_e.nr_robot.CURRENT_PUMP_SPEEDS}")
    print(f"Trying to set aspirate speed to: {params['aspirate_speed']}")
    
    print("Step 3: Attempting aspiration (this automatically gets the pipet tip)...")
    lash_e.nr_robot.aspirate_from_vial("liquid_source", 0.05 + params["overaspirate_vol"], parameters=aspirate_params)
    print("✅ Aspiration successful")
    
    print("Step 4: Attempting dispensing...")
    measurement = lash_e.nr_robot.dispense_into_vial("measurement_vial_0", 0.05 + params["overaspirate_vol"], parameters=dispense_params, measure_weight=True)
    print(f"✅ Dispensing successful - measurement: {measurement}")
    
except Exception as e:
    print(f"❌ FAILED at current step: {e}")
    import traceback
    traceback.print_exc()

finally:
    print("Cleaning up...")
    try:
        lash_e.nr_robot.remove_pipet()
        lash_e.nr_robot.return_vial_home("measurement_vial_0")
        lash_e.nr_robot.move_home()
    except:
        pass
