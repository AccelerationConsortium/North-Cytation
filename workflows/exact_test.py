#!/usr/bin/env python3
"""
Exact copy test - run ONE condition exactly like calibration_sdl_short.py
"""

# EXACT imports from calibration_sdl_short.py
from calibration_sdl_base import *
import sys
sys.path.append("../utoronto_demo")
import os
import logging
from master_usdl_coordinator import Lash_E
import recommenders.pipeting_optimizer_v2 as recommender
import slack_agent
from datetime import datetime
import recommenders.llm_optimizer as llm_opt

# EXACT settings from calibration_sdl_short.py  
LIQUID = "glycerol"
SIMULATE = False
REPLICATES = 3  # Same as SDL
DENSITY_LIQUID = LIQUIDS[LIQUID]["density"]
NEW_PIPET_EACH_TIME_SET = LIQUIDS[LIQUID]["refill_pipets"]
INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/calibration_vials_short.csv"

# EXACT initialization from calibration_sdl_short.py
lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=SIMULATE, initialize_biotek=False)
lash_e.nr_robot.check_input_file()
lash_e.nr_robot.move_vial_to_location("measurement_vial_0", "clamp", 0)

# EXACT state from calibration_sdl_short.py
state = {
    "measurement_vial_index": 0,
    "measurement_vial_name": "measurement_vial_0"
}

# EXACT volume and expectations from calibration_sdl_short.py
volume = 0.05
expected_mass = volume * DENSITY_LIQUID
expected_time = volume * 10.146 + 9.5813

# Test parameters - same as your first condition
params = {
    'aspirate_speed': 23.0,
    'aspirate_wait_time': 25.0, 
    'dispense_speed': 9.0,
    'dispense_wait_time': 3.0,
    'pre_asp_air_vol': 0.1,
    'post_asp_air_vol': 0.0,
    'retract_speed': 1.0,
    'overaspirate_vol': 0.025
}

raw_measurements = []

print("Testing EXACT same call as calibration_sdl_short.py...")

try:
    # EXACT function call from calibration_sdl_short.py line 108
    result = pipet_and_measure(lash_e, 'liquid_source', state["measurement_vial_name"], volume, params, expected_mass, expected_time, REPLICATES, SIMULATE, None, raw_measurements, LIQUID, NEW_PIPET_EACH_TIME_SET)
    print(f"SUCCESS: {result}")
except Exception as e:
    print(f"ERROR: {e}")
    
# EXACT cleanup from calibration_sdl_short.py
lash_e.nr_robot.remove_pipet()
lash_e.nr_robot.return_vial_home(state["measurement_vial_name"])
lash_e.nr_robot.move_home()
