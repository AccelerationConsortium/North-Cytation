# --- cmc_workflow.py ---
import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import time

INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/microgel_inputs.csv" #This file contains the status of the vials used for the workflow
MEASUREMENT_PROTOCOL_FILE = r"C:\Protocols\abs_300_800_sweep.prt" #This is the measurement protocol developed in the Cytation software
LOGGING_FOLDER = "../utoronto_demo/logs/"
simulate = False
enable_logging = False

with Lash_E(INPUT_VIAL_STATUS_FILE, simulate=simulate, logging=enable_logging) as lash_e:
    
    lash_e.nr_robot.check_input_file() #Check the status of the input vials
    #lash_e.nr_track.check_input_file() #Check the status of the wellplates (for multiple wellplate assays)


    #Step 1. Dispense the solid into the reactor
    lash_e.mass_dispense_into_vial('reaction_mixture', mass_mg=20)

    #Step 2. Add the HmIm slowly
    lash_e.nr_robot.dispense_from_vial_into_vial('HmIm', 'reaction_mixture', volume=1.0)

    time.sleep(1*60*60)

    #Step 3. Add the Zn(OAc)2
    lash_e.nr_robot.dispense_from_vial_into_vial('Zn(OAc)2', 'reaction_mixture', 1.0)

    time.sleep(16*60*60)

    

    