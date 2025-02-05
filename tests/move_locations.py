import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import pandas as pd
import numpy as np
from Locator import *

#Define your workflow! Make sure that it has parameters that can be changed!
def sample_workflow(input_vial_status_file, source_vials, aspirate_volumes, dest_vial_position, wp_dispense_volume, well_plate_array,reactor_time,cytation_protocol_file_path):
  
    # Initial State of your Vials, so the robot can know where to pipet
    vial_status = pd.read_csv(input_vial_status_file, sep=r"\t", engine="python")
    print(vial_status)
    input("Only hit enter if the status of the vials (including open/close) is correct, otherwise hit ctrl-c")

    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(input_vial_status_file)

    #lash_e.nr_robot.get_pipet()

    lash_e.nr_robot.c9.goto_safe(PR_PIP_1)

    input()

    lash_e.nr_robot.c9.goto_safe(PR_PIP_1)



    # #Transfer the well plate to the cytation and measure
    # lash_e.measure_wellplate(cytation_protocol_file_path)
    
#Execute the sample workflow. Pipet from vial 0 to vial 1, then to positions 0,1,2 in the well plate. The total pipetted volume is 0.6 mL or 600 uL. 300 uL will go to vial 1, 100 uL will go to each well.
#Note I will have a conversion of "A1" to 0 and "A2" to 1 for the future, so you could do ["A1", "A2", "A3"] if you prefer that over 0,1,2
#Your protocol needs to be made inside the gen5 software, including the automated export
sample_workflow("../utoronto_demo/status/sample_input_vials.txt", [0,1], [0.6,0.6], 2, 0.3, [0,1,2], 5, r"C:\Protocols\Quick_Measurement.prt")
