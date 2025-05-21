import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import pandas as pd
import numpy as np


def aspirate_dispense_vials_wp(input_vial_status_file): #dispenses from vial to vial & vial to wellplate
  
    # Initial State of your Vials, so the robot can know where to pipet
    vial_status = pd.read_csv(input_vial_status_file, sep=r",", engine="python")
    print(vial_status)
    input("Only hit enter if the status of the vials (including open/close) is correct, otherwise hit ctrl-c")

    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(input_vial_status_file)

    lash_e.nr_robot.dispense_from_vial_into_vial(4,0,0.8)
    lash_e.nr_robot.aspirate_from_vial(4,0.9)
    lash_e.nr_robot.dispense_into_wellplate([0,1,2],[0.2,0.2,0.2])
    lash_e.nr_robot.dispense_into_vial(0,0.3)
    lash_e.nr_robot.remove_pipet()
    
    lash_e.nr_robot.move_home()



aspirate_dispense_vials_wp("../utoronto_demo/status/color_matching_vials.csv")
