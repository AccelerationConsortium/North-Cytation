import re
import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import pandas as pd
import numpy as np

VIAL_FILE = "../utoronto_demo/status/sample_input_vials.csv"  # Vials used

#Define your workflow! Make sure that it has parameters that can be changed!
def test_vortex(input_vial_status_file, target_vial_num,vortex_time):
  
    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(input_vial_status_file)

    lash_e.nr_robot.vortex_vial(target_vial_num, vortex_time=vortex_time)
    lash_e.nr_robot.return_vial_home(target_vial_num) #slight bug
    lash_e.nr_robot.move_home()

  

test_vortex(VIAL_FILE,2, 3)