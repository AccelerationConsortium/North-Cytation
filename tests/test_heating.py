import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import pandas as pd
import numpy as np
import time

#Define your workflow! 
#In this case we have two parameters: 
def sample_workflow():
  
    INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/sample_capped_vial.txt"
    MEASUREMENT_PROTOCOL_FILE = r"C:\Protocols\Quick_Measurement.prt"

    # Initial State of your Vials, so the robot can know where to pipet
    vial_status = pd.read_csv(INPUT_VIAL_STATUS_FILE, sep=",")

    print(vial_status)
    input("Only hit enter if the status of the vials (including open/close) is correct, otherwise hit ctrl-c")

    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(INPUT_VIAL_STATUS_FILE,initialize_robot=True,initialize_biotek=False)

    lash_e.temp_controller.set_temp(0,50)

    time_e=0
    while time_e < 600:
        time.sleep(1)
        print("Temperature: ", lash_e.temp_controller.get_temp(0))
        time_e+=1

#Execute the sample workflow.
#Specify that we are going to aspirate 0.6 from our two sample vials. We could also set the number of replicates to some other number than 3
#e.g. sample_workflow(aspiration_volume=0.6,replicates=5)
sample_workflow()
