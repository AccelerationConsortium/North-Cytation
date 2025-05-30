import sys
from tkinter import NO
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
from North_Safe import North_Powder
from north import NorthC9
import pandas as pd
import time

#Define your workflow! 
#In this case we have two parameters: 
def sample_workflow():

    c9 = NorthC9("A", network_serial="AU06CNCF")
    powder_disp = North_Powder(c9)
    #powder_disp.activate_powder_channel(0)
    powder_disp.dispense_powder(0,1000)

#Execute the sample workflow.
#Specify that we are going to aspirate 0.6 from our two sample vials. We could also set the number of replicates to some other number than 3
#e.g. sample_workflow(aspiration_volume=0.6,replicates=5)
sample_workflow()
