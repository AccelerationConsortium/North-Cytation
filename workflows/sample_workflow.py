import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import pandas as pd
import numpy as np
import time

INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/sample_input_vials.txt"
MEASUREMENT_PROTOCOL_FILE = r"C:\Protocols\Quick_Measurement.prt"
REACTOR_NUM = 1 #Green reactor

#Define your workflow! 
#In this case we have two parameters: 
def check_input_file(input_file):  
    # Initial State of your Vials, so the robot can know where to pipet
    vial_status = pd.read_csv(input_file, sep=",")
    print(vial_status)
    input("Only hit enter if the status of the vials (including open/close) is correct, otherwise hit ctrl-c")

def sample_workflow(aspiration_volume, replicates=3):
  
    INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/sample_input_vials.txt"
    MEASUREMENT_PROTOCOL_FILE = r"C:\Protocols\Quick_Measurement.prt"
    REACTOR_NUM = 1 #Green reactor

    # Initial State of your Vials, so the robot can know where to pipet
    vial_status = pd.read_csv(INPUT_VIAL_STATUS_FILE, sep=",")

    print(vial_status)
    input("Only hit enter if the status of the vials (including open/close) is correct, otherwise hit ctrl-c")

    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(INPUT_VIAL_STATUS_FILE)

    #The vial indices are numbers that are used to track the vials. For the sake of clarity, these are stored in the input vial file but accessed here
    target_vial_index = lash_e.nr_robot.get_vial_index_from_name('target_vial') #Get the ID of our target reactor
    reservoir_A_index = lash_e.nr_robot.get_vial_index_from_name('source_vial_a')
    reservoir_B_index = lash_e.nr_robot.get_vial_index_from_name('source_vial_b')

    #Grab a new wellplate!
    #lash_e.grab_new_wellplate()
    
    #This uncaps our target vessel to receive liquids in the clamp
    lash_e.nr_robot.move_vial_to_location(target_vial_index,location='clamp',location_index=0)
    lash_e.nr_robot.uncap_clamp_vial()
      
    #Transfer our aspiration_volume from reservoir A to our target vial (This is set when we call the method)
    lash_e.nr_robot.dispense_from_vial_into_vial(reservoir_A_index,target_vial_index,aspiration_volume)
    lash_e.nr_robot.remove_pipet()

    #Transfer aspiration_volume from reservoir B to our target vial (This is set when we call the method)
    lash_e.nr_robot.dispense_from_vial_into_vial(reservoir_B_index,target_vial_index,aspiration_volume)
    lash_e.nr_robot.remove_pipet()

    #Mix your vessel using vortexing for ~5 seconds
    lash_e.nr_robot.grab_vial(target_vial_index)
    lash_e.nr_robot.vortex_vial(target_vial_index,vortex_time=5)

    # Move the vial to the photoreactor. Since there are multiple reactors, we are going to reactor 0.
    lash_e.nr_robot.drop_off_vial(target_vial_index,location='photoreactor_array',location_index=0)
    lash_e.photoreactor.turn_on_reactor_led(reactor_num=REACTOR_NUM,intensity=100) #Let's turn on the photoreactor to intensity 100
    lash_e.photoreactor.turn_on_reactor_led(reactor_num=REACTOR_NUM,rpm=600) #Let's start stirring at 600 rpm
    lash_e.nr_robot.move_home()
    time.sleep(3)

    #Let's aspirate 50% of our target volume. Note this will cause an error if this volume is higher than 1 mL
    lash_e.photoreactor.turn_off_reactor_fan(reactor_num=REACTOR_NUM) #Turn off stirring before aspiration
    lash_e.nr_robot.aspirate_from_vial(target_vial_index, aspiration_volume*0.5)

    #Let's divide up the amount we are aspirating to dispense into the wells. 
    well_dispense_amount = aspiration_volume*0.5/replicates 

    #Both the well_plate_array and the volumes are Arrays, meaning that you can dispense into multiple wells in one action
    well_indices = range (0, replicates) #The range function gives us a list of well indexes from 0 until our replicate number... eg 0,1,2 for replicates=3
    lash_e.nr_robot.dispense_into_wellplate(well_indices, [well_dispense_amount]*replicates) #Dispense into the wells
    lash_e.nr_robot.remove_pipet()

    #Turn off the photoreactor
    lash_e.photoreactor.turn_off_reactor_led(reactor_num=REACTOR_NUM)
    
    #Send the target_vial back to its home position
    lash_e.nr_robot.return_vial_home(target_vial_index)

    #Transfer the well plate to the cytation and measure
    lash_e.measure_wellplate(MEASUREMENT_PROTOCOL_FILE)
    
#Execute the sample workflow.
#Specify that we are going to aspirate 0.6 from our two sample vials. We could also set the number of replicates to some other number than 3
#e.g. sample_workflow(aspiration_volume=0.6,replicates=5)
sample_workflow(aspiration_volume=0.6)
