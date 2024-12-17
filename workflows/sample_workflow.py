import sys
sys.path.append("../utoronto_demo")
import master_usdl_coordinator as lash_e
import pandas as pd
import time

#Define your workflow! Make sure that it has parameters that can be changed!
def sample_workflow(input_vial_status_file, source_vial_position, dest_vial_position, aspirate_volume_mL, well_plate_array,reactor_time,cytation_protocol_file_path):
  
    # Initial State of your Vials, so the robot can know where to pipet
    vial_status = pd.read_csv(input_vial_status_file)
    print(vial_status)
    input("Only hit enter if the status of the vials (including open/close) is correct, otherwise hit ctrl-c")

    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e.initialize(vial_status)

    #Mix the target vial by vortexing for 5 seconds
    #lash_e.nr_robot has methods specific to the north robot
    #lash_e.c9 has even more specific methods from the north API
    lash_e.nr_robot.vortex_vial(source_vial_position, 5)

    #Aspirate from your vial
    lash_e.nr_robot.aspirate_from_vial(source_vial_position, aspirate_volume_mL)

    #Dispense half (0.5x) of the amount into another vial
    lash_e.nr_robot.dispense_into_vial(dest_vial_position, aspirate_volume_mL*0.5)

    #Dispense 1/6th of the aspirated amount into 3 different wells
    #Both the well_plate_array and the volumes are Arrays, meaning that you can dispense into multiple wells in one action
    lash_e.nr_robot.dispense_into_wellplate(well_plate_array, [aspirate_volume_mL/6, aspirate_volume_mL/6, aspirate_volume_mL/6])

    #Remove the pipet and return the vial if needed
    lash_e.nr_robot.finish_pipetting()

    #Move the vial to the photoreactor and run photoreactor, then return. Note there are two photoreactors 1 and 2. The LED colors must be switched manually. 4 options: White, Blue, Green, Violet
    #Note: DO NOT look indirectly at the violet light
    lash_e.run_photoreactor(dest_vial_position,target_rpm=600,intensity=100,duration=reactor_time,reactor_num=1)

    #Transfer the well plate to the cytation and measure
    lash_e.measure_wellplate(cytation_protocol_file_path)
    
#Execute the sample workflow. Pipet from vial 0 to vial 1, then to positions 0,1,2 in the well plate. The total pipetted volume is 0.6 mL or 600 uL. 300 uL will go to vial 1, 100 uL will go to each well.
#Note I will have a conversion of "A1" to 0 and "A2" to 1 for the future, so you could do ["A1", "A2", "A3"] if you prefer that over 0,1,2
#Your protocol needs to be made inside the gen5 software, including the automated export
sample_workflow("../utoronto_demo/status/sample_input_vials.csv", 0, 1, 0.6, [0,1,2], 10, r"C:\Protocols\Testing_Protocol.prt")
