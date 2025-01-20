import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import pandas as pd
import numpy as np

#Define your workflow! Make sure that it has parameters that can be changed!
def sample_workflow(input_vial_status_file, source_vials, aspirate_volumes, dest_vial_position, wp_dispense_volume, well_plate_array,reactor_time,cytation_protocol_file_path):
  
    # Initial State of your Vials, so the robot can know where to pipet
    vial_status = pd.read_csv(input_vial_status_file, sep=r"\t", engine="python")
    print(vial_status)
    input("Only hit enter if the status of the vials (including open/close) is correct, otherwise hit ctrl-c")

    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(input_vial_status_file)

    #Grab a new wellplate!
    lash_e.grab_new_wellplate()
    
    #This uncaps our target vessel to receive liquids in the clamp
    lash_e.nr_robot.move_vial_to_clamp(dest_vial_position)
    lash_e.nr_robot.uncap_clamp_vial()

    #Aspirate from your two vials and dispense into your clamp vial
    for i in range (0, 2):
        lash_e.nr_robot.aspirate_from_vial(source_vials[i], aspirate_volumes[i])
        lash_e.nr_robot.dispense_into_vial(dest_vial_position, aspirate_volumes[i])
        lash_e.nr_robot.remove_pipet()

    #Remove the pipet and return the vial if needed
    lash_e.nr_robot.finish_pipetting()

    #Mix your vessel using vortexing... Currently this causes an error
    #lash_e.nr_robot.vortex_vial(dest_vial_position,5)

    #Move the vial to the photoreactor and run photoreactor, then return. Note there are two photoreactors 1 and 2. The LED colors must be switched manually. 4 options: White, Blue, Green, Violet
    #Note: DO NOT look indirectly at the violet light
    lash_e.run_photoreactor(dest_vial_position,target_rpm=600,intensity=100,duration=reactor_time,reactor_num=1)

    volume_in_vessel = np.sum(aspirate_volumes)
    aspirate_volume = volume_in_vessel/2
    lash_e.nr_robot.aspirate_from_vial(dest_vial_position,wp_dispense_volume)

    #Dispense 30% of the amount into wp
    #Both the well_plate_array and the volumes are Arrays, meaning that you can dispense into multiple wells in one action
    lash_e.nr_robot.dispense_into_wellplate(well_plate_array, [wp_dispense_volume*0.3, wp_dispense_volume*0.3, wp_dispense_volume*0.3])

    #Dispense the rest back into the vial
    lash_e.nr_robot.dispense_into_vial(dest_vial_position,0.1*wp_dispense_volume)

    lash_e.nr_robot.finish_pipetting()

    #Transfer the well plate to the cytation and measure
    lash_e.measure_wellplate(cytation_protocol_file_path)
    
#Execute the sample workflow. Pipet from vial 0 to vial 1, then to positions 0,1,2 in the well plate. The total pipetted volume is 0.6 mL or 600 uL. 300 uL will go to vial 1, 100 uL will go to each well.
#Note I will have a conversion of "A1" to 0 and "A2" to 1 for the future, so you could do ["A1", "A2", "A3"] if you prefer that over 0,1,2
#Your protocol needs to be made inside the gen5 software, including the automated export
sample_workflow("../utoronto_demo/status/sample_input_vials.txt", [0,1], [0.6,0.6], 2, 0.2, [0,1,2], 5, r"C:\Protocols\Quick_Measurement.prt")
