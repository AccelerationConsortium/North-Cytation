import sys
sys.path.append("../utoronto_demo")
from North_Safe import North_Robot
from North_Safe import North_Track
from north import NorthC9
import pandas as pd
from biotek import Biotek
import time

#Define your workflow! Make sure that it has parameters that can be changed!
def sample_workflow(source_vial_position, dest_vial_position, aspirate_volume_mL, well_plate_array,protocol_file_path):
  
    # Initial State of your Vials, so the robot can know where to pipet
    VIAL_FILE = "../utoronto_demo/status/sample_input_vials.csv"  
    vial_df = pd.read_csv(VIAL_FILE)
    
    #Initialize the robot, track, and cytation 5
    c9 = NorthC9("A", network_serial="AU06CNCF")
    nr_robot = North_Robot(c9, vial_df)
    nr_track = North_Track(c9)
    cytation = Biotek()

    #Reset the robot state
    nr_robot.reset_after_initialization()

    #Mix the target vial by vortexing for 5 seconds
    nr_robot.vortex_vial(source_vial_position, 5)

    #Move a vial from the rack in position 0 to the clamp
    nr_robot.move_vial_to_clamp(source_vial_position)

    #Uncap the vial in the clamp
    nr_robot.uncap_clamp_vial()

    #Pipet from the source vial a specific amount in mL
    nr_robot.aspirate_from_vial(source_vial_position, aspirate_volume_mL)

    #Dispense half (0.5x) of the amount into another vial
    nr_robot.dispense_into_vial(dest_vial_position, aspirate_volume_mL*0.5)

    #Dispense 1/6th of the aspirated amount into 3 different wells
    #Both the well_plate_array and the volumes are Arrays, meaning that you can dispense into multiple wells in one action
    nr_robot.dispense_into_wellplate(well_plate_array, [aspirate_volume_mL/6, aspirate_volume_mL/6, aspirate_volume_mL/6])

    #Remove the pipet. Note that we didn't grab a pipet but this happens by default if you try to aspirate and there is no pipet. However, the removal needs to be specified. 
    nr_robot.remove_pipet()

    #Cap the vial in the clamp then return back
    nr_robot.recap_clamp_vial()
    nr_robot.return_vial_from_clamp()

    #Methods to be added in the future:
    #nr_robot.move_vial_to_photoreactor(vial_num)
    #nr_robot.run_photoreactor(target_rpm, target_time, relative_intensity)

    #Transfer the well plate to the cytation. Position 0 is the default well plate tray.
    nr_track.grab_well_plate_from_nr(0)
    nr_track.move_gripper_to_cytation()
    cytation.CarrierOut()
    nr_track.release_well_plate_in_cytation()
    cytation.CarrierIn()

    #Run your characterization protocol
    plate = cytation.load_protocol(protocol_file_path)
    run = cytation.run_protocol(plate)
    while cytation.protocol_in_progress(run):
        print("Read in progress...")
        time.sleep(10)

    #Transfer the sample back from cytation
    cytation.CarrierOut()
    nr_track.grab_well_plate_from_cytation()
    cytation.CarrierIn()
    nr_track.return_well_plate_to_nr(0)  
    
#Execute the sample workflow. Pipet from vial 0 to vial 1, then to positions 0,1,2 in the well plate. The total pipetted volume is 0.6 mL or 600 uL. 300 uL will go to vial 1, 100 uL will go to each well.
#Note I will have a conversion of "A1" to 0 and "A2" to 1 for the future, so you could do ["A1", "A2", "A3"] if you prefer that over 0,1,2
#Your protocol needs to be made inside the gen5 software, including the automated export
sample_workflow(0, 1, 0.6, [0,1,2], r"C:\Protocols\Testing_Protocol.prt")
