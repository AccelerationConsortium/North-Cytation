"""
Sample workflow for North Robotics automation.
This script demonstrates a typical liquid handling and measurement sequence.
Edit the parameters at the bottom to change the workflow.
"""
import sys
sys.path.append("../utoronto_demo") #Add the parent folder to the system path
import time #For pauses
from master_usdl_coordinator import Lash_E # This is the main class that coordinates the robot, photoreactor, and cytation

def sample_workflow(aspiration_volume: float, replicates: int = 3):
    """
    Run a sample workflow.

    Arguments:
        aspiration_volume (float): Volume (in mL) to transfer from each source vial to the target vial.
        replicates (int): Number of wells to dispense into (default: 3).

    Steps:
    1. Initialize the workstation (robot, track, cytation, photoreactors)
    2. Check the status of the input vials (user confirmation)
    3. Prepare the target vial to receive liquids
    4. Transfer liquid from source_vial_a to target_vial
    5. Transfer liquid from source_vial_b to target_vial
    6. Vortex the target vial
    7. Move the vial to the photoreactor and start the reaction
    8. Aspirate from the target vial after reaction
    9. Dispense into the wellplate
    10. Turn off the photoreactor
    11. Return the target vial to its home position
    12. Move the wellplate into the plate reader for measurements
    """

    # 1. Initialize the workstation (robot for pipetting, track for moving the wellplate, cytation for measuring the wellplate, photoreactors)
    """ 
    The argument here (INPUT_VIAL_STATUS_FILE) is the path to a file that tracks what vials are used in the workflow.
    The file should be a CSV with the following vial_names:
        - source_vial_a: Name of the first source vial 
        - source_vial_b: Name of the second source vial
        - target_vial: Name of the target vial
    See the example file  ../utoronto_demo/status/sample_input_vials.csv
    Note: You should create your own csv file with the vials you want to use in your workflow. 
    Note: If you set simulate=True, you can run your code without the robot, photoreactor, or cytation to see if there are any errors.
    """
    INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/sample_input_vials.csv"
    lash_e = Lash_E(INPUT_VIAL_STATUS_FILE,initialize_t8=True,initialize_p2=True,simulate=False) # Initialize the Lash_E class with the input vial status file

    # 2. Check the status of the input vials 
    lash_e.nr_robot.check_input_file() #outputs the values in sample_input_vials.csv and user must confirm by typing Enter if everything looks ok to proceed

    lash_e.temp_controller.set_temp(40) # Set the temperature of the heater to 40 degrees Celsius

    # 3. Prepare Source Vial A by adding solid and liquid (Note that in theory priming for the reservoir dispense is needed, but this is not done here)
    lash_e.mass_dispense_into_vial('source_vial_a', 20, return_home=False)
    lash_e.nr_robot.dispense_into_vial_from_reservoir(reservoir_index=1, vial_index = 'source_vial_a', volume=5.0) #Dispense 5 mL of water into source_vial_a from reservoir 0
    lash_e.nr_robot.vortex_vial(vial_name='source_vial_a', vortex_time=5) #Vortex source_vial_a for 5 seconds to mix the solid and liquid
    lash_e.nr_robot.move_vial_to_location(vial_name='source_vial_a', location='heater', location_index=0) #Move source_vial_a to the heater
    lash_e.temp_controller.turn_on_stirring()

    # 4. Transfer liquid from source_vial_a to target_vial, then remove the pipet tip
    lash_e.nr_robot.dispense_from_vial_into_vial(source_vial_name='source_vial_b', dest_vial_name='target_vial', volume=aspiration_volume) #pipet the amount specified in aspiration_volume from source_vial_a to target_vial
    lash_e.nr_robot.remove_pipet() #remove the pipet tip it carries

    # 5. Transfer liquid from source_vial_b to target_vial, then remove the pipet
    lash_e.nr_robot.dispense_from_vial_into_vial(source_vial_name='source_vial_a', dest_vial_name='target_vial', volume=aspiration_volume) #pipet the amount specified in aspiration_volume from source_vial_b to target_vial
    lash_e.nr_robot.remove_pipet() #remove pipet tip

    # 6. Mix the target vial
    lash_e.nr_robot.vortex_vial(vial_name='target_vial', vortex_time=2) # vortex the target_vial for 2 seconds

    # 7. Move the vial to the photoreactor and then turn on the fan to mix the stir bar
    REACTOR_NUM = 1  # Green reactor
    lash_e.nr_robot.move_vial_to_location(vial_name='target_vial', location='photoreactor_array', location_index=0) #Move the target vial to the photoreactor
    lash_e.photoreactor.turn_on_reactor_led(reactor_num=REACTOR_NUM, intensity=100) # Turn on the LED of the green reactor at 100% intensity
    lash_e.photoreactor.stir_reactor(reactor_num=REACTOR_NUM, rpm=600) # Turn on the fan of the green reactor for stirring at 600 RPM 
    lash_e.nr_robot.move_home() # Move the robot to home position
    time.sleep(1) # Wait for 1 second (while the reactor is on)

    # 8. Aspirate from the target vial, ensuring the stirring is first turned off
    total_volume_for_wells = aspiration_volume * 0.5 
    lash_e.photoreactor.turn_off_stirring(reactor_num=REACTOR_NUM) # Turn off the stirring
    lash_e.nr_robot.aspirate_from_vial('target_vial', total_volume_for_wells) # Aspirate half the total volume from the target vial

    # 9. Dispense into the wellplate, then remove the pipet
    well_dispense_amount = total_volume_for_wells / replicates #This divides the total volume by the number of replicates
    well_indices = range(0, replicates) #This creates a list of indices from 0 to replicates-1
    dispense_volumes = [well_dispense_amount] * replicates #This creates a list which repeats the dispense amount for each well
    lash_e.nr_robot.dispense_into_wellplate(dest_wp_num_array=well_indices, amount_mL_array=dispense_volumes) #dispense the volumes (specified in dispense_volumes) into the wells (specified in well_indices)
    lash_e.nr_robot.remove_pipet() #remove pipet tip

    # 10. Turn off the photoreactor & heater
    lash_e.photoreactor.turn_off_reactor_led(reactor_num=REACTOR_NUM)
    lash_e.temp_controller.turn_off_heating() #Turn off the heater
    lash_e.temp_controller.turn_off_stirring()

    # 11. Return the target vial to its home position
    lash_e.nr_robot.return_vial_home('source_vial_a')
    lash_e.nr_robot.return_vial_home('target_vial')

    # 12. Measure the wellplate
    MEASUREMENT_PROTOCOL_FILE = r"C:\Protocols\Quick_Measurement.prt" #Cytation protocol to run
    data = lash_e.measure_wellplate(MEASUREMENT_PROTOCOL_FILE, well_indices) #Move the wellplate to the cytation, run Quick_Measurement.prt for wells specified in well_indices, and return the wellplate

if __name__ == "__main__": #This is the main function that runs when the script is executed
    """
    Run the sample workflow with a 0.6 mL aspiration volume
    Edit the aspiration volume here to change the amount of liquid transferred; eg by writing sample_workflow(aspiration_volume=0.8) for 0.8 mL    
    You can also change the number of replicates here; eg by writing sample_workflow(aspiration_volume=0.6, replicates=4) for 4 replicates
    The default number of replicates is 3, if you don't specify it
    Note that by setting parameters you can quickly and easily change the workflow without having to edit the code (except here)
    """
    sample_workflow(aspiration_volume=0.6)
