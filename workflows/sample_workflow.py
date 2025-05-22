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
    6. Mix the target vial
    7. Move the vial to the photoreactor and start the reaction
    8. Aspirate from the target vial after reaction
    9. Dispense into the wellplate
    10. Turn off the photoreactor
    11. Return the target vial to its home position
    12. Measure the wellplate
    """

    # 1. Initialize the workstation (robot for pipetting, track for moving the wellplate, cytation for measuring the wellplate, photoreactors)
    """ 
    The argument here is the path to file that tracks what vials are used in the workflow.
    The file should be a CSV with the following vial_names:
        - source_vial_a: Name of the first source vial 
        - source_vial_b: Name of the second source vial
        - target_vial: Name of the target vial
    See the example file  ../utoronto_demo/status/sample_input_vials.csv
    Note: You should create your own csv file with the vials you want to use in your workflow. 
    Note: If you set simulate=True, you can run your code without the robot, photoreactor, or cytation to see if there are any errors.
    """
    INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/sample_input_vials.csv"
    lash_e = Lash_E(INPUT_VIAL_STATUS_FILE,simulate=True) # Initialize the Lash_E class with the input vial status file

    # 2. Check the status of the input vials (user must confirm in terminal that everything looks ok)
    lash_e.nr_robot.check_input_file()

    # 3. Prepare the target vial to receive liquids
    """
    move_vial_to_location(vial_name: str, location: str, location_index: int)
        - vial_name: Name of the vial to move
        - location: Description of the location to move to (e.g., 'clamp', 'photoreactor_array', 'main_8mL_rack', 'heater')
        - location_index: Index of the location to move to (e.g., 0 for the first position, 1 for the second position, etc.)
    """
    lash_e.nr_robot.move_vial_to_location(vial_index='target_vial', location='clamp', location_index=0)
    lash_e.nr_robot.uncap_clamp_vial() # Instruct the robot to uncap the vial in the clamp

    # 4. Transfer liquid from source_vial_a to target_vial, then remove the pipet
    """
    dispense_from_vial_into_vial(source_vial: str, dest_vial: str, volume: float)
        - source_vial: Name of the source vial to aspirate from
        - dest_vial: Name of the destination vial to dispense into
        - volume: Volume (in mL) to transfer
    """
    lash_e.nr_robot.dispense_from_vial_into_vial('source_vial_a', 'target_vial', aspiration_volume)
    lash_e.nr_robot.remove_pipet() #Instruct the robot to remove the pipet it carries

    # 5. Transfer liquid from source_vial_b to target_vial, then remove the pipet
    lash_e.nr_robot.dispense_from_vial_into_vial('source_vial_b', 'target_vial', aspiration_volume)
    lash_e.nr_robot.remove_pipet() #Instruct the robot to remove the pipet it carries

    # 6. Mix the target vial
    """
    vortex_vial(vial_name: str, vortex_time: float)
        - vial_name: Name of the vial to vortex
        - vortex_time: Time (in seconds) to vortex
    """
    lash_e.nr_robot.recap_clamp_vial() # Instruct the robot to recap the vial in the clamp
    lash_e.nr_robot.vortex_vial('target_vial', vortex_time=2)

    # 7. Move the vial to the photoreactor and then turn on the fan to mix the stir bar
    """
    turn_on_reactor_led(reactor_num: int, intensity: int)
        - reactor_num: Number of the reactor to turn on (e.g., 1 for the first reactor)
        - intensity: Intensity of the LED (0-100)
    turn_on_reactor_fan(reactor_num: int, rpm: int)
        - reactor_num: Number of the reactor to turn on (e.g., 1 for the first reactor)
        - rpm: target RPM of the fan (e.g., 600)
    """
    REACTOR_NUM = 1  # Green reactor
    lash_e.nr_robot.move_vial_to_location('target_vial', location='photoreactor_array', location_index=0) #Move the target vial to the photoreactor
    lash_e.photoreactor.turn_on_reactor_led(reactor_num=REACTOR_NUM, intensity=100)
    lash_e.photoreactor.stir_reactor(reactor_num=REACTOR_NUM, rpm=600)
    lash_e.nr_robot.move_home() # Move the robot to home position
    time.sleep(1) # Wait for 1 second

    # 8. Aspirate from the target vial, ensuring the stir bar is first turned off
    """
    turn_off_reactor_fan(reactor_num: int)
        - reactor_num: Number of the reactor to turn off (e.g., 1 for the first reactor)
    aspirate_from_vial(vial_name: str, volume: float)
        - vial_name: Name of the vial to aspirate from
        - volume: Volume (in mL) to aspirate, here we use half of the aspiration volume
    """
    total_volume_for_wells = aspiration_volume * 0.5
    lash_e.photoreactor.turn_off_stirring(reactor_num=REACTOR_NUM)
    lash_e.nr_robot.aspirate_from_vial('target_vial', total_volume_for_wells)

    # 9. Dispense into the wellplate, then remove the pipet
    """
    dispense_into_wellplate(well_indices: list or range, volumes: list[float])
        - well_indices: Indices of the wells to dispense into (e.g., [0, 1, 2])
        - volumes: Volumes (in mL) to dispense into each well (e.g., [0.1, 0.1, 0.1])
    """
    well_dispense_amount = total_volume_for_wells / replicates #This divides the total volume by the number of replicates
    well_indices = range(0, replicates) #This creates a list of indices from 0 to replicates-1
    dispense_volumes = [well_dispense_amount] * replicates #This duplicates the dispense amount for each well
    lash_e.nr_robot.dispense_into_wellplate(well_indices, dispense_volumes)
    lash_e.nr_robot.remove_pipet()

    # 10. Turn off the photoreactor
    """
    turn_off_reactor_led(reactor_num: int)
        - reactor_num: Number of the reactor to turn off (e.g., 1 for the first reactor)
    """
    lash_e.photoreactor.turn_off_reactor_led(reactor_num=REACTOR_NUM)

    # 11. Return the target vial to its home position
    """
    return_vial_home(vial_name: str)
        - vial_name: Name of the vial to return home
    """
    lash_e.nr_robot.return_vial_home('target_vial')

    # 12. Measure the wellplate
    """
    measure_wellplate(protocol_file: str, well_indices: list or range)
        - protocol_file: Path to the measurement protocol file (e.g., 'path/to/protocol.prt')
        - well_indices: Indices of the wells to measure (e.g., [0, 1, 2])
    """
    MEASUREMENT_PROTOCOL_FILE = r"C:\Protocols\Quick_Measurement.prt"
    lash_e.measure_wellplate(MEASUREMENT_PROTOCOL_FILE, well_indices)

if __name__ == "__main__": #This is the main function that runs when the script is executed
    """
    Run the sample workflow with a 0.6 mL aspiration volume
    Edit the aspiration volume here to change the amount of liquid transferred; eg by writing sample_workflow(aspiration_volume=0.8) for 0.8 mL    
    You can also change the number of replicates here; eg by writing sample_workflow(aspiration_volume=0.6, replicates=4) for 4 replicates
    The default number of replicates is 3, if you don't specify it
    Note that by setting parameters you can quickly and easily change the workflow without having to edit the code (except here)
    """
    sample_workflow(aspiration_volume=0.6)
