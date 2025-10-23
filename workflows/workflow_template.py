"""
Template for creating new workflows in the North Robotics automation system.
Copy this file and modify it to create your own workflow.

Author: North Robotics Team
Date: {DATE}
"""
import sys
sys.path.append("../utoronto_demo")
import time
from master_usdl_coordinator import Lash_E
from pipetting_data.pipetting_parameters import PipettingParameters

def your_workflow_name(param1: float, param2: int = 3, simulate: bool = True):
    """
    [REPLACE] Brief description of what this workflow does.
    
    Args:
        param1 (float): [REPLACE] Description of parameter 1
        param2 (int): [REPLACE] Description of parameter 2 (default: 3)
        simulate (bool): Run in simulation mode without hardware (default: True)
    
    Workflow Steps:
        1. [REPLACE] Initialize system
        2. [REPLACE] Describe each major step
        3. [REPLACE] ...
        4. [REPLACE] Final measurements and cleanup
    """
    
    # === CONFIGURATION SECTION ===
    # [REPLACE] Modify these file paths and parameters for your workflow
    INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/your_vials.csv"  # Create your own CSV
    MEASUREMENT_PROTOCOL_FILE = r"C:\Protocols\Your_Protocol.prt"      # Your Cytation protocol
    
    # Experimental parameters - modify as needed
    TARGET_TEMPERATURE = 25.0  # Celsius
    REACTION_TIME = 30         # seconds
    VORTEX_TIME = 5           # seconds
    
    # === INITIALIZATION ===
    print(f"🤖 Starting {your_workflow_name.__name__} workflow...")
    print(f"   Parameters: param1={param1}, param2={param2}")
    print(f"   Simulation mode: {simulate}")
    
    # Initialize the system - adjust initialization flags as needed
    lash_e = Lash_E(
        INPUT_VIAL_STATUS_FILE,
        initialize_t8=True,      # Temperature controller
        initialize_p2=True,      # Photoreactor
        simulate=simulate
    )
    
    # === SAFETY CHECKS ===
    # Always validate input files before starting
    lash_e.nr_robot.check_input_file()
    lash_e.nr_track.check_input_file()
    
    try:
        # === STEP 1: SETUP ===
        print("📋 Step 1: System setup...")
        
        # Set temperature if needed
        lash_e.temp_controller.set_temp(TARGET_TEMPERATURE)
        
        # Get a new wellplate
        lash_e.grab_new_wellplate()
        
        # === STEP 2: SAMPLE PREPARATION ===
        print("🧪 Step 2: Sample preparation...")
        
        # [REPLACE] Add your sample preparation steps here
        # Examples:
        # lash_e.mass_dispense_into_vial('source_vial', mass_mg=20, return_home=False)
        # lash_e.nr_robot.dispense_into_vial_from_reservoir(
        #     reservoir_index=0, 
        #     vial_index='source_vial', 
        #     volume=5.0
        # )
        # lash_e.nr_robot.vortex_vial(vial_name='source_vial', vortex_time=VORTEX_TIME)
        
        # === STEP 3: LIQUID HANDLING ===
        print("💧 Step 3: Liquid handling operations...")
        
        # [REPLACE] Add your liquid handling steps here
        # Examples:
        # lash_e.nr_robot.dispense_from_vial_into_vial(
        #     source_vial_name='source_vial_a',
        #     dest_vial_name='target_vial', 
        #     volume=param1
        # )
        
        # === STEP 4: REACTIONS (if applicable) ===
        print("⚗️ Step 4: Reaction processing...")
        
        # [REPLACE] Add reaction steps if needed
        # Examples:
        # REACTOR_NUM = 1
        # lash_e.nr_robot.move_vial_to_location('target_vial', 'photoreactor_array', 0)
        # lash_e.photoreactor.turn_on_reactor_led(reactor_num=REACTOR_NUM, intensity=100)
        # lash_e.photoreactor.stir_reactor(reactor_num=REACTOR_NUM, rpm=600)
        # time.sleep(REACTION_TIME)
        # lash_e.photoreactor.turn_off_reactor_led(reactor_num=REACTOR_NUM)
        # lash_e.photoreactor.turn_off_stirring(reactor_num=REACTOR_NUM)
        
        # === STEP 5: WELLPLATE PREPARATION ===
        print("🧫 Step 5: Wellplate preparation...")
        
        # [REPLACE] Add your wellplate dispensing logic here
        well_indices = list(range(param2))  # Use param2 as number of wells
        # dispense_volume = param1 / param2   # Example calculation
        
        # Example dispensing:
        # lash_e.nr_robot.aspirate_from_vial('target_vial', total_volume)
        # lash_e.nr_robot.dispense_into_wellplate(
        #     dest_wp_num_array=well_indices, 
        #     amount_mL_array=[dispense_volume] * param2
        # )
        # lash_e.nr_robot.remove_pipet()
        
        # === STEP 6: MEASUREMENTS ===
        print("📊 Step 6: Measurements...")
        
        # [REPLACE] Modify the measurement protocol and wells as needed
        if not simulate:
            data = lash_e.measure_wellplate(MEASUREMENT_PROTOCOL_FILE, well_indices)
            print(f"   Measurement data collected for {len(well_indices)} wells")
        else:
            print("   Simulation: Skipping actual measurements")
            data = None
        
        # === STEP 7: CLEANUP ===
        print("🧹 Step 7: Cleanup...")
        
        # Return vials to home positions
        # [REPLACE] Add your vials here
        # lash_e.nr_robot.return_vial_home('source_vial')
        # lash_e.nr_robot.return_vial_home('target_vial')
        
        # Turn off equipment
        lash_e.temp_controller.turn_off_heating()
        lash_e.temp_controller.turn_off_stirring()
        
        # Discard wellplate
        lash_e.discard_used_wellplate()
        
        print("✅ Workflow completed successfully!")
        return data
        
    except Exception as e:
        print(f"❌ Workflow failed with error: {e}")
        # Emergency cleanup
        try:
            lash_e.temp_controller.turn_off_heating()
            lash_e.temp_controller.turn_off_stirring()
            lash_e.photoreactor.emergency_stop()
            lash_e.nr_robot.move_home()
        except:
            pass
        raise

def main():
    """
    Main function to run the workflow.
    [REPLACE] Modify the parameters here to test your workflow.
    """
    # [REPLACE] Adjust these parameters for your specific workflow
    param1_value = 0.5      # Example: volume in mL
    param2_value = 6        # Example: number of replicates
    simulate_mode = True    # Set to False for real hardware
    
    # Run the workflow
    result = your_workflow_name(
        param1=param1_value,
        param2=param2_value,
        simulate=simulate_mode
    )
    
    if result is not None:
        print(f"📈 Analysis results: {len(result)} measurements collected")
    
if __name__ == "__main__":
    main()