import sys
import time
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E 
import pandas as pd
import slack_agent
from pathlib import Path
from pipetting_data.embedded_calibration_validation import validate_pipetting_accuracy

def dispense_from_photoreactor_into_sample(lash_e,reaction_mixture_index,sample_index,volume=0.2):
    print("\nDispensing from photoreactor into sample: ", sample_index)
    lash_e.photoreactor.turn_off_reactor_fan(reactor_num=0)
    lash_e.nr_robot.dispense_from_vial_into_vial(reaction_mixture_index,sample_index,volume=volume, liquid='water')
    mix_current_sample(lash_e,sample_index,volume=0.8)
    lash_e.nr_robot.remove_pipet()
    lash_e.photoreactor.turn_on_reactor_fan(reactor_num=0,rpm=600)
    lash_e.nr_robot.move_home()
    lash_e.nr_robot.home_robot_components()
    print()

def transfer_samples_into_wellplate_and_characterize(lash_e,sample_index,first_well_index,cytation_protocol_file_path,replicates,output_dir,simulate=False,well_volume=0.2):
    print("\nTransferring sample: ", sample_index, " to wellplate at well index: ", first_well_index)
    lash_e.nr_robot.move_vial_to_location(sample_index, location="main_8mL_rack", location_index=44) #Move sample to safe pipetting position
    lash_e.nr_robot.aspirate_from_vial(sample_index, well_volume*replicates, track_height=True, liquid='water')
    wells = range(first_well_index,first_well_index+replicates)
    lash_e.nr_robot.dispense_into_wellplate(wells, [well_volume]*replicates, liquid='water')
    lash_e.nr_robot.remove_pipet()
    lash_e.nr_robot.return_vial_home(sample_index) #Return sample to home position
    data_out = lash_e.measure_wellplate(cytation_protocol_file_path, wells_to_measure=wells)
    # output_file = r'C:\Users\Imaging Controller\Desktop\SQ\output_'+str(first_well_index)+'.txt'
    if not simulate:
        output_file = output_dir / f'output_{first_well_index}.txt'
        data_out.to_csv(output_file, sep=',')
        #Use analyzer to analyze the data
    print()

def mix_current_sample(lash_e, sample_index, new_pipet=False,repeats=3, volume=0.25):
    print("\nMixing sample: ", sample_index)
    # if new_pipet:
    #     lash_e.nr_robot.remove_pipet()
    # lash_e.nr_robot.dispense_from_vial_into_vial(sample_index,sample_index,volume=volume,move_to_dispense=False,buffer_vol=0)
    # for _ in range (repeats-1):
    #     lash_e.nr_robot.dispense_from_vial_into_vial(sample_index,sample_index,volume=volume,move_to_aspirate=False,move_to_dispense=False,buffer_vol=0)
    # lash_e.nr_robot.remove_pipet() # This step is for pipetting up and down *3 to simulate mixing.
    lash_e.nr_robot.remove_pipet()
    lash_e.nr_robot.vortex_vial(sample_index, 5)
    print()


def get_time(simulate,current_time=None):
    if not simulate:
        return time.time()
    else:
        if current_time is None:
            return 0
        else:
            return current_time + 1

def validate_peroxide_pipetting(lash_e, output_dir=None, replicates=5, simulate=False):
    """
    Run pipetting validation for all solutions used in peroxide workflow.
    Uses actual volume ranges from the workflow for realistic validation.
    
    Args:
        lash_e: Initialized Lash_E coordinator
        output_dir: Directory for validation results (default: current directory)
        replicates: Number of replicates per volume test (default: 5)
        simulate: Whether running in simulation mode
    
    Returns:
        dict: Validation results for all solutions
    """
    if output_dir is None:
        output_dir = "output"
        
    print("\n" + "="*60)
    print("RUNNING PIPETTING VALIDATION BEFORE EXPERIMENT")
    print("="*60)
    print("Testing all solutions with volumes actually used in the workflow...")
    
    validation_results = {}
    
    # Setup validation vial for mass measurements (use an empty vial)
    validation_vial = "validation_vial"
    
    try:
        # 1. WATER VALIDATION (Large volumes: 950uL used in workflow)
        print("\n--- Validating Water Pipetting ---")
        water_volumes = [0.2, 0.8, 0.95]  # 200uL, 800uL (mixing), 950uL (actual workflow volume)
        water_results = validate_pipetting_accuracy(
            lash_e=lash_e,
            source_vial="Water",
            destination_vial="Water",
            liquid_type="water",
            volumes_ml=water_volumes,
            replicates=replicates,
            output_folder=output_dir,
            plot_title="Water Validation - Peroxide Workflow",
            switch_pipet=False  # Fresh tip for each measurement
        )
        validation_results['water'] = water_results
        print(f"Water validation complete: Accuracy {water_results['mean_accuracy_pct']:.1f}%, Precision {water_results['mean_precision_cv_pct']:.1f}%")
        
        # 2. REAGENT A+B VALIDATION (Medium volumes: 150uL used in workflow)
        print("\n--- Validating Reagent A+B Pipetting ---")
        reagent_volumes = [0.15]  # 50uL, 100uL, 150uL (actual), 200uL
        reagent_results = validate_pipetting_accuracy(
            lash_e=lash_e,
            source_vial="Reagent_A+B",
            destination_vial="Reagent_A+B",
            liquid_type="water",  # Assuming aqueous reagent
            volumes_ml=reagent_volumes,
            replicates=replicates,
            output_folder=output_dir,
            plot_title="Reagent A+B Validation - Peroxide Workflow",
            switch_pipet=False
        )
        validation_results['reagent_ab'] = reagent_results
        print(f"Reagent A+B validation complete: Accuracy {reagent_results['mean_accuracy_pct']:.1f}%, Precision {reagent_results['mean_precision_cv_pct']:.1f}%")
        
        # 3. REACTION MIXTURE VALIDATION (Medium volumes: 200uL used in workflow)
        # Note: This validates pipetting FROM photoreactor position
        print("\n--- Validating Reaction Mixture Pipetting ---")
        rxn_volumes = [0.2]  # 50uL, 100uL, 200uL (actual), 300uL
        rxn_results = validate_pipetting_accuracy(
            lash_e=lash_e,
            source_vial="Rxn_Mixture", 
            destination_vial="Rxn_Mixture", 
            liquid_type="water",  # Assuming aqueous reaction mixture
            volumes_ml=rxn_volumes,
            replicates=replicates,
            output_folder=output_dir,
            plot_title="Reaction Mixture Validation - Peroxide Workflow",
            switch_pipet=False
        )
        validation_results['rxn_mixture'] = rxn_results
        print(f"Reaction mixture validation complete: Accuracy {rxn_results['mean_accuracy_pct']:.1f}%, Precision {rxn_results['mean_precision_cv_pct']:.1f}%")
        
        # SUMMARY
        print("\n=== VALIDATION SUMMARY ===")
        for solution, results in validation_results.items():
            print(f"{solution.upper():15s}: Accuracy {results['mean_accuracy_pct']:6.1f}% | Precision {results['mean_precision_cv_pct']:6.1f}% | R² {results['r_squared']:.3f}")
        
        # Check if any validation failed significantly
        failed_validations = []
        for solution, results in validation_results.items():
            if abs(results['mean_accuracy_pct']) > 10.0:  # >10% error
                failed_validations.append(f"{solution} (accuracy: {results['mean_accuracy_pct']:.1f}%)")
            if results['mean_precision_cv_pct'] > 15.0:  # >15% CV
                failed_validations.append(f"{solution} (precision: {results['mean_precision_cv_pct']:.1f}% CV)")
                
        if failed_validations:
            print("\n⚠️  WARNING: Some validations failed quality thresholds:")
            for failure in failed_validations:
                print(f"   - {failure}")
            print("Consider recalibrating pipetting parameters before running workflow.")
        else:
            print("\n✅ All validations passed quality thresholds!")
        
        # Pause for user to review validation results (only in hardware mode)
        if not simulate:
            print("\n" + "="*60)
            print("VALIDATION COMPLETE - PLEASE REVIEW RESULTS")
            print("="*60)
            print("Check the validation plots and statistics above.")
            print("If validation looks good, press ENTER to continue with experiment.")
            print("If validation failed, press Ctrl+C to abort and recalibrate.")
            input("Press ENTER to continue or Ctrl+C to abort: ")
                       
        return validation_results
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        if not simulate:
            print("Please fix the validation issues before running the experiment.")
            input("Press ENTER to continue anyway (not recommended) or Ctrl+C to abort: ")
        return {'error': str(e)}

#Define your workflow! Make sure that it has parameters that can be changed!
def peroxide_workflow(reagent_incubation_time=20*60,sample_incubation_time=18*60,interval=5*60,replicates=3): #Reagent incubation time=20 mins; sample incubation time is 18 mins; sample platereading interval is 5 mins.
  
    # Initial State of your Vials, so the robot can know where to pipet
    INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/peroxide_assay_vial_status.csv"
    MEASUREMENT_PROTOCOL_FILE =r"C:\Protocols\SQ_Peroxide.prt"

    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    SIMULATE =  True #Set to True if you want to simulate the robot, False if you want to run it on the real robot
    lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=SIMULATE)

    # input("Only hit enter if the status of the vials (including open/close) is correct, otherwise hit ctrl-c")    
    lash_e.nr_robot.check_input_file()

    # RUN PIPETTING VALIDATION FOR ALL SOLUTIONS
    if not SIMULATE:
        validation_output_dir = output_dir / "pipetting_validation" if 'output_dir' in locals() else "output"
    else:
        validation_output_dir = "output"
        
    validate_peroxide_pipetting(lash_e=lash_e, output_dir=validation_output_dir, simulate=SIMULATE)

    print("\n" + "="*60)
    print("STARTING PEROXIDE ASSAY EXPERIMENT")
    print("="*60)

    sample_times = [0,5,10,15,20,25,45,60] #in minutes
    sample_indices = [f"{t}_min_Reaction" for t in sample_times]

    #create file name for the output data
    if not SIMULATE:
        exp_name = input("Experiment name: ")
        output_dir = Path(r'C:\Users\Imaging Controller\Desktop\SQ') / exp_name #appends exp_name to the output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        print("Output directory created at:", output_dir)
        slack_agent.send_slack_message("Peroxide workflow started!")
    else:
        output_dir = None

    #-> Start from here! 
    #Step 2.5: Add 950 µL water from water_index (vial_index 45) into vial_index 0-5.
    for i in sample_indices:
        lash_e.nr_robot.dispense_from_vial_into_vial('Water',i,volume=0.950, liquid='water')
    lash_e.nr_robot.remove_pipet()

    # # # #Step 3: Add 150 µL "Working Reagent(reagent A+B)" (vial_index 44) to 950 µL deionized water (vial_index 0-5) to dilute the Working Reagent.
    for i in sample_indices: 
        lash_e.nr_robot.dispense_from_vial_into_vial('Reagent_A+B',i,volume=0.150, liquid='water')
        mix_current_sample(lash_e,i,new_pipet=True,volume=0.8)
    
    #Step 4: Move the reaction mixture vial (vial_index 43) to the photoreactor to start the reaction.
    lash_e.nr_robot.move_vial_to_location('Rxn_Mixture', location="photoreactor_array", location_index=0)
    #Turn on photoreactor
    lash_e.photoreactor.turn_on_reactor_led(reactor_num=0,intensity=100)
    lash_e.photoreactor.turn_on_reactor_fan(reactor_num=0,rpm=600)

    #Step 5: Add 200 µL "reaction mixture" (vial in the photoreactor) to "Diluted Working Reagent" (vial_index 0-5). 
    # Six aliquots added at 0, 5, 10, 15, 20, 25 min time marks for incubation (incubation=18 min).
    
    SCHEDULE_FILE = r"c:\Users\owenm\OneDrive\Desktop\North Robotics\utoronto_demo\utoronto_demo\schedule.csv"  # Hardcoded path to schedule file
    schedule = pd.read_csv(SCHEDULE_FILE, sep=",") #Read the schedule file

    schedule = schedule.sort_values(by='start_time') #sort in ascending time order
    print("Schedule: ", schedule)

    start_time = current_time = get_time(SIMULATE)
    print("Starting timed portion at: ", start_time)
    #Let's complete the items one at a time
    items_completed = 0
    starting_well_index = 0
    time_increment = 60
    while items_completed < schedule.shape[0]: #While we still have items to complete in our schedule
        active_item = schedule.iloc[items_completed]
        time_required = active_item['start_time']
        action_required = active_item['action']
        sample_index = active_item['sample_index']
        current_time = get_time(SIMULATE,current_time)
        measured_items = 0

        #If we reach the triggered item's required time:
        if current_time - start_time > time_required:
            print("\nEvent triggered: " + action_required + f" from sample {sample_index}")
            print(f"Current Elapsed Time: {(current_time - start_time)/60} minutes")
            print(f"Intended Elapsed Time: {(time_required)/60} minutes")

            if action_required=="dispense_from_reactor":
                dispense_from_photoreactor_into_sample(lash_e,'Rxn_Mixture',sample_index,volume=0.2)
                items_completed+=1
            elif action_required=="measure_samples":
                transfer_samples_into_wellplate_and_characterize(lash_e,sample_index,starting_well_index,MEASUREMENT_PROTOCOL_FILE,replicates, output_dir,simulate=SIMULATE)
                starting_well_index += replicates
                items_completed+=1
                measured_items+=1
        elif current_time - start_time > time_increment:
            print(f"I'm Alive! Current Elapsed Time: {(current_time - start_time)/60} minutes")
            time_increment=time_increment+60
        
        if not SIMULATE:
            time.sleep(1)
    lash_e.nr_robot.move_home()   

    lash_e.nr_robot.return_vial_home('Rxn_Mixture')
    lash_e.photoreactor.turn_off_reactor_fan(reactor_num=0)
    lash_e.photoreactor.turn_off_reactor_led(reactor_num=0)
    lash_e.nr_robot.move_home()

    if not SIMULATE:   
        slack_agent.send_slack_message("Peroxide workflow completed!")

peroxide_workflow() #Run your workflow

