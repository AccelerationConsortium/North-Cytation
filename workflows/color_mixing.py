import sys

from matplotlib.pylab import False_
sys.path.append("../utoronto_demo")
from backups.surfactant_grid_adaptive_concentrations_backup_consolidated_data_organization_20260203_172735 import VALIDATE_LIQUIDS
from master_usdl_coordinator import Lash_E
import pandas as pd
import numpy as np
import time
from datetime import datetime

INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/color_mixing_vials.csv"

def condition_tip(lash_e, vial_name, conditioning_volume_ul=200, liquid_type='water'):
    """Condition a pipette tip by aspirating and dispensing into source vial multiple times
    
    Args:
        lash_e: Lash_E robot controller
        vial_name: Name of vial to condition tip with
        conditioning_volume_ul: Volume for conditioning (default 200 uL)
        liquid_type: Type of liquid for pipetting parameters ('water', 'DMSO', etc.)
    """
    try:
        # Calculate volume per conditioning cycle
        cycles = 3
        volume_per_cycle_ul = conditioning_volume_ul
        volume_per_cycle_ml = volume_per_cycle_ul / 1000

        lash_e.logger.info(f"Conditioning tip with {vial_name} ({conditioning_volume_ul} uL, {cycles} cycles)")
        
        for cycle in range(cycles):
            # Aspirate from vial
            lash_e.nr_robot.aspirate_from_vial(
                source_vial_name=vial_name,
                amount_mL=volume_per_cycle_ml,
                liquid=liquid_type,
            )
            
            # Dispense back to same vial
            lash_e.nr_robot.dispense_into_vial(
                dest_vial_name=vial_name,
                amount_mL=volume_per_cycle_ml,
                liquid=liquid_type
            )
        
        lash_e.logger.info(f"Tip conditioning complete for {vial_name}")
        
    except Exception as e:
        lash_e.logger.warning(f"Could not condition tip with {vial_name}: {e}")

def generate_random_matrix(rows, cols, row_sum, divisible_by):
    if row_sum % divisible_by != 0:
        raise ValueError("Row sum must be a multiple of divisible_by.")
    
    if row_sum < (cols - 1) * divisible_by:
        raise ValueError("Row sum is too small to distribute among columns.")
    
    matrix = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        remaining_sum = row_sum
        values = []

        for j in range(cols - 1):
            max_val = remaining_sum - (cols - j - 1) * divisible_by
            max_val = max(max_val, 0)
            
            choices = np.arange(0, max_val + divisible_by, divisible_by)
            value = np.random.choice(choices)
            values.append(value)
            remaining_sum -= value

        # Last column takes the remaining sum (guaranteed to be a multiple of divisible_by)
        values.append(remaining_sum)
        
        # Shuffle to randomize placement of zeros and values
        np.random.shuffle(values)
        matrix[i, :] = values
    
    return matrix

def mix_wells(lash_e, wells, wash_vial='wash', wash_volume=0.150, repeats=1,replicates=6):
    for well in wells:
        move_to_wellplate = False  # Default: don't move (stay at current position)
        if well%replicates==0:
            lash_e.nr_robot.aspirate_from_vial(wash_vial,wash_volume)
            lash_e.nr_robot.dispense_into_vial(wash_vial,wash_volume,initial_move=False)
            move_to_wellplate = True  # Now DO move for this first well in group
        #for i in range (0,repeats):
        #    lash_e.nr_robot.dispense_from_vial_into_vial(wash_vial,wash_vial,wash_volume,move_to_aspirate=False,move_to_dispense=False,buffer_vol=0)

        lash_e.nr_robot.pipet_from_wellplate(well,wash_volume,move_to_aspirate=move_to_wellplate)
        lash_e.nr_robot.pipet_from_wellplate(well,wash_volume,aspirate=False,move_to_aspirate=False)
        for i in range (0, repeats):
            lash_e.nr_robot.pipet_from_wellplate(well,wash_volume,move_to_aspirate=False)
            lash_e.nr_robot.pipet_from_wellplate(well, wash_volume,aspirate=False,move_to_aspirate=False)

def sample_workflow(number_samples=6,replicates=6,colors=4,resolution_vol=10,well_volume=240):
  

    SIMULATE = True
    VALIDATE_LIQUIDS = False

    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(INPUT_VIAL_STATUS_FILE,simulate=SIMULATE)

    #data_colors_uL = generate_random_matrix(number_samples, colors, well_volume, resolution_vol)/1000
    data_colors_uL = pd.read_csv("C:\\Users\\Imaging Controller\\Desktop\\ECON_MIXING\\color_mixing_composition_orig.txt", sep=',',index_col=0)/1000

    print(data_colors_uL)

    #data_pd_save = data_colors_uL*1000
    #data_pd_save = pd.DataFrame(data=data_pd_save,columns=['water','red','blue','yellow'])
    #data_pd_save.to_csv("../utoronto_demo/output/color_mixing_composition.csv",sep=',')

    print("Row sums:", np.sum(data_colors_uL * 1000, axis=1))  # Should all equal 250

    data_colors_uL = np.repeat(data_colors_uL, replicates, axis=0)
    # Convert back to DataFrame to allow column selection by name
    data_colors_uL = pd.DataFrame(data_colors_uL, columns=['water','red','blue','yellow'])

    sum_colors = np.sum(data_colors_uL,0)
    print("Total volume per vial:", sum_colors) #how much of each volume is used

    num_wells = data_colors_uL.shape[0]
    print("Number samples: ", num_wells)
    wells = range(0, num_wells)  

    input("Waiting...")

    if VALIDATE_LIQUIDS:
        # Validate vials with 200 uL tip conditioning
        from pipetting_data.embedded_calibration_validation import validate_pipetting_accuracy
        validation_folder = f"output/pipetting_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        vials = ['water', 'yellow', 'red', 'blue']
        for vial_name in vials:
            try:
                print(f"Validating {vial_name} with 200 uL tip conditioning...")
                results = validate_pipetting_accuracy(
                    lash_e=lash_e,
                    source_vial=vial_name,
                    destination_vial=vial_name,
                    liquid_type="water",
                    volumes_ml=[0.15, 0.1, 0.05, 0.02, 0.01],  # Convert 10 µL to 0.01 mL
                    replicates=5,
                    output_folder=validation_folder,
                    plot_title=f"Pipetting Validation - {vial_name}",
                    switch_pipet=False,
                    adaptive_correction=True,
                    condition_tip_enabled=True,  # Enable tip conditioning
                    conditioning_volume_ul=170   # Use 200 uL conditioning 
                )
                lash_e.logger.info(f"{vial_name} validation: R²={results['r_squared']:.4f}, "
                            f"Accuracy={results['mean_accuracy_pct']:.2f}%")
            except Exception as e:
                lash_e.logger.warning(f"Could not validate {vial_name}: {e}")


    start_time = time.perf_counter()

    for color in ['water','red','blue','yellow']:
        color_volumes = data_colors_uL[color].values  # Get array of volumes for this color
        print(f"Dispensing {color}: volumes = {color_volumes}")

        # Move vial to clamp position
        lash_e.nr_robot.move_vial_to_location(color, 'clamp', 0)
        lash_e.nr_robot.c9.close_clamp()  # Ensure clamp is closed to hold vial in place

        
        # Get tip and condition it with 200 uL for better accuracy
        condition_tip(lash_e, color, conditioning_volume_ul=170, liquid_type='water')
        
        # Manual loop through wells with this color
        for well_index, volume in enumerate(color_volumes):
            if volume > 0.0:  # Only dispense if volume > 0
                # Aspirate from vial
                lash_e.nr_robot.aspirate_from_vial(color, volume, liquid='water')
                # Dispense into wellplate
                lash_e.nr_robot.dispense_into_wellplate([well_index], [volume], liquid='water')
        
        # Remove tip and return vial when done with this color
        lash_e.nr_robot.remove_pipet()
        lash_e.nr_robot.c9.open_clamp()
        lash_e.nr_robot.return_vial_home(color)

    mix_wells(lash_e, wells,replicates=replicates)
    
    # Note: Tips are now removed per color, so no global remove_pipet needed

    end_time = time.perf_counter()

    print("Time to complete: ", end_time - start_time)
    
    #Transfer the well plate to the cytation and measure
    results = lash_e.measure_wellplate(r"C:\\Protocols\\Econ_Mixing.prt",wells_to_measure=range(0,96))

    print(results)

    if results is not None:
        # Create timestamped filename to avoid overwrites
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_filename = f"../utoronto_demo/output/color_mixing_results_{timestamp}.csv"
        
        try:
            # Handle MultiIndex columns by flattening them
            if isinstance(results.columns, pd.MultiIndex):
                # Flatten MultiIndex columns to single level
                results.columns = ['_'.join(map(str, col)).strip() for col in results.columns]
            
            results.to_csv(results_filename, sep=',')
            print(f"Results saved to: {results_filename}")
            
        except Exception as e:
            # Fallback: save as pickle if CSV fails
            pickle_filename = f"../utoronto_demo/output/color_mixing_results_{timestamp}.pkl"
            results.to_pickle(pickle_filename)
            print(f"CSV save failed ({e}), results saved as pickle: {pickle_filename}")
    else:
        print("No results to save (results was None)")
    
#Execute the sample workflow.
sample_workflow()
