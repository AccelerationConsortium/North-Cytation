import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import numpy as np
import time

INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/calibration_vials.csv"
MEASUREMENT_PROTOCOL_FILE = r"C:\Protocols\Ilya_Measurement.prt"
SIMULATE = True
REPLICATES = 3
VOLUMES = [0.02,0.05,0.10,0.15,0.20] #For example
DENSITY_LIQUID = 0.997
EXPECTED_MASSES = VOLUMES*DENSITY_LIQUID*1000
EXPECTED_ABS = None #Need help from Ilya, is there a formula?
EXPECTED_TIME = None #How long does it take to aspirate and dispense 1 mL? How much is the move_to, the move_back, the aspirate and the dispense. 
method = "mass" #Could also be "absorbance"
well_count = 0

#Define your workflow! 
def sample_workflow():
  
    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(INPUT_VIAL_STATUS_FILE,simulate=SIMULATE)

    #Check input status is correct
    lash_e.nr_robot.check_input_file()

    #Iterate through the volumes
    for i in range (0, len(VOLUMES)):
        volume = VOLUMES[i]
        measurement_results = [] #Track the outputs for each volume

        start_time = time.time() #start ticks

        if method == "mass":
            lash_e.nr_robot.move_vial_to_location('measurement_vial', 'clamp', 0) #Move a vial to the clamp for measurement
            for j in range (0, REPLICATES):
                lash_e.nr_robot.aspirate_from_vial('liquid_source', volume)
                mass_measured = lash_e.nr_robot.dispense_into_vial('measurement_vial', volume, measure_weight=True)
                measurement_results.append(mass_measured)

        if method == 'absorbance':
            wells = range(well_count, well_count+REPLICATES)
            for well in wells:
                lash_e.nr_robot.aspirate_from_vial('liquid_source', volume)
                lash_e.nr_robot.dispense_into_wellplate([well],[volume])
            measurement_results = lash_e.measure_wellplate(MEASUREMENT_PROTOCOL_FILE,wells)
            well_count += REPLICATES

        end_time = time.time() #end ticks

        variation = np.stdev(measurement_results) / np.average(measurement_results) * 100

        if method == "mass":
            deviation = ( np.average(measurement_results) - EXPECTED_MASSES[i] ) / EXPECTED_MASSES[i] * 100
        if method == "absorbance":
            deviation = ( np.average(measurement_results) - EXPECTED_ABS[i] ) / EXPECTED_ABS[i] * 100

        time_elapsed = (end_time - start_time) / REPLICATES

        time_score = 1 / (1 + np.exp((time_elapsed - EXPECTED_TIME[i]))) #Varies from 1 to 0

        print(f"Volume deposited: {volume} mL with {REPLICATES} replicates")
        print("Measurement type: " + method)
        print(f"Variation: {variation} % [0 is optimal]")
        print(f"Deviation: {deviation} % [0 is optimal]")
        print(f"Elapsed Time: {time_elapsed} seconds per replicate")
        print(f"Time score: {time_score} units [1 is ideal]")
        
#Execute the sample workflow.
sample_workflow()
