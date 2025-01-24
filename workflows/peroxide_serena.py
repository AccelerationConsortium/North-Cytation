import sys
import time
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E 
import pandas as pd
import numpy as np


#Define your workflow! Make sure that it has parameters that can be changed!
def sample_workflow(input_vial_status_file,cytation_protocol_file_path):
  
    # Initial State of your Vials, so the robot can know where to pipet
    vial_status = pd.read_csv(input_vial_status_file, sep=r"\t", engine="python")
    print(vial_status)
    input("Only hit enter if the status of the vials (including open/close) is correct, otherwise hit ctrl-c")

    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(input_vial_status_file)

    #lash_e.grab_new_wellplate()
    
    #Step 1: Add 20 µL "reagent A" (vial 0) to "reagent B" (vial 1).
    lash_e.nr_robot.aspirate_from_vial(0, 0.02)
    lash_e.nr_robot.dispense_into_vial(1, 0.02)
    lash_e.remove_pipet()
    #OAM: Better to do 20 uL with the small tip, then remove it
    #The small tip can't do the 500 uL in the next steps, so it has to be 200 uL or switch tips
    
    for _ in range (3):
        lash_e.nr_robot.aspirate_from_vial(1, 0.25)
        lash_e.nr_robot.dispense_into_vial(1, 0.25)
    lash_e.nr_robot.remove_pipet() # This step is for pipetting up and down *3 to simulate mixing.
    
    #Step 2: incubate reagent A + B = "Working Reagent" (Vial 1) for 20 min
    time.sleep(1200)

    #Step 3: Add 150 µL "Woring Reagent" (vial 1) to 950 µL deionized water (Vials 6-11) to dilute the Working Reagent.
    for i in range (6, 7): 
        lash_e.nr_robot.aspirate_from_vial(1, 0.150)
        lash_e.nr_robot.dispense_into_vial(i, 0.150)
        for _ in range (3):
            lash_e.nr_robot.aspirate_from_vial(i, 0.25)
            lash_e.nr_robot.dispense_into_vial(i, 0.25)
    
    #Step 4: Move the reaction mixture vial (vial 2) to the photoreactor to start the reaction.
    lash_e.run_photoreactor(2,target_rpm=600,intensity=100,duration=60,reactor_num=1)

    #Step 5: Add 200 µL "reaction mixture" (vial in the photoreactor) to "Diluted Working Reagent" (Vials 6-11). 
            # Six aliquots need to be taken from the "reaction mixture" and added to the "diluted working reagent" at 0, 5, 10, 15, 20, 25 time marks for incubation (18 min).
    #interval = 5 * 60
    #incubation_time = 18 * 60
    #incubation_tracker = []
    #start_time = time.time()
    #next_plate_reading = incubation_tracker["start_time"] + incubation_time
 
    for i in range (6, 7):
        #next_addition_time = start_time + (i-6) * interval
        if True: #time.time() == next_addition_time:
            lash_e.nr_robot.aspirate_from_vial(1, 0.02)
            lash_e.nr_robot.dispense_into_vial(i, 0.02)
            for _ in range (3):
                lash_e.nr_robot.aspirate_from_vial(i, 0.25)
                lash_e.nr_robot.dispense_into_vial(i, 0.25)
            lash_e.nr_robot.remove_pipet()

          #  incubation_tracker.append({
          #       "start_time": time.time(),
          #       "vial": i
          #      })  #Here I'm trying to let the system record the start time of the incubation (the time when the "Reaction mixture" is added to the "diluted working reagent")
                    #It is for calculating when to add the incubated mixture to the wellplate based on the satrt time and the set incubation time in Step 6.
    
    #Step 6: When the incubation is complete (18 min after each addition in step 5), add 200 µL "reaction mixture + diluted working reagent" (Vials 6-11) to the well plate & take absbance measurements.
            current_time = time.time()
            wells = [(i-6)*3,(i-6)*3+1,(i-6)*3+2]  # Defining the wells for each reaction mixture vial (e.g. vial 6 is added to wells 0,1,2)
            if True: #current_time - incubation_tracker["start_time"] == next_plate_reading:
                lash_e.nr_robot.aspirate_from_vial(i, 0.6)
                lash_e.nr_robot.dispense_into_wellplate(wells, [0.2,0.2,0.2])
                lash_e.nr_robot.finish_pipetting()
                lash_e.measure_wellplate(cytation_protocol_file_path) #Step 7: Transfer the well plate to the cytation and measure
                
                #incubation_tracker.remove(i)
        
#Note I will have a conversion of "A1" to 0 and "A2" to 1 for the future, so you could do ["A1", "A2", "A3"] if you prefer that over 0,1,2
#Your protocol needs to be made inside the gen5 software, including the automated export
sample_workflow(".txt", r"C:\Protocols\Spectral_Automation.prt")

