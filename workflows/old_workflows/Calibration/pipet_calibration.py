import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import pandas as pd
import numpy as np

#from pathlib import Path



#export_directory = Path("C:\Users\Imaging Controller\Desktop\utoronto_demo\Calibration Data\Calibration Jan 2025") #edit


EXPORT_FILEPATH_1 = r"C:\Users\Imaging Controller\Desktop\utoronto_demo\Calibration Data\Calibration Jan 2025\calibration_water_250_multi_constant_extra.txt" #put solvent, dispense_method, into notes
#ASPIRATE_VOLUMES = [0.3,0.25,0.2,0.15,0.1,0.05,0.025]
#ASPIRATE_VOLUMES = [0.04, 0.03, 0.02, 0.01]
ASPIRATE_VOLUMES = [0.075, 0.05, 0.025, 0.01, 0.005]
REPLICATES = 3


def consistent_v_multi(volumes, replicates, first_buffer=0):
    multi_volumes=[]
    for v in volumes: 
        if first_buffer >0:
            temp_volumes = [first_buffer]
        else:
            temp_volumes = []
        for i in range(replicates):
            temp_volumes.append(v)
            
        multi_volumes.append(temp_volumes)
    return multi_volumes

ASPIRATE_VOLUMES_MULTI = consistent_v_multi([0.075, 0.05, 0.025, 0.01, 0.005], 3, first_buffer=0.01) #for constant V

#ASPIRATE_VOLUMES_MULTI = [[0.075, 0.05, 0.025, 0.01, 0.005], [0.075, 0.05, 0.025, 0.01, 0.005], [0.075, 0.05, 0.025, 0.01, 0.005]]


def dispense_all_calibration(input_vial_status_file, source_vial, aspirate_volumes, replicates, dest_vial_position, export_path, buffer = 0):
  
    # Initial State of your Vials, so the robot can know where to pipet
    vial_status = pd.read_csv(input_vial_status_file, sep=r"\t", engine="python")
    print(vial_status)

    input("Only hit enter if the status of the vials (including open/close) is correct, otherwise hit ctrl-c")

    lash_e = Lash_E(input_vial_status_file)

    calibration_data = [['Expected Volume (mL)', 'Actual mass (g)']]

    try:
        #This uncaps our target vessel to receive liquids in the clamp
        lash_e.nr_robot.c9.move_z(292)

        lash_e.nr_robot.c9.DEFAULT_VEL = 20
        lash_e.nr_robot.move_vial_to_clamp(dest_vial_position)
        lash_e.nr_robot.uncap_clamp_vial()

        #Aspirate from vial and dispense into clamped vial
        for volume in aspirate_volumes:
            print(f"------ \nVolume calibrating: {volume} mL\n")

            for i in range(0, replicates):

                lash_e.nr_robot.aspirate_from_vial(source_vial, volume+buffer) #aspirate already gets pipet
                mass = lash_e.nr_robot.dispense_into_vial(dest_vial_position, volume, measure_weight = True)

                mass_str = str(mass)[0:5] #round to 5 digits

                if buffer > 0:
                    lash_e.nr_robot.dispense_into_vial(source_vial, amount_mL=buffer) #dispense extra buffer back into source

                print(f"Replicate #{i}: mass = {mass_str}")
                calibration_data.append([volume, mass_str]) 
        
        lash_e.nr_robot.remove_pipet()

        #Remove the pipet and return the vial if needed
        lash_e.nr_robot.finish_pipetting()

        lash_e.nr_robot.c9.move_z(292)

        np.savetxt(export_path, calibration_data,delimiter='\t', fmt ='% s') #export file

    
    except KeyboardInterrupt:
        lash_e.nr_robot.c9 = None

def multi_dispense_calibration(input_vial_status_file, source_vial, aspirate_volume_list, dest_vial_position, export_path, buffer = 0):
  
    vial_status = pd.read_csv(input_vial_status_file, sep=r"\t", engine="python")
    print(vial_status)
    print(aspirate_volume_list)

    input("Only hit enter if the status of the vials (including open/close) is correct, otherwise hit ctrl-c")

    lash_e = Lash_E(input_vial_status_file)

    calibration_data = [['Expected Volume (mL)', 'Actual mass (g)']]

    try:
        #This uncaps our target vessel to receive liquids in the clamp
        lash_e.nr_robot.c9.DEFAULT_VEL= 20
        lash_e.nr_robot.move_vial_to_clamp(dest_vial_position)
        lash_e.nr_robot.uncap_clamp_vial()

        #Aspirate from vial and dispense into clamped vial
        for volume_l in aspirate_volume_list:
            print(f"------ \n Volume calibrating: {volume_l} mL + buffer:{buffer} \n")

            lash_e.nr_robot.aspirate_from_vial(source_vial, (sum(volume_l)+buffer)) #gets pipet & aspirates sum of amounts in list

            for i in range(0, len(volume_l)): #dispense each in the volume list (for that aspirate/dispense)
                mass = lash_e.nr_robot.dispense_into_vial(dest_vial_position, volume_l[i], measure_weight = True)

                mass_str = str(mass)[0:5] #round to 5 digits

                print(f"Dispensing {volume_l[i]}: mass = {mass_str}")
                calibration_data.append([volume_l[i], mass_str]) 

            if buffer > 0:
                lash_e.nr_robot.dispense_into_vial(source_vial, amount_mL=buffer) #dispense extra buffer back into source

        
        lash_e.nr_robot.remove_pipet()

        #Remove the pipet and return the vial if needed
        lash_e.nr_robot.finish_pipetting()

        lash_e.nr_robot.c9.move_z(292)
        np.savetxt(export_path, calibration_data,delimiter='\t', fmt ='% s') #export file

    
    except KeyboardInterrupt:
        lash_e = None
    

multi_dispense_calibration(input_vial_status_file="../utoronto_demo/status/vial_status_wellplate - test.txt", source_vial = 0, aspirate_volume_list=ASPIRATE_VOLUMES_MULTI, dest_vial_position=1, export_path = EXPORT_FILEPATH_1, buffer=0)