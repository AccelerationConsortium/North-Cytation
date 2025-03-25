import sys
sys.path.append("../utoronto_demo")
#from master_usdl_coordinator import Lash_E
import pandas as pd

INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/ilya_input_vials.txt"
MEASUREMENT_PROTOCOL_FILE = r"C:\Protocols\Ilya_Measurement.prt"
INSTRUCTIONS_FILE = "../utoronto_demo/status/ilya_input.csv"

#Define your workflow! 
def sample_workflow():
  
    # Initial State of your Vials, so the robot can know where to pipet
    input_data = pd.read_csv(INSTRUCTIONS_FILE, sep=',')
    vial_status = pd.read_csv(INPUT_VIAL_STATUS_FILE, sep=",")
    print(vial_status)

    input("Only hit enter if the status of the vials (including open/close) is correct, otherwise hit ctrl-c")

    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(INPUT_VIAL_STATUS_FILE)

    #The vial indices are numbers that are used to track the vials. For the sake of clarity, these are stored in the input vial file but accessed here
    ethanol_index = lash_e.nr_robot.get_vial_index_from_name('ethanol') 
    ethanol_dye_index = lash_e.nr_robot.get_vial_index_from_name('ethanol_dye')
    water_index = lash_e.nr_robot.get_vial_index_from_name('water')
    water_dye_index = lash_e.nr_robot.get_vial_index_from_name('water_dye') 
    glycerol_index = lash_e.nr_robot.get_vial_index_from_name('glycerol')
    glycerol_dye_index = lash_e.nr_robot.get_vial_index_from_name('glycerol_dye')
    
    input_indices = [water_dye_index,water_index,glycerol_dye_index,glycerol_index,ethanol_dye_index,ethanol_index]

    for i in range (0, 3):
        lash_e.nr_robot.dispense_from_vials_into_wellplate(input_data,input_indices)

        if i==1:
            wells = range(48,96) 
        else:
            wells = range(0,48)

        input_data['Well'] = wells
        #Transfer the well plate to the cytation and measure
        data_output = lash_e.measure_wellplate(MEASUREMENT_PROTOCOL_FILE,wells_to_measure=wells)

        save_file = 'output_data_'+str(i)+'.txt'
        data_output.to_csv(save_file, sep=',')

        input("Pause waiting for next round... If second round finished please add new well-plate")
    
#Execute the sample workflow.
sample_workflow()
