import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import pandas as pd

INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/ilya_input_vials.txt"
MEASUREMENT_PROTOCOL_FILE = r"C:\Protocols\Ilya_Measurement.prt"
INSTRUCTIONS_FILE = "../utoronto_demo/status/ilya_input.csv"

#Define your workflow! 
def sample_workflow():
  
    # Initial State of your Vials, so the robot can know where to pipet
    input_data = pd.read_csv(INSTRUCTIONS_FILE, sep=',',index_col="Well")/1000
    vial_status = pd.read_csv(INPUT_VIAL_STATUS_FILE, sep=",")
    print(vial_status)

    print(input_data)

    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(INPUT_VIAL_STATUS_FILE)

    input("Only hit enter if the status of the vials (including open/close) is correct, otherwise hit ctrl-c")

    #The vial indices are numbers that are used to track the vials. For the sake of clarity, these are stored in the input vial file but accessed here
    ethanol_index = lash_e.nr_robot.get_vial_index_from_name('ethanol') 
    ethanol_dye_index = lash_e.nr_robot.get_vial_index_from_name('ethanol_dye')
    water_index = lash_e.nr_robot.get_vial_index_from_name('water')
    water_dye_index = lash_e.nr_robot.get_vial_index_from_name('water_dye') 
    glycerol_index = lash_e.nr_robot.get_vial_index_from_name('glycerol')
    glycerol_dye_index = lash_e.nr_robot.get_vial_index_from_name('glycerol_dye')
    
    input_indices = [water_dye_index,water_index,glycerol_dye_index,glycerol_index,ethanol_dye_index,ethanol_index]

    for i in range (0, 3):
        

        if i==1:
            wells = range(48,96)
        else:
            wells = range(0,48)

        input_data.index = wells

        water_df = input_data[(input_data['water'] > 0) | (input_data['water_dye'] > 0)]
        glycerol_df = input_data[(input_data['glycerol'] > 0) | (input_data['glycerol_dye'] > 0)]
        ethanol_df = input_data[(input_data['ethanol'] > 0) | (input_data['ethanol_dye'] > 0)]

        # Display the resulting dataframes
        print("Water DataFrame:")
        print(water_df)
        print("\nGlycerol DataFrame:")
        print(glycerol_df)
        print("\nEthanol DataFrame:")
        print(ethanol_df)

        lash_e.nr_robot.dispense_from_vials_into_wellplate(water_df,input_indices)
        lash_e.nr_robot.dispense_from_vials_into_wellplate(glycerol_df,input_indices,dispense_speed=30,wait_time=5)
        lash_e.nr_robot.dispense_from_vials_into_wellplate(ethanol_df,input_indices,asp_cycles=2)


        #Transfer the well plate to the cytation and measure
        data_output = lash_e.measure_wellplate(MEASUREMENT_PROTOCOL_FILE,wells_to_measure=wells,meas_type="read")

        save_file = 'output_data_'+str(i)+'.txt'
        data_output.to_csv(save_file, sep=',')

        input("Pause waiting for next round... If second round finished please add new well-plate")
    
#Execute the sample workflow.
sample_workflow()
