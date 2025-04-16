import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import pandas as pd

INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/CMC_workflow_input.csv"
MEASUREMENT_PROTOCOL_FILE = r"C:\Protocols\Quick_Measurement.prt" #Will need to create a measurement protocol
LAST_WP_INDEX = 0

#Define your workflow! 
#In this case we have two parameters: 
def check_input_file(input_file):  
    # Initial State of your Vials, so the robot can know where to pipet
    vial_status = pd.read_csv(input_file, sep=",")
    print(vial_status)
    input("Only hit enter if the status of the vials (including open/close) is correct, otherwise hit ctrl-c")

def mix_surfactants(surfactant_index_list, surfactant_volumes, target_vial_index, mix_ratio=0.75): #Mix the different surfactants + water into a new vial
    for i in range (0, len(surfactant_index_list)):
        surfactant_index = surfactant_index_list[i]
        surfactant_volume = surfactant_volumes[i]
        lash_e.nr_robot.dispense_from_vial_into_vial(surfactant_index,target_vial_index,surfactant_volume)
        lash_e.nr_robot.mix_vial(target_vial_index,surfactant_volume*mix_ratio)

def create_wellplate_samples(surfactant_mixture_index, surfactant_mixture_volume, DMSO_pyrene_index,DMSO_pyrene_volume,replicates,last_wp_index): #Add the DMSO_pyrene and surfactant mixture to well plates
    well_indices = range (last_wp_index,last_wp_index+replicates)
    dispense_indices = [surfactant_mixture_index,DMSO_pyrene_index]
    data = {'Volume A': [surfactant_mixture_volume]*replicates, 'Volume B': [DMSO_pyrene_volume]*replicates}
    # Convert to DataFrame
    df = pd.DataFrame(data)
    df.index = well_indices
    lash_e.nr_robot.dispense_from_vials_into_wellplate(df,dispense_indices)



def sample_workflow():
    #Check the input to confirm that it's OK!
    check_input_file(INPUT_VIAL_STATUS_FILE)

    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(INPUT_VIAL_STATUS_FILE)

    #The vial indices are numbers that are used to track the vials. I will be implementing a dictionary system so this won't be needed
    surfactant_1_index = lash_e.nr_robot.get_vial_index_from_name('surfactant_1_stock') #Get the ID of our target reactor
    surfactant_2_index = lash_e.nr_robot.get_vial_index_from_name('surfactant_2_stock')
    surfactant_mixture_index = lash_e.nr_robot.get_vial_index_from_name('surfactant_mixture_1')
    pyrene_DMSO_index = lash_e.nr_robot.get_vial_index_from_name('pyrene_DMSO')
    water_index = lash_e.nr_robot.get_vial_index_from_name('water')

    surfactant_index_list = [surfactant_1_index, surfactant_2_index, water_index]
    surfactant_volumes = [0.2,0.3,0.5]
    DMSO_VOLUME = 0.10
    replicates=3

    mix_surfactants(surfactant_index_list,surfactant_volumes,surfactant_mixture_index)
    create_wellplate_samples(surfactant_mixture_index,DMSO_VOLUME,pyrene_DMSO_index,replicates=replicates,last_wp_index=LAST_WP_INDEX)

    #Transfer the well plate to the cytation and measure
    lash_e.measure_wellplate(MEASUREMENT_PROTOCOL_FILE,wells_to_measure=range(LAST_WP_INDEX,LAST_WP_INDEX+replicates))
    
#Execute the sample workflow.
sample_workflow()
