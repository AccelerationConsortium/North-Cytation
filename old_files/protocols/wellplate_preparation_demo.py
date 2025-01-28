import North_Safe
from Locator import *
import numpy as np
import pandas as pd
import math

#Input data
BLUE_DIMS = [20,77]
DEFAULT_DIMS = [25,85]
FILTER_DIMS = [24,98]


PUMP_SPEED = 15 #pipetting pump speed
REPLICATES = 5 #constant for all samples #need to update to read from vial df
MAX_SOLUTIONS = 1 #max number of solutions that are added into each well-- can maybe add function to detect "solution" from column names
pipet_count = 0

#needed files: 1. vial_status 2.wellplate_recipe
VIAL_FILE = "vial_status_wellplate - test.txt" #txt
RECIPE_FILE = "wellplate_recipe - demo.csv" #csv

#TODO: could implement translating from the recipe kind of file (with columns saying which solutions to add)

def generate_wellplate_placement(recipe_df, n = REPLICATES, starting_row = 0) -> list: #TODO: EDIT & FINISH!!!
    """
    Generates a list with the well-plate coordinates for each sample (to be pipetted into).
    Generates an empty list, if there are too many samples.

    Starting_row (int):

    Returns list with locations for each of the samples -- will append to dataframe afterwards.

    """
    k = len(recipe_df) #number of samples
    

    placement_list = []
    max_horizontal_groups = math.floor(12/n) #maximum number of replicate groups that will fit in each row 
    
    
    max_k = max_horizontal_groups*(8-starting_row) #maximum number of samples that can fit in wellplate



    if (k*n>96):
        return placement_list
    
    elif (k>max_k):
        return placement_list
    
    # else: #assuming all samples will fit in plate
    #     sample_placement = []
    
    return placement_list

def convert_wp_placement_to_num(coordinate: str) -> int:
    """
    Converts wellplate location (ex. A1, G4) to numerical coordinates 
    Returns numberical value from 1-96 (**())
    Ex. A1 -> 1, G12 -> 96
    """
    coordinate = coordinate.replace(" ", "")
    conversion = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}
    return (conversion[coordinate[0]]*12 + (int(coordinate[1:])-1)) #need 2 indices!

def get_wp_num_list(s) -> list:
    """
    Converts string of wellplate coordinates (ex. [["A1, A2, A3"],["B1, B2, B3"]]) to list of wellplate well number (ex. [0,1,2])
    """
    wp_num_list = []

    if s != "nan": 
        s = str(s)
        s_list = s.split(",")
        for l in s_list:
            if l != "nan":
                new = convert_wp_placement_to_num(l)
                wp_num_list.append(new)
        
    return wp_num_list

def get_amounts_list(s) -> list: #can merge with get_wp_num_list later on
    """
    Converts string of amounts to a list with amounts
    """
    amounts_list = []

    if s != "nan": 
        s = str(s)
        s_list = s.split(",")
        for l in s_list:
            amounts_list.append(float(l))
        
    return amounts_list


def get_non_empty_vial_num(name, vial_df) -> int:
    """
    Returns list of the indices in vial dataframe which have the same name as indicated.
    """
    print (vial_df[vial_df['vial name']==name])
    vial_indices = vial_df.index[vial_df['vial name']==name].tolist() #list of indices with same name
    print(vial_indices)
    i = 0
    while (vial_df["vial volume (mL)"][vial_indices[i]]==0 and i < len(vial_indices)): #looping through indices to find non-empty vial
        i = i+1
    return vial_indices[i]

def get_vial_indices(name, vial_df) -> list:
    return vial_df[vial_df['vial name']==name].index


def check_enough_volume(vial_df, recipe_df) -> list:
    """
    Checks if enough solution to prepare samples indicated
    Returns a list of sample names that there is not enough volume of (will have 1mL of buffer, as pipetting is not very accurate at low volumes)
    - Returns empty list if there is enough volume for all
    """

    insufficient_volume_list = []

    unique_vials = vial_df['vial name'].drop_duplicates()

    unique_vials_df = pd.DataFrame()
    unique_vials_df["Unique vial name"] = unique_vials

    volumes = []

    for name in unique_vials_df["Unique vial name"]: #for each solution
        rows_index = vial_df[vial_df['vial name']==name].index 

        total_volume = 0
        for i in rows_index: #for each vial with same name 
            total_volume += (vial_df["vial volume (mL)"][i]-1) #leaves 1mL buffer for each vial
        
        volumes.append(total_volume)
    
    unique_vials_df["Total Volume"] = volumes

    print(unique_vials_df)

    required_volumes = []

    for index, row in unique_vials_df.iterrows(): #for each solution
        curr_sol_name = row["Unique vial name"]
        
        required_vol = 0

        for j in range(MAX_SOLUTIONS):
            sol_column = "Solution " + str(j+1) #naming starts at 1 -- column name
            amount_column = "Amount " + str(j+1) + " (mL)" #column name for the amount column

            solutions_needed = recipe_df[recipe_df[sol_column]==curr_sol_name] #df with rows in which the solution needed is the same as the vial sol.

            for index2, row2 in solutions_needed.iterrows(): 
                curr_amount = float(row2[amount_column])
                required_vol += curr_amount*REPLICATES
        
        required_volumes.append(required_vol)
        available_volume = row["Total Volume"]
            
        if required_vol > available_volume: #insufficient volume, add vial name to list
            insufficient_volume_list.append(curr_sol_name)
            

    unique_vials_df["Required Volume"] = required_volumes
    print(unique_vials_df)
    
    return insufficient_volume_list

def check_next_vial(recipe_df, curr_column, curr_vial_name, curr_step) :
    """ Checks if next step in recipe is pipeted from the same vial. If so, no capping nor change of pipettes (returns true & i++).
    """
    if curr_step == len(recipe_df)-1: 
        return False, curr_step
    elif curr_vial_name == recipe_df[curr_column].loc[curr_step+1]: #TODO: check if enough volume (or if check_enough_volume already does)
        return True, curr_step+1
    else:
        return False, curr_step
    


#Loading data
vial_df = pd.read_csv(VIAL_FILE, delimiter='\t', index_col='vial index') #Edit this
vial_df.astype({'vial volume (mL)': 'float'})
samples_df = pd.read_csv(RECIPE_FILE, delimiter=',') #assumes all values are valid 
samples_df["Wellplate Index"] = samples_df["Location"].apply(get_wp_num_list)


print("vial_df: \n", vial_df)
print("Samples_df: \n", samples_df)



#Initializing Robot
nr = North_Safe.North_Robot(vial_df)

nr.c9.open_clamp()
nr.reset_after_initialization()


nr.set_pipet_tip_type(BLUE_DIMS, 0) #SET!!
nr.c9.set_pump_speed(0,PUMP_SPEED)

i = 0
while i < len(samples_df): #for each sample in samples_df
    print("**--------------------------------------------------**")
    print("Preparing Sample", i, ":", samples_df['Solution Name'][i])
    


    for j in range(MAX_SOLUTIONS): #for each solution (ex. Solution 1, Solution 2) to be added to well plate 
        sol_column = "Solution " + str(j+1) #naming starts at 1
        curr_vial_name = str(samples_df[sol_column][i]) #name of solution (vial) to be added -- still in sample i (but j changes the column to access)
        print(curr_vial_name)

        amount_column = "Amount " + str(j+1) + " (mL)"
        #curr_amount = float(samples_df[amount_column][i]) #amount (PER WELL) to be added
        curr_amounts = get_amounts_list(samples_df[amount_column][i])

        curr_dispense_type = "None"
        curr_aspirate_extra = False

        curr_replicates = len(samples_df["Wellplate Index"][i])

        if ("nan" not in str(samples_df["Type"][i]).lower()):
            curr_dispense_type = str(samples_df["Type"][i])
            print("Getting dispense type -- ", curr_dispense_type)
            if curr_dispense_type.lower() == "drop" or curr_dispense_type.lower() == "drop-touch":
                nr.c9.set_pump_speed(0, 20)
        
            elif curr_dispense_type.lower() == "slow":
                nr.c9.set_pump_speed(0,15)
            


        if "nan" in curr_vial_name.lower():
            print('break!') #TODO: Remove later
            break
        
        curr_vial_num = get_non_empty_vial_num(curr_vial_name, vial_df) #see how to fix up for multiple vials...
        print("Vial num", curr_vial_num)


        nr.move_vial_to_clamp(curr_vial_num) #open clamp at the end
        nr.uncap_clamp_vial() #opens clamp at the end 
        nr.c9.close_clamp()
        
        check_next_vial_bool = True #default, so it runs the first time, but doesn't change i
        
        while check_next_vial_bool: #keeps pipetting when next step transfers from same vial
            nr.set_robot_speed(10)
            num_replicates = len(samples_df["Wellplate Index"][i])
            nr.aspirate_from_vial(curr_vial_num, 0.375)
            nr.dispense_into_wellplate(samples_df["Wellplate Index"][i], curr_amounts,num_replicates, dispense_type = curr_dispense_type)
            
            check_next_vial_bool, i = check_next_vial(samples_df, sol_column, curr_vial_name, curr_step=i) #returns next i value
            
            #update values for next run with new i
            if check_next_vial_bool: #not changing pipette tips for next step in protocol, update num_duplicates, amount & dispense_type
                curr_amounts = get_amounts_list(samples_df[amount_column][i])
                curr_replicates = len(samples_df["Wellplate Index"][i])
                curr_dispense_type = str(samples_df["Type"][i])
                #print(samples_df["Type"][i])
                
                if curr_dispense_type.lower() == "drop" or curr_dispense_type.lower() == "drop-touch":
                    nr.c9.set_pump_speed(0, 30) #TODO: change back to 20
            
                elif curr_dispense_type.lower() == "slow":
                    nr.c9.set_pump_speed(0,15)

            print("Check next:", check_next_vial_bool)
            print("updated i:", i)
            

        nr.c9.default_vel=40
        nr.remove_pipet()
        nr.recap_clamp_vial()
        nr.return_vial_from_clamp(curr_vial_num)
        pipet_count += 1
    i = i+1 #made a while loop, so can update i to skip some iterations

nr.c9.move_z(292)