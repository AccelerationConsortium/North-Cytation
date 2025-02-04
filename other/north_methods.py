from north import NorthC9
from Locator import *
import numpy as np

#Global variables, must be referenced with 
HAS_PIPET = False
GRIPPER_STATUS = "Open"
PIPETS_USED = 0
CLAMPED_VIAL = "None"
OPEN_VIALS = [1,2,3] #List of vials that do not have caps. These cannot be moved


#Initialize function
c9 = NorthC9('A', network_serial='AU06CNCF')
c9.default_vel = 30

def initialize():
    #c9 = NorthC9('A', network_serial='AU06CNCF')
    print("Initializing")
    c9.home_pump(0)
    c9.open_gripper()  #just in case
    remove_pipet() #just in case

def remove_pipet(): #Remove the pipet tip
    global HAS_PIPET
    c9.goto_safe(p_remove_approach)
    c9.goto(p_remove_cap, vel=5)
    c9.move_z(292, vel=20)
    HAS_PIPET = False

def get_pipet(pipet_num): #Add a pipet tip
    c9.goto_safe(p_capture_grid[pipet_num])
    global HAS_PIPET, PIPETS_USED
    HAS_PIPET = True
    PIPETS_USED += 1
    
#We may need to do a check to see if vials are uncapped or not
def pipet_from_vial_into_vial(source_vial_num, dest_vial_num, amount_mL):
    #Check if has pipet
    if HAS_PIPET == False:
        get_pipet(PIPETS_USED)
    
    source_vial_clamped = (CLAMPED_VIAL == source_vial_num) #Is the source vial clamped?
    dest_vial_clamped = (CLAMPED_VIAL == dest_vial_num) #Is the destination vial clamped?
    
    #Aspirate from source
    if source_vial_clamped:
        c9.goto_safe(vial_clamp_pip)
    else:
        c9.goto_safe(rack_pip[source_vial_num])
    c9.aspirate_ml(0, amount_mL)
    
    #Dispense at destination
    if dest_vial_clamped:
        c9.goto_safe(vial_clamp_pip)
    else:
        c9.goto_safe(rack_pip[dest_vial_num])
    c9.dispense_ml(0, amount_mL)

#We will need to check if the vial is capped before moving
def move_vial_to_clamp(vial_num):
    global CLAMPED_VIAL, GRIPPER_STATUS
    print("Moving vial " + str(vial_num) + " to clamp")
    #Check that the robot gripper is empty
    if GRIPPER_STATUS == "Open":
        #Check to make sure there is no vial
        if CLAMPED_VIAL == "None":
            goto_location_if_not_there(rack[vial_num]) #move to vial
            c9.close_gripper() #grip vial
            c9.goto_safe(vial_clamp) #move vial to clamp
            c9.close_clamp() #clamp vial
            c9.open_gripper() #release vial
            CLAMPED_VIAL = vial_num
        else:
            print("Cannot move vial to clamp, clamp full")
    else:
        print("Cannot move vial to clamp, gripper full")

#We will need to check if the vial is capped before moving
def return_vial_from_clamp(vial_num):
    global CLAMPED_VIAL, GRIPPER_STATUS
    print("Moving vial " + str(vial_num) + " from clamp")
   #Check that the robot gripper is empty
    if GRIPPER_STATUS == "Open":
        #Check to make sure there is a vial
        if CLAMPED_VIAL != "None":
            goto_location_if_not_there(vial_clamp) #Maybe check if it is already there or not 
            c9.close_gripper() #Grab vial
            c9.open_clamp() #unclamp vial
            c9.goto_safe(rack[vial_num]) #Move back to vial rack
            c9.open_gripper() #Release vial
            CLAMPED_VIAL = "None"
        else:
            print("Cannot return vial from clamp, no vial in clamp")
    else:
        print("Cannot return vial from clamp, gripper full")

def uncap_clamp_vial():
    global GRIPPER_STATUS, CLAMPED_VIAL
    print ("Removing cap from clamped vial")
    if GRIPPER_STATUS == "Open":
        if CLAMPED_VIAL != "None":
            goto_location_if_not_there(vial_clamp) #Maybe check if it is already there or not   
            c9.close_clamp() #clamp vial
            c9.close_gripper()
            c9.uncap()
            GRIPPER_STATUS = "Cap"
        else:
            print("Cannot decap vial, no vial in clamp")
    else:
        print("Cannot decap, gripper full")

def recap_clamp_vial():
    print("Recapping clamped vial")
    
    global GRIPPER_STATUS, CLAMPED_VIAL, HAS_PIPET
    error_check_list = [] #List of specific errors for this method
    error_check_list.append([GRIPPER_STATUS, "Cap", "Cannot recap, no cap in gripper"])
    error_check_list.append([HAS_PIPET, False, "Can't recap vial, holding pipet"])
    error_check_list.append([str(CLAMPED_VIAL).isnumeric(), True, "Cannot recap, no vial in clamp"])
    
    if check_for_errors(error_check_list) == False:
        goto_location_if_not_there(vial_clamp) #Maybe check if it is already there or not
        c9.close_clamp() #Make sure vial is clamped
        c9.cap() #Cap the vial
        c9.open_gripper() #Open the gripper to release the cap
        GRIPPER_STATUS = "Open"

#Checks first that you aren't already there... This mostly applies for cap/decap
def goto_location_if_not_there(location):
    difference_threshold = 550
    if get_location_distance(location, c9.get_robot_positions()) > difference_threshold:
        c9.goto_safe(location)

def get_location_distance(loc_1, loc_2):
    difference = np.sum(np.absolute(np.array(loc_2)[1:4] - np.array(loc_1)[1:4]))
    return difference
        
def print_status():
    print("Gripper Status: " + GRIPPER_STATUS)
    print("Clamp Status: " + str(CLAMPED_VIAL))
    print("Pipets Used: " + str(PIPETS_USED))
    print("Has Pipet: " + str(HAS_PIPET))

#Removes the target vial, vortexes it, then puts it back
def vortex_vial(vial_num, vortex_rads):
    global GRIPPER_STATUS, CLAMPED_VIAL, HAS_PIPET
    print("Vortexing Vial: " + str(vial_num))
    vial_clamped = (CLAMPED_VIAL == vial_num) #Is the vial clamped?
    
    error_check_list = []
    error_check_list.append([GRIPPER_STATUS, "Open", "Can't Vortex, gripper is used"])
    error_check_list.append([HAS_PIPET, False, "Can't vortex vial, holding pipet"])
    
    if check_for_errors(error_check_list) == False:
        #Get vial
        if vial_clamped:
            goto_location_if_not_there(vial_clamp)
            c9.close_gripper()
            c9.open_clamp()
        else:
            goto_location_if_not_there(rack[vial_num])
            c9.close_gripper()
    
        c9.move_z(292) #Move to a higher height
        #Rotate
        c9.move_axis(c9.GRIPPER, vortex_rads, vel=100)
        c9.move_axis(c9.GRIPPER, 0, vel=100)
        
        #Return vial
        if vial_clamped:
            c9.goto_safe(vial_clamp)
            c9.close_clamp()
            c9.open_gripper()
        else:
            c9.goto_safe(rack[vial_num])
            c9.open_gripper()

def check_for_errors(error_check_list):
    error_occured = False
    for error_check in error_check_list:
        if error_check[0] != error_check[1]:
            error_occured = True
            print(error_check[2])
    return error_occured


initialize()
move_vial_to_clamp(0)
uncap_clamp_vial()
pipet_from_vial_into_vial(1, 0, 0.5)
remove_pipet()
pipet_from_vial_into_vial(5, 1, 0.5)
remove_pipet()
recap_clamp_vial()
return_vial_from_clamp(0)




    
    

