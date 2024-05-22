from north import NorthC9
from Locator import *
import numpy as np

class North_Robot:

    #Global variables, must be referenced with 
    HAS_PIPET = False
    GRIPPER_STATUS = "Open"
    PIPETS_USED = 0
    CLAMPED_VIAL = "None"
    OPEN_VIALS = []
    VIAL_NAMES = [] #List of vials that do not have caps. These cannot be moved
    c9 = None
    DROP_AMOUNT = 0.02 #Droplet amount
    
    #Initialize function
    def __init__(self, open_vials, vial_names):
        print("Initializing")
        self.c9 = NorthC9('A', network_serial='AU06CNCF')
        self.OPEN_VIALS = open_vials
        self.VIAL_NAMES = vial_names
        self.c9.default_vel = 40 #Could make this a parameter

    #Reset positions and get rid of any pipet tips
    def reset_after_initialization(self):
        self.c9.home_pump(0) #Home the pump
        #self.c9.home_carousel() #Home the carousel
        self.c9.open_gripper()  #In case there's something
        self.c9.zero_scale() #Zero the scale
        self.remove_pipet() #Remove the pipet, if there's one for some reason   

    #Remove the pipet tip
    def remove_pipet(self):
        print("Removing pipet")
        self.c9.goto_safe(p_remove_approach)
        self.c9.goto(p_remove_cap, vel=5)
        self.c9.move_z(292, vel=20)
        self.HAS_PIPET = False

    #Add a pipet tip
    def get_pipet(self):
        print("Getting pipet number: " +str(self.PIPETS_USED))
        self.c9.goto_safe(p_capture_grid[self.PIPETS_USED])
        self.HAS_PIPET = True
        self.PIPETS_USED += 1
        
    #We may need to do a check to see if vials are uncapped or not
    def pipet_from_vial_into_vial(self, source_vial_num, dest_vial_num, amount_mL, dispense_type="all_at_once",
                                  measure_weight=False, use_calibration=False, aspirate_conditioning=False):
        
        error_check_list = [] #List of specific errors for this method
        error_check_list.append([self.check_if_vials_are_open([source_vial_num, dest_vial_num]), True, "Can't pipet, at least one vial is capped"])
        initial_mass = 0
        final_mass = 0

        if self.check_for_errors(error_check_list) == False:
            #Check if has pipet, get one if needed
            if self.HAS_PIPET == False:
                self.get_pipet()
            
            print("Pipetting from vial " + self.get_vial_name(source_vial_num) + " to vial " + self.get_vial_name(dest_vial_num) + ", Method: " + dispense_type.replace("_", " "))
            #print("Dispense type: " + dispense_type)
            
            source_vial_clamped = (self.CLAMPED_VIAL == source_vial_num) #Is the source vial clamped?
            dest_vial_clamped = (self.CLAMPED_VIAL == dest_vial_num) #Is the destination vial clamped?
                    
            #If the destination vial is at the clamp and you want the weight, measure prior to pipetting
            if measure_weight and dest_vial_clamped:
                initial_mass = self.c9.read_steady_scale()
            
            #Aspirate from source
            if source_vial_clamped:
                self.c9.goto_xy_safe(vial_clamp_pip)
                self.c9.move_z(120)
                self.c9.goto_z(vial_clamp_pip, vel=15)
            else:
                self.c9.goto_xy_safe(rack_pip[source_vial_num])
                self.c9.move_z(120)
                self.c9.goto_z(rack_pip[source_vial_num], vel=15)
            
            #This is for volatile solvents at larger volumes... "Harry's Method"
            if aspirate_conditioning:
                for i in range (0, 2):
                    self.c9.aspirate_ml(0, amount_mL)
                    self.c9.dispense_ml(0, amount_mL)

            self.c9.aspirate_ml(0, amount_mL)
                    
            #Dispense at destination
            if dest_vial_clamped:
                self.c9.goto_xy_safe(vial_clamp_pip)
                self.c9.goto_z(vial_clamp_pip)
            else:
                self.c9.goto_xy_safe(rack_pip[dest_vial_num])
                self.c9.goto_z(rack_pip[dest_vial_num])
            
            if dispense_type=="all_at_once":
                self.c9.dispense_ml(0, amount_mL)
            elif dispense_type == "by_drop":
                for i in range (0, int(amount_mL/self.DROP_AMOUNT)):
                    self.c9.dispense_ml(0,self.DROP_AMOUNT)
            
            #If the destination vial is at the clamp and you want the weight, measure after pipetting
            if measure_weight and dest_vial_clamped:
                final_mass = self.c9.read_steady_scale()
            
        measured_mass = final_mass - initial_mass
                
        return measured_mass

    #We will need to check if the vial is capped before moving
    def move_vial_to_clamp(self, vial_num):
        print("Moving vial " + self.get_vial_name(vial_num) + " to clamp")

        error_check_list = [] #List of specific errors for this method
        error_check_list.append([self.GRIPPER_STATUS, "Open", "Cannot move vial to clamp, gripper full"])
        error_check_list.append([self.HAS_PIPET, False, "Cannot move vial to clamp, robot holding pipet"])
        error_check_list.append([str(self.CLAMPED_VIAL).isnumeric(), False, "Cannot move vial to clamp, clamp full"])
        error_check_list.append([self.check_if_vials_are_open([vial_num]), False, "Can't move vial, vial is uncapped"])

        if self.check_for_errors(error_check_list) == False:
            self.goto_location_if_not_there(rack[vial_num]) #move to vial
            self.c9.close_gripper() #grip vial
            self.c9.goto_safe(vial_clamp) #move vial to clamp
            #self.c9.close_clamp() #clamp vial
            self.c9.open_gripper() #release vial
            self.CLAMPED_VIAL = vial_num

    #We will need to check if the vial is capped before moving
    def return_vial_from_clamp(self, vial_num):
        print("Moving vial " + self.get_vial_name(vial_num) + " from clamp")

        error_check_list = [] #List of specific errors for this method
        error_check_list.append([self.GRIPPER_STATUS, "Open", "Cannot return vial from clamp, gripper full"])
        error_check_list.append([self.HAS_PIPET, False, "Cannot return vial from clamp, robot holding pipet"])
        error_check_list.append([str(self.CLAMPED_VIAL).isnumeric(), True, "Cannot return vial from clamp, no vial in clamp"])
        error_check_list.append([self.check_if_vials_are_open([self.CLAMPED_VIAL]), False, "Can't return vial, vial is uncapped"])

        if self.check_for_errors(error_check_list) == False:
            self.goto_location_if_not_there(vial_clamp) #Maybe check if it is already there or not 
            self.c9.close_gripper() #Grab vial
            self.c9.open_clamp() #unclamp vial
            self.c9.goto_safe(rack[vial_num]) #Move back to vial rack
            self.c9.open_gripper() #Release vial
            self.CLAMPED_VIAL = "None"

    #Uncap the vial in the clamp
    def uncap_clamp_vial(self):
        print ("Removing cap from clamped vial")

        error_check_list = [] #List of specific errors for this method
        error_check_list.append([self.GRIPPER_STATUS, "Open", "Cannot uncap, gripper full"])
        #error_check_list.append([self.HAS_PIPET, False, "Can't uncap vial, holding pipet"])
        error_check_list.append([str(self.CLAMPED_VIAL).isnumeric(), True, "Cannot uncap, no vial in clamp"])
        error_check_list.append([self.check_if_vials_are_open([self.CLAMPED_VIAL]), False, "Can't uncap, vial is uncapped already"])

        if self.check_for_errors(error_check_list) == False:
            self.goto_location_if_not_there(vial_clamp) #Maybe check if it is already there or not   
            self.c9.close_clamp() #clamp vial
            self.c9.close_gripper()
            self.c9.uncap()
            self.GRIPPER_STATUS = "Cap"
            self.c9.open_clamp()
            self.OPEN_VIALS.append(self.CLAMPED_VIAL) #track open vials

    #Recap the vial in the clamp
    def recap_clamp_vial(self):
        print("Recapping clamped vial")
        
        error_check_list = [] #List of specific errors for this method
        error_check_list.append([self.GRIPPER_STATUS, "Cap", "Cannot recap, no cap in gripper"])
        #error_check_list.append([self.HAS_PIPET, False, "Can't recap vial, holding pipet"])
        error_check_list.append([self.check_if_vials_are_open([self.CLAMPED_VIAL]), True, "Can't recap, vial is capped already"])
        error_check_list.append([str(self.CLAMPED_VIAL).isnumeric(), True, "Cannot recap, no vial in clamp"])
        
        if self.check_for_errors(error_check_list) == False:
            self.goto_location_if_not_there(vial_clamp) #Maybe check if it is already there or not
            self.c9.close_clamp() #Make sure vial is clamped
            self.c9.cap() #Cap the vial
            self.c9.open_gripper() #Open the gripper to release the cap
            self.GRIPPER_STATUS = "Open"

            self.OPEN_VIALS.remove(self.CLAMPED_VIAL)

    #Checks first that you aren't already there... This mostly applies for cap/decap
    def goto_location_if_not_there(self, location):
        difference_threshold = 550
        if self.get_location_distance(location, self.c9.get_robot_positions()) > difference_threshold:
            self.c9.goto_safe(location)

    #Measurement for how far two points are
    def get_location_distance(self, loc_1, loc_2):
        difference = np.sum(np.absolute(np.array(loc_2)[1:4] - np.array(loc_1)[1:4]))
        return difference
    
    #print the global variables
    def print_status(self):
        print("Gripper Status: " + self.GRIPPER_STATUS)
        print("Clamp Status: " + str(self.CLAMPED_VIAL))
        print("Pipets Used: " + str(self.PIPETS_USED))
        print("Has Pipet: " + str(self.HAS_PIPET))

    #Removes the target vial, vortexes it, then puts it back
    def vortex_vial(self, vial_num, vortex_rads):
        print("Vortexing Vial: " + self.get_vial_name(vial_num))
        vial_clamped = (self.CLAMPED_VIAL == vial_num) #Is the vial clamped?
        
        error_check_list = []
        error_check_list.append([self.GRIPPER_STATUS, "Open", "Can't Vortex, gripper is used"])
        error_check_list.append([self.check_if_vials_are_open([vial_num]), False, "Can't vortex, vial not capped"])
        error_check_list.append([self.HAS_PIPET, False, "Can't vortex vial, holding pipet"])
        
        if self.check_for_errors(error_check_list) == False:
            #Get vial
            if vial_clamped:
                self.goto_location_if_not_there(vial_clamp)
                self.c9.close_gripper()
                self.c9.open_clamp()
            else:
                self.goto_location_if_not_there(rack[vial_num])
                self.c9.close_gripper()
        
            self.c9.move_z(292) #Move to a higher height
            #Rotate
            self.c9.move_axis(self.c9.GRIPPER, vortex_rads, vel=100)
            self.c9.move_axis(self.c9.GRIPPER, 0, vel=100)
            
            #Return vial
            if vial_clamped:
                self.c9.goto_safe(vial_clamp)
                self.c9.close_clamp()
                self.c9.open_gripper()
            else:
                self.c9.goto_safe(rack[vial_num])
                self.c9.open_gripper()

    #This is just to formalize the process of error checking so its more extensible
    #Error check list is as follows error_check[0] is the value, error_check[1] is the target value, error_check[2] is the error message
    def check_for_errors(self, error_check_list):
        error_occured = False
        for error_check in error_check_list:
            if error_check[0] != error_check[1]:
                error_occured = True
                print(error_check[2])
        return error_occured

    def get_vial_name(self, vial_num):
        name = ""
        try:
            name = str(vial_num) + " (" + self.VIAL_NAMES[vial_num] + ")"
        except:
            name = str(vial_num)
        return name
            
    def check_if_vials_are_open(self, vials_to_check):
        all_open = True
        for vial in vials_to_check:
            if (vial not in self.OPEN_VIALS):
                all_open = False
        return all_open

    def dispense_liquid_into_clamped_vial():
        return None
    def dispense_solid_into_clamped_vial():
        return None
    def move_vial_to_heater():
        return None
    def set_heater_temperature():
        return None
    def dispense_from_vial_into_wellplate():
        return None

    
    

