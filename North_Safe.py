from north import NorthC9
from Locator import *
import numpy as np
import time
import math

class North_Robot:

    #Global variables, must be referenced with 
    HAS_PIPET = False
    GRIPPER_STATUS = "Open"
    PIPETS_USED = 0
    CLAMPED_VIAL = "None"
    PIPET_DIMS = [] #Item 0 = Length, Item 1 = Distance from top of rack
    VIAL_NAMES = [] #List of vials that do not have caps. These cannot be moved
    VIAL_DF = None #Dataframe tracking vial information

    c9 = None
    
    #Initialize function
    def __init__(self, vial_df, pipet_dims):
        print("Initializing")
        self.c9 = NorthC9('A', network_serial='AU06CNCF')
        self.VIAL_DF = vial_df
        self.VIAL_NAMES = vial_df['vial name']
        self.PIPET_DIMS = pipet_dims
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

    #Add a pipet tip... 
    #TBD: Adjust the height based on the size of the pipet tip
    def get_pipet(self):
        
        num = (self.PIPETS_USED%16)*3+math.floor(self.PIPETS_USED/16)
        print("Getting pipet number: " +str(num))

        #First move to the xy location (disabled for now)
        self.c9.goto_safe(p_capture_grid[num])
        
        

        #Second move to z location, based off of the height (ignore p_capture_grid height here)
        #Standard height + delta-Z (PIPET_DIM[0])
        #self.c9.move_z()

        self.HAS_PIPET = True
        self.PIPETS_USED += 1
        
    #Pipet from a vial into another vial
    #Measure_Weight gives you the option to report the mass dispensed
    #Use calibration (not implemented) is if you want to adjust the volume based off a known calibration
    #Aspirate conditioning is an alternate way to aspirate (up and down twice)
    #Dispense_type affects how you dispense... either all_at_once or by_drip but in theory this could be done just by adjusting the dispense speed
    def pipet_from_vial_into_vial(self, source_vial_num, dest_vial_num, amount_mL, measure_weight=False, use_calibration=False, 
                                  aspirate_conditioning=False, track_height=True, wait_over_vial=False, aspirate_extra=False, aspirate_speed=11, dispense_speed=11):
        
        error_check_list = [] #List of specific errors for this method
        error_check_list.append([self.check_if_vials_are_open([source_vial_num, dest_vial_num]), True, "Can't pipet, at least one vial is capped"])
        initial_mass = 0
        final_mass = 0

        if self.check_for_errors(error_check_list) == False:
            #Check if has pipet, get one if needed
            if self.HAS_PIPET == False:
                self.get_pipet()
            
            print("Pipetting from vial " + self.get_vial_name(source_vial_num) + " to vial " + self.get_vial_name(dest_vial_num))
            #print("Dispense type: " + dispense_type)
            
            source_vial_clamped = (self.CLAMPED_VIAL == source_vial_num) #Is the source vial clamped?
            dest_vial_clamped = (self.CLAMPED_VIAL == dest_vial_num) #Is the destination vial clamped?
                    
            #If the destination vial is at the clamp and you want the weight, measure prior to pipetting
            if measure_weight and dest_vial_clamped:
                initial_mass = self.c9.read_steady_scale()
            
            #Aspirate from source... Need to adjust aspiration height based off of existing volume
            if source_vial_clamped:
                self.c9.goto_xy_safe(vial_clamp_pip)
                self.c9.move_z(120) #safe height above vial
                #self.c9.move_z(self.get_aspirate_height(self.VIAL_DF.at[source_vial_num,'vial volume (mL)'], amount_mL, buffer = 0.5), vel=15)
                self.c9.move_z(vial_clamp_pip, vel=15) #Needs a different base height for the 
            else:
                self.c9.goto_xy_safe(rack_pip[source_vial_num])
                self.c9.move_z(120) #safe height above vial
                source_vial_volume = self.VIAL_DF.at[source_vial_num,'vial volume (mL)']
                if track_height:
                    self.c9.move_z(self.get_aspirate_height(source_vial_volume, amount_mL, 0.5, 76), vel=15)
                else:
                    self.c9.goto_z(rack_pip[source_vial_num], vel=15)
            
            #This is for volatile solvents at larger volumes... "Harry's Method"
            if aspirate_conditioning:
                for i in range (0, 2):
                    self.c9.aspirate_ml(0, amount_mL)
                    self.c9.dispense_ml(0, amount_mL)

            #self.c9.set_pump_speed(0, aspirate_speed)
            self.c9.aspirate_ml(0, amount_mL)

            self.c9.move_z(140) #Safe height
            
            if wait_over_vial:
                time.sleep(5)

            air_buffer_mL = 0
            if aspirate_extra:
                self.c9.aspirate_mL(0, 0.05)
                air_buffer_mL = 0.05

            #Track the removed volume in the dataframe
            original_amount = self.VIAL_DF.at[source_vial_num,'vial volume (mL)']
            self.VIAL_DF.at[source_vial_num,'vial volume (mL)']=(original_amount-amount_mL)

            #Dispense at destination
            if dest_vial_clamped:
                self.c9.goto_xy_safe(vial_clamp_pip)
                self.c9.goto_z(vial_clamp_pip)
            else:
                self.c9.goto_xy_safe(rack_pip[dest_vial_num])
                self.c9.goto_z(rack_pip[dest_vial_num])
            
            #self.c9.set_pump_speed(0, dispense_speed)
            self.c9.dispense_ml(0, amount_mL+air_buffer_mL)

            

            #Track the added volume in the dataframe
            self.VIAL_DF.at[dest_vial_num,'vial volume (mL)']=self.VIAL_DF.at[dest_vial_num,'vial volume (mL)']+amount_mL


            #If the destination vial is at the clamp and you want the weight, measure after pipetting
            if measure_weight and dest_vial_clamped:
                final_mass = self.c9.read_steady_scale()
            
        measured_mass = final_mass - initial_mass  
        return measured_mass

    def pipet_from_vial_into_wellplate(self, source_vial_num, dest_wp_num, amount_mL, replicate =1, dispense_type = "None", use_calibration=False, 
                                  aspirate_conditioning=False, track_height=True, wait_over_vial=False, aspirate_extra = False):
        """
        To pipette from source vial into well plate given the source_vial_num (from dataframe & txt file), dest_wp_num (array of numbers for wellplate coordinates), amount_mL (
        amount to be dispensed PER WELL!), replicates
        
        amount_mL is PER WELL!!!

        Need to provide dest_wp_num in a list, to accomodate for multiple repeats~ (so only change pipetting once)

        Returns True when pipetting is complete

        """

        #source_vial, dest_wp, amount_mL, replicates
        # NO NEED: mesaure_weight, track_height
        #implement later: wait_over_vial -- when it needs to

        error_check_list = [] #List of specific errors for this method
        error_check_list.append([self.check_if_vials_are_open([source_vial_num]), True, "Can't pipet, at least one vial is capped"])

        if self.check_for_errors(error_check_list) == False:
            #Check if has pipet, get one if needed
            if self.HAS_PIPET == False:
                self.get_pipet()

            
            source_vial_clamped = (self.CLAMPED_VIAL == source_vial_num) #Is the source vial clamped? (should be)
            source_vial_volume = self.VIAL_DF.at[source_vial_num,'vial volume (mL)']

            #Aspirate from source... Need to adjust aspiration height based off of existing volume
            if source_vial_clamped:
                self.c9.goto_xy_safe(vial_clamp_pip)
                self.c9.move_z(160) #safe height above vial
                
                #self.c9.move_z(self.get_aspirate_height(self.VIAL_DF.at[source_vial_num,'vial volume (mL)'], amount_mL, buffer = 0.5), vel=15)
                #self.c9.move_z(vial_clamp_pip, vel=15) #Needs a different base height for the 

                if track_height:
                    self.c9.move_z(self.get_aspirate_height(source_vial_volume, amount_mL, 0.5, 125), vel=15)
                else:
                    self.c9.goto_z(160, vel=15)

            else:
                self.c9.goto_xy_safe(rack_pip[source_vial_num])
                self.c9.move_z(160) #safe height above vial
                if track_height:
                    self.c9.move_z(self.get_aspirate_height(source_vial_volume, amount_mL, 0.5, 125), vel=15)
                else:
                    self.c9.goto_z(125, vel=15)
            
            #This is for volatile solvents at larger volumes... "Harry's Method"
            if aspirate_conditioning:
                for i in range (0, 2):
                    self.c9.aspirate_ml(0, amount_mL)
                    self.c9.dispense_ml(0, amount_mL)

#             if dispense_type.lower() == "drop":
#                 self.c9.set_pump_speed(0, 25)
#                 print("dropped")
#             
#             elif dispense_type.lower() == "slow":
#                 self.c9.set_pump_speed(0,20)
#     

            #self.c9.set_pump_speed(0, aspirate_speed)
            self.c9.aspirate_ml(0, amount_mL*replicate)

            #self.c9.move_z(180) #Safe height
            
            if wait_over_vial:
                time.sleep(5)

            air_buffer_mL = 0
            if aspirate_extra:
                self.c9.aspirate_mL(0, 0.05)
                air_buffer_mL = 0.05

            #Track the removed volume in the dataframe
            original_amount = self.VIAL_DF.at[source_vial_num,'vial volume (mL)']
            self.VIAL_DF.at[source_vial_num,'vial volume (mL)']=original_amount-(amount_mL*replicate)
            #print(self.VIAL_DF)

            self.c9.default_vel = 15 
            #Dispense at wellplate
            for i in range(replicate):
                if i == 0:
                    self.c9.goto_xy_safe(well_plate_grid[dest_wp_num[i]])
                    self.c9.goto_z(well_plate_grid[dest_wp_num[i]])
                else:
                    self.c9.goto(well_plate_grid[dest_wp_num[i]])

                self.c9.dispense_ml(0, amount_mL+air_buffer_mL)
                print("Pipetting from vial " + self.get_vial_name(source_vial_num) + " to well #" + str(dest_wp_num[i]))
                
                if dispense_type.lower() == "drop" and i == replicate-1: #last replicate
                    self.c9.default_vel=40
                    self.c9.move_z(125)
                    time.sleep(2)
                    print("last drop")
                    
                elif dispense_type.lower() == "drop" or dispense_type.lower() == "slow":
                    time.sleep(2)
                    

                print("Dispense type: " + dispense_type)
        return True

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
            #self.OPEN_VIALS.append(self.CLAMPED_VIAL) #track open vials

            self.VIAL_DF.at[self.CLAMPED_VIAL, 'open']=True
            #print(self.VIAL_DF)

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

            #self.OPEN_VIALS.remove(self.CLAMPED_VIAL)
            self.VIAL_DF.at[self.CLAMPED_VIAL, 'open']=False
            #print(self.VIAL_DF)

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
            self.c9.move_axis(self.c9.GRIPPER, vortex_rads, vel=50)
            self.c9.move_axis(self.c9.GRIPPER, 0, vel=50)
            
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
            if not (self.VIAL_DF.at[vial, 'open']):
                all_open = False
        return all_open

    #Get adjust the aspiration height based on how much is there
    def get_aspirate_height(self, amount_vial, amount_to_withdraw, buffer, base_height):
        target_height = base_height + (6*(amount_vial - amount_to_withdraw - buffer))
        print(target_height)
        if target_height > base_height:
            return target_height
        else:
            return base_height

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

    
    

