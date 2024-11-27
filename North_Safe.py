from north import NorthC9
from Locator import *
import numpy as np
import time
import math
import pandas as pd
import sys
sys.path.append("C:\\Users\\Imaging Controller\\Desktop\\utoronto_demo")
sys.path.append("C:\\Users\\Imaging Controller\\Desktop\\utoronto_demo\\status")

class North_Track:

    #Controller
    c9 = None
    MAX_HEIGHT = 0
    RELEASE_DISTANCE_Y = 2400
    LID_DISTANCE_Y = 4000

    #Well plate active areas
    NR_WELL_PLATE_X = [131854,105860,81178]
    NR_WELL_PLATE_Y = [88396, 86965, 89155]
    #NR_WELL_PLATE_Y_RELEASE = 86000
    
    #Transit constants
    WELL_PLATE_TRANSFER_Y = 75000
    CYT_SAFE_X = 47747
    
    #Release into cytation
    CYT_TRAY_Y_RELEASE = 5500
    CYT_TRAY_Y_GRAB = 7500
    CYT_TRAY_X = 68608

    QUARTZ_WP_OFFSET = 1300

    #Speed horizontal
    DEFAULT_X_SPEED = 50

    #Speed vertical
    DEFAULT_Y_SPEED = 50

    def __init__(self, c9):
        self.c9 = c9

    def set_horizontal_speed(self,vel):
        self.c9.DEFAULT_X_SPEED = vel

    def set_vertical_speed(self,vel):
        self.c9.DEFAULT_Y_SPEED = vel    

    def open_gripper(self):
        self.c9.set_output(4, True)  
        self.c9.set_output(5, False)
        self.c9.delay(2)
    def close_gripper(self):
        self.c9.set_output(5, True)  #gripper close
        self.c9.set_output(4, False)
        self.c9.delay(2)

    def grab_well_plate_from_nr(self, well_plate_num, grab_lid=False,quartz_wp=False):
        move_up = 0
        if grab_lid:
            move_up = self.LID_DISTANCE_Y 
        if quartz_wp:
            move_up = -self.QUARTZ_WP_OFFSET #
        self.open_gripper()
        self.c9.move_axis(7, self.NR_WELL_PLATE_X[well_plate_num], vel=self.DEFAULT_X_SPEED) #left to WP
        self.c9.move_axis(6, self.NR_WELL_PLATE_Y[well_plate_num]-move_up, vel=self.DEFAULT_Y_SPEED) #down
        self.close_gripper()
        self.c9.move_axis(6, self.WELL_PLATE_TRANSFER_Y, vel=self.DEFAULT_Y_SPEED) #up slightly

    def move_gripper_to_cytation(self):
        self.c9.move_axis(7, self.CYT_SAFE_X, vel=self.DEFAULT_X_SPEED) #past Cytation loading
        self.c9.move_axis(6,self.MAX_HEIGHT,vel=self.DEFAULT_Y_SPEED) #up to max height
    
    def release_well_plate_in_cytation(self,quartz_wp=False):
        move_up = 0
        if quartz_wp:
            move_up = -self.QUARTZ_WP_OFFSET #
        #OPEN CYTATION TRAY
        self.c9.move_axis(7, self.CYT_TRAY_X, vel=self.DEFAULT_X_SPEED) #to well plate loading
        self.c9.move_axis(6, self.CYT_TRAY_Y_RELEASE-move_up,vel=25) #down slightly
        self.open_gripper()
        self.c9.move_axis(6,self.MAX_HEIGHT, vel=self.DEFAULT_Y_SPEED) #back to max height
        #CLOSE CYTATION TRAY

    def grab_well_plate_from_cytation(self,quartz_wp=False):
        #OPEN CYTATION TRAY
        move_up = 0
        if quartz_wp:
            move_up = -self.QUARTZ_WP_OFFSET #
        self.open_gripper()
        self.c9.move_axis(6, self.CYT_TRAY_Y_GRAB-move_up,vel=25) #down slightly
        self.close_gripper()
        self.c9.move_axis(6,self.MAX_HEIGHT,vel=self.DEFAULT_Y_SPEED) #up to max height
        self.c9.move_axis(7, self.CYT_SAFE_X, vel=self.DEFAULT_X_SPEED) #past Cytation loading
        #CLOSE CYTATION TRAY
        
    def return_well_plate_to_nr(self, well_plate_num, grab_lid=False,quartz_wp=False):
        move_up = 0
        if grab_lid:
            move_up = self.LID_DISTANCE_Y 
        if quartz_wp:
            move_up = -self.QUARTZ_WP_OFFSET #
        self.c9.move_axis(6, self.WELL_PLATE_TRANSFER_Y, vel=self.DEFAULT_Y_SPEED) #down
        self.c9.move_axis(7, self.NR_WELL_PLATE_X[well_plate_num] , vel=self.DEFAULT_X_SPEED) #left to WP
        self.c9.move_axis(6, self.NR_WELL_PLATE_Y[well_plate_num]-self.RELEASE_DISTANCE_Y-move_up, vel=50) #down
        self.open_gripper()
        self.c9.move_axis(6, self.WELL_PLATE_TRANSFER_Y, vel=self.DEFAULT_Y_SPEED) #up slightly
    
    def origin(self):
        self.c9.move_axis(6, self.MAX_HEIGHT, vel=self.DEFAULT_Y_SPEED) #max_height
        self.c9.move_axis(7, 0, vel=self.DEFAULT_X_SPEED)

class North_Robot:

    #Global variables, must be referenced with 
    HAS_PIPET = False
    GRIPPER_STATUS = "Open"
    CLAMPED_VIAL = "None"
    PIPET_DIMS = [] #Item 0 = Length, Item 1 = Distance from top of rack
    VIAL_NAMES = [] #List of vials that do not have caps. These cannot be moved
    VIAL_DF = None #Dataframe tracking vial information
    PIPET_ARRAY = None

    VIAL_FILE = None
    PIPET_FILE = None

    #Pipet rack data
    ACTIVE_PIPET_RACK = 0 #There are two racks... Rack 0 and Rack 1
    PIPETS_USED = [0,0] #Tracker for each rack

    #Pipet tip dimensions (Future... Let's keep a spreadsheet for this not list them all here)
    BLUE_DIMS = [20,77]
    DEFAULT_DIMS = [25,85]
    FILTER_DIMS = [24,98]

    #Controller
    c9 = None
    
    #Initialize function
    def __init__(self, c9, vial_file=None,delim=',',pipet_file=None):
        print("Initializing North Robot")
        self.c9 = c9
        try:
            self.VIAL_FILE = vial_file
            self.VIAL_DF = pd.read_csv(vial_file, delimiter=delim)
            self.VIAL_NAMES = self.VIAL_DF['vial name']
        except Exception as e:
            print("No vial file inputted", e)
        self.PIPET_DIMS = self.DEFAULT_DIMS
        self.c9.default_vel = 20 #Could make this a parameter
        if pipet_file:
            self.PIPET_FILE = pipet_file
        else:
            self.PIPET_FILE = "../utoronto_demo/status/pipets.txt"
        try:
            with open(self.PIPET_FILE) as f:
                pipets = np.array(f.read().split(','))
            self.PIPET_ARRAY = pipets.astype(np.int16)
        except:    
            print("Cannot open pipet file... using default sequence")

    #Reset positions and get rid of any pipet tips
    def reset_after_initialization(self):
        self.c9.move_z(200)
        self.c9.home_pump(0) #Home the pump
        #self.c9.home_carousel() #Home the carousel
        self.c9.open_gripper()  #In case there's something
        self.c9.zero_scale() #Zero the scale
        #self.remove_pipet() #Remove the pipet, if there's one for some reason   

    #Change the speed of the robot
    def set_robot_speed(self, speed_in):
        self.c9.default_vel = speed_in

    #Adjust the pipet tips used and also which rack you are drawing from
    def set_pipet_tip_type(self, pipet_dims, active_rack):
        self.ACTIVE_PIPET_RACK = active_rack
        self.PIPET_DIMS = pipet_dims

    #Remove the pipet tip
    def remove_pipet(self):
        print("Removing pipet")
        self.c9.goto_safe(p_remove_approach)
        self.c9.goto(p_remove_cap, vel=5)
        remove_pipet_height = 292 #Constant height to remove the pipet (doesn't change with the pipet type, just moving up)
        self.c9.move_z(remove_pipet_height, vel=20)
        self.HAS_PIPET = False

    #Take a pipet tip from the active rack with the active pipet tip dimensions 
    def get_pipet(self):
        
        try:
            active_pipet_num = self.PIPET_ARRAY[0] #First available pipet
        except:
            active_pipet_num = self.PIPETS_USED[self.ACTIVE_PIPET_RACK]

        num = (active_pipet_num%16)*3+math.floor(active_pipet_num/16)
        print(f"Getting pipet number: {active_pipet_num} from rack {self.ACTIVE_PIPET_RACK}")

        #First move to the xy location 
        
        if self.ACTIVE_PIPET_RACK == 0:
            location = p_capture_grid[num]
        elif self.ACTIVE_PIPET_RACK == 1:
            location = p_capture_grid_high[num]

        self.c9.goto_xy_safe(location)

        #Second move to z location, based off of the height
        base_height = self.c9.counts_to_mm(3, location[3])
        self.c9.move_z(base_height - self.DEFAULT_DIMS[0] + self.PIPET_DIMS[0]) #Adjust height based off of the distance from the default tip (which the measurements were done for)

        self.HAS_PIPET = True
        self.PIPETS_USED[self.ACTIVE_PIPET_RACK] += 1

        self.PIPET_ARRAY = np.delete(self.PIPET_ARRAY, 0)
        self.save_pipet_status(self.PIPET_FILE)

        #print ("Pipets: ", self.PIPETS_USED)

    def pipet_from_location(self, amount, pump_speed, height, aspirate=True, move_speed=15):
        self.c9.move_z(height, vel=move_speed)
        #self.c9.set_pump_speed(0, pump_speed)
        if aspirate:
            self.c9.aspirate_ml(0,amount)
        else:
            self.c9.dispense_ml(0,amount)

    #Pipet from a vial into another vial
    #Use calibration (not implemented) is if you want to adjust the volume based off a known calibration
    #Aspirate conditioning is an alternate way to aspirate (up and down some number of cycles)
    def aspirate_from_vial(self, source_vial_num, amount_mL, use_calibration=False, asp_disp_cycles=0, track_height=True, vial_wait_time=0, aspirate_speed=11):
        error_check_list = [] #List of specific errors for this method
        error_check_list.append([self.check_if_vials_are_open([source_vial_num]), True, "Can't pipet, at least one vial is capped"])

        if self.check_for_errors(error_check_list) == False:
            #Check if has pipet, get one if needed
            if self.HAS_PIPET == False:
                self.get_pipet()
            
            print("Pipetting from vial " + self.get_vial_name(source_vial_num))
            #print("Dispense type: " + dispense_type)
            
            source_vial_clamped = (self.CLAMPED_VIAL == source_vial_num) #Is the source vial clamped?
            source_vial_volume = self.VIAL_DF.at[source_vial_num,'vial volume (mL)']
                    
            #Aspirate from source... Need to adjust aspiration height based off of existing volume
            if source_vial_clamped:
                location = vial_clamp_pip
                
                if track_height:
                    height = self.get_aspirate_height(source_vial_volume, amount_mL, 0.5, 125)
                else:
                    height = 125
            else:
                location = rack_pip[source_vial_num]
                if track_height:
                    height = self.get_aspirate_height(source_vial_volume, amount_mL, 0.5, 76)
                else:
                    height = rack_pip[source_vial_num] #Edit this
             
            height_shift_pipet = self.PIPET_DIMS[1] - self.DEFAULT_DIMS[1] #Adjust height based on the pipet dimensions
            height += height_shift_pipet

            #Move to the correct location and pipet
            self.c9.goto_xy_safe(location)
            #If you want to have extra aspirate and dispense steps
            for i in range (0, asp_disp_cycles):
                self.pipet_from_location(amount_mL, aspirate_speed, height, True)
                self.pipet_from_location(amount_mL, aspirate_speed, height, False)
            self.pipet_from_location(amount_mL, aspirate_speed, height, True)

            #Record the volume change
            original_amount = self.VIAL_DF.at[source_vial_num,'vial volume (mL)']
            self.VIAL_DF.at[source_vial_num,'vial volume (mL)']=(original_amount-amount_mL)

            #Move to a safe height
            if vial_wait_time > 0:
                self.c9.move_z(160, vel=15) 
            
                #Wait above the vial for specified time
                time.sleep(vial_wait_time)
            
            #Wait above the vial for specified time
            time.sleep(vial_wait_time)
        try:
            self.save_vial_status(self.VIAL_FILE)
        except:
            print("Cannot save updated vial status")
    
    #Measure_Weight gives you the option to report the mass dispensed
    def dispense_into_vial(self, dest_vial_num, amount_mL, dispense_speed=11, measure_weight=False):     
        error_check_list = [] #List of specific errors for this method
        error_check_list.append([self.check_if_vials_are_open([dest_vial_num]), True, "Can't pipet, at least one vial is capped"])    
        
        if self.check_for_errors(error_check_list) == False:
            print("Pipetting into vial " + self.get_vial_name(dest_vial_num))
            
            dest_vial_clamped = (self.CLAMPED_VIAL == dest_vial_num) #Is the destination vial clamped?
            dest_vial_volume = self.VIAL_DF.at[dest_vial_num,'vial volume (mL)']

            #Dispense at destination
            if dest_vial_clamped:
                location = vial_clamp_pip
            else:
                location = rack_pip[dest_vial_num]
            height = self.c9.counts_to_mm(3, location[3]) #get the z height of these locations

            height_shift_pipet = self.PIPET_DIMS[1] - self.DEFAULT_DIMS[1] #Adjust height based on the pipet dimensions
            height += height_shift_pipet

            #If the destination vial is at the clamp and you want the weight, measure prior to pipetting
            if measure_weight and dest_vial_clamped:
                initial_mass = self.c9.read_steady_scale()

            self.c9.goto_xy_safe(location, vel=15)
            self.pipet_from_location(amount_mL, dispense_speed, height+20, aspirate = False)

            #Track the added volume in the dataframe
            self.VIAL_DF.at[dest_vial_num,'vial volume (mL)']=self.VIAL_DF.at[dest_vial_num,'vial volume (mL)']+amount_mL

            #If the destination vial is at the clamp and you want the weight, measure after pipetting
            if measure_weight and dest_vial_clamped:
                final_mass = self.c9.read_steady_scale()
            
        #Move to a safe height
        self.c9.move_z(height+60, vel=15) #dispense is always near the top

        if measure_weight:
            measured_mass = final_mass - initial_mass  
            return measured_mass
        else:
            return True

    def dispense_into_wellplate(self, dest_wp_num_array, amount_mL_array, dispense_type = "None", dispense_speed=15,wait_time=1):
        """
        To pipette from source vial into well plate given the source_vial_num (from dataframe & txt file), dest_wp_num (array of numbers for wellplate coordinates), amount_mL (
        amount to be dispensed PER WELL!), replicates, dispense_type options ("touch", "drop-touch", "drop", "slow")
        
        amount_mL is PER WELL!!!

        Need to provide dest_wp_num in a list, to accomodate for multiple repeats~ (so only change pipetting once)

        Returns True when pipetting is complete

        """
        #Dispense at wellplate
        for i in range(0, len(dest_wp_num_array)):    

            location = well_plate_new_grid[dest_wp_num_array[i]] #Where are we dispensing
            amount_mL = amount_mL_array[i] #What amount for this well

            height = self.c9.counts_to_mm(3, location[3])
        
            height_shift_pipet = self.PIPET_DIMS[1] - self.DEFAULT_DIMS[1] #Adjust height based on the pipet dimensions
            height += height_shift_pipet    
            
            #amount_mL = amounts[i] #*Uncomment if want to dispense multiple amounts in same move(demo)

            if i == 0:
                self.c9.goto_xy_safe(location, vel=5)
                #print("Z before dispense", self.c9.counts_to_mm(3, self.c9.get_axis_position(3))) #z-axis value
                #print("x at location", self.get_x_mm()) #x-axis value
                
            else: #need to update location with new height for the different pipette tips (so it doesn't keep going up to safe height and back down)
                location_copy = location.copy()
                location_copy[3] = self.c9.mm_to_counts(3, height) #to use the adjusted height & not the default one (we adjust the height in this function)
                self.c9.goto(location_copy, vel=5) #doesn't move the height
                
            #self.move_rel_x(2) #This shouldn't need to happen... Something is up

            if dispense_type.lower() == "drop-touch" or dispense_type.lower() == "touch": #move lower & towards side of well before dispensing
                height -= 5 #goes 5mm lower when dispensing
                
                #print("x before dispense", self.get_x_mm())

            self.pipet_from_location(amount_mL, dispense_speed, height, False) #dispense
            time.sleep(wait_time)
            
            #print("Z at dispense", self.c9.counts_to_mm(3, self.c9.get_axis_position(3)))
            print("Transfering", amount_mL, "mL into well #" + str(dest_wp_num_array[i]))

            if dispense_type.lower() == "drop" and i == len(dest_wp_num_array)-1: #last replicate
                #self.c9.default_vel=40
                self.c9.move_z(125) #move to safe height
                time.sleep(1)
                print("last drop")
                
            elif dispense_type.lower() == "drop" or dispense_type.lower() == "slow":
                time.sleep(2)
                
            elif dispense_type.lower() == "drop-touch":
                self.move_rel_x(-1.75) #-1.75 for quartz WP, -3 for PS
                time.sleep(0.2)
                self.move_rel_z(5) #move back up before moving (goto() -- unsafe) to next well
            
            elif dispense_type.lower() == "touch":
                self.move_rel_x(2.5) #1.75 for quartz WP, 3 for PS
                time.sleep(0.2)
                self.move_rel_z(10)           
                
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
    def return_vial_from_clamp(self):
        print("Moving vial " + self.get_vial_name(self.CLAMPED_VIAL) + " from clamp")

        error_check_list = [] #List of specific errors for this method
        error_check_list.append([self.GRIPPER_STATUS, "Open", "Cannot return vial from clamp, gripper full"])
        error_check_list.append([self.HAS_PIPET, False, "Cannot return vial from clamp, robot holding pipet"])
        error_check_list.append([str(self.CLAMPED_VIAL).isnumeric(), True, "Cannot return vial from clamp, no vial in clamp"])
        error_check_list.append([self.check_if_vials_are_open([self.CLAMPED_VIAL]), False, "Can't return vial, vial is uncapped"])

        if self.check_for_errors(error_check_list) == False:
            self.goto_location_if_not_there(vial_clamp) #Maybe check if it is already there or not 
            self.c9.close_gripper() #Grab vial
            self.c9.open_clamp() #unclamp vial
            self.c9.goto_safe(rack[int(self.CLAMPED_VIAL)]) #Move back to vial rack
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
            try:
                self.save_vial_status(self.VIAL_FILE)
            except:
                print("Cannot save vial file")

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
            self.c9.cap(revs=2, torque_thresh = 1300) #Cap the vial #Cap the vial
            self.c9.open_gripper() #Open the gripper to release the cap
            self.GRIPPER_STATUS = "Open"

            #self.OPEN_VIALS.remove(self.CLAMPED_VIAL)
            self.VIAL_DF.at[self.CLAMPED_VIAL, 'open']=False
            #print(self.VIAL_DF)

            try:
                self.save_vial_status(self.VIAL_FILE)
            except:
                print("Cannot save vial file")

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

    def move_rel_x(self, x_distance:float):
        """
        Move x_distance (in mm) along the x-axis. Relative position (not absolute). Left is positive, right is negative.
        """
        current_loc_mm = self.c9.n9_fk(self.c9.get_axis_position(0), self.c9.get_axis_position(1), self.c9.get_axis_position(2))
        #tuple (x (mm), y (mm), theta (mm))
        target_x =  current_loc_mm[0] + x_distance
        self.c9.move_xy(target_x, current_loc_mm[1])
    
    def move_rel_y(self, y_distance:float):
        """
        Move y_distance (in mm) along the y-axis. Relative position (not absolute). Forward is positive, backwards is negative.
        """
        current_loc_mm = self.c9.n9_fk(self.c9.get_axis_position(0), self.c9.get_axis_position(1), self.c9.get_axis_position(2))
        #tuple (x (mm), y (mm), theta (mm))
        target_y =  current_loc_mm[1] + y_distance
        self.c9.move_xy(current_loc_mm[0], target_y)
        
    def move_rel_z(self, z_distance:float):
        """
        Move z_distance (in mm) along z-axis. Positive z_distance = up
        """
        curr_z = self.c9.counts_to_mm(3, self.c9.get_axis_position(3))
        target_z = curr_z + z_distance
        self.c9.move_z(target_z)
        
    def get_x_mm(self) -> float:
        current_loc_mm = self.c9.n9_fk(self.c9.get_axis_position(0), self.c9.get_axis_position(1), self.c9.get_axis_position(2))
        return current_loc_mm[0]
    
    def save_vial_status(self,file):
        self.VIAL_DF.to_csv(file, index=False,sep='\t')

    def save_pipet_status(self,file):
        save_data = ','.join(map(str, self.PIPET_ARRAY.flatten()))
        with open(file, "w") as output:
            output.write(save_data.replace('\0', ''))

    def reset_robot(self):
        self.c9.open_gripper()
        self.c9.open_clamp()
        self.remove_pipet()