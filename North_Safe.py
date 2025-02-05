from north import NorthC9
from Locator import *
import numpy as np
import time
import math
import pandas as pd
import sys
import slack_agent
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

    #Source wellplate stack constants
    SOURCE_X = 2950
    SOURCE_Y = [82800, 76500, 69800, 63400, 57000, 50300, 43800]

    SAFE_MOVE_SOURCE_X = 22000
    SAFE_MOVE_SOURCE_Y = 4000 #up before moving to SAFE_MOVE_SOURCE_X

    num_source = 0 #number of wellplates in source stack
    well_plate_df = None

    #Let's initialize the number of well plates from a file
    def __init__(self, c9):
        self.c9 = c9
        self.well_plate_df = pd.read_csv("../utoronto_demo/status/wellplate_storage_status.txt", sep=r",", engine="python")
        #print(self.well_plate_df)
        self.num_source = int(self.well_plate_df.loc[self.well_plate_df['Location']=='Input']['Status'].values)
        #print(self.num_source)

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
    
    def get_next_WP_from_source(self): 
        #TODO: change velocities back to default ones
        #TODO: continue adding heights after 7 WP
        #TODO: save the new num_source to external file?

        if self.num_source>0: #have at least one WP in stack
            print(f"Getting Wellplate #{self.num_source} from source stack at Y: {self.SOURCE_Y[self.num_source-1]}")
            self.c9.move_axis(6, 0, vel=20)
            self.open_gripper() 

            #to source wp stack
            self.c9.move_axis(7, self.SOURCE_X, vel=30)
            self.c9.move_axis(6, self.SOURCE_Y[self.num_source-1], vel=30)
            self.close_gripper()

            #up to "safe" area and move down 
            self.c9.move_axis(6, self.SAFE_MOVE_SOURCE_Y, vel=30)
            self.c9.move_axis(7, self.SAFE_MOVE_SOURCE_X, vel=30)

            self.num_source -= 1
            
            self.well_plate_df.loc[self.well_plate_df['Location']=='Input','Status'] = self.num_source
            print(self.well_plate_df)
            self.well_plate_df.to_csv("../utoronto_demo/status/wellplate_storage_status.txt", index=False,sep=',')
            print(self.num_source)

            print(f"# of wellplates remaining in source stack: {self.num_source}")
        else:
            print("Cannot get wellplate from empty stack")

class North_Robot:
    #Global variables, must be referenced with 
    GRIPPER_STATUS = "Open"
    CLAMPED_VIAL = "None"
    PR_VIALS = [None, None]
    
    VIAL_NAMES = [] #List of vials that do not have caps. These cannot be moved
    VIAL_DF = None #Dataframe tracking vial information
    VIAL_FILE = None
    
    #Pipet rack data
    PIPET_DIMS = [] #Item 0 = Length, Item 1 = Distance from top of rack
    BLUE_DIMS = [20,77] #Large blue tips (mm)
    DEFAULT_DIMS = [25,85] #Default Tips from MT (mm)
    FILTER_DIMS = [24,98] #Large filter tips (mm)

    HAS_PIPET = False
    PIPET_FILE = None
    LOWER_PIPET_ARRAY_INDEX = 0 #Track the status of the pipets used (in the lower rack)
    HIGHER_PIPET_ARRAY_INDEX = 1 #Track the status of the pipets used (in the upper rack)
    PIPETS_USED = [0,0] #Tracker for each rack... This may become deprecated
    DEFAULT_SMALL_TIP_DELTA_Z = -20 #TODO: This value needs to be measured
    HELD_PIPET_INDEX = None   

    #Controller
    c9 = None
   
    #Initialize function
    def __init__(self, c9,vial_file=None,pipet_file=None):
        print("Initializing North Robot")
        self.c9 = c9
        try:
            self.VIAL_FILE = vial_file
            self.VIAL_DF = pd.read_csv(vial_file, sep=r"\t", engine="python")
            self.VIAL_DF.index = self.VIAL_DF['vial index'].values
            print(self.VIAL_DF)
            #print(self.VIAL_DF['vial volume (mL)'])
            self.VIAL_NAMES = self.VIAL_DF['vial name']
            #print(self.VIAL_NAMES)
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
                self.PIPETS_USED[self.LOWER_PIPET_ARRAY_INDEX]=int(pipets[0])
                self.PIPETS_USED[self.HIGHER_PIPET_ARRAY_INDEX]=int(pipets[1])
            #self.PIPET_ARRAY = pipets.astype(np.int16)
        except:    
            print("Cannot open pipet file... using default sequence")

    #Reset positions and get rid of any pipet tips
    def reset_after_initialization(self):
        self.c9.home_pump(0) #Home the pump
        #self.c9.home_carousel() #Home the carousel
        self.c9.open_gripper()  #In case there's something
        self.c9.zero_scale() #Zero the scale
        self.c9.move_z(292)
        self.remove_pipet() #Remove the pipet, if there's one for some reason   

    #Change the speed of the robot. Note: I think this function is rarely used. 
    def set_robot_speed(self, speed_in):
        self.c9.default_vel = speed_in

    #Adjust the pipet tips used and also which rack you are drawing from. Note: This could end up deprecated depending on how we do the data structures. 
    def set_pipet_tip_type(self, pipet_dims, active_rack):
        self.ACTIVE_PIPET_RACK = active_rack
        self.PIPET_DIMS = pipet_dims

    #Remove the pipet tip
    def remove_pipet(self):
        print("Removing pipet")
        self.c9.goto_safe(p_remove_approach,vel=30)
        if self.HELD_PIPET_INDEX==self.LOWER_PIPET_ARRAY_INDEX:
            self.c9.goto(p_remove_cap, vel=5)
        elif self.HELD_PIPET_INDEX==self.HIGHER_PIPET_ARRAY_INDEX:
            self.c9.goto(p_remove_small, vel=5)
        remove_pipet_height = 292 #Constant height to remove the pipet (doesn't change with the pipet type, just moving up)
        self.c9.move_z(remove_pipet_height, vel=20)
        self.HAS_PIPET = False

    #Take a pipet tip from the active rack with the active pipet tip dimensions 
    def get_pipet(self, pipet_rack_index=LOWER_PIPET_ARRAY_INDEX):
        error_check_list = [] #List of specific errors for this method
        error_check_list.append([self.HAS_PIPET, False, "Can't get pipet, already have pipet tip"])

        if self.check_for_errors(error_check_list) == False:
            
            active_pipet_num = self.PIPETS_USED[pipet_rack_index] #First available pipet

            #This is to pause the program and send a slack message when the pipets are out!
            MAX_PIPETS=47
            if active_pipet_num > MAX_PIPETS:
                slack_agent.send_slack_message("The North Robot is out of pipets! Please refill pipets then hit enter on the terminal!")
                input("Paused! Please refill ALL pipets then hit enter!")
                self.PIPETS_USED=[0,0]
                self.save_pipet_status(self.PIPET_FILE)
                active_pipet_num=0

            num = (active_pipet_num%16)*3+math.floor(active_pipet_num/16)
            print(f"Getting pipet number: {active_pipet_num} from rack {pipet_rack_index}")

            #First move to the xy location 
            #TODO: The higher_grid needs to be defined as a location
            if pipet_rack_index == self.LOWER_PIPET_ARRAY_INDEX:
                location = p_capture_grid[num]
            elif pipet_rack_index == self.HIGHER_PIPET_ARRAY_INDEX: 
                location = p_capture_high[num]

            self.c9.goto_xy_safe(location)

            #Second move to z location, based off of the height
            base_height = self.c9.counts_to_mm(3, location[3])
            self.c9.move_z(base_height - self.DEFAULT_DIMS[0] + self.PIPET_DIMS[0]) #Adjust height based off of the distance from the default tip (which the measurements were done for)

            #We have a pipet. What kind of pipet do we have? How many pipets are left in the rack?
            self.HAS_PIPET = True
            self.HELD_PIPET_INDEX = pipet_rack_index
            self.PIPETS_USED[pipet_rack_index] += 1

            #self.PIPET_ARRAY = np.delete(self.PIPET_ARRAY, 0)
            self.save_pipet_status(self.PIPET_FILE)
            #print ("Pipets: ", self.PIPETS_USED) 

    #The only actual method where the pump is called to aspirate or dispense
    def pipet_from_location(self, amount, pump_speed, height, aspirate=True, move_speed=15, initial_move=True):
        if initial_move:
            self.c9.move_z(height, vel=move_speed)
        #self.c9.set_pump_speed(0, pump_speed)
        if aspirate:
            if amount <= 1:
                try:
                    self.c9.aspirate_ml(0,amount)
                except:
                    print("Cannot aspirate (North Safe)... See previous error")
            else:
                print("Cannot aspirate more than 1 mL")
        else:
            try:
                self.c9.dispense_ml(0,amount)
            except: #If there's not enough to dispense
                self.c9.move_pump(0,0)

        #raise RuntimeError(f'Cannot move pump to {new_pos}, pump too empty/full to dispense {ml} mL')

    #Default selection of pipet tip... Make extensible in the future
    def select_pipet_tip(self, volume, specified_tip):
        if specified_tip is None:
            if volume <= 0.200:
                pipet_rack_index=self.HIGHER_PIPET_ARRAY_INDEX
            elif volume <= 1.0:
                pipet_rack_index=self.LOWER_PIPET_ARRAY_INDEX
            else:
                print("Cannot get tip automatically as amount is out of range")
        else:
            pipet_rack_index=specified_tip
        return pipet_rack_index
    
    #Check if the aspiration volume is within limits... Make extensible in the future
    def aspiration_volume_unacceptable(self,amount_mL):
        error_check_list = []
        error_check_list.append([self.HELD_PIPET_INDEX==self.HIGHER_PIPET_ARRAY_INDEX and amount_mL>=0.25,False,"Can't pipet more than 0.25 mL from small pipet"])
        error_check_list.append([self.HELD_PIPET_INDEX==self.HIGHER_PIPET_ARRAY_INDEX and amount_mL<0.01,False,"Can't pipet less than 10 uL from small pipet"])
        error_check_list.append([self.HELD_PIPET_INDEX==self.LOWER_PIPET_ARRAY_INDEX and amount_mL>=1.00,False,"Can't pipet more than 1.00 mL from large pipet"])
        error_check_list.append([self.HELD_PIPET_INDEX==self.LOWER_PIPET_ARRAY_INDEX and amount_mL<0.025,False,"Can't pipet less than 25 uL from large pipet"])
        return self.check_for_errors(error_check_list,True) #Return True if issue

    #Integrating error messages and deliberate pauses
    def pause_after_error(self,err_message,send_slack=True):
        print(err_message)
        if send_slack:
            slack_agent.send_slack_message(err_message)
        input("Waiting for user to press enter or quit after error...")

    #Pipet from a vial into another vial
    #Use calibration (not implemented) is if you want to adjust the volume based off a known calibration
    #Aspirate conditioning is an alternate way to aspirate (up and down some number of cycles)
    def aspirate_from_vial(self, source_vial_num, amount_mL, move_to_aspirate=True, specified_tip=None, track_height=True, vial_wait_time=0, aspirate_speed=11,asp_disp_cycles=0):
        
        error_check_list = [] #List of specific errors for this method
        error_check_list.append([self.is_vial_pipetable(source_vial_num), True, "Can't pipet from vial. Vial may be marked as closed."])

        #Move the vial to clamp if needed to aspirate
        if self.check_for_errors(error_check_list):
            print("Moving vial to clamp to uncap")
            self.move_vial_to_clamp(source_vial_num)
            self.uncap_clamp_vial()        

        #Check if has pipet, get one if needed based on volume being aspirated (or if tip is specified)
        if self.HAS_PIPET == False:
            pipet_rack_index = self.select_pipet_tip(amount_mL,specified_tip)           
            self.get_pipet(pipet_rack_index)
        
        #Check for an issue with the pipet and the specified amount
        if self.HAS_PIPET == True:
            if self.aspiration_volume_unacceptable(amount_mL):
                return False

        #Get current volume
        source_vial_volume = self.VIAL_DF.at[source_vial_num,'vial volume (mL)']

        #Reject aspiration if the volume is not high enough
        if source_vial_volume < amount_mL:
            self.pause_after_error("Cannot aspirate more volume than in vial")
            return False

        print("Pipetting from vial " + self.get_vial_name(source_vial_num) + ", amount: "  + str(amount_mL) + " mL")
        
        #The type of rack is relevant for the volume adjustment, but this should probably be streamlined in the future
        source_vial_rack = self.VIAL_DF.at[source_vial_num,'rack'] #A or B
        if source_vial_rack=='A':
            height_volume_constant=6
        elif source_vial_rack=='B':
            height_volume_constant=2

        #Where are we pipetting from?  
        location,base_height = self.get_vial_pipetting_location(source_vial_num)

        #Adjust height based on the amount that is in the vial
        if track_height:
            height = self.get_aspirate_height(source_vial_volume,amount_mL,base_height,height_volume_constant)
        else:
            height = base_height #Go to the minimum height
  
        #Adjust for different pipet type
        height = self.adjust_height_based_on_pipet_held(height)

        print("Aspirate height: ", height)

        #TODO: Check to make sure we aren't going too low for the small pipet tips only. Ideally we wouldn't need this. 
        if self.HELD_PIPET_INDEX==self.HIGHER_PIPET_ARRAY_INDEX:
            MIN_SMALLPIP_HEIGHT_VIAL_RACK = 60.70 #At  ~3 mL
            MIN_SMALLPIP_HEIGHT_VIAL_RACK_LEFT_EDGE = 47.80 #At ~0.5 mL
            MIN_SMALLPIP_HEIGHT_CLAMP = 92.03 #At ~0 mL
            MIN_SMALLPIP_HEIGHT_PR = 64.28 #At ~ 2 mL

        #Move to the correct location and pipet
        if move_to_aspirate:
            self.c9.goto_xy_safe(location,vel=15)
        
        #If you want to have extra aspirate and dispense steps
        for i in range (0, asp_disp_cycles):
            self.pipet_from_location(amount_mL, aspirate_speed, height, True)
            self.pipet_from_location(amount_mL, aspirate_speed, height, False)
        
        #Main aspiration
        self.pipet_from_location(amount_mL, aspirate_speed, height, True, initial_move=move_to_aspirate)

        #Record the volume change
        original_amount = self.VIAL_DF.at[source_vial_num,'vial volume (mL)']
        self.VIAL_DF.at[source_vial_num,'vial volume (mL)']=(original_amount-amount_mL)

        #Move to a safe height and wait if required
        if vial_wait_time > 0:
            self.c9.move_z(160, vel=15) 
            time.sleep(vial_wait_time)
        
        #Update the new volume in memory
        try:
            self.save_vial_status(self.VIAL_FILE)
        except:
            print("Cannot save updated vial status")

        return True
    
    def adjust_height_based_on_pipet_held(self,height):
        height_shift_pipet = 0
        if self.HELD_PIPET_INDEX == self.LOWER_PIPET_ARRAY_INDEX:
            height_shift_pipet = self.PIPET_DIMS[1] - self.DEFAULT_DIMS[1] #Adjust height based on the pipet dimensions
        elif self.HELD_PIPET_INDEX == self.HIGHER_PIPET_ARRAY_INDEX:
            height_shift_pipet = self.DEFAULT_SMALL_TIP_DELTA_Z #Adjust height based on difference from default dims
        height += height_shift_pipet
        return height

    #TODO: Move the vial to a new area, for example to make it more accessible. 
    def move_vial(initial_index, final_index):
        None

    #This method dispenses from a vial into another vial, using buffer transfer to improve accuracy if needed.
    def dispense_from_vial_into_vial(self,source_vial_index,dest_vial_index,volume,move_to_aspirate=True,move_to_dispense=True,buffer_vol=0.02):
        if volume < 0.2 and volume >= 0.02:
            tip_type = self.HIGHER_PIPET_ARRAY_INDEX
            max_volume = 0.25
        elif volume >= 0.2 and volume <= 1.00:
            tip_type = self.LOWER_PIPET_ARRAY_INDEX
            max_volume = 1.00
        else:
            print(f"Cannot accurately dispense: {volume} mL")
            return False
        
        extra_aspirate = 0
        if max_volume-volume >= 2*buffer_vol:
            extra_aspirate = 2*buffer_vol
        
        self.aspirate_from_vial(source_vial_index,volume+extra_aspirate,move_to_aspirate=move_to_aspirate,specified_tip=tip_type)
        if extra_aspirate > 0:
            self.dispense_into_vial(source_vial_index,buffer_vol,initial_move=False)
        
        self.dispense_into_vial(dest_vial_index,volume,initial_move=move_to_dispense)
        
        if extra_aspirate > 0:
            self.dispense_into_vial(source_vial_index,buffer_vol,initial_move=move_to_dispense)

    #TODO add error checks and safeguards
    def pipet_from_wellplate(self,wp_index,volume,aspirate_speed=10,aspirate=True,move_to_aspirate=True):
        location = well_plate_new_grid[wp_index]
        height = self.c9.counts_to_mm(3, location[3])
        height = self.adjust_height_based_on_pipet_held(height) 
        if aspirate:
            height = height - 12 #Go to the bottom of the well

        if move_to_aspirate:
                self.c9.goto_xy_safe(location, vel=15)

        self.pipet_from_location(volume, aspirate_speed, height, aspirate = aspirate, initial_move=move_to_aspirate)

    #Dispense an amount into a vial
    def dispense_into_vial(self, dest_vial_num,amount_mL,initial_move=True,dispense_speed=11,measure_weight=False):     
        error_check_list = [] #List of specific errors for this method
        error_check_list.append([self.is_vial_pipetable(dest_vial_num), True, "Can't pipet, at least one vial is capped"])    
        print("Pipetting into vial " + self.get_vial_name(dest_vial_num) + ", amount: " + str(amount_mL) + " mL")

        successful_dispense = False
        measured_mass = None
        if self.check_for_errors(error_check_list) == False:
            dest_vial_clamped = (self.CLAMPED_VIAL == dest_vial_num) #Is the destination vial clamped?
            dest_vial_volume = self.VIAL_DF.at[dest_vial_num,'vial volume (mL)'] #What is the current vial volume

            #If the destination vial is at the clamp and you want the weight, measure prior to pipetting
            if measure_weight and dest_vial_clamped:
                initial_mass = self.c9.read_steady_scale()

            #Where is the vial?
            location,base_height = self.get_vial_pipetting_location(dest_vial_num)
            
            #What height do we need to go to?
            height = self.c9.counts_to_mm(3, location[3]) #baseline z-height
            height = self.adjust_height_based_on_pipet_held(height)

            #Move to the location if told to (Could change this to an auto-check)
            if initial_move:               
                self.c9.goto_xy_safe(location, vel=15)   
            
            #Pipet into the vial
            self.pipet_from_location(amount_mL, dispense_speed, height, aspirate = False, initial_move=initial_move)

            #Track the added volume in the dataframe
            self.VIAL_DF.at[dest_vial_num,'vial volume (mL)']=self.VIAL_DF.at[dest_vial_num,'vial volume (mL)']+amount_mL

            #If the destination vial is at the clamp and you want the weight, measure after pipetting
            if measure_weight and dest_vial_clamped:
                final_mass = self.c9.read_steady_scale()
                measured_mass = final_mass - initial_mass  

            successful_dispense = True

        return successful_dispense,measured_mass

    def dispense_into_wellplate(self, dest_wp_num_array, amount_mL_array, dispense_type = "None", dispense_speed=11,wait_time=1):
        #Dispense at wellplate
        first_dispense = True
        for i in range(0, len(dest_wp_num_array)):    
            try:
                location = well_plate_new_grid[dest_wp_num_array[i]] #Where are we dispensing
            except:
                location = well_plate_new_grid[self.convert_well_into_index(dest_wp_num_array[i])]

            amount_mL = amount_mL_array[i] #What amount for this well

            if amount_mL == 0:
                continue

            height = self.c9.counts_to_mm(3, location[3])
            height = self.adjust_height_based_on_pipet_held(height) 

            if first_dispense:
                self.c9.goto_xy_safe(location, vel=15)
                first_dispense = False
                
            else:
                print(height)
                self.c9.goto_xy_safe(location, vel=5, safe_height=height)

            if dispense_type.lower() == "drop-touch" or dispense_type.lower() == "touch": #move lower & towards side of well before dispensing
                height -= 5 #goes 5mm lower when dispensing

            if self.HELD_PIPET_INDEX == self.HIGHER_PIPET_ARRAY_INDEX:
                dispense_speed = 8 #Use lower dispense speed for smaller tip

            print("Transfering", amount_mL, "mL into well #" + str(dest_wp_num_array[i]))

            #Dispense and then wait
            self.pipet_from_location(amount_mL, dispense_speed, height, False)
            time.sleep(wait_time)

            #OAM Notes: This code does not appear extensible but keep for now. 
            if dispense_type.lower() == "drop" and i == len(dest_wp_num_array)-1: #last replicate
                self.c9.move_z(125) #move to safe height
                time.sleep(1)
                
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
            
        return True

    def dispense_from_vials_into_wellplate(self, well_plate_df, vial_indices, low_volume_cutoff=0.05, buffer_vol=0.02):

        #Step 1: Determine which vials correspond to the columns in well_plate_df, make sure that there's enough liquid in each

        well_plate_df_low = well_plate_df.where(well_plate_df<low_volume_cutoff).fillna(0) # Create a new DataFrame with values below the cutoff
        well_plate_df_high = well_plate_df.mask(well_plate_df < low_volume_cutoff, 0) #Create a dataframe where the values are above the cutoff

        well_plate_instructions = [[well_plate_df_high,1.0,self.LOWER_PIPET_ARRAY_INDEX],[well_plate_df_low,0.25,self.HIGHER_PIPET_ARRAY_INDEX]] #Magic numbers for now

        for well_plate_instruction in well_plate_instructions:
            
            well_plate_df = well_plate_instruction[0]
            max_volume = well_plate_instruction[1]
            pipet_index = well_plate_instruction[2]
            
            well_plate_dispense_2d_array = well_plate_df.values
            well_plate_indices = well_plate_df.index.tolist() #TODO: This should be more flexible potentially
            print("Well plate indices: ", well_plate_indices)

            #Step 2: Systematically dispense based one liquid at a time
            for i in range (0, len(vial_indices)):

                vial_index = vial_indices[i]          

                vol_needed = np.sum(well_plate_dispense_2d_array[:, i])

                vial_open = self.is_vial_pipetable(vial_index)

                if not vial_open:
                    print(f"Vial {vial_index} not open, moving to clamp")
                    self.move_vial_to_clamp(vial_index)
                    self.uncap_clamp_vial()
                
                print("Aspirating from Vial: ", vial_index)
                print("Total volume needed: ", vol_needed)
                last_index = 0
                vol_dispensed = 0
                vol_remaining=0
                while round(vol_dispensed,3) < round(vol_needed,3) and vol_needed>0:
                    dispense_vol=0
                    dispense_array = []
                    well_plate_array = []
                    processing=True
                    while processing:
                        try:
                            volume = round(well_plate_dispense_2d_array[last_index,i],3)

                            if dispense_vol+volume<=max_volume:
                                dispense_vol+=volume
                                dispense_array.append(round(float(volume),3))
                                well_plate_array.append(well_plate_indices[last_index])
                                last_index+=1
                            else:
                                processing=False
                        except:
                            processing=False
                    print(f"Amount to Dispense:{dispense_vol}")
            
                    extra_aspirate_vol=0 #Might be nice to aspirate a bit extra if possible, especially for many dispenses
                    sacrificial_dispense_vol=0 #Amount to dispense back
                    if vol_dispensed == 0:
                        extra_aspirate_vol = min(max_volume-dispense_vol,buffer_vol*2) #Either 20 uL or MAX-DISPENSE (which could be as low as 0 uL)
                        if extra_aspirate_vol == buffer_vol*2:
                            sacrificial_dispense_vol=buffer_vol
                    else:
                        extra_aspirate_vol = min(buffer_vol, max_volume-dispense_vol-vol_remaining) #Either 20 uL or less
                        if extra_aspirate_vol == buffer_vol:
                            sacrificial_dispense_vol = extra_aspirate_vol #Just do this if there is enough
                    
                    vol_remaining = vol_remaining+extra_aspirate_vol-sacrificial_dispense_vol #What's leftover?
                    print("Extra Aspiration: ", extra_aspirate_vol)
                    print("Sacrificial Dispense: ", sacrificial_dispense_vol)
                    print("Remaining volume: ", vol_remaining)                  

                    print(f"Aspirating solution {vial_index}: {dispense_vol+extra_aspirate_vol} uL")
                    self.aspirate_from_vial(vial_index,dispense_vol+extra_aspirate_vol,specified_tip=pipet_index)
                    
                    if extra_aspirate_vol > 0:
                        self.dispense_into_vial(vial_index,sacrificial_dispense_vol,initial_move=False)
  
                    #indices to dispense... this is not right
                    #well_plate_array = np.arange((last_index-len(dispense_array)),last_index,1)
                    #well_plate_array = [int(x) for x in well_plate_array]

                    print("Indices dipensed:", well_plate_array)
                    print("Dispense volumes:", dispense_array)
                    print("Dispense sum", np.sum(dispense_array))

                    self.dispense_into_wellplate(well_plate_array,dispense_array)

                    vol_dispensed += dispense_vol

                    print(f"Solution Dispensed {vial_index}: {vol_dispensed} uL")     
                
                print("Vol remaining, returning to vial: ", vol_remaining)
                
                
                if vol_needed>0 and vol_remaining>0:
                    self.dispense_into_vial(vial_index,vol_remaining)
                if vol_needed>0:
                    self.remove_pipet()

                if str(self.CLAMPED_VIAL).isnumeric():
                    self.recap_clamp_vial()
                    self.return_vial_from_clamp()
                
                print()

        return True

    #We will need to check if the vial is capped before moving
    def move_vial_to_clamp(self, vial_num):
        print("Moving vial " + self.get_vial_name(vial_num) + " to clamp")

        error_check_list = [] #List of specific errors for this method
        error_check_list.append([self.GRIPPER_STATUS, "Open", "Cannot move vial to clamp, gripper full"])
        error_check_list.append([self.HAS_PIPET, False, "Cannot move vial to clamp, robot holding pipet"])
        error_check_list.append([str(self.CLAMPED_VIAL).isnumeric(), False, "Cannot move vial to clamp, clamp full"])
        error_check_list.append([self.is_vial_movable(vial_num), True, "Can't move vial, vial is uncapped."])

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
        error_check_list.append([self.is_vial_movable(self.CLAMPED_VIAL), True, "Can't return vial, vial is uncapped"])

        if self.check_for_errors(error_check_list) == False:
            self.goto_location_if_not_there(vial_clamp) #Maybe check if it is already there or not 
            self.c9.close_gripper() #Grab vial
            self.c9.open_clamp() #unclamp vial
            self.c9.goto_safe(rack[int(self.CLAMPED_VIAL)]) #Move back to vial rack
            self.c9.open_gripper() #Release vial
            self.CLAMPED_VIAL = "None"

    #Move a vial to the photoreactor. To do: Add a tracker for whether the reactors are full. 
    def move_vial_to_photoreactor(self, vial_num, reactor_num):
        print("Moving vial " + self.get_vial_name(vial_num) + " to photoreactor " + str(reactor_num))

        location = None
        if reactor_num==1:
            location = photo_reactor_1
        elif reactor_num==2:
            location = photo_reactor_2


        error_check_list = [] #List of specific errors for this method
        error_check_list.append([self.GRIPPER_STATUS, "Open", "Cannot move vial to clamp, gripper full"])
        error_check_list.append([self.HAS_PIPET, False, "Cannot move vial to clamp, robot holding pipet"])
        #error_check_list.append([str(self.CLAMPED_VIAL).isnumeric(), False, "Cannot move vial to clamp, clamp full"])
        error_check_list.append([self.is_vial_movable(vial_num), True, "Can't move vial, vial is uncapped"])

        if self.check_for_errors(error_check_list) == False:
            self.goto_location_if_not_there(rack[vial_num]) #move to vial
            self.c9.close_gripper() #grip vial
            self.c9.goto_safe(location,vel=30) #move vial to clamp
            #self.c9.close_clamp() #clamp vial
            self.c9.open_gripper() #release vial
            #self.CLAMPED_VIAL = vial_num       
            # 
            self.PR_VIALS[reactor_num-1]=vial_num

    #Return a vial from the photoreactor to the tray. To do: Add a tracker for whether the reactors are full. 
    def return_vial_from_photoreactor(self, vial_num, reactor_num):
        print("Moving vial " + self.get_vial_name(vial_num) + " from photoreactor " + str(reactor_num))

        location = None
        if reactor_num==1:
            location = photo_reactor_1
        elif reactor_num==2:
            location = photo_reactor_2

        error_check_list = [] #List of specific errors for this method
        error_check_list.append([self.GRIPPER_STATUS, "Open", "Cannot return vial from clamp, gripper full"])
        error_check_list.append([self.HAS_PIPET, False, "Cannot return vial from clamp, robot holding pipet"])
        #error_check_list.append([str(self.CLAMPED_VIAL).isnumeric(), True, "Cannot return vial from clamp, no vial in clamp"])
        #error_check_list.append([self.check_if_vials_are_open([self.CLAMPED_VIAL]), False, "Can't return vial, vial is uncapped"])

        if self.check_for_errors(error_check_list) == False:
            self.goto_location_if_not_there(location) #Maybe check if it is already there or not 
            self.c9.close_gripper() #Grab vial
            self.c9.open_clamp() #unclamp vial
            self.c9.goto_safe(rack[vial_num]) #Move back to vial rack
            self.c9.open_gripper() #Release vial
            #self.CLAMPED_VIAL = "None"
            self.PR_VIALS[reactor_num-1]=None

    #Uncap the vial in the clamp
    def uncap_clamp_vial(self):
        print ("Removing cap from clamped vial")

        error_check_list = [] #List of specific errors for this method
        error_check_list.append([self.GRIPPER_STATUS, "Open", "Cannot uncap, gripper full"])
        #error_check_list.append([self.HAS_PIPET, False, "Can't uncap vial, holding pipet"])
        error_check_list.append([str(self.CLAMPED_VIAL).isnumeric(), True, "Cannot uncap, no vial in clamp"])
        error_check_list.append([self.is_vial_movable(self.CLAMPED_VIAL), True, "Can't uncap, vial is uncapped already"])

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
        error_check_list.append([self.is_vial_movable(self.CLAMPED_VIAL), False, "Can't recap, vial is capped already"])
        error_check_list.append([str(self.CLAMPED_VIAL).isnumeric(), True, "Cannot recap, no vial in clamp"])
        
        if self.check_for_errors(error_check_list) == False:
            self.goto_location_if_not_there(vial_clamp) #Maybe check if it is already there or not
            self.c9.close_clamp() #Make sure vial is clamped
            self.c9.cap(revs=1.9, torque_thresh = 400) #Cap the vial #Cap the vial
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
            self.c9.goto_safe(location,vel=30)

    #Measurement for how far two points are
    def get_location_distance(self, loc_1, loc_2):
        difference = np.sum(np.absolute(np.array(loc_2)[1:4] - np.array(loc_1)[1:4]))
        return difference
    
    #Removes the target vial, vortexes it, then puts it back
    def vortex_vial(self, vial_num, vortex_time, vortex_speed=70):
        print("Vortexing Vial: " + self.get_vial_name(vial_num))
        vial_clamped = (self.CLAMPED_VIAL == vial_num) #Is the vial clamped?
        
        error_check_list = []
        error_check_list.append([self.GRIPPER_STATUS, "Open", "Can't Vortex, gripper is used"])
        error_check_list.append([self.is_vial_movable(vial_num), True, "Can't vortex, vial not capped"])
        error_check_list.append([self.HAS_PIPET, False, "Can't vortex vial, holding pipet"])
        
        #Correlate vortex_rads to time and speed
        # vortex_rads = (vortex_time-0.158)/9.95E-4*vortex_speed

        # print(vortex_rads)

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

            #self.c9.spin_axis(self.c9.GRIPPER, 1000*vortex_speed, 500000)
            #time.sleep(vortex_time)
            #self.c9.spin_axis(self.c9.GRIPPER, 0, 500000)

            self.c9.move_axis(self.c9.GRIPPER, 1000*vortex_time*vortex_speed, vel=vortex_speed,accel=10000)
            self.c9.reduce_axis_position(axis=self.c9.GRIPPER)
           
            #Return vial
            if vial_clamped:
                self.c9.goto_safe(vial_clamp)
                self.c9.close_clamp()
                self.c9.open_gripper()
            else:
                self.c9.goto_safe(rack[vial_num])
                self.c9.open_gripper()

    #This is just to formalize the process of error checking so its more extensible
    #This may end up deprecated
    #Error check list is as follows error_check[0] is the value, error_check[1] is the target value, error_check[2] is the error message
    def check_for_errors(self, error_check_list, stop_after_error=False):
        error_occured = False
        for error_check in error_check_list:
            if error_check[0] != error_check[1]:
                error_occured = True
                print(error_check[2])
                if stop_after_error:
                    self.pause_after_error(error_check[2])
        return error_occured

   #Check to see if we can move the vial         
    def is_vial_movable(self, vial_index):
        movable = False
        try: #New syntax for the vial file
            movable = (self.VIAL_DF.at[vial_index, 'cap_status'] == "close_cap" or  self.VIAL_DF.at[vial_index, 'cap_status'] == "open_cap")
        except: #Old syntax for the vial file
            movable = not self.VIAL_DF.at[vial_index, 'open']
        return movable
    
    #Check to see if the pipet can have liquids added/removed
    def is_vial_pipetable(self, vial_index):
        pipetable = False
        try: #New syntax for the vial file
            pipetable = (self.VIAL_DF.at[vial_index, 'cap_status'] == "no_cap" or  self.VIAL_DF.at[vial_index, 'cap_status'] == "open_cap")
        except: #Old syntax for the vial file
            pipetable = self.VIAL_DF.at[vial_index, 'open']
        return pipetable

    #Get adjust the aspiration height based on how much is there
    def get_aspirate_height(self, amount_vial, amount_to_withdraw, base_height, height_volume_constant=6, buffer=0.5):      
        target_height = base_height + (height_volume_constant*(amount_vial - amount_to_withdraw - buffer))
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

    #Translate in the x direction
    def move_rel_x(self, x_distance:float):
        """
        Move x_distance (in mm) along the x-axis. Relative position (not absolute). Left is positive, right is negative.
        """
        current_loc_mm = self.c9.n9_fk(self.c9.get_axis_position(0), self.c9.get_axis_position(1), self.c9.get_axis_position(2))
        #tuple (x (mm), y (mm), theta (mm))
        target_x =  current_loc_mm[0] + x_distance
        self.c9.move_xy(target_x, current_loc_mm[1])
    
    #Translate in the y direction
    def move_rel_y(self, y_distance:float):
        """
        Move y_distance (in mm) along the y-axis. Relative position (not absolute). Forward is positive, backwards is negative.
        """
        current_loc_mm = self.c9.n9_fk(self.c9.get_axis_position(0), self.c9.get_axis_position(1), self.c9.get_axis_position(2))
        #tuple (x (mm), y (mm), theta (mm))
        target_y =  current_loc_mm[1] + y_distance
        self.c9.move_xy(current_loc_mm[0], target_y)
        
    #Translate in the z direction    
    def move_rel_z(self, z_distance:float):
        """
        Move z_distance (in mm) along z-axis. Positive z_distance = up
        """
        curr_z = self.c9.counts_to_mm(3, self.c9.get_axis_position(3))
        target_z = curr_z + z_distance
        self.c9.move_z(target_z)
  
    #Save the status of the vials to a file
    def save_vial_status(self,file):
        self.VIAL_DF.to_csv(file, index=False,sep='\t')

    #Save the status of the pipets to a file
    def save_pipet_status(self,file):
        save_data = ','.join(map(str, self.PIPETS_USED))
        with open(file, "w") as output:
            output.write(save_data.replace('\0', ''))

    #Just get everything back to normal
    def reset_robot(self):
        self.c9.move_pump(0,0)
        self.c9.open_gripper()
        self.c9.open_clamp()
        self.HELD_PIPET_INDEX = 0 
        self.remove_pipet()
        self.HELD_PIPET_INDEX = 1
        self.remove_pipet()

    #Remove pipets and move any clamped vials back to their normal positions
    def finish_pipetting(self):
        if self.HAS_PIPET:
            self.remove_pipet()
        if self.GRIPPER_STATUS=="Cap" and str(self.CLAMPED_VIAL).isnumeric():
            self.recap_clamp_vial()
            self.return_vial_from_clamp()

    #Move the robot to the home position    
    def move_home(self):
        self.c9.goto_safe(home)

    def get_vial_name(self,vial_num):
        return self.VIAL_DF.at[vial_num,'vial name']

    #Just a method to find the position of a vial that may be in a different position
    #Better to update this method then have this in the methods that find vials... More extensible
    def get_vial_pipetting_location(self,vial_num):
        #The height at which the pipet touches the ground for the 1 mL pipet
        CLAMP_BASE_HEIGHT = 114.5
        VIAL_RACK_BASE_HEIGHT = 67.25
        PR_BASE_HEIGHT = 80 #Need to fine tune this
        LARGE_VIAL_BASE_HEIGHT = 55    

        vial_clamped = (self.CLAMPED_VIAL == vial_num) #Is the vial clamped?
        vial_rack = self.VIAL_DF.at[vial_num,'rack'] #A or B  
        pr_index = None
        location = None
        min_height = None   

        #Which photoreactor are we in?
        for i in range (0, len(self.PR_VIALS)): 
            if self.PR_VIALS[i] is not None:
                pr_index = i
        if vial_clamped:
                location = vial_clamp_pip
                min_height = CLAMP_BASE_HEIGHT
        elif pr_index is not None:
            if pr_index == 0:
                location = PR_PIP_1
            elif pr_index == 1:
                location = PR_PIP_2
            min_height = PR_BASE_HEIGHT
        else:
            if vial_rack=="A":
                location = rack_pip[vial_num]
                min_height = VIAL_RACK_BASE_HEIGHT
            elif vial_rack=="B":
                location = large_vial_pip[vial_num]
                min_height = LARGE_VIAL_BASE_HEIGHT
        return location,min_height