from torch import clamp
from north import NorthC9
from Locator import *
import numpy as np
import time
import math
import pandas as pd
import sys
import slack_agent
import yaml
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
    CYT_TRAY_Y_GRAB = 8500
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
        self.num_source = int(self.well_plate_df.loc[self.well_plate_df['Location']=='Input']['Status'].values)
        #Load yaml data
        self.reset_after_initialization()
    
    def reset_after_initialization(self):
        None
        #Return well plate if well-plate in gripper
        #Send used well plate to trash
        #Send unused well plate back to source

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

class North_T8:

    t8 = None

    def __init__(self,c9):
        self.t8 = NorthC9('B', network=c9.network)
        self.t8.get_info()
    
    def autotune(self,channel, target_temp):
        self.t8.disable_channel(channel)
        self.t8.set_temp(channel, target_temp)
        self.t8.enable_channel(channel)
        self.t8.temp_autotune(channel, True)

    def set_temp(self,channel,target_temp):
        self.t8.set_temp(channel, target_temp)
        self.t8.enable_channel(channel)
    
    def get_temp(self,channel):
        return self.t8.get_temp(channel)

    def turn_off_heating(self,channel):
        self.t8.disable_channel(channel)

class North_Robot:
    ROBOT_STATUS_FILE = "../utoronto_demo/status/robot_status.yaml" #Store the state of the robot. Update this after every method that alters the state. 

    #What's in the robot's gripper
    GRIPPER_STATUS = None #Could be None, "Cap" or "Open"
    GRIPPER_VIAL_INDEX = None #Could be None or the index of the vial/cap
    
    VIAL_DF = None #Track the status of all vials part of the experiment. 
    VIAL_FILE = None #File that we save the vial data in 
    
    #All of this data doesn't seem to need to be tracked here. TODO: Compartmentalize and place appropriately. 
    DEFAULT_SMALL_TIP_DELTA_Z = -20 #This is the height difference between the bottom of the small pipet tip and the large tip
    LOWER_PIPET_ARRAY_INDEX = 0 #Label representing the lower rack at the back with 1000 uL tips
    HIGHER_PIPET_ARRAY_INDEX = 1 #Label representing the upper rack at the back with 250 uL tips

    HELD_PIPET_INDEX = None #What kind of pipet do we have  
    PIPETS_USED = [0,0] #Tracker for each rack. TODO: This isn't very extensible or innately clear. 
    PIPET_FLUID_VIAL_INDEX = None #What vial was the fluid aspirated from?
    PIPET_FLUID_VOLUME = 0 #What is the held volume in the pipet?

    #Track the pump speed. This seems to help with pump errors. 
    CURRENT_PUMP_SPEED = 11

    #Controller
    c9 = None
   
    #Initialize the status of the robot. 
    def __init__(self,c9,vial_file=None):
        print("Initializing North Robot...")
        self.c9 = c9
        self.VIAL_FILE = vial_file
        self.get_robot_status() #Update the status of the robot from memory
        self.reset_after_initialization() #Reset everything that may not be as desired, eg return to "Home"
        #sys.excepthook = self.global_exception_handler

    #Reset positions and get rid of any pipet tips
    def reset_after_initialization(self):
        print("Physical initialization of North Robot...")
        self.c9.default_vel = 20 #Set the default speed of the robot
        #self.c9.zero_scale() #TODO this is broken
        self.c9.open_clamp()
        
        try: #Try resetting. If a home is required, some of these actions may fail. 
            if self.PIPET_FLUID_VIAL_INDEX is not None and self.PIPET_FLUID_VOLUME > 0: #If we still have liquid leftover
                print("The robot reports having liquid in its tip... Returning that liquid...")
                vial_index = self.PIPET_FLUID_VIAL_INDEX 
                volume = self.PIPET_FLUID_VOLUME
                self.dispense_into_vial(vial_index,volume)
            
            if self.HELD_PIPET_INDEX is not None: #If we've got a pipet tip, let's get rid of it
                print("The robot reports having a tip, removing the tip")
                self.remove_pipet()
            
            if self.GRIPPER_STATUS is not None: #If the gripper is full, let's address that
                vial_index = self.GRIPPER_VIAL_INDEX
                if self.GRIPPER_STATUS == "Cap": #If we have the cap, the vial must be in the gripper.
                    print("The robot reports having a cap in its gripper... Recapping the clamp vial...")
                    self.recap_clamp_vial()
                    self.return_vial_home(vial_index)
                elif self.GRIPPER_STATUS == "Vial": #We need to know where this vial is intended to be!
                    print("The robot reports having a vial in the gripper... Returning that vial home...")
                    location = self.get_vial_info(vial_index,'home_location')
                    location_index = self.get_vial_info(vial_index,'home_location_index')
                    self.drop_off_vial(vial_index,location=location,location_index=location_index)
            self.move_home()
        except Exception as e:
            print("An error occured during initialization, homing components: ", e)
            self.c9.home_carousel() #Home the carousel
            self.c9.home_robot() #Home the robot
            self.c9.home_pump(0) #Home the pump
            for i in range (6,8): #Home the track
                self.c9.home_axis(i)
            self.reset_after_initialization()
        
        self.c9.open_gripper()
        self.c9.home_pump(0)

    #Save the status of the robot to memory
    def save_robot_status(self):
        self.VIAL_DF.to_csv(self.VIAL_FILE, index=False,sep=',') #Save the status of the vial dataframe

        # Robot status data
        robot_status = {
            "gripper_status": self.GRIPPER_STATUS,
            "gripper_vial_index": self.GRIPPER_VIAL_INDEX,
            "held_pipet_index": self.HELD_PIPET_INDEX,
            "pipets_used": {"lower_rack_1": self.PIPETS_USED[self.LOWER_PIPET_ARRAY_INDEX], "upper_rack_1": self.PIPETS_USED[self.HIGHER_PIPET_ARRAY_INDEX]},
            "pipet_fluid_vial_index": self.PIPET_FLUID_VIAL_INDEX,
            "pipet_fluid_volume": float(self.PIPET_FLUID_VOLUME)
        }

        # Writing to a file
        with open(self.ROBOT_STATUS_FILE, "w") as file:
            yaml.dump(robot_status, file, default_flow_style=False)

    #Update the status of the robot from memory
    def get_robot_status(self):
        # Get the vial dataframe
        try:
            self.VIAL_DF = pd.read_csv(self.VIAL_FILE, sep=",")
            self.VIAL_DF.index = self.VIAL_DF['vial_index'].values
        except:
            self.pause_after_error("Issue reading vial status", False)

        # Get the robot status
        with open(self.ROBOT_STATUS_FILE, "r") as file:
            robot_status = yaml.safe_load(file)

        # Convert "None" or "null" strings to actual None. CHATGPT
        def convert_none(value):
            if isinstance(value, dict):
                return {k: convert_none(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [convert_none(v) for v in value]
            elif value in ["None", "null"]:
                return None
            return value

        robot_status = convert_none(robot_status)

        try:
            self.GRIPPER_STATUS = robot_status['gripper_status']
            self.GRIPPER_VIAL_INDEX = robot_status['gripper_vial_index']
            self.HELD_PIPET_INDEX = robot_status['held_pipet_index']
            self.PIPETS_USED[self.LOWER_PIPET_ARRAY_INDEX] = robot_status['pipets_used']['lower_rack_1']
            self.PIPETS_USED[self.HIGHER_PIPET_ARRAY_INDEX] = robot_status['pipets_used']['upper_rack_1']
            self.PIPET_FLUID_VOLUME = robot_status['pipet_fluid_volume']
        except:
            self.pause_after_error("Issue reading robot status", False)

    #Remove the pipet tip
    def remove_pipet(self):
        print("Removing pipet")
        self.c9.goto_safe(p_remove_approach,vel=30)
        #Slightly different removal location depending on what type of tip you have
        if self.HELD_PIPET_INDEX==self.LOWER_PIPET_ARRAY_INDEX:
            self.c9.goto(p_remove_cap, vel=5)
        elif self.HELD_PIPET_INDEX==self.HIGHER_PIPET_ARRAY_INDEX:
            self.c9.goto(p_remove_small, vel=5)
        remove_pipet_height = 292 #Constant height to remove the pipet (doesn't change with the pipet type, just moving up)
        self.c9.move_z(remove_pipet_height, vel=20)
        self.HELD_PIPET_INDEX = None
        self.PIPET_FLUID_VIAL_INDEX = None
        self.PIPET_FLUID_VOLUME = 0
        self.save_robot_status() #Update in memory

    #Take a pipet tip from the active rack with the active pipet tip dimensions 
    def get_pipet(self, pipet_rack_index):
        
        error_check_list = [] #List of specific errors for this method
        error_check_list.append([self.HELD_PIPET_INDEX is None, True, "Can't get pipet, already have pipet tip"])
        self.check_for_errors(error_check_list,True)
            
        active_pipet_num = self.PIPETS_USED[pipet_rack_index] #First available pipet

        #This is to pause the program and send a slack message when the pipets are out!
        MAX_PIPETS=47
        if active_pipet_num > MAX_PIPETS:
            self.pause_after_error("The North Robot is out of pipets! Please refill pipets then hit enter on the terminal!")
            self.PIPETS_USED=[0,0]
            self.save_robot_status()
            active_pipet_num=0

        #This conversion is neccessary to take the tips in the correct order.
        num = (active_pipet_num%16)*3+math.floor(active_pipet_num/16)
        print(f"Getting pipet number: {active_pipet_num} from rack {pipet_rack_index}")

        #First move to the xy location 
        if pipet_rack_index == self.LOWER_PIPET_ARRAY_INDEX:
            location = p_capture_grid[num]
        elif pipet_rack_index == self.HIGHER_PIPET_ARRAY_INDEX: 
            location = p_capture_high[num]

        #Move to get the pipet tip
        self.c9.goto_xy_safe(location)
        base_height = self.c9.counts_to_mm(3, location[3])
        self.c9.move_z(base_height) 

        self.c9.move_z(292,vel=5)

        #We have a pipet. What kind of pipet do we have? How many pipets are left in the rack?
        self.HELD_PIPET_INDEX = pipet_rack_index
        self.PIPETS_USED[pipet_rack_index] += 1

        #Update the status of the robot in memory
        self.save_robot_status()

    #The only actual method where the pump is called to aspirate or dispense
    def pipet_from_location(self, amount, pump_speed, height, aspirate=True, move_speed=15, initial_move=True):
        if initial_move:
            self.c9.move_z(height, vel=move_speed)
        if pump_speed != self.CURRENT_PUMP_SPEED:
            self.c9.set_pump_speed(0, pump_speed)
        if aspirate:
            if amount <= 1:
                try:
                    self.c9.aspirate_ml(0,amount)
                except:
                    self.pause_after_error("Cannot aspirate. Likely a pump position issue", True)
            else:
                self.pause_after_error("Cannot aspirate more than 1 mL", True)
        else:
            try:
                self.c9.dispense_ml(0,amount)
            except: #If there's not enough to dispense
                print("Dispense exceeded limit: Dispensing all liquid")
                self.c9.move_pump(0,0)

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
    def check_if_aspiration_volume_unacceptable(self,amount_mL):
        error_check_list = []
        error_check_list.append([self.HELD_PIPET_INDEX==self.HIGHER_PIPET_ARRAY_INDEX and amount_mL>0.25,False,"Can't pipet more than 0.25 mL from small pipet"])
        error_check_list.append([self.HELD_PIPET_INDEX==self.HIGHER_PIPET_ARRAY_INDEX and amount_mL<0.01,False,"Can't pipet less than 10 uL from small pipet"])
        error_check_list.append([self.HELD_PIPET_INDEX==self.LOWER_PIPET_ARRAY_INDEX and amount_mL>1.00,False,"Can't pipet more than 1.00 mL from large pipet"])
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
    def aspirate_from_vial(self, source_vial_num, amount_mL,move_to_aspirate=True,specified_tip=None,track_height=True,vial_wait_time=0,aspirate_speed=11,asp_disp_cycles=0):
        
        error_check_list = [] #List of specific errors for this method
        error_check_list.append([self.is_vial_pipetable(source_vial_num), True, "Can't pipet from vial. Vial may be marked as closed."])

        #Move the vial to clamp if needed to aspirate
        if self.check_for_errors(error_check_list):
            print("Moving vial to clamp to uncap")
            self.move_vial_to_location(source_vial_num,location='clamp',location_index=0)
            self.uncap_clamp_vial()        

        #Check if has pipet, get one if needed based on volume being aspirated (or if tip is specified)
        if self.HELD_PIPET_INDEX is None:
            pipet_rack_index = self.select_pipet_tip(amount_mL,specified_tip)           
            self.get_pipet(pipet_rack_index)
        
        #Check for an issue with the pipet and the specified amount, pause and send slack message if so
        self.check_if_aspiration_volume_unacceptable(amount_mL) 

        #Get current volume
        source_vial_volume = self.get_vial_info(source_vial_num,'vial_volume')

        #Reject aspiration if the volume is not high enough
        if source_vial_volume < amount_mL:
            self.pause_after_error("Cannot aspirate more volume than in vial")

        print("Pipetting from vial " + self.get_vial_info(source_vial_num,'vial_name') + ", amount: "  + str(amount_mL) + " mL")

        #Where are we pipetting from?  
        location = self.get_vial_location(source_vial_num,True)
        base_height = self.get_min_pipetting_height(source_vial_num)
        vial_type = self.get_vial_info(source_vial_num, 'vial_type')

        #The vial type affects the depth for aspiration. TODO: There should be a data structure for vial types that holds this. 
        if vial_type=='8_mL':
            height_volume_constant=6
        elif vial_type=='20_mL':
            height_volume_constant=2

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
        
        #If you want to have extra aspirate and dispense steps. TODO: This isn't going to work right now as is. 
        for i in range (0, asp_disp_cycles):
            self.pipet_from_location(amount_mL, aspirate_speed, height, True)
            self.pipet_from_location(amount_mL, aspirate_speed, height, False)
        
        #Main aspiration
        self.pipet_from_location(amount_mL, aspirate_speed, height, True, initial_move=move_to_aspirate)

        #Record the volume change
        self.VIAL_DF.at[source_vial_num,'vial_volume']=(source_vial_volume-amount_mL)

        #Move to a safe height and wait if required. TODO: 160 is a magic number and should be replaced with a SAFE_HEIGHT or some constant
        if vial_wait_time > 0:
            self.c9.move_z(160, vel=15) 
            time.sleep(vial_wait_time)
        
        #Update the new volume in memory
        self.PIPET_FLUID_VIAL_INDEX = int(source_vial_num)
        self.PIPET_FLUID_VOLUME += amount_mL
        self.save_robot_status()
    
    #TODO: I've changed this, so that all that matters is the delta_z. Let's make this extensible to other tips in the future
    def adjust_height_based_on_pipet_held(self,height):
        height_shift_pipet = 0
        if self.HELD_PIPET_INDEX == self.HIGHER_PIPET_ARRAY_INDEX:
            height_shift_pipet = self.DEFAULT_SMALL_TIP_DELTA_Z #Adjust height based on difference from default dims
        height += height_shift_pipet
        return height

    #This method dispenses from a vial into another vial, using buffer transfer to improve accuracy if needed.
    def dispense_from_vial_into_vial(self,source_vial_index,dest_vial_index,volume,move_to_aspirate=True,move_to_dispense=True,buffer_vol=0.02):
        if volume < 0.2 and volume >= 0.02:
            tip_type = self.HIGHER_PIPET_ARRAY_INDEX
            max_volume = 0.25
        elif volume >= 0.2 and volume <= 1.00:
            tip_type = self.LOWER_PIPET_ARRAY_INDEX
            max_volume = 1.00
        else:
            self.pause_after_error(f"Cannot accurately aspirate: {volume} mL under specified conditions.")
        
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
        
        measured_mass = None
        error_check_list = [] #List of specific errors for this method
        error_check_list.append([self.is_vial_pipetable(dest_vial_num), True, "Can't pipet, at least one vial is capped"])    
        self.check_for_errors(error_check_list,True) #This will cause a pause if there's an issue

        print("Pipetting into vial " + self.get_vial_info(dest_vial_num,'vial_name') + ", amount: " + str(amount_mL) + " mL")
        
        dest_vial_clamped = self.get_vial_info(dest_vial_num,'location')=='clamp' #Is the destination vial clamped?
        dest_vial_volume = self.get_vial_info(dest_vial_num,'vial_volume') #What is the current vial volume?

        #If the destination vial is at the clamp and you want the weight, measure prior to pipetting
        if measure_weight and dest_vial_clamped:
            initial_mass = self.c9.read_steady_scale()

        #Where is the vial?
        location= self.get_vial_location(dest_vial_num,True)
        
        #What height do we need to go to?
        height = self.c9.counts_to_mm(3, location[3]) #baseline z-height
        height = self.adjust_height_based_on_pipet_held(height)

        #Move to the location if told to (Could change this to an auto-check)
        if initial_move:               
            self.c9.goto_xy_safe(location, vel=15)   
        
        #Pipet into the vial
        self.pipet_from_location(amount_mL, dispense_speed, height, aspirate = False, initial_move=initial_move)

        #Track the added volume in the dataframe
        self.VIAL_DF.at[dest_vial_num,'vial_volume']=self.VIAL_DF.at[dest_vial_num,'vial_volume']+amount_mL
        self.PIPET_FLUID_VOLUME -= amount_mL
        self.save_robot_status()

        #If the destination vial is at the clamp and you want the weight, measure after pipetting
        if measure_weight and dest_vial_clamped:
            final_mass = self.c9.read_steady_scale()
            measured_mass = final_mass - initial_mass  

        return measured_mass

    #Dispense into a series of wells (dest_wp_num_array) a specific set of amounts (amount_mL_array)
    def dispense_into_wellplate(self, dest_wp_num_array, amount_mL_array, dispense_type = "None", dispense_speed=11,wait_time=1):
        #Dispense at wellplate
        first_dispense = True
        for i in range(0, len(dest_wp_num_array)):    
            try:
                location = well_plate_new_grid[dest_wp_num_array[i]] #Where are we dispensing
            except:
                location = well_plate_new_grid[self.convert_well_into_index(dest_wp_num_array[i])]

            amount_mL = amount_mL_array[i] #What amount for this well

            if amount_mL == 0: #Skip empty dispenses
                continue

            height = self.c9.counts_to_mm(3, location[3])
            height = self.adjust_height_based_on_pipet_held(height) 

            if first_dispense:
                self.c9.goto_xy_safe(location, vel=15)
                first_dispense = False
                
            else:
                self.c9.goto_xy_safe(location, vel=5, safe_height=height) #Use safe_height here!

            if dispense_type.lower() == "drop-touch" or dispense_type.lower() == "touch": #move lower & towards side of well before dispensing
                height -= 5 #goes 5mm lower when dispensing

            if self.HELD_PIPET_INDEX == self.HIGHER_PIPET_ARRAY_INDEX:
                dispense_speed = 13 #Use lower dispense speed for smaller tip

            print("Transferring", amount_mL, "mL into well #" + str(dest_wp_num_array[i]))

            #Dispense and then wait
            self.pipet_from_location(amount_mL, dispense_speed, height, False)
            time.sleep(wait_time)

            #OAM Notes: Different techniques. drop and slow could both be just longer wait_time              
            if dispense_type.lower() == "drop-touch":
                self.move_rel_x(-1.75) #-1.75 for quartz WP, -3 for PS
                time.sleep(0.2)
                self.move_rel_z(5) #move back up before moving to next well
            
            elif dispense_type.lower() == "touch":
                self.move_rel_x(2.5) #1.75 for quartz WP, 3 for PS
                time.sleep(0.2)
                self.move_rel_z(10)           

        self.PIPET_FLUID_VOLUME -= np.sum(amount_mL_array) 
        self.save_robot_status()    
        return True

    #This is a custom method that takes a "well_plate_df" as an array of destinations and some "vial_indices" which are the different dispensed liquids
    #This method will use both the large and small tips, with a specified low_volume_cutoff between the two
    #This method does use multiple dispenses per aspiration for efficiency
    def dispense_from_vials_into_wellplate(self, well_plate_df, vial_indices, low_volume_cutoff=0.05, buffer_vol=0.02):

        #Step 1: Determine which vials correspond to the columns in well_plate_df, make sure that there's enough liquid in each
        well_plate_dispense_2d_array=well_plate_df.values
        vols_required = np.sum(well_plate_dispense_2d_array, axis=0)
        print("Total volumes needed: ", vols_required)
        for i in range (0, len(vial_indices)):
            vial = vial_indices[i]
            volume_needed = vols_required[i]
            volume = self.get_vial_info(vial,'vial_volume')
            print(vial)
            print(volume_needed)
            print(volume)
            if volume < volume_needed:
                self.pause_after_error(f"Dispensing from Vials to Wellplate, not enough of solution: {vial}",True)

        #Split the dispenses based on the tip-styles available
        well_plate_df_low = well_plate_df.where(well_plate_df<low_volume_cutoff).fillna(0) # Create a new DataFrame with values below the cutoff
        well_plate_df_high = well_plate_df.mask(well_plate_df < low_volume_cutoff, 0) #Create a dataframe where the values are above the cutoff
        well_plate_instructions = [[well_plate_df_high,1.0,self.LOWER_PIPET_ARRAY_INDEX],[well_plate_df_low,0.25,self.HIGHER_PIPET_ARRAY_INDEX]] #Magic numbers for now

        #For each pipet tip type. Could potentially do this differently
        for well_plate_instruction in well_plate_instructions:
            
            well_plate_df = well_plate_instruction[0] #List of target dispenses
            max_volume = well_plate_instruction[1] #What's the maximum we should aspirate to?
            pipet_index = well_plate_instruction[2] #What pipet are we using?
            
            well_plate_dispense_2d_array = well_plate_df.values
            well_plate_indices = well_plate_df.index.tolist() #This is the list of wells that are being dispensed to
            print("Well plate indices: ", well_plate_indices)

            #Step 2: Systematically dispense based one liquid at a time
            for i in range (0, len(vial_indices)):

                vial_index = vial_indices[i] #This is the vial_index, not the location         
                vol_needed = np.sum(well_plate_dispense_2d_array[:, i]) #What's the total volume required here?
                vial_open = self.is_vial_pipetable(vial_index) #Can we pipet from the target vial?

                print("Aspirating from Vial: ", vial_index)
                print("Total volume needed: ", vol_needed)

                #Open the vial at the clamp to pipet if need be
                return_vial=False
                if not vial_open:
                    print(f"Vial {vial_index} not open, moving to clamp")
                    self.move_vial_to_location(vial_index,location='clamp',location_index=0)
                    self.uncap_clamp_vial()
                    return_vial=True
                
                last_index = 0
                vol_dispensed = 0
                vol_remaining=0
                #This loops until we don't need to dispense anymore. The rounding is to make sure the loop ends.
                while round(vol_dispensed,3) < round(vol_needed,3) and vol_needed>0:
                    dispense_vol=0
                    dispense_array = []
                    well_plate_array = []
                    processing=True
                    #Determines how to maximize the dispenses in one pipet tip. 
                    while processing:
                        try:
                            volume = round(well_plate_dispense_2d_array[last_index,i],3) #Round to the nearest uL
                            if dispense_vol+volume<=max_volume:
                                dispense_vol+=volume
                                dispense_array.append(volume)
                                well_plate_array.append(well_plate_indices[last_index])
                                last_index+=1
                            else:
                                processing=False
                        except:
                            processing=False
                    print(f"Amount to Dispense:{dispense_vol}")
            
                    #These steps are to calculate whether or not we should add a buffer, or to aspirate/dispense a bit to improve pipeting accuracy. 
                    extra_aspirate_vol=0 #Might be nice to aspirate a bit extra if possible, especially for many dispenses
                    sacrificial_dispense_vol=0 #Amount to dispense back
                    if vol_dispensed == 0:
                        extra_aspirate_vol = min(max_volume-dispense_vol,buffer_vol*2) #At the start, get 2x buffer, then dispense back a bit (if possible)
                        if extra_aspirate_vol == buffer_vol*2:
                            sacrificial_dispense_vol=buffer_vol
                    else:
                        extra_aspirate_vol = min(buffer_vol, max_volume-dispense_vol-vol_remaining) #Either 1x buffer (to put right back), or some negative value (in case we need to go close to max_volume)
                        if extra_aspirate_vol == buffer_vol:
                            sacrificial_dispense_vol = extra_aspirate_vol #Just do this if there is enough
                    
                    vol_remaining = vol_remaining+extra_aspirate_vol-sacrificial_dispense_vol #What's leftover?
                    print("Extra Aspiration: ", extra_aspirate_vol)
                    print("Sacrificial Dispense: ", sacrificial_dispense_vol)
                    print("Remaining volume: ", vol_remaining)                  

                    #Let's get our solution and any extra we need
                    print(f"Aspirating solution {vial_index}: {dispense_vol+extra_aspirate_vol} uL")
                    self.aspirate_from_vial(vial_index,dispense_vol+extra_aspirate_vol,specified_tip=pipet_index)
                    
                    #Put back the extra if there is any
                    if sacrificial_dispense_vol > 0:
                        self.dispense_into_vial(vial_index,sacrificial_dispense_vol,initial_move=False)
  
                    print("Indices to dipense:", well_plate_array)
                    print("Dispense volumes:", dispense_array)
                    print("Dispense sum", np.sum(dispense_array))

                    self.dispense_into_wellplate(well_plate_array,dispense_array) #Dispense into the wellplate

                    vol_dispensed += dispense_vol #Track how much we've dispensed so far
                    print(f"Solution Dispensed {vial_index}: {vol_dispensed} uL")     
                
                print("Vol remaining, returning to vial: ", vol_remaining)
                
                if vol_needed>0 and vol_remaining>0: #Put back the buffer if there is any
                    self.dispense_into_vial(vial_index,vol_remaining)
                if vol_needed>0:
                    self.remove_pipet()

                if return_vial: #If we moved the vial, move it back
                    self.recap_clamp_vial()
                    self.return_vial_home(vial_index)
                print()
        return True

    #Check the original status of the vial in order to send it to its home location
    def return_vial_home(self,vial_num):
        home_location = self.get_vial_info(vial_num,'home_location')
        home_location_index = self.get_vial_info(vial_num,'home_location_index')
        
        vial_location = self.get_vial_info(vial_num,'location')
        if vial_location == 'clamp' and self.GRIPPER_STATUS == "Cap":
            self.recap_clamp_vial()
        self.move_vial_to_location(vial_num,home_location,home_location_index)
        self.save_robot_status()

    #Drop off a vial at a location that you already have
    def drop_off_vial(self, vial_num, location, location_index):
        destination = self.get_location(False,location,location_index)
        destination_empty = self.get_vial_in_location(location,location_index) is None

        self.check_for_errors([[destination_empty, True, "Cannot move vial to destination, destination full"]],True)

        self.c9.goto_safe(destination) #move vial to destination
        self.c9.open_gripper() #release vial
        
        self.VIAL_DF.at[vial_num, 'location']=location
        self.VIAL_DF.at[vial_num, 'location_index']=location_index
        self.GRIPPER_STATUS = None #We no longer have the vial
        self.GRIPPER_VIAL_INDEX = None
        self.save_robot_status() #Update in memory

    def grab_vial(self,vial_num):
        initial_location = self.get_vial_location(vial_num, False)

        error_check_list = [] #List of specific errors for this method
        error_check_list.append([self.GRIPPER_STATUS is None, True, "Cannot move vial to destination, gripper full"])
        error_check_list.append([self.HELD_PIPET_INDEX is None, True, "Cannot move vial to destination, robot holding pipet"])
        error_check_list.append([self.is_vial_movable(vial_num), True, "Can't move vial, vial is uncapped."])  

        self.check_for_errors(error_check_list,True) #Check for an error, and pause if there's an issue

        #self.open_gripper()
        self.goto_location_if_not_there(initial_location) #move to vial
        self.c9.close_gripper() #grip vial
        
        self.GRIPPER_STATUS = "Vial" #Update the status of the robot
        self.GRIPPER_VIAL_INDEX = vial_num
        self.save_robot_status() #Save the status of the robot

    #Send the vial to a specified location
    def move_vial_to_location(self,vial_num,location,location_index):
        print("Moving vial " + self.get_vial_info(vial_num,'vial_name') + " to " + location + ": " + str(location_index))
        self.grab_vial(vial_num) #Grab the vial
        self.drop_off_vial(vial_num,location,location_index) #Drop off the vial

    def get_vial_in_location(self, location_name, location_index):
        # Filter rows where both conditions match
        mask = (self.VIAL_DF['location'] == location_name) & (self.VIAL_DF['location_index'] == location_index)
        
        # Get the matching values
        matching_vials = self.VIAL_DF.loc[mask, 'vial_index'].values

        # Return the first match or None if no match is found
        return int(matching_vials[0]) if len(matching_vials) > 0 else None

    #Uncap the vial in the clamp
    def uncap_clamp_vial(self):
        print ("Removing cap from clamped vial")

        clamp_vial_index = self.get_vial_in_location('clamp',0)

        error_check_list = [] #List of specific errors for this method
        error_check_list.append([self.GRIPPER_STATUS is None, True, "Cannot uncap, gripper full"])
        error_check_list.append([self.HELD_PIPET_INDEX is None, True, "Can't uncap vial, holding pipet"])
        error_check_list.append([clamp_vial_index is None, False, "Cannot uncap, no vial in clamp"])
        error_check_list.append([self.is_vial_movable(clamp_vial_index), True, "Can't uncap, vial is uncapped already"])

        self.check_for_errors(error_check_list,True) #Check for an error and pause if there is one
        
        self.goto_location_if_not_there(vial_clamp) #Maybe check if it is already there or not   
        self.c9.close_clamp() #clamp vial
        self.c9.close_gripper()
        self.c9.uncap(revs=3)
        self.GRIPPER_STATUS = "Cap"
        self.c9.open_clamp()

        self.VIAL_DF.at[clamp_vial_index, 'capped']=False
        self.GRIPPER_VIAL_INDEX = clamp_vial_index
        self.save_robot_status()

    #Recap the vial in the clamp
    def recap_clamp_vial(self):
        print("Recapping clamped vial")
        
        clamp_vial_index = self.get_vial_in_location('clamp',0)

        error_check_list = [] #List of specific errors for this method
        error_check_list.append([self.GRIPPER_STATUS, "Cap", "Cannot recap, no cap in gripper"])
        error_check_list.append([self.HELD_PIPET_INDEX is None, True, "Can't recap vial, holding pipet"])
        error_check_list.append([clamp_vial_index is None, False, "Cannot recap, no vial in clamp"])
        error_check_list.append([self.is_vial_movable(clamp_vial_index), False, "Can't recap, vial is capped already"])
        
        self.check_for_errors(error_check_list,True) #Let's pause if there is an error

        self.goto_location_if_not_there(vial_clamp)
        self.c9.close_clamp() #Make sure vial is clamped
        self.c9.cap(revs=2.0, torque_thresh = 600) #Cap the vial #Cap the vial
        self.c9.open_gripper() #Open the gripper to release the cap
        self.GRIPPER_STATUS = None
        self.c9.open_clamp()

        self.VIAL_DF.at[clamp_vial_index, 'capped']=True #Update the vial status
        self.GRIPPER_VIAL_INDEX = None
        self.save_robot_status()

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
        print("Vortexing Vial: " + self.get_vial_info(vial_num,'vial_name'))
        
        if not self.GRIPPER_STATUS == "Vial": #Get vial if the vial isn't already held
            if self.GRIPPER_VIAL_INDEX == vial_num and self.GRIPPER_STATUS == "Cap":
                self.recap_clamp_vial()
            vial_location = self.get_vial_location(vial_num,False)
            self.grab_vial(vial_num)
    
        #Vortex
        self.c9.move_z(292) #Move to a higher height
        self.c9.move_axis(self.c9.GRIPPER, 1000*vortex_time*vortex_speed, vel=vortex_speed,accel=10000)
        self.c9.reduce_axis_position(axis=self.c9.GRIPPER)
        
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
        movable = self.get_vial_info(vial_index,'capped') == True and self.get_vial_info(vial_index,'vial_type') == "8_mL"
        return movable
    
    #Check to see if the pipet can have liquids added/removed
    def is_vial_pipetable(self, vial_index):
        pipetable = False
        pipetable = self.get_vial_info(vial_index,'capped') == False or self.get_vial_info(vial_index,'cap_type') == "open"
        return pipetable

    #Get adjust the aspiration height based on how much is there
    def get_aspirate_height(self, amount_vial, amount_to_withdraw, base_height, height_volume_constant=6, buffer=1.0):      
        target_height = base_height + (height_volume_constant*(amount_vial - amount_to_withdraw - buffer))
        if target_height > base_height:
            return target_height
        else:
            return base_height

    def dispense_liquid_into_vial(target_vial_index, dispense_amountyes): #Can work on this
        return None
    
    def dispense_solid_into_vial(target_vial_index, dispense_mass): #Can work on this
        return None
    
    def set_heater_temperature(target_temperature): #Can work on this
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
  
    #Move the robot to the home position    
    def move_home(self):
        print("Moving robot to home position")
        self.c9.goto_safe(home)

    #Get some piece of information about a vial
    #vial_index,vial_name,location,location_index,vial_volume,capped,cap_type,vial_type
    def get_vial_info(self,vial_num,column_name):
        values = self.VIAL_DF.loc[self.VIAL_DF['vial_index'] == vial_num, column_name].values
        if len(values) > 0:
            return values[0]  # Return the first match
        else:
            return None  # Handle case where no match is found    

    def get_vial_index_from_name(self,vial_name):
        values = self.VIAL_DF.loc[self.VIAL_DF['vial_name'] == vial_name, 'vial_index'].values
        if len(values) > 0:
            return values[0]  # Return the first match
        else:
            return None  # Handle case where no match is found  

    #Physical method that get's hard-coded minimum heights for pipetting    
    def get_min_pipetting_height(self,vial_num):
        #The height at which the pipet touches the ground for the 1 mL pipet
        min_height = None   

        location_name = self.get_vial_info(vial_num,'location')

        #These constants should be considered properties that we can take in from files later
        CLAMP_BASE_HEIGHT = 114.5
        VIAL_RACK_BASE_HEIGHT = 67.25
        PR_BASE_HEIGHT = 85 #Need to fine tune this
        LARGE_VIAL_BASE_HEIGHT = 55    
        SMALL_VIAL_BASE_HEIGHT = 97
    
        if location_name=='main_8mL_rack':
            min_height = VIAL_RACK_BASE_HEIGHT 
        elif location_name=='large_vial_rack':
            min_height = LARGE_VIAL_BASE_HEIGHT
        elif location_name=='small_vial_rack':
            min_height = SMALL_VIAL_BASE_HEIGHT
        elif location_name=='photoreactor_array':
            min_height = PR_BASE_HEIGHT
        elif location_name=='clamp':
            min_height = CLAMP_BASE_HEIGHT

        return min_height

    #Get the position of a vial right now
    def get_vial_location(self,vial_num,use_pipet):
        location_name = self.get_vial_info(vial_num,'location')
        location_index = self.get_vial_info(vial_num,'location_index')
        return self.get_location(use_pipet,location_name,location_index)

    #Translate the location names and indexes to hard-coded locations for pipets or gripping vials
    def get_location(self,use_pipet,location_name,location_index):
        location = None
        #Perhaps we can streamline this association later
        if use_pipet: #For the pipet
            if location_name=='main_8mL_rack':
                location = rack_pip[location_index]
            elif location_name=='large_vial_rack':
                location = large_vial_pip[location_index]
            elif location_name=='small_vial_rack':
                location = small_vial_pip[location_index]
            elif location_name=='photoreactor_array':
                if location_index==0:
                    location=PR_PIP_0
                elif location_index==1:
                    location=PR_PIP_1
            elif location_name=='clamp':
                location = vial_clamp_pip
        else: #For the gripper
           if location_name=='main_8mL_rack':
                location = rack[location_index]
           elif location_name=='large_vial_rack':
                print("No defined location")
           elif location_name=='small_vial_rack':
               print("No defined location")
           elif location_name=='photoreactor_array':
                if location_index==0:
                    location=photo_reactor_0
                elif location_index==1:
                    location=photo_reactor_1
           elif location_name=='clamp':
                location = vial_clamp 
           elif location_name=='heater':
               location = heater_grid[location_index]
        return location
    
    def global_exception_handler(self,exc_type, exc_value, exc_traceback):
        if "MOTOR FAULT" in str(exc_value):
            # Implement the restart mechanism here
            self.pause_after_error("Critical motor fault detected! Restarting controller...")
        else:
            print(f"Unhandled Exception: {exc_type.__name__}: {exc_value}")