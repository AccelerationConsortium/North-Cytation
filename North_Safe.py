from Locator import *
import numpy as np
import time
import math
import pandas as pd
import slack_agent
import yaml
from unittest.mock import MagicMock

class North_Track:

    #Controller
    c9 = None
    MAX_HEIGHT = 0
    RELEASE_DISTANCE_Y = 2400
    LID_DISTANCE_Y = 4000

    #Well plate active areas
    NR_WELL_PLATE_X = [131854,105860,81178]
    NR_WELL_PLATE_Y = [88750, 86965, 89155]
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

    """New double WP stack"""
    DOUBLE_SOURCE_X = 50 
    DOUBLE_WASTE_X = 14550 
    DOUBLE_TRANSFER_X = 28000 
    DOUBLE_SOURCE_Y_96 = [83500, 76800, 70700] #first element = height when 1 WP is in stack
    DOUBLE_WASTE_Y_96 = [83000, 76350, 70200] #first element = height when dropping off 1st WP to waste
    SOURCE_HEIGHTS_DICT = {"96 WELL PLATE": DOUBLE_SOURCE_Y_96}
    WASTE_HEIGHTS_DICT = {"96 WELL PLATE": DOUBLE_WASTE_Y_96}

    #num_source = 0 #number of wellplates in source stack 
    well_plate_df = None

    #Let's initialize the number of well plates from a file
    def __init__(self, c9):
        self.c9 = c9
        #self.well_plate_df = pd.read_csv("../utoronto_demo/status/wellplate_storage_status.txt", sep=r",", engine="python") #TODO: not sure if needed anymore?
        #self.num_source = int(self.well_plate_df.loc[self.well_plate_df['Location']=='Input']['Status'].values)

        self.NUM_SOURCE = 0
        self.NUM_WASTE = 0
        self.CURRENT_WP_TYPE = "96 WELL PLATE"
        self.NR_OCCUPIED = False

        #Load yaml data
        self.TRACK_STATUS_FILE = "../utoronto_demo/status/track_status.yaml"
        self.get_track_status() #set NUM_SOURCE, NUM_WASTE, CURRENT_WP_TYPE and NR_OCCUPIED from yaml file
        self.reset_after_initialization()
    
    def check_input_file(self, pause_after_check=True):
        """
        Prints the well plate status values for user to confirm the initial state of your well plates.
        """
        #edit: what we want to output (wellplate type -- all of dictionary basically...)
        print(f"--Wellplate status-- \n Wellplate type: {self.CURRENT_WP_TYPE} \n Number in source: {self.NUM_SOURCE} \n Number in waste: {self.NUM_WASTE} \n Is the pipetting area occupied: {self.NR_OCCUPIED}")

        if pause_after_check:
            input("Only hit enter if the status of the well plates is correct, otherwise hit ctrl-c")

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
        """Return the well plate to the NR station. Assumes already holding a wellplate, will move down to WELL_PLATE_TRANSFER_Y directly when called.
        
        Args:
            `well_plate_num`(int): The number identifying which wp stand to return to (0 for the pipetting one)
            `grab_lid`(bool): if grabbing lid (default = False)
            `quartz_wp`(bool): if the wellplate is quartz (default = False)
        """
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
    
    def get_next_WP_from_source(self):  #OLD
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
    
    #Save the status of the robot to memory
    def save_track_status(self):
        track_status = {
            "num_in_source": self.NUM_SOURCE,
            "num_in_waste": self.NUM_WASTE,
            "wellplate_type": self.CURRENT_WP_TYPE,
            "nr_occupied": self.NR_OCCUPIED
        }

        # Writing to a file
        with open(self.TRACK_STATUS_FILE, "w") as file:
            yaml.dump(track_status, file, default_flow_style=False)

    #Update the status of the robot from yaml file
    def get_track_status(self):
        """Get the track status from the yaml file."""
        # Get the track status
        with open(self.TRACK_STATUS_FILE, "r") as file:
            track_status = yaml.safe_load(file)

        # Convert "None" or "null" strings to actual None. CHATGPT
        def convert_none(value):
            if isinstance(value, dict):
                return {k: convert_none(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [convert_none(v) for v in value]
            elif value in ["None", "null"]:
                return None
            return value

        track_status = convert_none(track_status)

        try:
            self.NUM_SOURCE = track_status['num_in_source']
            self.NUM_WASTE = track_status['num_in_waste']
            self.CURRENT_WP_TYPE = track_status['wellplate_type']
            self.NR_OCCUPIED = track_status['nr_occupied']
        except:
            self.pause_after_error("Issue reading robot status", False)

    def get_new_wellplate(self): #double WP stack 
        """Get a new wellplate from the source stack (in the double stack holder) and move to north robot pipetting area"""
        DOUBLE_SOURCE_Y = self.SOURCE_HEIGHTS_DICT[self.CURRENT_WP_TYPE] 
        MAX_IN_SOURCE = len(DOUBLE_SOURCE_Y) #the number of valid WPs that can be stored or have been initialized

        if self.NUM_SOURCE > 0 and self.NUM_SOURCE <= MAX_IN_SOURCE:
            print(f"Getting {self.NUM_SOURCE}th wellplate from source")
            
            #move to source stack and grab wellplate
            self.open_gripper()
            self.c9.move_axis(6, self.MAX_HEIGHT, vel=25) #up to max height
            self.c9.move_axis(7, self.DOUBLE_SOURCE_X, vel=20) #above source
            self.c9.move_axis(6, DOUBLE_SOURCE_Y[self.NUM_SOURCE-1], vel=15) #down to WP height
            self.close_gripper()
            
            #move up from source stack to "safe" area and move down
            self.c9.move_axis(6, self.MAX_HEIGHT, vel=15) #up to max height
            self.c9.move_axis(7, self.DOUBLE_TRANSFER_X, vel=10) #to "safe" area
            self.c9.move_axis(6, self.WELL_PLATE_TRANSFER_Y, vel=10) #to transfer area #TODO: See if need to height adjust?
            
            self.NUM_SOURCE -= 1
            self.save_track_status() #update yaml

            self.return_well_plate_to_nr(0)
            #TODO: move to NR wellplate station (existing function?) -> self.NR_OCCUPIED = True, self.save_track_status()
        
        else:
            if self.NUM_SOURCE <= 0:
                print("Wellplate stack is empty")
            else: #self.NUM_SOURCE > maximum
                print("Stack is overfilled / height is not initialized")

    
    def discard_wellplate(self): #double WP stack
        DOUBLE_WASTE_Y = self.WASTE_HEIGHTS_DICT[self.CURRENT_WP_TYPE] 
        MAX_IN_WASTE = len(DOUBLE_WASTE_Y) #the number of valid WPs that can be stored / have been initialized
        
        #TODO: assumes already holding wellplate (at transfer area)
        
        if self.NUM_WASTE < MAX_IN_WASTE: 
            print(f"Discarding wellplate as the {self.NUM_WASTE}th WP in waste stack.")

            self.c9.move_axis(6, self.MAX_HEIGHT, vel=25) #up to max height
            self.c9.move_axis(7, self.DOUBLE_WASTE_X, vel=15) #above waste stack
            self.c9.move_axis(6, DOUBLE_WASTE_Y[self.NUM_WASTE], vel=15)
            self.open_gripper()
            self.c9.move_axis(6, self.MAX_HEIGHT, vel=25)

            self.NUM_WASTE += 1
            self.save_track_status()
        else:
            print("Wellplate stack is too full for discarding another well plate.")
        
class North_T8:

    t8 = None

    def __init__(self,c9):
        from north import NorthC9
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
    DEFAULT_SMALL_TIP_DELTA_Z = -21 #This is the height difference between the bottom of the small pipet tip and the large tip
    LOWER_PIPET_ARRAY_INDEX = 0 #Label representing the lower rack at the back with 1000 uL tips
    HIGHER_PIPET_ARRAY_INDEX = 1 #Label representing the upper rack at the back with 250 uL tips

    HELD_PIPET_INDEX = None #What kind of pipet do we have  
    PIPETS_USED = [0,0] #Tracker for each rack. TODO: This isn't very extensible
    PIPET_FLUID_VIAL_INDEX = None #What vial was the fluid aspirated from?
    PIPET_FLUID_VOLUME = 0 #What is the held volume in the pipet?

    #Track the pump speed. This seems to help with pump errors. 
    CURRENT_PUMP_SPEED = 11

    #Controller
    c9 = None
    simulate = False
   
    #Initialize the status of the robot. 
    def __init__(self,c9,vial_file=None,simulate=False):
        print("Initializing North Robot...")
        self.c9 = c9
        self.VIAL_FILE = vial_file
        self.get_robot_status() #Update the status of the robot from memory
        self.reset_after_initialization() #Reset everything that may not be as desired, eg return to "Home"
        self.simulate = simulate
        self.load_pumps() #Load the pumps
        #sys.excepthook = self.global_exception_handler

    #Load the pumps and set volumes
    def load_pumps(self):
        self.c9.pumps[0]['volume'] = 1
        self.c9.pumps[1]['volume'] = 2.5
        self.c9.set_pump_speed(1, 20)

    #Check the status of the input vial file
    def check_input_file(self,pause_after_check=True):
        """
        Prints the vial status dataframe for user to confirm the initial state of your vials.
        """
        vial_status = pd.read_csv(self.VIAL_FILE, sep=",")
        print(vial_status)
        if pause_after_check:
            input("Only hit enter if the status of the vials (including open/close) is correct, otherwise hit ctrl-c")

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
            self.c9.home_pump(1) #Home reservoir 1
            for i in range (6,8): #Home the track
                self.c9.home_axis(i)
            self.reset_after_initialization()
        
        self.c9.open_gripper()
        self.c9.home_pump(0)

    #Save the status of the robot to memory
    def save_robot_status(self):
        # Robot status data
        robot_status = {
            "gripper_status": self.GRIPPER_STATUS,
            "gripper_vial_index": self.GRIPPER_VIAL_INDEX,
            "held_pipet_index": self.HELD_PIPET_INDEX,
            "pipets_used": {"lower_rack_1": self.PIPETS_USED[self.LOWER_PIPET_ARRAY_INDEX], "upper_rack_1": self.PIPETS_USED[self.HIGHER_PIPET_ARRAY_INDEX]},
            "pipet_fluid_vial_index": self.PIPET_FLUID_VIAL_INDEX,
            "pipet_fluid_volume": float(self.PIPET_FLUID_VOLUME)
        }

        if not self.simulate: 
            # Writing to a file
            self.VIAL_DF.to_csv(self.VIAL_FILE, index=False,sep=',') #Save the status of the vial dataframe
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
        MAX_PIPETS=47 #This is based off the racks
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

        self.c9.move_z(292,vel=5) #Move to a safe height (292)

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
        if send_slack and not isinstance(self.c9, MagicMock):
            slack_agent.send_slack_message(err_message)
        input("Waiting for user to press enter or quit after error...")

    def normalize_vial_index(self, vial):
        """Accepts either a vial index (int) or vial name (str) and returns the vial index (int)."""
        if isinstance(vial, str):
            vial_idx = self.get_vial_index_from_name(vial)
            if vial_idx is None:
                raise ValueError(f"Vial name '{vial}' not found in VIAL_DF.")
            return vial_idx
        return vial

    #Pipet from a vial into another vial
    #Use calibration (not implemented) is if you want to adjust the volume based off a known calibration
    #Aspirate conditioning is an alternate way to aspirate (up and down some number of cycles)
    def aspirate_from_vial(self, source_vial_name, amount_mL,move_to_aspirate=True,specified_tip=None,track_height=True,wait_time=0,aspirate_speed=11,asp_disp_cycles=0):
        """
        Aspirate amount_ml from a source vial.
        Args:
            `source_vial_name`(str): Name of the source vial to aspirate from
            `amount_mL`(float): Amount to aspirate in mL
        """
        
        source_vial_num = self.normalize_vial_index(source_vial_name) #Convert to int if needed
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

        print("Pipetting from vial " + self.get_vial_info(source_vial_num,'vial_name') + ", amount: "  + str(round(amount_mL,3)) + " mL")

        #Where are we pipetting from?  
        location = self.get_vial_location(source_vial_num,True)
        base_height = self.get_min_pipetting_height(source_vial_num)
        vial_type = self.get_vial_info(source_vial_num, 'vial_type')

        #The vial type affects the depth for aspiration. TODO: There should be a data structure for vial types that holds this. 
        if vial_type=='8_mL':
            height_volume_constant=6
        elif vial_type=='20_mL':
            height_volume_constant=2
        else:
            height_volume_constant=0

        #Adjust height based on the amount that is in the vial
        if track_height:
            height = self.get_aspirate_height(source_vial_volume,amount_mL,base_height,height_volume_constant)
        else:
            height = base_height #Go to the minimum height
  
        #Adjust for different pipet type
        height = self.adjust_height_based_on_pipet_held(height)
        #print("Aspirate height: ", height)

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
            self.pipet_from_location(amount_mL, aspirate_speed, height, True,initial_move=move_to_aspirate)
            self.pipet_from_location(amount_mL, aspirate_speed, height, False,initial_move=False)
            move_to_aspirate = False
        
        #Main aspiration
        self.pipet_from_location(amount_mL, aspirate_speed, height, True, initial_move=move_to_aspirate)

        #Record the volume change
        self.VIAL_DF.at[source_vial_num,'vial_volume']=(source_vial_volume-amount_mL)

        #Wait if required
        if wait_time > 0:
            time.sleep(wait_time)
        
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
    def dispense_from_vial_into_vial(self,source_vial_name,dest_vial_name,volume,move_to_aspirate=True,move_to_dispense=True,buffer_vol=0.02,track_height=True):
        """
        Dispenses a specified volume from source_vial_name to dest_vial_name, with optional buffer transfer for accuracy.

        Args:
            `source_vial_name` (str): Name of the source vial to aspirate from.
            `dest_vial_name` (str): Name of the destination vial to dispense into.
            `volume` (float): Volume (in mL) to transfer.
            `move_to_aspirate` (bool): Whether to move to the source vial before aspirating.
            `move_to_dispense` (bool): Whether to move to the destination vial before dispensing.
            `buffer_vol` (float): Buffer volume (in mL)
            `track_height` (bool): Whether to track the volume & height to aspirate from, in the source vial.
        """
        
        source_vial_index = self.normalize_vial_index(source_vial_name) #Convert to int if needed
        dest_vial_index = self.normalize_vial_index(dest_vial_name) #Convert to int if needed
        if volume < 0.2 and volume >= 0.01:
            tip_type = self.HIGHER_PIPET_ARRAY_INDEX
            max_volume = 0.25
        elif volume >= 0.2 and volume <= 1.00:
            tip_type = self.LOWER_PIPET_ARRAY_INDEX
            max_volume = 1.00
        elif volume == 0:
            print("Cannot dispense 0 mL")
            return
        else:
            self.pause_after_error(f"Cannot accurately aspirate: {volume} mL under specified conditions.")
        
        extra_aspirate = 0
        if max_volume-volume >= 2*buffer_vol:
            extra_aspirate = 2*buffer_vol
        
        self.aspirate_from_vial(source_vial_index,round(volume+extra_aspirate,3),move_to_aspirate=move_to_aspirate,specified_tip=tip_type,track_height=track_height)
        if extra_aspirate > 0:
            self.dispense_into_vial(source_vial_index,buffer_vol,initial_move=False)
        
        self.dispense_into_vial(dest_vial_index,volume,initial_move=move_to_dispense)
        
        if extra_aspirate > 0:
            self.dispense_into_vial(source_vial_index,buffer_vol,initial_move=move_to_dispense)

    def select_wellplate_grid(self,well_plate_type):
        if well_plate_type == "96 WELL PLATE":
            location = well_plate_new_grid
        elif well_plate_type == "48 WELL PLATE":
            location = grid_48
        else:
            print("Unknown well plate type")
            location = None
        return location

    #TODO add error checks and safeguards
    def pipet_from_wellplate(self,wp_index,volume,aspirate_speed=10,aspirate=True,move_to_aspirate=True,stay_low=False,well_plate_type="96 WELL PLATE"):
        location = self.select_wellplate_grid(well_plate_type)[wp_index]
        
        height = self.c9.counts_to_mm(3, location[3])
        height = self.adjust_height_based_on_pipet_held(height) 

        if well_plate_type == "96 WELL PLATE":
            height_adjust = 16
        elif well_plate_type == "48 WELL PLATE":
            height_adjust = 19

        if aspirate:
            height = height - height_adjust #Go to the bottom of the well

        if move_to_aspirate:
                if not stay_low:
                    self.c9.goto_xy_safe(location, vel=15)
                else:
                    self.move_rel_z(height_adjust)
                    self.c9.goto(location, vel=15)

        self.pipet_from_location(volume, aspirate_speed, height, aspirate = aspirate, initial_move=move_to_aspirate)

    #Mix the well
    def mix_well_in_wellplate(self,wp_index,volume,repeats=3,well_plate_type="96 WELL PLATE"):
        self.pipet_from_wellplate(wp_index,volume,well_plate_type=well_plate_type)
        self.pipet_from_wellplate(wp_index,volume,aspirate=False,move_to_aspirate=False,well_plate_type=well_plate_type)
        for i in range (1, repeats):
            self.pipet_from_wellplate(wp_index,volume,move_to_aspirate=False,well_plate_type=well_plate_type)
            self.pipet_from_wellplate(wp_index,volume,aspirate=False,move_to_aspirate=False,well_plate_type=well_plate_type)

    #Mix in a vial
    def mix_vial(self,vial_name,volume,repeats=3):
        vial_index= self.normalize_vial_index(vial_name)
        self.aspirate_from_vial(vial_index,volume,3)
        self.dispense_into_vial(vial_index,volume,initial_move=False)
        for i in range (1,repeats):
            self.dispense_from_vial_into_vial(vial_index,vial_index,volume,move_to_aspirate=False,move_to_dispense=False,buffer_vol=0)

    #Dispense an amount into a vial
    def dispense_into_vial(self, dest_vial_name,amount_mL,initial_move=True,dispense_speed=11,measure_weight=False,wait_time=0):     
        
        dest_vial_num = self.normalize_vial_index(dest_vial_name) #Convert to int if needed

        measured_mass = None
        error_check_list = [] #List of specific errors for this method
        error_check_list.append([self.is_vial_pipetable(dest_vial_num), True, "Can't pipet, at least one vial is capped"])    
        self.check_for_errors(error_check_list,True) #This will cause a pause if there's an issue

        print("Pipetting into vial " + self.get_vial_info(dest_vial_num,'vial_name') + ", amount: " + str(round(amount_mL,3)) + " mL")
        
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

        if wait_time > 0:
            time.sleep(wait_time)

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
    def dispense_into_wellplate(self, dest_wp_num_array, amount_mL_array, dispense_type = "None", dispense_speed=11,wait_time=1,well_plate_type="96 WELL PLATE"):
        """
        Dispenses specified amounts into a series of wells in a well plate.
        Args:
            `dest_wp_num_array` (list or range): Array of well indices to dispense into (e.g., [0, 1, 2])
            `amount_mL_array` (list[float]): Array of amounts (in mL) to dispense into each well (e.g., [0.1, 0.2, 0.3])
        """
        first_dispense = True
        for i in range(0, len(dest_wp_num_array)):    
            try:
                location = self.select_wellplate_grid(well_plate_type)[dest_wp_num_array[i]]
            except:
                #location = well_plate_new_grid[self.convert_well_into_index(dest_wp_num_array[i])]
                self.pause_after_error("Can't parse wellplate wells in non-indexed form for now")

            amount_mL = amount_mL_array[i] #What amount for this well

            if amount_mL == 0: #Skip empty dispenses
                continue

            height = self.c9.counts_to_mm(3, location[3])
            height = self.adjust_height_based_on_pipet_held(height) 

            if first_dispense:
                self.c9.goto_xy_safe(location, vel=15)
                first_dispense = False
                
            else:
                self.c9.goto_xy_safe(location, vel=5, accel = 1, safe_height=height) #Use safe_height here!

            if dispense_type.lower() == "drop-touch" or dispense_type.lower() == "touch": #move lower & towards side of well before dispensing
                height -= 5 #goes 5mm lower when dispensing

            if self.HELD_PIPET_INDEX == self.HIGHER_PIPET_ARRAY_INDEX and dispense_speed == 11: #Adjust this later
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

        self.PIPET_FLUID_VOLUME -= np.sum(amount_mL_array)  # <-- Add this line back
        self.save_robot_status()    
        return True

    #This is a custom method that takes a "well_plate_df" as an array of destinations and some "vial_indices" which are the different dispensed liquids
    #This method will use both the large and small tips, with a specified low_volume_cutoff between the two
    #This method does use multiple dispenses per aspiration for efficiency
    def dispense_from_vials_into_wellplate(self, well_plate_df, vial_names, low_volume_cutoff=0.05, buffer_vol=0.02,
                                            dispense_speed=11, wait_time=1, asp_cycles=0, track_height=True,
                                            well_plate_type="96 WELL PLATE", pipet_back_and_forth=False):
        vial_indices = [self.normalize_vial_index(v) for v in vial_names]  # Normalize vial indices
        
        # Step 1: Check if there's enough liquid in each vial
        well_plate_dispense_2d_array = well_plate_df.values
        vols_required = np.sum(well_plate_dispense_2d_array, axis=0)
        print("Total volumes needed (mL):", vols_required)

        for i, vial in enumerate(vial_indices):
            volume_needed = vols_required[i]
            volume = self.get_vial_info(vial, 'vial_volume')
            if volume < volume_needed - 1e-6:
                self.pause_after_error(f"Not enough solution in vial: {vial}", True)

        # Step 2: Split dispenses based on pipette range
        well_plate_df_low = well_plate_df.where(well_plate_df < low_volume_cutoff).fillna(0)
        well_plate_df_high = well_plate_df.mask(well_plate_df < low_volume_cutoff, 0)
        well_plate_instructions = [
            [well_plate_df_low, 0.25, self.HIGHER_PIPET_ARRAY_INDEX],
            [well_plate_df_high, 1.0, self.LOWER_PIPET_ARRAY_INDEX]
        ]

        # Step 3: Process each pipette configuration
        for well_plate_df, max_volume, pipet_index in well_plate_instructions:
            if well_plate_df.empty:
                continue

            well_plate_dispense_2d_array = well_plate_df.values
            well_plate_indices = well_plate_df.index.tolist()

            for i, vial_index in enumerate(vial_indices):
                vol_needed = np.sum(well_plate_dispense_2d_array[:, i])
                if vol_needed < 1e-6:
                    continue

                print(f"\nAspirating from Vial: {vial_index}")
                print(f"Total volume needed: {vol_needed:.3f} mL")

                return_vial = False
                if not self.is_vial_pipetable(vial_index):
                    print(f"Vial {vial_index} not open, moving to clamp")
                    self.move_vial_to_location(vial_index, location='clamp', location_index=0)
                    self.uncap_clamp_vial()
                    return_vial = True

                last_index = 0
                vol_dispensed = 0.0
                vol_remaining = 0.0

                while vol_dispensed < vol_needed - 1e-6:
                    dispense_array = []
                    well_plate_array = []
                    dispense_vol = 0.0

                    # Dispensing logic
                    if pipet_back_and_forth:
                        while last_index < len(well_plate_indices):
                            volume = well_plate_dispense_2d_array[last_index, i]
                            if volume > 1e-6:
                                dispense_array = [volume]
                                well_plate_array = [well_plate_indices[last_index]]
                                dispense_vol = volume
                                well_plate_dispense_2d_array[last_index, i] = 0.0
                                last_index += 1
                                break
                            last_index += 1
                    else:
                        while last_index < len(well_plate_indices):
                            volume = well_plate_dispense_2d_array[last_index, i]
                            if dispense_vol + volume <= max_volume + 1e-6 and volume > 1e-6:
                                dispense_vol += volume
                                dispense_array.append(volume)
                                well_plate_array.append(well_plate_indices[last_index])
                                well_plate_dispense_2d_array[last_index, i] = 0.0
                                last_index += 1
                            else:
                                break

                    if dispense_vol < 1e-6:
                        break  # Nothing to dispense

                    # Optional buffer logic
                    extra_aspirate_vol = 0.0
                    sacrificial_dispense_vol = 0.0
                    if not pipet_back_and_forth:
                        if math.isclose(vol_dispensed, 0.0, abs_tol=1e-6):
                            extra_aspirate_vol = min(max_volume - dispense_vol, buffer_vol * 2)
                            if math.isclose(extra_aspirate_vol, buffer_vol * 2, abs_tol=1e-6):
                                sacrificial_dispense_vol = buffer_vol
                        else:
                            extra_aspirate_vol = min(buffer_vol, max_volume - dispense_vol - vol_remaining)
                            if math.isclose(extra_aspirate_vol, buffer_vol, abs_tol=1e-6):
                                sacrificial_dispense_vol = buffer_vol

                    vol_remaining += extra_aspirate_vol - sacrificial_dispense_vol

                    total_aspirate = dispense_vol + extra_aspirate_vol
                    print(f"Aspirating {total_aspirate:.3f} mL from vial {vial_index}")
                    self.aspirate_from_vial(vial_index, total_aspirate, specified_tip=pipet_index,
                                            aspirate_speed=dispense_speed, wait_time=wait_time,
                                            asp_disp_cycles=asp_cycles, track_height=track_height)

                    if sacrificial_dispense_vol > 0:
                        self.dispense_into_vial(vial_index, sacrificial_dispense_vol, initial_move=False,
                                                dispense_speed=dispense_speed, wait_time=wait_time)

                    print("Dispensing to wells:", well_plate_array)
                    print("Dispense volumes:", dispense_array)
                    self.dispense_into_wellplate(well_plate_array, dispense_array,
                                                dispense_speed=dispense_speed, wait_time=wait_time,
                                                well_plate_type=well_plate_type)

                    vol_dispensed += dispense_vol
                    print(f"Total dispensed so far: {vol_dispensed:.3f} mL")

                # Final buffer return
                if vol_remaining > 1e-6:
                    print(f"Returning remaining volume: {vol_remaining:.3f} mL to vial {vial_index}")
                    self.dispense_into_vial(vial_index, vol_remaining,
                                            dispense_speed=dispense_speed, wait_time=wait_time)

                self.remove_pipet()

                if return_vial:
                    self.recap_clamp_vial()
                    self.return_vial_home(vial_index)

        return True

    #Prime the line from the reservoir to the vial. In theory this could happen automatically. Probably good to do it if you are using a reservoir. 
    def prime_reservoir_line(self, reservoir_index, overflow_vial, volume=0.5):
        overflow_vial = self.normalize_vial_index(overflow_vial) #Convert to int if needed
        print(f"Priming reservoir {reservoir_index} line into vial {overflow_vial}: {volume} mL")
        self.dispense_into_vial_from_reservoir(reservoir_index,overflow_vial,volume)

    def dispense_into_vial_from_reservoir(self,reservoir_index,vial_index,volume):
        
        vial_index = self.normalize_vial_index(vial_index) #Convert to int if needed
        print(f"Dispensing into vial {vial_index} from reservoir {reservoir_index}: {volume} mL")

        #Step 1: move the vial to the clamp
        self.move_vial_to_location(vial_index,'clamp',0)
        self.uncap_clamp_vial()
        self.move_home()
        #Step 2: move the carousel
        self.c9.move_carousel(45,70) #This will take some work. Note that for now I'm just doing for position 0
        #Step 3: aspirate and dispense from the reservoir
        max_volume = self.c9.pumps[reservoir_index]['volume']
        num_dispenses = math.ceil(volume/max_volume)
        dispense_vol = volume/num_dispenses
        print(f"Dispensing {dispense_vol} mL {num_dispenses} times")
        for i in range (0, num_dispenses):        
             self.c9.set_pump_valve(reservoir_index,self.c9.PUMP_VALVE_LEFT)
             self.c9.aspirate_ml(reservoir_index,dispense_vol)
             self.c9.set_pump_valve(reservoir_index,self.c9.PUMP_VALVE_RIGHT)
             self.c9.dispense_ml(reservoir_index,dispense_vol)
        time.sleep(1)
        vial_volume = self.get_vial_info(vial_index,'vial_volume')
        self.VIAL_DF.at[vial_index,'vial_volume']=(vial_volume+volume)
        self.save_robot_status()

        #Step 4: Return the vial back to home
        self.c9.move_carousel(0,0)
        self.recap_clamp_vial()
        self.return_vial_home(vial_index)

    #Check the original status of the vial in order to send it to its home location
    def return_vial_home(self,vial_name):
        """
        Return the specified vial to its home location.
        Args:
            `vial_name` (str): Name of the vial to return home.
        """
        vial_index = self.normalize_vial_index(vial_name) #Convert to int if needed
        
        home_location = self.get_vial_info(vial_index,'home_location')
        home_location_index = self.get_vial_info(vial_index,'home_location_index')
        
        vial_location = self.get_vial_info(vial_index,'location')
        if vial_location == 'clamp' and self.GRIPPER_STATUS == "Cap":
            self.recap_clamp_vial()
        self.move_vial_to_location(vial_index,home_location,home_location_index)
        self.save_robot_status()

    #Drop off a vial at a location that you already have
    def drop_off_vial(self, vial_name, location, location_index):

        vial_index = self.normalize_vial_index(vial_name) #Convert to int if needed

        destination = self.get_location(False,location,location_index)
        occupying_vial = self.get_vial_in_location(location,location_index)
        # Allow drop-off if the location is empty or occupied by the same vial
        destination_empty = (occupying_vial is None) or (occupying_vial == vial_index)

        self.check_for_errors([[destination_empty, True, "Cannot move vial to destination, destination full"]],True)

        self.c9.goto_safe(destination) #move vial to destination
        self.c9.open_gripper() #release vial
        
        self.VIAL_DF.at[vial_index, 'location']=location
        self.VIAL_DF.at[vial_index, 'location_index']=location_index
        self.GRIPPER_STATUS = None #We no longer have the vial
        self.GRIPPER_VIAL_INDEX = None
        self.save_robot_status() #Update in memory

    def grab_vial(self,vial_name):
        
        vial_index = self.normalize_vial_index(vial_name) #Convert to int if needed
        
        print("Grabbing vial")
        initial_location = self.get_vial_location(vial_index, False)
        loc = self.get_vial_info(vial_index,'location')

        if loc == 'clamp' and self.GRIPPER_STATUS == "Cap":
            self.recap_clamp_vial()

        error_check_list = [] #List of specific errors for this method
        error_check_list.append([self.GRIPPER_STATUS is None, True, "Cannot move vial to destination, gripper full"])
        error_check_list.append([self.HELD_PIPET_INDEX is None, True, "Cannot move vial to destination, robot holding pipet"])
        error_check_list.append([self.is_vial_movable(vial_index), True, "Can't move vial, vial is uncapped."])  

        self.check_for_errors(error_check_list,True) #Check for an error, and pause if there's an issue

        #self.open_gripper()
        self.goto_location_if_not_there(initial_location) #move to vial
        self.c9.close_gripper() #grip vial
        
        self.GRIPPER_STATUS = "Vial" #Update the status of the robot
        self.GRIPPER_VIAL_INDEX = vial_index
        self.save_robot_status() #Save the status of the robot

    #Send the vial to a specified location
    def move_vial_to_location(self,vial_name:str,location:str,location_index:int):
        """
        Moves vial to specified location

        Args:
            vial_name (str): Name of the vial to move
            location (str): Description of the location to move to (e.g., 'clamp', 'photoreactor_array', 'main_8mL_rack', 'heater')
            location_index (int): Index of the location to move to (e.g., 0 for the first position, 1 for the second position, etc.)
         """
        
        vial_index = self.normalize_vial_index(vial_name) #Convert to int if needed

        print("Moving vial " + self.get_vial_info(vial_index,'vial_name') + " to " + location + ": " + str(location_index))
        self.grab_vial(vial_index) #Grab the vial
        self.drop_off_vial(vial_index,location,location_index) #Drop off the vial

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
        if self.simulate:
            return 0
        difference = np.sum(np.absolute(np.array(loc_2)[1:4] - np.array(loc_1)[1:4]))
        return difference
    
    #Removes the target vial, vortexes it, then puts it back
    def vortex_vial(self, vial_name, vortex_time, vortex_speed=70):
        """
        Moves target vial up, vortexes for a specified time and speed and returns it to its original location.
        Args:
            `vial_name` (str): Name of the vial to vortex
            `vortex_time` (float): Time (in seconds) to vortex
            `vortex_speed` (float): Speed of vortexing (default is 70)
        """
        
        vial_index = self.normalize_vial_index(vial_name) #Convert to int if needed

        print("Vortexing Vial: " + self.get_vial_info(vial_index,'vial_name'))
        
        #Check to see if the vial is capped
        if self.GRIPPER_VIAL_INDEX == vial_index  and self.GRIPPER_STATUS == "Cap":
            self.recap_clamp_vial()
        self.grab_vial(vial_index)
    
        #Vortex
        self.c9.move_z(292) #Move to a higher height
        self.c9.move_axis(self.c9.GRIPPER, 1000*vortex_time*vortex_speed, vel=vortex_speed,accel=10000)
        self.c9.reduce_axis_position(axis=self.c9.GRIPPER)

        location = self.get_vial_info(vial_index,'location')
        location_index = self.get_vial_info(vial_index,'location_index')

        #Move the vial back to its original location
        self.drop_off_vial(vial_index, location, location_index)
        
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
    def is_vial_movable(self, vial_name):

        vial_index = self.normalize_vial_index(vial_name) #Convert to int if neede

        movable = False
        movable = self.get_vial_info(vial_index,'capped') == True and self.get_vial_info(vial_index,'vial_type') == "8_mL"
        return movable
    
    #Check to see if the pipet can have liquids added/removed
    def is_vial_pipetable(self, vial_name):

        vial_index = self.normalize_vial_index(vial_name) #Convert to int if needed

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
  
    def dispense_solid_into_vial(target_vial_index, dispense_mass): #Can work on this
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
    def get_vial_info(self,vial_name,column_name):

        vial_index = self.normalize_vial_index(vial_name) #Convert to int if needed

        values = self.VIAL_DF.loc[self.VIAL_DF['vial_index'] == vial_index, column_name].values
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
    def get_min_pipetting_height(self,vial_name):

        vial_index = self.normalize_vial_index(vial_name) #Convert to int if needed

        #The height at which the pipet touches the ground for the 1 mL pipet
        min_height = None   

        location_name = self.get_vial_info(vial_index,'location')

        #These constants should be considered properties that we can take in from files later
        CLAMP_BASE_HEIGHT = 114.5
        VIAL_RACK_BASE_HEIGHT = 68.5
        PR_BASE_HEIGHT = 85 #Need to fine tune this
        LARGE_VIAL_BASE_HEIGHT = 55    
        SMALL_VIAL_BASE_HEIGHT = 98
    
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
        elif location_name == '12_well_ilya':
            min_height = 72.00

        return min_height

    #Get the position of a vial right now
    def get_vial_location(self,vial_name,use_pipet):
        vial_index = self.normalize_vial_index(vial_name) #Convert to int if needed
        location_name = self.get_vial_info(vial_index,'location')
        location_index = self.get_vial_info(vial_index,'location_index')
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
            elif location_name == '12_well_ilya':
                location = ilya_wellplate[location_index]
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