from re import S
from Locator import *
import numpy as np
import time
import math
import pandas as pd
import slack_agent
import yaml
from unittest.mock import MagicMock
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class North_Track:

    #Controller
    c9 = None
    MAX_HEIGHT = 0
    RELEASE_DISTANCE_Y = 2400
    LID_DISTANCE_Y = 4000

    #Well plate active areas
    NR_WELL_PLATE_X = [131854,105860,81178]
    #NR_WELL_PLATE_Y = [88750, 86965, 89155]
    NR_WELL_PLATE_Y = [88750, 88000, 89155]
    #NR_WELL_PLATE_Y_RELEASE = 86000
    
    #Transit constants
    WELL_PLATE_TRANSFER_Y = 75000
    CYT_SAFE_X = 47747
    
    #Release into cytation
    CYT_TRAY_Y_RELEASE = 5500
    CYT_TRAY_Y_GRAB = 8500
    CYT_TRAY_X = 68608

    #QUARTZ_WP_OFFSET = 1300
    QUARTZ_WP_OFFSET = 3000

    #Speed horizontal
    DEFAULT_X_SPEED = 50

    #Speed vertical
    DEFAULT_Y_SPEED = 50

    """New double WP stack"""
    DOUBLE_SOURCE_X = 175 #50 with old holder (but scratches WP) 
    DOUBLE_WASTE_X = 14550 
    DOUBLE_TRANSFER_X = 28000 
    DOUBLE_SOURCE_Y_96 = [83500, 76800, 70700] #first element = height when 1 WP is in stack
    DOUBLE_WASTE_Y_96 = [83000, 76350, 70200] #first element = height when dropping off 1st WP to waste
    DOUBLE_SOURCE_Y_48 =  [83500, 74000, 64500]
    DOUBLE_WASTE_Y_48 = [83000, 73500, 64000]

    SOURCE_HEIGHTS_DICT = {"96 WELL PLATE": DOUBLE_SOURCE_Y_96, "48 WELL PLATE": DOUBLE_SOURCE_Y_48}
    WASTE_HEIGHTS_DICT = {"96 WELL PLATE": DOUBLE_WASTE_Y_96, "48 WELL PLATE": DOUBLE_WASTE_Y_48}

    #num_source = 0 #number of wellplates in source stack 
    well_plate_df = None

    #Let's initialize the number of well plates from a file
    def __init__(self, c9, simulate = False, logger=None):
        self.c9 = c9
        self.logger = logger
        self.logger.info("Initializing North Track...")
        #self.well_plate_df = pd.read_csv("../utoronto_demo/status/wellplate_storage_status.txt", sep=r",", engine="python") #TODO: not sure if needed anymore?
        #self.num_source = int(self.well_plate_df.loc[self.well_plate_df['Location']=='Input']['Status'].values)

        self.NUM_SOURCE = 0
        self.NUM_WASTE = 0
        self.CURRENT_WP_TYPE = "96 WELL PLATE"
        self.NR_OCCUPIED = False
        self.simulate = simulate

        #Load yaml data
        self.logger.debug("Loading track status from file: %s", "../utoronto_demo/status/track_status.yaml")
        self.TRACK_STATUS_FILE = "../utoronto_demo/status/track_status.yaml"
        self.get_track_status() #set NUM_SOURCE, NUM_WASTE, CURRENT_WP_TYPE and NR_OCCUPIED from yaml file
        self.reset_after_initialization()
    
    def check_input_file(self, pause_after_check=True, visualize=True):
        self.logger.info(f"--Wellplate status-- \n Wellplate type: {self.CURRENT_WP_TYPE} \n Number in source: {self.NUM_SOURCE} \n Number in waste: {self.NUM_WASTE} \n NR Pipetting area occupied: {self.NR_OCCUPIED}")

        if visualize:
            self.logger.info("Visualizing wellplate status...")
            fig, ax = plt.subplots(figsize=(10, 6))

            plate_width = 2.5
            plate_height = 0.4
            spacing = 0.1

            for i in range(self.NUM_SOURCE):
                rect = plt.Rectangle((1, i * (plate_height + spacing)), plate_width, plate_height,
                                    edgecolor='black', facecolor='lightblue')
                ax.add_patch(rect)
                ax.text(1 + plate_width / 2, i * (plate_height + spacing) + plate_height / 2,
                        self.CURRENT_WP_TYPE, ha='center', va='center', fontsize=8)

            for i in range(self.NUM_WASTE):
                rect = plt.Rectangle((5, i * (plate_height + spacing)), plate_width, plate_height,
                                    edgecolor='black', facecolor='lightcoral')
                ax.add_patch(rect)
                ax.text(5 + plate_width / 2, i * (plate_height + spacing) + plate_height / 2,
                        self.CURRENT_WP_TYPE, ha='center', va='center', fontsize=8)

            if self.NR_OCCUPIED:
                rect = plt.Rectangle((9, 0), plate_width, plate_height,
                                    edgecolor='black', facecolor='khaki')
                ax.add_patch(rect)
                ax.text(9 + plate_width / 2, plate_height / 2,
                        "Occupied", ha='center', va='center', fontsize=8)
                ax.text(9 + plate_width / 2, plate_height + 0.2, "NR Pipette Area",
                        ha='center', va='bottom', fontsize=10, weight='bold')

            ax.text(1 + plate_width / 2, self.NUM_SOURCE * (plate_height + spacing) + 0.2, "Source Stack",
                    ha='center', va='bottom', fontsize=10, weight='bold')
            ax.text(5 + plate_width / 2, self.NUM_WASTE * (plate_height + spacing) + 0.2, "Waste Stack",
                    ha='center', va='bottom', fontsize=10, weight='bold')

            ax.set_xlim(0, 12)
            ax.set_ylim(0, max(self.NUM_SOURCE, self.NUM_WASTE) * (plate_height + spacing) + 1)
            ax.axis('off')
            ax.set_title("-- Please Confirm Wellplate Status --", fontsize=14, weight='bold')
            plt.tight_layout()
            plt.show()

        if pause_after_check and not self.simulate:
            input("Only hit enter if the status of the well plates is correct, otherwise hit ctrl-c")

    def reset_after_initialization(self):
        None
        #Return well plate if well-plate in gripper
        #Send used well plate to trash
        #Send unused well plate back to source

    def set_horizontal_speed(self,vel):
        self.logger.debug("Setting horizontal speed to %d", vel)
        self.c9.DEFAULT_X_SPEED = vel

    def set_vertical_speed(self,vel):
        self.logger.debug("Setting vertical speed to %d", vel)
        self.c9.DEFAULT_Y_SPEED = vel    

    def open_gripper(self):
        self.logger.debug("Opening gripper")
        self.c9.set_output(4, True)  
        self.c9.set_output(5, False)
        self.c9.delay(2)
    
    def close_gripper(self):
        self.logger.debug("Closing gripper")
        self.c9.set_output(5, True)  #gripper close
        self.c9.set_output(4, False)
        self.c9.delay(2)

    def grab_well_plate_from_nr(self, well_plate_num, grab_lid=False,quartz_wp=False):
        """
        Grab a well plate from the designated wellplate stand. Currently directly moves horizontally, then down to the wellplate stand, grabs wp and ends by moving up slightly to WELL_PLATE_TRANSFER_Y.
        """
        self.logger.info("Grabbing well plate %d from NR", well_plate_num)
        move_up = 0
        if grab_lid:
            move_up = self.LID_DISTANCE_Y 
        if quartz_wp:
            move_up = -self.QUARTZ_WP_OFFSET #
        self.open_gripper() #TODO: move up to safe height first?
        self.c9.move_axis(7, self.NR_WELL_PLATE_X[well_plate_num], vel=self.DEFAULT_X_SPEED) #left to WP
        self.c9.move_axis(6, self.NR_WELL_PLATE_Y[well_plate_num]-move_up, vel=self.DEFAULT_Y_SPEED) #down
        self.close_gripper()
        self.c9.move_axis(6, self.WELL_PLATE_TRANSFER_Y, vel=self.DEFAULT_Y_SPEED) #up slightly

        #save status
        self.NR_OCCUPIED = False
        self.save_track_status() #update yaml

    def move_gripper_to_cytation(self):
        self.logger.debug("Moving gripper to Cytation")
        self.c9.move_axis(7, self.CYT_SAFE_X, vel=self.DEFAULT_X_SPEED) #past Cytation loading
        self.c9.move_axis(6,self.MAX_HEIGHT,vel=self.DEFAULT_Y_SPEED) #up to max height
    
    def release_well_plate_in_cytation(self,quartz_wp=False):
        self.logger.info("Releasing well plate in Cytation")
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
        self.logger.info("Grabbing well plate from Cytation")
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
        self.logger.info("Returning well plate to NR well plate stand %d", well_plate_num)
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

        self.NR_OCCUPIED = True
        self.save_track_status() #update yaml
    
    def origin(self):
        self.c9.move_axis(6, self.MAX_HEIGHT, vel=self.DEFAULT_Y_SPEED) #max_height
        self.c9.move_axis(7, 0, vel=self.DEFAULT_X_SPEED)
        self.logger.debug("Moving North Track to home position")
      
    #Save the status of the robot to memory
    def save_track_status(self):
        #self.logger.debug("Saving track status to file: %s", self.TRACK_STATUS_FILE)
        track_status = {
            "num_in_source": self.NUM_SOURCE,
            "num_in_waste": self.NUM_WASTE,
            "wellplate_type": self.CURRENT_WP_TYPE,
            "nr_occupied": self.NR_OCCUPIED
        }

        if not self.simulate: #not simulating
            # Writing to a file
            with open(self.TRACK_STATUS_FILE, "w") as file:
                yaml.dump(track_status, file, default_flow_style=False)

    #Update the status of the robot from yaml file
    def get_track_status(self):
        self.logger.debug("Getting track status from file: %s", self.TRACK_STATUS_FILE)
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

    def calculate_wp_stack_height(self, num, wp_type, pickup):
        self.logger.debug("Calculating well plate stack height for num: %d, wp_type: %s, pickup: %s", num, wp_type, pickup)
        wp_parameters = {"96 WELL PLATE": {"y-int": 83500, "slope": -6400}, "48 WELL PLATE": {"y-int": 83500, "slope":-9500}}
        height = wp_parameters[wp_type]["y-int"] + wp_parameters[wp_type]["slope"]*(num-1) 
        MAX_HEIGHT = 20000 #TODO: need to confirm (seems close to the top)

        #print(f"HEIGHT = {height}")

        if height >= MAX_HEIGHT and height <= wp_parameters[wp_type]["y-int"]: #beneath max height and higher than the lowest source
            if pickup == True:
                return height
            else: #for dropping off
                if num == 0:
                    return height #base height is the same
                else:
                    return height-500 #drop of slightly higher than drop off
        else:
            return -1 #invalid input
        
    def get_new_wellplate(self, move_home_afterwards = True): #double WP stack 
        self.logger.info("Getting new wellplate from source stack")
        """Get a new wellplate from the source stack (in the double stack holder) and move to north robot pipetting area"""
        DOUBLE_SOURCE_Y = self.calculate_wp_stack_height(self.NUM_SOURCE, self.CURRENT_WP_TYPE, pickup=True)
        

        if self.NUM_SOURCE > 0 and DOUBLE_SOURCE_Y != -1 and self.NR_OCCUPIED == False: #still have well plates in stack & pipetting area is empty
            self.logger.debug(f"Getting {self.get_ordinal(self.NUM_SOURCE)} wellplate from source at {DOUBLE_SOURCE_Y}")
            
            #move to source stack and grab wellplate
            self.open_gripper()
            self.c9.move_axis(6, self.MAX_HEIGHT, vel= self.DEFAULT_Y_SPEED) #up to max height
            self.c9.move_axis(7, self.DOUBLE_SOURCE_X, vel=self.DEFAULT_X_SPEED) #above source
            self.c9.move_axis(6, DOUBLE_SOURCE_Y, vel=20) #down to WP height (slowed)
            self.close_gripper()
            
            #move up from source stack to "safe" area and move down
            self.c9.move_axis(6, self.MAX_HEIGHT, vel=self.DEFAULT_Y_SPEED) #up to max height
            self.c9.move_axis(7, self.DOUBLE_TRANSFER_X, vel=self.DEFAULT_X_SPEED) #to "safe" area
            self.c9.move_axis(6, self.WELL_PLATE_TRANSFER_Y, vel=20) #to transfer area (slowed)
            
            self.NUM_SOURCE -= 1
            self.save_track_status() #update yaml

            self.return_well_plate_to_nr(0)

            if move_home_afterwards:
                self.origin()
        
        else:
            if self.NUM_SOURCE <= 0:
                self.pause_after_error("Wellplate stack is empty")
            elif DOUBLE_SOURCE_Y == -1:
                self.pause_after_error("Invalid height calculated (too many wellplates in stack)")
            if self.NR_OCCUPIED == True:
                self.pause_after_error("Cannot get wellplate, NR pipetting area is occupied")
    
    def discard_wellplate(self, wellplate_num=0, move_home_afterwards = True): #double WP stack
        """Grabs a wellplate (from desginated wellplate stack) and discards it into the waste stack (in the double stack holder). ENSURE North Robot is homed!!!
        
        Args:
            `wellplate_num`(int): The number identifying which wellplate stand to discard a wellplate from (0 for the pipetting one) 
        """
        DOUBLE_WASTE_Y = self.calculate_wp_stack_height(self.NUM_WASTE+1, self.CURRENT_WP_TYPE, pickup=False)
        
        if DOUBLE_WASTE_Y != -1 and self.NR_OCCUPIED == True: #can still hold an additional wellplate & pipetting area is occupied
            self.logger.info(f"Discarding wellplate as the {self.get_ordinal(self.NUM_WASTE+1)} WP in waste stack at height: {DOUBLE_WASTE_Y}")
            
            #move up to max height, then to NR area to grab wellplate
            self.c9.move_axis(6, self.MAX_HEIGHT, vel=self.DEFAULT_Y_SPEED) #up to max height
            self.grab_well_plate_from_nr(wellplate_num) #move to wellplate area to grab wellplate

            #cross the cytation -- move down to transfer_Y
            self.c9.move_axis(6, self.WELL_PLATE_TRANSFER_Y, vel=self.DEFAULT_Y_SPEED) #down to height for passing cytation
            self.c9.move_axis(7, self.DOUBLE_TRANSFER_X, vel=self.DEFAULT_X_SPEED) #to "safe" area

            #move up to waste stack and go down to drop off wp
            self.c9.move_axis(6, self.MAX_HEIGHT, vel=self.DEFAULT_Y_SPEED) #up to max height
            self.c9.move_axis(7, self.DOUBLE_WASTE_X, vel=self.DEFAULT_X_SPEED) #above waste stack
            self.c9.move_axis(6, DOUBLE_WASTE_Y, vel=20) #down to drop off wellplate (slowed)
            self.open_gripper()
            self.c9.move_axis(6, self.MAX_HEIGHT, vel=self.DEFAULT_Y_SPEED) #move back up to max height

            self.NUM_WASTE += 1
            self.save_track_status()

            if move_home_afterwards: #home after dropping off wellplate into waste stack
                self.origin()
        else:
            if DOUBLE_WASTE_Y == -1:
                self.pause_after_error("Wellplate stack is too full for discarding another well plate.")
            if self.NR_OCCUPIED == False:
                self.pause_after_error("Cannot discard wellplate, designated wellplate stand is empty.")
        

        #Integrating error messages and deliberate pauses
    
    def pause_after_error(self,err_message,send_slack=True):
        self.logger.error(err_message)
        if send_slack and not isinstance(self.c9, MagicMock):
            slack_agent.send_slack_message(err_message)
        if not self.simulate:  # not simulating
            input("Waiting for user to press enter or quit after error...")
    
    def get_ordinal(self,n): #convert n into an ordinal number (ex. 1st, 2nd, 4th)  -- from chatgpt
        if 10 <= n % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        return f"{n}{suffix}"

class North_Powder:

    p2 = None
    c9 = None

    def __init__(self, c9,simulate = False, logger=None):
        
        self.c9 = c9
        self.simulate = simulate
        self.logger = logger
        self.logger.debug("Initializing North Powder Dispenser...")
        if not simulate:
            from north import NorthC9
            self.p2 = NorthC9('C', network=c9.network)

    def activate_powder_channel(self, channel=0):
        self.logger.debug(f"Activating powder channel: {channel}")
        self.p2.activate_powder_channel(channel)
        self.p2.home_OL_stepper(0, 300)

    # shake the cartridge for t ms, with frequency f [40, 180] hz, amplitude [60, 100] %
    def shake(self,t, f=80, a=80, wait=True):
        self.logger.debug("Shaking dispenser cartridge...")
        return self.p2.amc_pwm(int(f), int(t), int(a), wait=wait)
    
    def set_opening(self, channel, deg):
        self.logger.debug(f"Setting opening for channel {channel} to {deg} degrees")
        self.p2.move_axis(channel, deg*(1000/360.0))

    def dispense_powder_time(self, channel, time):
        self.logger.info(f"Dispensing powder for {time} ms on channel {channel}")
        self.activate_powder_channel(channel)
        self.c9.move_carousel(70,70)
        
        initial_mass = self.c9.read_steady_scale()
        self.set_opening(channel,45)
        self.shake(time)
        self.set_opening(channel,0)
        final_mass = self.c9.read_steady_scale()
        dispensed_mass = final_mass - initial_mass
        self.logger.info("Mass dispensed: ", dispensed_mass)
        
        self.c9.move_carousel(0,0)

    def cl_pow_dispense(self,mg_target, channel, protocol=None, zero_scale=False, max_tries=20):
        import settings.powder_settings as settings
        start_t = time.perf_counter()
        mg_togo = mg_target

        if protocol is None:
            protocol = settings.default_ps
        ps = protocol.fast_settings
        if mg_togo < protocol.slow_settings.thresh:
            ps = protocol.slow_settings
        elif mg_togo < protocol.med_settings.thresh:
            ps = protocol.med_settings

        # intialize
        self.set_opening(channel,0)  # make sure everything starts closed
        prev_mass = 0
        delta_mass = 0
        shake_t = ps.min_shake_t

        if zero_scale:
            self.c9.zero_scale()
            self.c9.delay(protocol.scale_delay)
        tare = self.c9.read_steady_scale() * 1000

        meas_mass = 0
        count = 0
        while mg_togo > protocol.tol and count < max_tries:
            count += 1

            self.set_opening(channel,ps.opening_deg)
            self.shake(shake_t, ps.freq, ps.amplitude)
            if ps.shut_valve:
                self.set_opening(channel,0)
            self.c9.delay(0.5)
            self.c9.read_steady_scale()  # dummy read to wait for steady
            self.c9.delay(protocol.scale_delay)  # delay after steady to allow for more settling time
            meas_mass = self.c9.read_steady_scale() * 1000 - tare

            mg_togo = mg_target - meas_mass
            delta_mass = meas_mass - prev_mass
            prev_mass = meas_mass

            if mg_togo < protocol.slow_settings.thresh:
                ps = protocol.slow_settings
            elif mg_togo < protocol.med_settings.thresh:
                ps = protocol.med_settings

            iter_target = (ps.percent_target * mg_togo)
            max_new_t = ps.max_growth * shake_t
            if delta_mass <= 0:
                shake_t = max_new_t
            else:
                shake_t *= (iter_target / delta_mass)
            shake_t = min(max_new_t, shake_t)  # no larger than max growth allows
            shake_t = max(ps.min_shake_t, shake_t)  # no shorter than min time
            shake_t = min(ps.max_shake_t, shake_t)  # no longer than max time

            self.logger.debug(f'Iteration {count}:')
            self.logger.debug(f'\tJust dispensed:  {delta_mass:.1f} mg')
            self.logger.debug(f'\tRemaining:       {mg_togo:.1f} mg')
            self.logger.debug(f'\tNext target:     {iter_target:.1f} mg')
            self.logger.debug(f'\tNext time:       {int(shake_t)} ms')
            self.logger.debug('')

            self.set_opening(channel, 0)

            self.logger.debug(f'Result:')
            self.logger.debug(f'\tLast iter:  {delta_mass:.1f} mg')
            self.logger.debug(f'\tDispensed: {meas_mass:.1f} mg')
            self.logger.debug(f'\tRemaining: {mg_togo:.1f} mg')
            self.logger.debug(f'\tTime:      {int(time.perf_counter() - start_t)} s')
            self.logger.debug('')

        return meas_mass

    def dispense_powder_mg(self, mass_mg, channel=0):
        self.logger.info(f"Dispensing {mass_mg} mg on channel {channel}")
        
        self.activate_powder_channel(channel)
        self.c9.move_carousel(66.5,70)
        #Dispense protocol
        mass_dispensed = self.cl_pow_dispense(mass_mg, channel)
        self.logger.info(f"Mass dispensed: {mass_dispensed} mg")

        self.c9.move_carousel(0,0)
        return mass_dispensed

class North_Temp:

    t8 = None
    c8 = None

    def __init__(self,c9,simulate = False, logger=None):
        self.simulate = simulate
        self.c9 = c9
        self.logger = logger

        self.logger.debug("Initializing temperature controller...")

        if not self.simulate:
            from north import NorthC9
            self.t8 = NorthC9('B', network=c9.network)
            self.c8 = NorthC9('D', network=c9.network)
        else:
            self.t8 = MagicMock()
            self.c8 = MagicMock()
    
    def autotune(self,target_temp,channel=0):
        self.logger.debug(f"Autotuning channel {channel} to target temperature {target_temp}C")
        self.t8.disable_channel(channel)
        self.t8.set_temp(channel, target_temp)
        self.t8.enable_channel(channel)
        self.t8.temp_autotune(channel, True)

    def set_temp(self,target_temp,channel=0):
        self.logger.debug(f"Setting channel {channel} to target temperature {target_temp}C")
        self.t8.set_temp(channel, target_temp)
        self.t8.enable_channel(channel)
    
    def get_temp(self,channel=0):
        return self.t8.get_temp(channel)

    def turn_off_heating(self,channel=0):
        self.logger.debug(f"Turning off heating for channel {channel}")
        self.t8.disable_channel(channel)

    def turn_on_stirring(self,speed=10000):
        self.logger.debug(f"Turning on stirring at speed {speed}")
        self.c8.spin_axis(0, speed)

    def turn_off_stirring(self):
        self.logger.debug("Turning off stirring")
        self.c8.spin_axis(0,0)

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
  
    #Initialize the status of the robot. 
    def __init__(self,c9,vial_file=None,simulate=False, logger=None):

        self.c9 = c9
        self.logger = logger
        self.VIAL_FILE = vial_file

        self.logger.info("Initializing North Robot...")

        self.get_robot_status() #Update the status of the robot from memory
        self.reset_after_initialization() #Reset everything that may not be as desired, eg return to "Home"
        self.simulate = simulate
        self.load_pumps() #Load the pumps
        #sys.excepthook = self.global_exception_handler

    #Load the pumps and set volumes
    def load_pumps(self):
        self.logger.debug("Loading pumps...")
        self.c9.pumps[0]['volume'] = 1
        self.c9.pumps[1]['volume'] = 2.5
        self.c9.set_pump_speed(1, 15)

    def visualize_racks(self, vial_status, fig_size_x=16, fig_size_y=10, xlim=12, ylim=6):
        self.logger.info("Visualizing vial racks...")
        fig, ax = plt.subplots(figsize=(fig_size_x, fig_size_y))
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-1, xlim)
        ax.set_ylim(-1, ylim)
        ax.invert_yaxis()
        ax.invert_xaxis()
        ax.grid(True)

        # Example Racks
        self.visualize_rack(
            ax=ax,
            rack_data=vial_status,
            rack_name='Main Rack',
            location_filter='main_8mL_rack',
            num_cols=8,
            num_rows=6,
            filled_color='lightgreen',
            offset_x=4.0,
            offset_y=0,
            title='Main Rack'
        )

        self.visualize_rack(
            ax=ax,
            rack_data=vial_status,
            rack_name='Second Rack',
            location_filter='large_vial_rack',
            num_cols=2,
            num_rows=3,
            filled_color='lightblue',
            offset_x=0.5,
            offset_y=0,
            title='20 mL Rack'
        )

        plt.tight_layout()
        plt.show()

    def visualize_rack(self,ax,rack_data,rack_name,location_filter,num_cols,num_rows,filled_color='lightgreen',offset_x=0,offset_y=0,title=None):
        self.logger.debug(f"Visualizing rack: {rack_name} with filter: {location_filter}")
        rack_data = rack_data[rack_data['location'] == location_filter]

        all_indices = set(range(num_cols * num_rows))
        present_indices = set(rack_data['location_index'])
        missing_indices = all_indices - present_indices

        # Draw filled/empty vials
        for _, row in rack_data.iterrows():
            index = row['location_index']
            col = index // num_rows
            row_pos = index % num_rows

            # Adjust for offset
            x = col + offset_x
            y = row_pos + offset_y

            fill_color = filled_color if row['vial_volume'] > 0 else 'white'
            circle = patches.Circle((x, y), 0.45, edgecolor='black', facecolor=fill_color)
            ax.add_patch(circle)

            if not row['capped']:
                cap_status = "uncapped"
            elif row['cap_type'] == 'open':
                cap_status = "open cap"
            elif row['cap_type'] == 'closed':
                cap_status = "closed cap"
            else:
                cap_status = "unknown cap"

            text = f"{row['vial_name']}\n{row['vial_volume']:.2f} mL\n{cap_status}"
            ax.text(x, y, text, ha='center', va='center', fontsize=8)

        # Draw dotted circles for missing vials
        for index in missing_indices:
            col = index // num_rows
            row_pos = index % num_rows

            x = col + offset_x
            y = row_pos + offset_y

            circle = patches.Circle((x, y), 0.45, edgecolor='black', facecolor='none', linestyle='dotted')
            ax.add_patch(circle)

        # Add bounding box and title for this rack
        rect = patches.Rectangle(
            (offset_x - 0.5, offset_y - 0.5),
            num_cols,
            num_rows,
            linewidth=2,
            edgecolor='gray',
            facecolor='none'
        )
        ax.add_patch(rect)

        if title:
            ax.text(
                offset_x + num_cols / 2 - 0.5,
                offset_y + num_rows - 0.3,
                title,
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )

    #Check the status of the input vial file
    def check_input_file(self,pause_after_check=True, visualize=True):
        """
        Prints the vial status dataframe for user to confirm the initial state of your vials.
        """
        vial_status = pd.read_csv(self.VIAL_FILE, sep=",")
        self.logger.info(vial_status)

        if visualize:
            self.visualize_racks(vial_status)

        if pause_after_check and not self.simulate:
            input("Only hit enter if the status of the vials (including open/close) is correct, otherwise hit ctrl-c")

    #Reset positions and get rid of any pipet tips... .Note that this does not do 100% of what it is intended to
    def reset_after_initialization(self):
        self.logger.debug("Physical initialization of North Robot...")
        self.c9.default_vel = 20 #Set the default speed of the robot
        self.c9.open_clamp()
        
        try: #Try resetting. If a home is required, some of these actions may fail. 
            if self.PIPET_FLUID_VIAL_INDEX is not None and self.PIPET_FLUID_VOLUME > 0: #If we still have liquid leftover
                self.logger.warning("The robot reports having liquid in its tip... Returning that liquid...")
                vial_index = self.PIPET_FLUID_VIAL_INDEX 
                volume = self.PIPET_FLUID_VOLUME
                self.dispense_into_vial(vial_index,volume)
            
            if self.HELD_PIPET_INDEX is not None: #If we've got a pipet tip, let's get rid of it
                self.logger.warning("The robot reports having a tip, removing the tip")
                self.remove_pipet()
            
            if self.GRIPPER_STATUS is not None: #If the gripper is full, let's address that
                vial_index = self.GRIPPER_VIAL_INDEX
                if self.GRIPPER_STATUS == "Cap": #If we have the cap, the vial must be in the gripper.
                    self.logger.warning("The robot reports having a cap in its gripper... Recapping the clamp vial...")
                    self.recap_clamp_vial()
                    self.return_vial_home(vial_index)
                elif self.GRIPPER_STATUS == "Vial": #We need to know where this vial is intended to be!
                    self.logger.warning("The robot reports having a vial in the gripper... Returning that vial home...")
                    location = self.get_vial_info(vial_index,'home_location')
                    location_index = self.get_vial_info(vial_index,'home_location_index')
                    self.drop_off_vial(vial_index,location=location,location_index=location_index)
            self.move_home()
        except Exception as e:
            self.logger.warning("An error occured during initialization, homing components: ", e)
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
        self.logger.debug("Saving robot status to file: %s", self.ROBOT_STATUS_FILE)
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
        self.logger.debug("Getting robot status from file: %s", self.ROBOT_STATUS_FILE)
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
        self.logger.debug("Removing pipet")
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

        print(f"Getting pipet number {active_pipet_num} from rack index: {pipet_rack_index}")

        #This is to pause the program and send a slack message when the pipets are out!
        MAX_PIPETS=95 # This is based off the racks
        if active_pipet_num > MAX_PIPETS:
            self.pause_after_error("The North Robot is out of pipets! Please refill pipets then hit enter on the terminal!")
            self.PIPETS_USED=[0,0]
            self.save_robot_status()
            active_pipet_num=0

        horizontal_rack_used = (active_pipet_num <= 47)

        self.logger.debug(f"Getting pipet number: {active_pipet_num} from rack {pipet_rack_index}")

        #This conversion is neccessary to take the tips in the correct order. Only for the horizontal rack. 
        if horizontal_rack_used:
            num = (active_pipet_num%16)*3+math.floor(active_pipet_num/16)
            
        else:
            num = (active_pipet_num - 48)
            num = (16 * (math.floor(num/16)+1) - 1) - num%16

        #First move to the xy location 
        if pipet_rack_index == self.LOWER_PIPET_ARRAY_INDEX:
            if horizontal_rack_used:
                location = p_capture_grid[num]
            else:
                location = pgrid_low_2[num]
        elif pipet_rack_index == self.HIGHER_PIPET_ARRAY_INDEX: 
            if horizontal_rack_used:
                location = p_capture_high[num]
            else:
                location = pgrid_high_2[num]

        #Move to get the pipet tip
        self.c9.goto_xy_safe(location)
        base_height = self.c9.counts_to_mm(3, location[3])
        self.c9.move_z(base_height) 

        if pipet_rack_index == self.LOWER_PIPET_ARRAY_INDEX:
            if horizontal_rack_used:
                location = p_capture_up[num]
                self.c9.goto(location, vel=5) #Move to the pipet tip
            else:
                self.move_rel_xyz(x_distance = -2, y_distance=2, z_distance=50, vel=5) #Test this out
                #self.c9.move_z(292,vel=5)
                   
        else:
            self.c9.move_z(292,vel=5) #Move to a safe height (292)

        #We have a pipet. What kind of pipet do we have? How many pipets are left in the rack?
        self.HELD_PIPET_INDEX = pipet_rack_index
        self.PIPETS_USED[pipet_rack_index] += 1

        #Update the status of the robot in memory
        self.save_robot_status()
 
    def adjust_pump_speed(self, pump, pump_speed):
        self.logger.debug(f"Adjusting pump {pump} speed to {pump_speed}")
        if pump_speed != self.CURRENT_PUMP_SPEED:
            self.c9.set_pump_speed(pump, pump_speed)

#New aspirate function
    def pipet_aspirate(self, amount, settling_time=0):
        
        self.logger.debug(f"Aspirating {amount} mL then waiting {settling_time} s")

        if amount <= 1:
            try:
                self.c9.aspirate_ml(0,amount)
            except:
                self.logger.warning("Aspirate exceeded limit: Aspirating to maximum")
                self.c9.move_pump(0,3000)
                slack_agent.send_slack_message(f"Aspirate was exceeded for {amount} mL. Aspirating to maximum volume of 1 mL.")
                #self.pause_after_error("Cannot aspirate. Likely a pump position issue", True)
        else:
            self.pause_after_error("Cannot aspirate more than 1 mL", True)

        if not self.simulate:
            time.sleep(settling_time) #Wait a bit after aspirating

#New dispense function
    def pipet_dispense(self, amount, settling_time=0, blowout_vol=0,blowout_speed=8):
        
        self.logger.debug(f"Dispensing {amount} mL then waiting {settling_time} s")
      
        try:
            self.c9.dispense_ml(0,amount)
        except: #If there's not enough to dispense
            self.logger.warning("Dispense exceeded limit: Dispensing all liquid")
            self.c9.move_pump(0,0)

        if not self.simulate:
            time.sleep(settling_time)

        if blowout_vol > 0:
            self.logger.debug(f"Blowing out {blowout_vol} mL at speed {blowout_speed}")
            self.adjust_pump_speed(0,blowout_speed)
            self.c9.set_pump_valve(0,self.c9.PUMP_VALVE_LEFT)
            self.c9.aspirate_ml(0,blowout_vol)
            self.c9.set_pump_valve(0,self.c9.PUMP_VALVE_RIGHT)
            self.c9.dispense_ml(0,blowout_vol)

    #Default selection of pipet tip... Make extensible in the future
    def select_pipet_tip(self, volume, specified_tip):
        if specified_tip is None:
            if volume <= 0.200:
                pipet_rack_index=self.HIGHER_PIPET_ARRAY_INDEX
            elif volume <= 1.0:
                pipet_rack_index=self.LOWER_PIPET_ARRAY_INDEX
            else:
                self.pause_after_error("Cannot get tip automatically as amount is out of range")
        else:
            pipet_rack_index=specified_tip
        return pipet_rack_index
    
    #Check if the aspiration volume is within limits... Make extensible in the future
    def check_if_aspiration_volume_unacceptable(self,amount_mL):
        error_check_list = []
        error_check_list.append([self.HELD_PIPET_INDEX==self.HIGHER_PIPET_ARRAY_INDEX and amount_mL>0.25,False,"Can't pipet more than 0.25 mL from small pipet"])
        #error_check_list.append([self.HELD_PIPET_INDEX==self.HIGHER_PIPET_ARRAY_INDEX and amount_mL<0.01,False,"Can't pipet less than 10 uL from small pipet"])
        error_check_list.append([self.HELD_PIPET_INDEX==self.LOWER_PIPET_ARRAY_INDEX and amount_mL>1.00,False,"Can't pipet more than 1.00 mL from large pipet"])
        error_check_list.append([self.HELD_PIPET_INDEX==self.LOWER_PIPET_ARRAY_INDEX and amount_mL<0.025,False,"Can't pipet less than 25 uL from large pipet"])
        return self.check_for_errors(error_check_list,True) #Return True if issue

    #Integrating error messages and deliberate pauses
    def pause_after_error(self,err_message,send_slack=True):
        self.logger.error(err_message)
        if send_slack and not isinstance(self.c9, MagicMock):
            slack_agent.send_slack_message(err_message)
        if not self.simulate:  # not simulating
            input("Waiting for user to press enter or quit after error...")

    def normalize_vial_index(self, vial):
        """Accepts either a vial index (int) or vial name (str) and returns the vial index (int)."""
        if isinstance(vial, str):
            vial_idx = self.get_vial_index_from_name(vial)
            if vial_idx is None:
                raise ValueError(f"Vial name '{vial}' not found in VIAL_DF.")
            return vial_idx
        return vial

    #Get adjust the aspiration height based on how much is there
    def get_aspirate_height(self, source_vial_num, amount_mL, track_height=True, buffer=1.0):

        #Get required information
        base_height = self.get_min_pipetting_height(source_vial_num)
        vial_type = self.get_vial_info(source_vial_num, 'vial_type')
        source_vial_volume = self.get_vial_info(source_vial_num,'vial_volume')

        #The vial type affects the depth for aspiration. 
        if vial_type=='8_mL':
            height_volume_constant=6
        elif vial_type=='20_mL':
            height_volume_constant=2
        else:
            height_volume_constant=0

        volume_adjusted_height = base_height + (height_volume_constant*(source_vial_volume - amount_mL - buffer)) #How low should we go?

        if track_height and volume_adjusted_height > base_height: #If we are adjusting for height and have a height above the base (can't go lower)
            return volume_adjusted_height
        else:
            return base_height #Default is the lowest position

    #Aspirate from a vial using the pipet tool
    def aspirate_from_vial(self, source_vial_name, amount_mL,move_to_aspirate=True,specified_tip=None,track_height=True,wait_time=1,aspirate_speed=11,asp_disp_cycles=0,retract_speed=5,pre_asp_air_vol=0,post_asp_air_vol=0,move_up=True):
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
            self.logger.debug("Moving vial to clamp to uncap")
            self.move_vial_to_location(source_vial_num,location='clamp',location_index=0)
            self.uncap_clamp_vial()        

        #Check if has pipet, get one if needed based on volume being aspirated (or if tip is specified)
        if self.HELD_PIPET_INDEX is None:
            pipet_rack_index = self.select_pipet_tip(amount_mL,specified_tip)           
            self.get_pipet(pipet_rack_index)
        
        #Check for an issue with the pipet and the specified amount, pause and send slack message if so
        self.check_if_aspiration_volume_unacceptable(amount_mL) 

        #Get current volume and vial location
        location = self.get_vial_location(source_vial_num,True)
        source_vial_volume = self.get_vial_info(source_vial_num,'vial_volume')

        #Reject aspiration if the volume is not high enough
        if source_vial_volume < amount_mL:
            self.pause_after_error("Cannot aspirate more volume than in vial")

        self.logger.info(f"Pipetting from vial {self.get_vial_info(source_vial_num, 'vial_name')}, amount: {round(amount_mL, 3)} mL")


        #Adjust the height based on the volume, then pipet type
        if move_to_aspirate:
            asp_height = self.get_aspirate_height(source_vial_name, amount_mL, track_height)
            asp_height = self.adjust_height_based_on_pipet_held(asp_height)

            #The dispense height is right above the location
            disp_height = self.c9.counts_to_mm(3, location[3]) 
            disp_height = self.adjust_height_based_on_pipet_held(disp_height)
        else:
            asp_height = disp_height = self.c9.counts_to_mm(3, self.c9.get_axis_position(3)) #IF we aren't moving, everything stays the same

        self.adjust_pump_speed(0,aspirate_speed) #Adjust the pump speed if needed

        #Move to the correct location in xy
        if move_to_aspirate:
            self.c9.goto_xy_safe(location,vel=15)
        
        # #If you want to have extra aspirate and dispense steps. TODO: This isn't going to work right now as is. 
        # for i in range (0, asp_disp_cycles):
        #     self.pipet_from_location(amount_mL, aspirate_speed, height, True,initial_move=move_to_aspirate)
        #     self.pipet_from_location(amount_mL, aspirate_speed, height, False,initial_move=False)
        #     move_to_aspirate = False
        
        #Main aspiration
        # self.pipet_from_location(amount_mL, aspirate_speed, height, True, initial_move=move_to_aspirate)

        #Step 1: Move to above the site and aspirate air if needed
        if pre_asp_air_vol > 0:
            self.c9.move_z(disp_height)
            self.pipet_aspirate(pre_asp_air_vol) 
        
        #Step 2: Move to inside the site and aspirate liquid
        self.c9.move_z(asp_height)
        self.pipet_aspirate(amount_mL,wait_time) #Main aspiration of liquid plus wait
        
        #Step 3: Retract and aspirate air if needed
        if move_up:
            self.c9.move_z(disp_height, vel=retract_speed) #Retract with a specific speed
        if post_asp_air_vol > 0:
            self.pipet_aspirate(post_asp_air_vol) 

        #Record the volume change
        self.VIAL_DF.at[source_vial_num,'vial_volume']=(source_vial_volume-amount_mL)

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
    #TODO: Maybe get rid of the buffer option here and replace with the other new parameters and potentially blowout
    def dispense_from_vial_into_vial(self,source_vial_name,dest_vial_name,volume,move_to_aspirate=True,move_to_dispense=True,buffer_vol=0,track_height=True,measure_mass=False, blowout_vol=0):
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
        self.logger.info(f"Dispensing {volume} mL from {source_vial_name} to {dest_vial_name} with buffer volume {buffer_vol} mL")

        source_vial_index = self.normalize_vial_index(source_vial_name) #Convert to int if needed
        dest_vial_index = self.normalize_vial_index(dest_vial_name) #Convert to int if needed
        if volume < 0.2 and volume >= 0.01:
            tip_type = self.HIGHER_PIPET_ARRAY_INDEX
            max_volume = 0.25
        elif volume >= 0.2 and volume <= 1.00:
            tip_type = self.LOWER_PIPET_ARRAY_INDEX 
            max_volume = 1.00
        elif volume == 0:
            self.logger.warning("Cannot dispense 0 mL")
            return
        else:
            self.pause_after_error(f"Cannot accurately aspirate: {volume} mL under specified conditions.")
        
        extra_aspirate = 0
        if max_volume-volume >= 2*buffer_vol:
            extra_aspirate = 2*buffer_vol
        
        self.aspirate_from_vial(source_vial_index,round(volume+extra_aspirate,3),move_to_aspirate=move_to_aspirate,specified_tip=tip_type,track_height=track_height)
        if extra_aspirate > 0:
            self.dispense_into_vial(source_vial_index,buffer_vol,initial_move=False)
        
        mass = self.dispense_into_vial(dest_vial_index,volume,initial_move=move_to_dispense,measure_weight=measure_mass,blowout_vol=blowout_vol)
        
        if extra_aspirate > 0:
            self.dispense_into_vial(source_vial_index,buffer_vol,initial_move=move_to_dispense)

        return mass

    def select_wellplate_grid(self,well_plate_type):
        if well_plate_type == "96 WELL PLATE":
            location = well_plate_new_grid
        elif well_plate_type == "48 WELL PLATE":
            location = grid_48
        else:
            self.pause_after_error("Unknown well plate type, cannot specify location...")
            location = None
        return location

    #TODO add error checks and safeguards
    def pipet_from_wellplate(self,wp_index,volume,aspirate_speed=10,aspirate=True,move_to_aspirate=True,well_plate_type="96 WELL PLATE"):
        self.logger.debug(f"Pipetting {'aspirating' if aspirate else 'dispensing'} {volume} mL from well plate index {wp_index} of type {well_plate_type}")
        
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
            self.c9.goto_xy_safe(location, vel=15)
            self.c9.move_z(height)

        self.adjust_pump_speed(0, aspirate_speed)
        if aspirate:
            self.pipet_aspirate(volume)
        else:
            self.pipet_dispense(volume)

    #Mix the well
    def mix_well_in_wellplate(self,wp_index,volume,repeats=3,well_plate_type="96 WELL PLATE"):
        self.logger.info(f"Mixing well {wp_index} in well plate of type {well_plate_type} with volume {volume} mL for {repeats} repeats")
        self.pipet_from_wellplate(wp_index,volume,well_plate_type=well_plate_type)
        self.pipet_from_wellplate(wp_index,volume,aspirate=False,move_to_aspirate=False,well_plate_type=well_plate_type)
        for i in range (1, repeats):
            self.pipet_from_wellplate(wp_index,volume,move_to_aspirate=False,well_plate_type=well_plate_type)
            self.pipet_from_wellplate(wp_index,volume,aspirate=False,move_to_aspirate=False,well_plate_type=well_plate_type)

    #Mix in a vial
    def mix_vial(self,vial_name,volume,repeats=3):
        self.logger.info(f"Mixing vial {vial_name} with volume {volume} mL for {repeats} repeats")
        vial_index= self.normalize_vial_index(vial_name)
        self.aspirate_from_vial(vial_index,volume,3,move_up=False,track_height=False)
        self.dispense_into_vial(vial_index,volume,initial_move=False)
        for i in range (1,repeats):
            self.dispense_from_vial_into_vial(vial_index,vial_index,volume,move_to_aspirate=False,move_to_dispense=False,buffer_vol=0)

    #Dispense an amount into a vial
    def dispense_into_vial(self, dest_vial_name,amount_mL,initial_move=True,dispense_speed=11,measure_weight=False,wait_time=0,blowout_vol=0, air_vol=0):     
        
        dest_vial_num = self.normalize_vial_index(dest_vial_name) #Convert to int if needed

        measured_mass = None
        error_check_list = [] #List of specific errors for this method
        error_check_list.append([self.is_vial_pipetable(dest_vial_num), True, "Can't pipet, at least one vial is capped"])    
        self.check_for_errors(error_check_list,True) #This will cause a pause if there's an issue

        self.logger.info(f"Pipetting into vial {self.get_vial_info(dest_vial_num, 'vial_name')}, amount: {round(amount_mL, 3)} mL")
        
        dest_vial_clamped = self.get_vial_info(dest_vial_num,'location')=='clamp' #Is the destination vial clamped?
        dest_vial_volume = self.get_vial_info(dest_vial_num,'vial_volume') #What is the current vial volume?

        #If the destination vial is at the clamp and you want the weight, measure prior to pipetting
        if measure_weight and dest_vial_clamped:
            if not self.simulate:
                initial_mass = self.c9.read_steady_scale()
            else:
                initial_mass = 0

        self.adjust_pump_speed(0,dispense_speed) #Adjust pump speed if needed

        #Where is the vial?
        location= self.get_vial_location(dest_vial_num,True)
        
        #What height do we need to go to?
        height = self.c9.counts_to_mm(3, location[3]) #baseline z-height
        height = self.adjust_height_based_on_pipet_held(height)

        #Move to the location if told to (Could change this to an auto-check)
        if initial_move:               
            self.c9.goto_xy_safe(location, vel=15)   
        
        #Pipet into the vial
        #self.pipet_from_location(amount_mL, dispense_speed, height, aspirate = False, initial_move=initial_move)
        if initial_move:
            self.c9.move_z(height)
        self.pipet_dispense(amount_mL+air_vol,wait_time, blowout_vol)

        #Track the added volume in the dataframe
        self.VIAL_DF.at[dest_vial_num,'vial_volume']=self.VIAL_DF.at[dest_vial_num,'vial_volume']+amount_mL
        self.PIPET_FLUID_VOLUME -= amount_mL
        self.save_robot_status()

        #If the destination vial is at the clamp and you want the weight, measure after pipetting
        if measure_weight and dest_vial_clamped:
            if not self.simulate:
                final_mass = self.c9.read_steady_scale()
            else:
                final_mass = 0
            measured_mass = final_mass - initial_mass  

        return measured_mass

    #Dispense into a series of wells (dest_wp_num_array) a specific set of amounts (amount_mL_array)
    def dispense_into_wellplate(self, dest_wp_num_array, amount_mL_array, dispense_type = "None", dispense_speed=11,wait_time=1,well_plate_type="96 WELL PLATE",blowout_vol=0,air_vol=0):
        """
        Dispenses specified amounts into a series of wells in a well plate.
        Args:
            `dest_wp_num_array` (list or range): Array of well indices to dispense into (e.g., [0, 1, 2])
            `amount_mL_array` (list[float]): Array of amounts (in mL) to dispense into each well (e.g., [0.1, 0.2, 0.3])
        """
        self.adjust_pump_speed(0,dispense_speed) #Adjust the pump speed if needed
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
                self.c9.move_z(height)
                first_dispense = False
                
            else:
                self.c9.goto_xy_safe(location, vel=5, accel = 1, safe_height=height) #Use safe_height here!

            if dispense_type.lower() == "drop-touch" or dispense_type.lower() == "touch": #move lower & towards side of well before dispensing
                height -= 5 #goes 5mm lower when dispensing

            if self.HELD_PIPET_INDEX == self.HIGHER_PIPET_ARRAY_INDEX and dispense_speed == 11: #Adjust this later
                dispense_speed = 13 #Use lower dispense speed for smaller tip

            self.logger.info(f"Transferring {amount_mL} mL into well #{dest_wp_num_array[i]} of {well_plate_type}")


            #Dispense and then wait
            self.pipet_dispense(amount_mL+air_vol,wait_time,blowout_vol)

            #OAM Notes: Different techniques. drop and slow could both be just longer wait_time              
            if dispense_type.lower() == "drop-touch":
                self.move_rel_x(-1.75) #-1.75 for quartz WP, -3 for PS
                if not self.simulate:
                    time.sleep(0.2)
                self.move_rel_z(5) #move back up before moving to next well
            elif dispense_type.lower() == "touch":
                self.move_rel_x(2.5) #1.75 for quartz WP, 3 for PS
                if not self.simulate:
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
                                            well_plate_type="96 WELL PLATE", pipet_back_and_forth=False, blowout_vol = 0, remove_pipet=True):
        vial_indices = [self.normalize_vial_index(v) for v in vial_names]  # Normalize vial indices
        
        self.logger.info(f"Dispensing from vials {vial_names} into well plate with type {well_plate_type} using automated algorithm")

        # Step 1: Check if there's enough liquid in each vial
        well_plate_dispense_2d_array = well_plate_df.values
        vols_required = np.sum(well_plate_dispense_2d_array, axis=0)
        self.logger.debug(f"Total volumes needed (mL): {vols_required}")

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

                self.logger.debug(f"\nAspirating from Vial: {vial_index}")
                self.logger.debug(f"Total volume needed: {vol_needed:.3f} mL")

                return_vial = False
                if not self.is_vial_pipetable(vial_index):
                    self.logger.debug(f"Vial {vial_index} not open, moving to clamp")
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
                            if dispense_vol + volume <= max_volume + 1e-6:
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
                    self.logger.debug(f"Aspirating {total_aspirate:.3f} mL from vial {vial_index}")
                    self.aspirate_from_vial(vial_index, total_aspirate, specified_tip=pipet_index,
                                            aspirate_speed=dispense_speed, wait_time=wait_time,
                                            asp_disp_cycles=asp_cycles, track_height=track_height)

                    if sacrificial_dispense_vol > 0:
                        self.dispense_into_vial(vial_index, sacrificial_dispense_vol, initial_move=False,
                                                dispense_speed=dispense_speed, wait_time=wait_time)

                    self.logger.debug(f"Dispensing to wells: {well_plate_array}")
                    self.logger.debug(f"Dispense volumes: {dispense_array}")
                    self.dispense_into_wellplate(well_plate_array, dispense_array,
                                                dispense_speed=dispense_speed, wait_time=wait_time,
                                                well_plate_type=well_plate_type, blowout_vol=blowout_vol)

                    vol_dispensed += dispense_vol
                    self.logger.debug(f"Total dispensed so far: {vol_dispensed:.3f} mL")

                # Final buffer return
                if vol_remaining > 1e-6:
                    self.logger.debug(f"Returning remaining volume: {vol_remaining:.3f} mL to vial {vial_index}")
                    self.dispense_into_vial(vial_index, vol_remaining,
                                            dispense_speed=dispense_speed, wait_time=wait_time)
                if remove_pipet:
                    self.remove_pipet()

                if return_vial:
                    self.recap_clamp_vial()
                    self.return_vial_home(vial_index)

        return True

    #Prime the line from the reservoir to the vial. In theory this could happen automatically. Probably good to do it if you are using a reservoir. 
    def prime_reservoir_line(self, reservoir_index, overflow_vial, volume=0.5):
        overflow_vial = self.normalize_vial_index(overflow_vial) #Convert to int if needed
        self.logger.info(f"Priming reservoir {reservoir_index} line into vial {overflow_vial}: {volume} mL")
        self.dispense_into_vial_from_reservoir(reservoir_index,overflow_vial,volume)

    def dispense_into_vial_from_reservoir(self,reservoir_index,vial_index,volume, measure_weight = False, return_home=True):
        
        vial_index = self.normalize_vial_index(vial_index) #Convert to int if needed
        self.logger.info(f"Dispensing into vial {vial_index} from reservoir {reservoir_index}: {volume} mL")
        measured_mass = None

        #Step 1: move the vial to the clamp
        if not self.get_vial_info(vial_index,'location')=='clamp':
            self.move_vial_to_location(vial_index,'clamp',0)
        if not self.is_vial_pipetable(vial_index):
            self.uncap_clamp_vial()
        self.move_home()

        if measure_weight: #weigh before dispense (if measure_weight = True)
            initial_mass = self.c9.read_steady_scale()

        #Step 2: move the carousel
        self.c9.move_carousel(45,70) #This will take some work. Note that for now I'm just doing for position 0
        #Step 3: aspirate and dispense from the reservoir
        if not self.simulate:
            max_volume = self.c9.pumps[reservoir_index]['volume']
        else:
            max_volume = 2.5
        num_dispenses = math.ceil(volume/max_volume)
        dispense_vol = volume/num_dispenses
        self.logger.debug(f"Dispensing {dispense_vol} mL {num_dispenses} times")
        for i in range (0, num_dispenses):        
             self.c9.set_pump_valve(reservoir_index,self.c9.PUMP_VALVE_LEFT)
             self.c9.aspirate_ml(reservoir_index,dispense_vol)
             self.c9.set_pump_valve(reservoir_index,self.c9.PUMP_VALVE_RIGHT)
             self.c9.dispense_ml(reservoir_index,dispense_vol)
        if not self.simulate:
            time.sleep(1)
        vial_volume = self.get_vial_info(vial_index,'vial_volume')
        self.VIAL_DF.at[vial_index,'vial_volume']=(vial_volume+volume)
        self.save_robot_status()

        #Step 4: Return the vial back to home
        self.c9.move_carousel(0,0)

        if measure_weight: #weigh after dispense (if measure_weight = True)
            final_mass = self.c9.read_steady_scale()
            measured_mass = final_mass - initial_mass

        if not self.get_vial_info(vial_index,'capped'):
            self.recap_clamp_vial()
        if return_home:
            self.return_vial_home(vial_index)
        return measured_mass

    #Check the original status of the vial in order to send it to its home location
    def return_vial_home(self,vial_name):
        """
        Return the specified vial to its home location.
        Args:
            `vial_name` (str): Name of the vial to return home.
        """
        self.logger.info(f"Returning vial {vial_name} to home location")

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

        self.logger.debug(f"Dropping off vial {vial_name} at location {location} with index {location_index}")

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
        
        self.logger.debug(f"Grabbing vial {vial_name} with index {vial_index}")
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

        self.logger.info(f"Moving vial {self.get_vial_info(vial_index, 'vial_name')} to {location}: {location_index}")
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
        self.logger.debug("Removing cap from clamped vial")

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
        self.logger.debug("Recapping clamped vial")
        
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

        self.logger.info(f"Vortexing Vial: {self.get_vial_info(vial_index,'vial_name')}")
        
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
                self.logger.warning(error_check[2])
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
 
    #CUrrently working!
    def move_rel_xyz(self, x_distance=0, y_distance=0, z_distance=0, vel=15, pipet=True):
        self.logger.debug(f"Moving robot relative to current position by x: {x_distance}, y: {y_distance}, z: {z_distance} mm, vel: {vel}")
        current_loc_mm = self.c9.n9_fk(self.c9.get_axis_position(0), self.c9.get_axis_position(1), self.c9.get_axis_position(2))
        #tuple (x (mm), y (mm), theta (mm))
        target_x =  current_loc_mm[0] + x_distance
        target_y =  current_loc_mm[1] + y_distance
        target_z =  self.c9.counts_to_mm(3, self.c9.get_axis_position(3)) + z_distance

        self.c9.move_xyz(target_x, target_y, target_z, vel=vel)

    #Move the robot to the home position    
    def move_home(self):
        self.logger.info("Moving robot to home position")
        self.c9.goto_safe(home)
        self.c9.move_carousel(0,0)

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
        LARGE_VIAL_BASE_HEIGHT = 93    
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
        elif location_name == 'heater':
            min_height = 144

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
            elif location_name == 'heater':
                location = heater_grid_pip[location_index]
        else: #For the gripper
           if location_name=='main_8mL_rack':
                location = rack[location_index]
           elif location_name=='large_vial_rack':
                self.pause_after_error("No defined location")
           elif location_name=='small_vial_rack':
               self.pause_after_erro("No defined location")
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
    
