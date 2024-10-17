from north import NorthC9
from Locator import *

def open_gripper():
    c9.set_output(4, True)  
    c9.set_output(5, False)
    c9.delay(2)
def close_gripper():
    c9.set_output(5, True)  #gripper close
    c9.set_output(4, False)
    c9.delay(2)

def grab_well_plate_from_nr():
    open_gripper()
    c9.move_axis(6, 131631, vel=95) #left to WP
    c9.move_axis(7, 83744, vel=50) #down
    close_gripper()
    c9.move_axis(7, 75000, vel=50) #up slightly

def move_gripper_to_cytation():
    c9.move_axis(6, 52000, vel=95) #past Cytation loading
    c9.move_axis(7,0,vel=50) #up to max height
 
def release_well_plate_in_cytation():
    #OPEN CYTATION TRAY
    c9.move_axis(6, 70650, vel=50) #to well plate loading
    c9.move_axis(7, 7500,vel=25) #down slightly
    open_gripper()
    c9.move_axis(7,0, vel=50) #back to max height
    #CLOSE CYTATION TRAY

def grab_well_plate_from_cytation():
    #OPEN CYTATION TRAY
    open_gripper()
    c9.move_axis(7, 7500,vel=25) #down slightly
    close_gripper()
    c9.move_axis(7,0,vel=50) #up to max height
    c9.move_axis(6, 52000, vel=50) #past Cytation loading
    #CLOSE CYTATION TRAY
    
def return_well_plate_to_nr():
    c9.move_axis(7, 75000, vel=50) #down
    c9.move_axis(6, 131631, vel=95) #left to WP
    c9.move_axis(7, 83744, vel=50) #down
    open_gripper()
    c9.move_axis(7, 0, vel=50) #max_height
    

c9 = NorthC9('A', network_serial='AU06CNCF')
c9.default_vel = 50  # percent, 75 max


open_gripper()
input("wait until enter...")
grab_well_plate_from_nr()
move_gripper_to_cytation()
input("Please open Cytation...")
release_well_plate_in_cytation()
grab_well_plate_from_cytation()
input("Please close Cytation")
return_well_plate_to_nr()


# #for "teaching" positions
#c9.axis_servo(6, False)
#print(c9.get_axis_position(6))  # record the position
#c9.home_axis(6)  # home again when done teaching before moving


#     c9.delay(2)
# 
#     c9.set_output(5, True)  #gripper close
#     c9.set_output(4, False)
#     
#     c9.delay(2)
'''
c9.home_axis(7)
c9.home_axis(6)


c9.move_axis(7,60000, vel=70) #move down above pickup

c9.delay(4)

c9.move_axis(6, 132000, vel=95) #move to first pickup
 
c9.move_axis(7,70145, vel=70) #move down above pickup
 
c9.set_output(3, True)  #gripper close
c9.set_output(2, False)  
c9.delay(.5)
 
c9.move_axis(7,60000, vel=70)
 
c9.move_axis(6,117604, vel=95)

c9.move_axis(7,82850, vel=70)

c9.set_output(2, True)  #gripper open
c9.set_output(3, False)
c9.delay(.5)

c9.move_axis(7,60000, vel=70)

c9.move_axis(6, 132000, vel=95)

c9.move_axis(7,77152, vel=70)

c9.set_output(3, True)  #gripper close
c9.set_output(2, False)  
c9.delay(1)

c9.move_axis(7,60000, vel=95)

c9.move_axis(6,30027, vel=95)

c9.move_axis(7,83493, vel=95)

c9.set_output(2, True)  #gripper open
c9.set_output(3, False)
c9.delay(1)

c9.move_axis(7,60000, vel=70)

c9.move_axis(6, 132000, vel=95)

c9.move_axis(7,83076, vel=70)

c9.set_output(3, True)  #gripper close
c9.set_output(2, False)  
c9.delay(.5)

c9.move_axis(7,60000, vel=70)

c9.move_axis(6,15513, vel=95)

c9.move_axis(7,83336, vel=70)

c9.set_output(2, True)  #gripper open
c9.set_output(3, False)
c9.delay(1)

c9.move_axis(7,60000, vel=70)

c9.move_sync(6, 7, 1000, 1000, vel=75)

'''


# #for "teaching" positions
#c9.axis_servo(6, False)
#print(c9.get_axis_position(6))  # record the position
#c9.home_axis(6)  # home again when done teaching before moving
