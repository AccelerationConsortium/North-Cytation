from north import NorthC9
import sys
sys.path.append("C:\\Users\\Imaging Controller\\Desktop\\utoronto_demo")
from Locator import *
from North_Safe import*

c9 = NorthC9('A', network_serial='AU06CNCF')
nt = North_Track(c9)

# c9.home_axis(7) #horizontal 
# c9.home_axis(6) #vertical

nt.set_horizontal_speed(5)
nt.set_vertical_speed(5)

#new
SOURCE_X = 2950 #counts for horizontal axis
SAFE_MOVE_SOURCE_X = 22000 #area for lowering WP before going past cytation
SOURCE_Y = 82800

#copy and pastED
WP_X = 131854
WP_Y = 88396
WELL_PLATE_TRANSFER_Y = 75000


nt.close_gripper()
c9.move_axis(6, 0, vel=5)
c9.move_axis(7, SAFE_MOVE_SOURCE_X, vel=5)
c9.move_axis(6, WELL_PLATE_TRANSFER_Y, vel=5)
c9.move_axis(7, WP_X, vel=5)
c9.move_axis(6, WP_Y, vel=5)
nt.open_gripper()
c9.move_axis(6, 0, vel=5)

#WORKS :)

#for "teaching" positions
#c9.axis_servo(7, False) #horizontal
#print(c9.get_axis_position(7))  # record the position
#c9.home_axis(7)  # home again when done teaching before moving

#load the positions of the different wellplates (try 3 for now)

