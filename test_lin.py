from north import NorthC9
from Locator import *

c9 = NorthC9('A', network_serial='AU06CNCF')
c9.default_vel = 50  # percent, 75 max

#c9.home_axis(6)
#c9.home_axis(7)
# #for "teaching" positions
#c9.axis_servo(6, False)
#print(c9.get_axis_position(6))  # record the position
#c9.home_axis(6)  # home again when done teaching before moving


c9.set_output(4, True)  #gripper open
c9.set_output(5, False)

c9.delay(1)

c9.move_axis(6, 130800, vel=95) #left to WP
c9.move_axis(7, 81300, vel=50) #down

c9.set_output(5, True)  #gripper close
c9.set_output(4, False)

c9.delay(2)

c9.move_axis(7, 70000, vel=50) #up slightly
c9.move_axis(6,50000,vel=95) #Past Cytation loading bay


'''
#move back to WP holder
c9.move_axis(6, 130800, vel=50) #left to WP
c9.move_axis(7, 81300, vel=50) #down

c9.set_output(4, True)  #gripper open
c9.set_output(5, False)

c9.delay(1)

c9.move_axis(7, 0, vel=50) #up

c9.set_output(5, True)  #gripper close
c9.set_output(4, False)
'''


'''
#     c9.delay(2)
# 
#     c9.set_output(5, True)  #gripper close
#     c9.set_output(4, False)
#     
#     c9.delay(2)

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
