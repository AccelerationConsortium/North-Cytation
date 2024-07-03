from north import NorthC9
from Locator import *

c9 = NorthC9('A', network_serial='AU06CNCF')
#c9.home_pump(0)
c9.default_vel = 25  # percent

# c9.move_xyz(100, 200, 200)
# c9.move_xyz(-100, 100, 292)

#c9.goto_safe(p_capture)

def remove_pipette():
    c9.goto_safe(p_remove_approach)
    c9.goto(p_remove_cap, vel=5)
    c9.move_z(292, vel=20)

def aspirate_to_clamp(vial_num, amount_mL, pipet_num):
    c9.goto_safe(p_capture_grid[pipet_num])
    c9.goto_safe(rack_pip[vial_num])
    c9.aspirate_ml(0, amount_mL)
    c9.goto_safe(vial_clamp_pip)
    c9.dispense_ml(0, amount_mL)
    remove_pipette()

#c9.home_pump(0)
#remove_pipette()



#Get vial
c9.goto_safe(rack[0])
c9.close_gripper()

#Move vial to clamp
#c9.goto_safe(vial_clamp)

c9.goto_safe(p_react_1)
c9.open_gripper()

c9.move_z(200)

c9.goto_safe(p_react_1)
c9.close_gripper()

c9.goto_safe(rack[0])
c9.open_gripper()

c9.move_z(200)

#Uncap vial
#c9.close_clamp()
#c9.uncap()

#Dispense liquid to vial
#aspirate_to_clamp(1, 0.3, 0)
#aspirate_to_clamp(2, 0.7, 1)
#aspirate_to_clamp(3, 0.3, 2)

#Recap
#c9.goto_safe(vial_clamp)
#c9.cap()

#Return and reset
#c9.open_gripper()
#c9.goto_safe(rack[0])
#c9.open_gripper()
#c9.move_z(200)