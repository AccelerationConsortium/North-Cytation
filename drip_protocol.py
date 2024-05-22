from north import NorthC9
from Locator import *
import numpy as np

calibration_file_name = 'Calibration_Data_Water_Drip_20uL.txt'
volume = 0.200
drip_amt = 0.02
repeats = 3


def remove_pipette():
    c9.goto_safe(p_remove_approach)
    c9.goto(p_remove_cap, vel=5)
    c9.move_z(292, vel=20)

def aspirate_to_clamp(vial_num, amount_mL):
    c9.goto_safe(rack_pip[vial_num])
    c9.aspirate_ml(0, amount_mL)
    c9.goto_safe(vial_clamp_pip)
    c9.dispense_ml(0, amount_mL)

#Start the C9
c9 = NorthC9('A', network_serial='AU06CNCF')
c9.home_pump(0)
#c9.home_robot()
c9.default_vel = 50  # percent

#Get vial
c9.goto_safe(rack[0])
c9.close_gripper()

#Move vial to clamp
c9.goto_safe(vial_clamp)

#Uncap vial
c9.close_clamp()
c9.uncap()

#Get pipet tip
c9.goto_safe(p_capture_grid[0])

c9.open_clamp()
c9.zero_scale()

calibration_data = [['Volume (mL)', 'Mass (g)']]

#Calibration Routine. Pipet from one vial to another, just water, get data
for i in range (0, repeats):
    num_drips = int(volume/drip_amt)
    c9.goto_safe(rack_pip[1])
    c9.aspirate_ml(0, volume)
    c9.goto_safe(vial_clamp_pip)
    for j in range (0, num_drips):

        #Measure initial weight
        initial_mass = c9.read_steady_scale()
       
        c9.dispense_ml(0, drip_amt)

        #Measure final weight
        final_mass = c9.read_steady_scale()

        mass = str(final_mass - initial_mass)[0:6]
        print("Try #" + str(i+1) + " mass : " + str(mass)[0:5] + " g")
        calibration_data.append([volume, mass])

np.savetxt(calibration_file_name, calibration_data, delimiter='\t', fmt ='% s')

#Remove pipet tip
remove_pipette()          

#Recap
c9.close_clamp()
c9.goto_safe(vial_clamp)
c9.cap()

#Return and reset
c9.open_clamp()
c9.goto_safe(rack[0])
c9.open_gripper()
c9.move_z(200)