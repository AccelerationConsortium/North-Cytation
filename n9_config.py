from north import NorthC9
from time import sleep

c9 = NorthC9("A", network_serial="AU06CNCF")  # enter the address of the robot's controller on the network

c9.get_info()

current_elbow = c9.get_home_offset(1)
current_shoulder = c9.get_home_offset(2)
current_z = c9.get_home_offset(3)

print("Current elbow offset is:", current_elbow)
print("Current shoulder offset is:", current_shoulder)
print("Current z_axis offset is:", current_z)

c9.set_home_offset(1, 0)
c9.set_home_offset(2, 0)
c9.set_home_offset(3, 0)

c9.home_robot()

print("")
input("Press Enter to close grippers on calibration tool...")
c9.close_gripper()
sleep(2)
c9.robot_servo(0)

print("")
print("Fasten the calibration post at (0, 225)")
input("Press Enter when the tool is aligned with the calibration post...")
elbow_1 = c9.get_axis_position(1)
shoulder_1 = c9.get_axis_position(2)

print("")
print("Flip the orientation of the elbow.")
input("Press Enter when the tool is aligned with the calibration post...")
elbow_2 = c9.get_axis_position(1)
shoulder_2 = c9.get_axis_position(2)

elbow_offset = 21250 - int((elbow_1 + elbow_2)/2)
shoulder_offset = 33667 - int((shoulder_1 + shoulder_2)/2)


print("")
print("With the arm in a safe configuration, move the z-axis to its lowest position.")
input("Press Enter when the z-axis is at its minimum...")
z_axis_max_cts = c9.get_axis_position(3)
z_axis_offset = 26200 - z_axis_max_cts


print("")
print("The elbow offset is:", elbow_offset)
print("The shoulder offset is:", shoulder_offset)
print("The z-axis offset is:", z_axis_offset)
if elbow_offset < 0 or shoulder_offset < 0 or z_axis_offset < 0:
    print("WARNING: Offsets are typically positive. Consider retrying the calibration.")

response = input("To save this calibration, enter [Y/y], otherwise enter [N/n]")
print("")
if response == "Y" or response == "y":
    c9.set_home_offset(1, elbow_offset)
    c9.set_home_offset(2, shoulder_offset)
    c9.set_home_offset(3, z_axis_offset)
    print("Calibration saved")
else:
    print("Calibration NOT saved")
    c9.set_home_offset(1, current_elbow)
    c9.set_home_offset(2, current_shoulder)
    c9.set_home_offset(3, current_z)
    print("Previous calibration restored")
    
c9.home_robot()