import sys
sys.path.append("C:\\Users\\Imaging Controller\\Desktop\\utoronto_demo")
sys.path.append("C:\\Users\\Imaging Controller\\Desktop\\utoronto_demo\\status")
import os

from north import NorthC9
import North_Safe
from Locator import *
import pandas as pd
import time

VIAL_FILE = "C:\\Users\\Imaging Controller\\Desktop\\utoronto_demo\\status\\vial_status_wellplate.txt"

c9 = NorthC9('A', network_serial='AU06CNCF')
vial_df = pd.read_csv(VIAL_FILE, delimiter='\t', index_col='vial index')
nr = North_Safe.North_Robot(c9,vial_df)
nr.set_robot_speed(15)

radians = [50000, 100000, 150000, 200000, 250000, 300000]
times = []
velocities = [55, 60, 65, 70, 75, 80, 85, 90]

vel_spin = 50
df_lists = []

try:
    nr.reset_after_initialization()
    for v in velocities:
        times = []
        times.append(v) #append velocity as first element
        vel_spin = v
        for r in radians:
            nr.goto_location_if_not_there(rack[0]) #go to rack
            nr.c9.close_gripper()

            nr.c9.move_z(292) #move up

            print(f"Velocity: {vel_spin}, radian: {r}")

            start_time = time.time() #start timer
            nr.c9.move_axis(nr.c9.GRIPPER, r, vel=vel_spin)
            end_time = time.time()
            times.append(end_time-start_time)

            nr.c9.goto_safe(rack[0])
            nr.c9.open_gripper()

        print(f"radians: {radians}")
        print(f"times (vel = {vel_spin}): {times}")

        df_lists.append(times)

    nr.c9.move_z(292)

    df = pd.DataFrame(df_lists, columns =(["Velocity"] + radians))
    print(df)
    df.to_csv("vortex_speed_times.csv")

except KeyboardInterrupt:
    c9 = None