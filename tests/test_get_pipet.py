import sys
sys.path.append("../utoronto_demo")
import North_Safe
from Locator import *
from north import NorthC9
import pandas as pd

#NOte update to new syntax

input_vial_status_file="../utoronto_demo/status/color_matching_vials.txt"
vial_status = pd.read_csv(input_vial_status_file, sep=",")

c9 = NorthC9('A', network_serial='AU06CNCF')
nr = North_Safe.North_Robot(c9, input_vial_status_file)

def test_get_pipet(index_list):
    try:
        for i in index_list:
            nr.get_pipet(i)
            #nr.remove_pipet()
            nr.move_home()
            input("Waiting to move tip...")
            nr.remove_pipet()
    except KeyboardInterrupt:
        
        c9 = None

test_get_pipet([nr.LOWER_PIPET_ARRAY_INDEX]*48)
