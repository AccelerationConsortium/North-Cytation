import sys

sys.path.append("../utoronto_demo")

import North_Safe
from Locator import *
from north import NorthC9
import time
import pandas as pd
import numpy as np

input_vial_status_file="../utoronto_demo/status/color_matching_vials.txt"
vial_status = pd.read_csv(input_vial_status_file, sep=",")

c9 = NorthC9('A', network_serial='AU06CNCF')
nr = North_Safe.North_Robot(c9, input_vial_status_file)

# try:
#     for i in range(47):
#         nr.get_pipet(0)
#         nr.remove_pipet()
# except KeyboardInterrupt:
#     c9 = None