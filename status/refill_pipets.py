import sys
import numpy as np
sys.path.append("../utoronto_demo")
sys.path.append("..\\utoronto_demo\\status")
PIPET_FILE = "../utoronto_demo/status/pipets.txt"

save_data = "0,0"

with open(PIPET_FILE, "w") as output:
    output.write(save_data)

print('Pipets refilled')

# # Robot status data
# robot_status = {
#     "pipets_used": {"lower_rack_1": 0, "upper_rack_1": 0},
# }

# # Writing to a file
# with open(self.ROBOT_STATUS_FILE, "w") as file:
#     yaml.dump(robot_status, file, default_flow_style=False)