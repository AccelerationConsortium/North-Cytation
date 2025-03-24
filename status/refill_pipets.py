import sys
import numpy as np
import yaml
sys.path.append("../utoronto_demo")
sys.path.append("..\\utoronto_demo\\status")
ROBOT_STATUS_FILE = "../utoronto_demo/status/robot_status.yaml"

# save_data = "0,0"

# with open(PIPET_FILE, "w") as output:
#     output.write(save_data)


# # Robot status data
# Get the robot status
with open(ROBOT_STATUS_FILE, "r") as file:
    robot_status = yaml.safe_load(file)

#set number of pipets used to 0
robot_status["pipets_used"]["lower_rack_1"] = 0
robot_status["pipets_used"]["upper_rack_1"] = 0

# # Writing to a file
with open(ROBOT_STATUS_FILE, "w") as file:
    yaml.dump(robot_status, file, default_flow_style=False)

print('Pipets refilled')


