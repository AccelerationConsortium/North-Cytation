import sys
import yaml
import os

# Get the path to the robot_state directory relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
robot_state_dir = os.path.join(script_dir, "..", "robot_state")
ROBOT_STATUS_FILE = os.path.join(robot_state_dir, "robot_status.yaml")
PIPET_RACKS_FILE = os.path.join(robot_state_dir, "pipet_racks.yaml")

# Load pipet racks configuration to get all rack names
try:
    with open(PIPET_RACKS_FILE, "r") as file:
        pipet_racks = yaml.safe_load(file)
    rack_names = list(pipet_racks.keys())
    print(f"Found {len(rack_names)} pipet racks: {', '.join(rack_names)}")
except FileNotFoundError:
    print(f"Warning: {PIPET_RACKS_FILE} not found, using fallback rack names")
    rack_names = ["large_tip_rack_1", "large_tip_rack_2", "small_tip_rack_1", "small_tip_rack_2"]

# Get the robot status
try:
    with open(ROBOT_STATUS_FILE, "r") as file:
        robot_status = yaml.safe_load(file)
except FileNotFoundError:
    print(f"Error: {ROBOT_STATUS_FILE} not found")
    sys.exit(1)

# Ensure pipets_used section exists
if "pipets_used" not in robot_status:
    robot_status["pipets_used"] = {}

# Set number of pipets used to 0 for all racks dynamically
reset_count = 0
for rack_name in rack_names:
    robot_status["pipets_used"][rack_name] = 0
    reset_count += 1
    print(f"  Reset {rack_name} to 0")

# Writing to a file
with open(ROBOT_STATUS_FILE, "w") as file:
    yaml.dump(robot_status, file, default_flow_style=False)

print(f'\nSuccess: {reset_count} pipet racks refilled (counters reset to 0)')


