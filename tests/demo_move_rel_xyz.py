"""
Demo: Move North Robot to home, then move to several XYZ positions with different tool orientations (no pipets attached).

Usage:
    python demo_move_rel_xyz.py

This script demonstrates:
- Homing the robot (move_home)
- Moving to several relative XYZ positions with different tool orientations
- No pipets or gripper attached
"""

import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import time

# Initialize robot (simulate mode for safety)
lash_e = Lash_E(None, initialize_biotek=False)

# Move to home position (no pipets)
lash_e.nr_robot.move_home()
print("Moved to home position.")

# Define a list of (x, y, z, tool_orientation) moves
moves = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (5, 5, 5),
]

for i, (dx, dy, dz) in enumerate(moves, 1):
    print(f"Move {i}: dx={dx}, dy={dy}, dz={dz}")
    # Move relative to current position with specified tool orientation
    lash_e.nr_robot.move_rel_xyz(dx, dy, dz)
    time.sleep(1)  # Pause for demo effect
    print(f"  Arrived at relative position {i} .")

print("Demo complete.")
