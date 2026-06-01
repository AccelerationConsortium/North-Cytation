"""
Demo: Move North Robot to home, then move to several XYZ positions with different tool orientations (degrees, no pipets attached).

Usage:
    python demo_move_rel_xyz.py

This script demonstrates:
- Homing the robot (move_home)
- Moving to several relative XYZ positions with different tool orientations (degrees)
- No pipets or gripper attached
"""

from North_Safe import North_Robot
import time

# Initialize robot (simulate mode for safety)
robot = North_Robot(simulate=True)

# Move to home position (no pipets)
robot.move_home()
print("Moved to home position.")

# Tool orientation in degrees (e.g., 0 = vertical, 90 = horizontal, 45 = angled)
moves = [
    (10, 0, 0, 0),      # vertical (0 degrees)
    (0, 10, 0, 90),     # horizontal (90 degrees)
    (0, 0, 10, 45),     # angled (45 degrees)
    (5, 5, 5, 0),       # vertical (0 degrees)
]


def awake_wait(seconds):
    interval = 30
    waited = 0
    while waited < seconds:
        sleep_time = min(interval, seconds - waited)
        time.sleep(sleep_time)
        waited += sleep_time
        print("I'm awake")

for i, (dx, dy, dz, orientation_deg) in enumerate(moves, 1):
    print(f"Move {i}: dx={dx}, dy={dy}, dz={dz}, tool_orientation={orientation_deg} degrees")
    robot.move_rel_xyz(dx, dy, dz, tool_orientation=orientation_deg)
    awake_wait(30)  # 30s demo wait with 'I'm awake' every 30s
    print(f"  Arrived at relative position {i} with tool orientation {orientation_deg} degrees.")


# No saving if simulate=True (demo does not save any files)
if not getattr(robot, 'simulate', False):
    # Example: robot.save_state('robot_state.json')
    pass

print("Demo complete.")
