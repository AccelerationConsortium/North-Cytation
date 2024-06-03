from north import NorthC9
from Locator import *

c9 = NorthC9('A', network_serial='AU06CNCF')
c9.default_vel = 30  # percent

c9.goto_xy_safe(p_capture_grid[0])
#c9.goto_xy_safe(rack_pip[0])

def remove_pipette():
    c9.goto_safe(p_remove_approach)
    c9.goto(p_remove_cap, vel=5)
    c9.move_z(292, vel=20)