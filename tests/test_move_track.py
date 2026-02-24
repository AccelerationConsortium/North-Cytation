import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E

#Initialize the workstation, which includes the robot, track, cytation and photoreactors
lash_e = Lash_E(None)
#lash_e.nr_track.move_through_path('pipetting_area')
