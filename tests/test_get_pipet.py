import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E

# Initialize Lash_E without vial status file (None is now supported)
lash_e = Lash_E(None, initialize_biotek=False)

for i in range(0, 4):
    lash_e.nr_robot.c9.home_axis(i)