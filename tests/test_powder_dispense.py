import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E

SIMULATE = False
VIAL_FILE = "status/powder_dispense_vials.csv"
MASS_MG = 50.0

lash_e = Lash_E(VIAL_FILE, simulate=SIMULATE, initialize_biotek=False, initialize_p2=True)
lash_e.mass_dispense_into_vial("reaction_vial", mass_mg=MASS_MG)
lash_e.nr_robot.move_home()
