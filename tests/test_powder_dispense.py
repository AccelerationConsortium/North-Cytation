import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E

SIMULATE = True
VIAL_FILE = "status/powder_dispense_vials.csv"
MASS_MG = 50.0

lash_e = Lash_E(VIAL_FILE, simulate=SIMULATE, initialize_biotek=False)
lash_e.mass_dispense_into_vial("reaction_vial", mass_mg=MASS_MG)
