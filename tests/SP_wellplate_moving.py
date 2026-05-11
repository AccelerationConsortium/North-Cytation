import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E

SIMULATE = False

# Initialize the workstation in simulation mode (no hardware movement)
lash_e = Lash_E(None, simulate=SIMULATE)

# Grab a wellplate from the source stack and deliver it to the pipetting area
lash_e.grab_new_wellplate()

# Move the wellplate from the pipetting area into the Cytation (Biotek) reader
lash_e.move_wellplate_to_cytation()

# Retrieve the wellplate from the Cytation back to the pipetting area
lash_e.move_wellplate_back_from_cytation()

# Move the wellplate from the pipetting area to the waste stack
lash_e.discard_used_wellplate()
