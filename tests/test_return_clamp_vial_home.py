import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E

VIAL_STATUS_FILE = "../utoronto_demo/status/calibration_vials.csv"
SIMULATE = False

lash_e = Lash_E(vial_file=VIAL_STATUS_FILE, simulate=SIMULATE, initialize_biotek=False)

# Return any vial currently in the clamp to its home position
for _, row in lash_e.nr_robot.VIAL_DF.iterrows():
    if row['location'] == 'clamp':
        vial_name = row['vial_name']
        lash_e.logger.info("Returning %s from clamp to home", vial_name)
        lash_e.nr_robot.return_vial_home(vial_name)
        lash_e.logger.info("Done: %s returned home", vial_name)
