import sys
import yaml
import pandas as pd
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
from North_Safe import PipettingParameters

VIAL_STATUS_FILE = "status/calibration_vials_short.csv"
HARDWARE_CONFIG_FILE = "sdl_pipette_calibration/protocols/north_robot_hardware.yaml"
SIMULATE = False

# Read the vial that was in use from hardware config (never gets reset)
with open(HARDWARE_CONFIG_FILE, 'r') as f:
    hw = yaml.safe_load(f)
clamp_vial_name = hw['vials']['measurement_vial']
print(f"Hardware config says measurement vial was: '{clamp_vial_name}'")

# Patch the CSV so the robot knows that vial is in the clamp
df = pd.read_csv(VIAL_STATUS_FILE)
in_clamp = df[df['location'] == 'clamp']
if len(in_clamp) == 0:
    print(f"CSV shows no vial in clamp - patching '{clamp_vial_name}' to clamp position")
    df.loc[df['vial_name'] == clamp_vial_name, 'location'] = 'clamp'
    df.loc[df['vial_name'] == clamp_vial_name, 'location_index'] = 0
    df.to_csv(VIAL_STATUS_FILE, index=False)
    print("CSV patched.")
else:
    clamp_vial_name = in_clamp.iloc[0]['vial_name']
    print(f"CSV already shows '{clamp_vial_name}' in clamp - no patch needed")

lash_e = Lash_E(vial_file=VIAL_STATUS_FILE, simulate=SIMULATE, initialize_biotek=False)

# Find vial in clamp and return it home
for _, row in lash_e.nr_robot.VIAL_DF.iterrows():
    if row['location'] == 'clamp':
        vial_name = row['vial_name']
        lash_e.logger.info("Dispensing tip contents back into %s", vial_name)
        lash_e.logger.info("Returning %s from clamp to home", vial_name)
        lash_e.nr_robot.return_vial_home(vial_name)
        lash_e.logger.info("Done: %s returned home", vial_name)
        lash_e.logger.info("Done: %s returned home", vial_name)
