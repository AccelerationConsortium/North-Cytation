from os import remove
import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
from pipetting_data.pipetting_parameters import PipettingParameters

#NOte update to new syntax

input_vial_status_file="../utoronto_demo/status/calibration_vials_short.csv"

lash_e = Lash_E(input_vial_status_file, initialize_biotek=False, simulate=False)

#lash_e.nr_robot.check_input_file()

num_replicates = 1
volume = 0.15

for i in range (0, num_replicates):

    # Create custom pipetting parameters
    custom_params = PipettingParameters(
        aspirate_speed=15,           # Slower aspiration
        dispense_speed=15,          # Medium dispense speed
        post_asp_air_vol=0.05,     # Small air gap
        pre_asp_air_vol=0.5,
        asp_disp_cycles=0
    )

    #lash_e.nr_robot.dispense_from_vial_into_vial('measurement_vial_0','measurement_vial_1', volume, parameters=custom_params)

lash_e.nr_robot.move_home()


# Create custom pipetting parameters
smalltip_2MeTHF_params = PipettingParameters(
    aspirate_speed=15,           # Slower aspiration
    dispense_speed=15,          # Medium dispense speed
    post_asp_air_vol=0.05,     # Small air gap
    pre_asp_air_vol=0.5,
    asp_disp_cycles=3 #tested w/o the cycles and it worked ok (no droplet, no dripping) - maybe not needed for 2MeTHF
)

# Create custom pipetting parameters
largetip_2MeTHF_params = PipettingParameters(
    aspirate_speed=13,           # Slower aspiration
    dispense_speed=13,          # Medium dispense speed
    post_asp_air_vol=0.05,     # Small air gap
    pre_asp_air_vol=0.3,
    asp_disp_cycles=3
)

# Create custom pipetting parameters
smalltip_2MeTHF_polymer_params = PipettingParameters(
    aspirate_speed=13,           # Slower aspiration
    dispense_speed=13,          # Medium dispense speed
    post_asp_air_vol=0.01,     # Small air gap
    pre_asp_air_vol=0.7,
    asp_disp_cycles=0
)

# Create custom pipetting parameters
smalltip_toluene_polymer_params = PipettingParameters(
    aspirate_speed=13,           # Slower aspiration
    dispense_speed=13,          # Medium dispense speed
    post_asp_air_vol=0.01,     # Small air gap
    pre_asp_air_vol=0.7,
    asp_disp_cycles=0
)# Same parameters sd 2MeTHF polymer solution. Worked really really well for toluene polymer solution.

smalltip_toluene_params = PipettingParameters(
    aspirate_speed=15,           # Slower aspiration
    dispense_speed=15,          # Medium dispense speed
    post_asp_air_vol=0.05,     # Small air gap
    pre_asp_air_vol=0.5,
    asp_disp_cycles=0
    )
#Lessons learned:
# 1. Polymer solutions are easier to pipet. Especially with small tips, which is all we need. 
# 2. Bare solvents may need asp_disp cycles to prevent dripping, but don't seem to drip.
# 3. Slowing down the robot seems to help?