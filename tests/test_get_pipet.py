import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E

# Initialize Lash_E without vial status file (None is now supported)
lash_e = Lash_E(None, initialize_biotek=False)

def test_get_pipet(tip_types):
    """
    Test pipet getting with proper tip type specification
    Args:
        tip_types: List of pipet tip types ('large_tip' or 'small_tip')
    """
    for tip_type in tip_types:
        print(f"Testing pipet type: {tip_type}")
        lash_e.nr_robot.get_pipet(tip_type)
        lash_e.nr_robot.move_home()
        lash_e.nr_robot.remove_pipet()

# Test with different pipet types (use strings, not indices)
# Available types: 'large_tip', 'small_tip'

# Test small tips multiple times
test_get_pipet(['small_tip'] * 5)

# Uncomment to test large tips:
# test_get_pipet(['large_tip'] * 3)

# Uncomment to test mixed types:
# test_get_pipet(['small_tip', 'large_tip', 'small_tip'])