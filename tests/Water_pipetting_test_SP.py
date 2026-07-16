import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E

VIAL_FILE = "status/water_test_vials.csv"
REPEATS = 3

TESTS = [
    #{"tip_type": "small_tip", "volume": 0.1},   # 100 uL
    {"tip_type": "large_tip", "volume": 0.4},   # 400 uL
]

lash_e = Lash_E(VIAL_FILE, simulate=False, initialize_biotek=False)

# Move water vial to clamp so scale measurements are possible
lash_e.nr_robot.move_vial_to_location("water", "clamp", 0)

for test in TESTS:
    tip_type = test["tip_type"]
    volume = test["volume"]
    print(f"\n=== {tip_type} | {volume*1000:.0f} uL x {REPEATS} reps ===")
    lash_e.nr_robot.get_pipet(tip_type)

    for i in range(REPEATS):
        # Read scale before aspiration
        mass_before = lash_e.nr_robot.c9.read_steady_scale()
        print(f"  Rep {i+1}: mass before aspiration = {mass_before*1000:.2f} mg")

        # Aspirate from clamped vial
        lash_e.nr_robot.aspirate_from_vial("water", volume)
        mass_after_asp = lash_e.nr_robot.c9.read_steady_scale()
        aspirated_mass = mass_before - mass_after_asp
        print(f"  Rep {i+1}: aspirated {volume*1000:.0f} uL | mass removed = {aspirated_mass*1000:.2f} mg")

        # Dispense back with weight measurement (vial is clamped)
        measured_mass, _ = lash_e.nr_robot.dispense_into_vial("water", volume, measure_weight=True)
        print(f"  Rep {i+1}: dispensed {volume*1000:.0f} uL | mass returned = {measured_mass*1000:.2f} mg")

    lash_e.nr_robot.remove_pipet()
    lash_e.nr_robot.move_home()

# Return vial home after all tests
lash_e.nr_robot.return_vial_home("water")
print("\nDone.")
lash_e.nr_robot.move_home()
