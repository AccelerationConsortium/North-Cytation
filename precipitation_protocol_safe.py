import North_Safe

#Input data
POLYMER_VOLUME = 0.1 #how much polymer solution in mL?
ANTISOLVENT_VOLUME = 2 #How much antisolvent solution in mL?
PIPET_VOLUME = 0.5 #How much volume do we drip in at a time?
open_vials = [1,2]
vial_names = ["Sample", "Polymer Solution", "Antisolvent"]

nr = North_Safe.North_Robot(open_vials, vial_names)

nr.reset_after_initialization()
nr.move_vial_to_clamp(0)
nr.uncap_clamp_vial()

nr.pipet_from_vial_into_vial(1, 0, POLYMER_VOLUME)
nr.remove_pipet()

num_pipets = int(ANTISOLVENT_VOLUME / PIPET_VOLUME)
for i in range (0, num_pipets):
    nr.pipet_from_vial_into_vial(2, 0, PIPET_VOLUME, dispense_type="by_drop")

nr.remove_pipet()
nr.recap_clamp_vial()
nr.return_vial_from_clamp(0)