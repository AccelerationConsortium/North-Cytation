import North_Safe

open_vials = [1,2,3]
nr = North_Safe.North_Robot(open_vials)

nr.reset_after_initialization()
nr.move_vial_to_clamp(0)
nr.uncap_clamp_vial()
nr.pipet_from_vial_into_vial(1, 0, 0.5)
nr.remove_pipet()
nr.pipet_from_vial_into_vial(5, 1, 0.5)
nr.remove_pipet()
nr.recap_clamp_vial()
nr.return_vial_from_clamp(0)