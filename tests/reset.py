#Reset the workstation to a known state

import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E

#Define your workflow! Make sure that it has parameters that can be changed!
def reset(input_vial_status_file):
  
    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(input_vial_status_file,initialize_biotek=False)
    lash_e.photoreactor.turn_off_reactor_led(reactor_num=1)
    lash_e.photoreactor.turn_off_reactor_led(reactor_num=0)
    lash_e.photoreactor.turn_off_reactor_fan(reactor_num=1)
    lash_e.photoreactor.turn_off_reactor_fan(reactor_num=0)

    lash_e.nr_track.origin()
    # lash_e.nr_robot.move_home()

    # lash_e.nr_track.open_gripper()

    #lash_e.nr_robot.c9.move_pump(1, 0)

reset("../utoronto_demo/status/sample_input_vials.csv")