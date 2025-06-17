#Vortex the vial
import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E


VIAL_FILE = "../utoronto_demo/status/sample_input_vials.csv"  # Vials used

#Define your workflow! Make sure that it has parameters that can be changed!
def test_vortex(input_vial_status_file, target_vial_num,vortex_time):
  
    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(input_vial_status_file)

    #lash_e.nr_robot.vortex_vial(target_vial_num, vortex_time=vortex_time)
    #lash_e.nr_robot.return_vial_home(target_vial_num) 
    #lash_e.nr_robot.move_home()
    lash_e.nr_robot.mix_vial('source_vial_a', 0.9, 3)
    lash_e.nr_robot.remove_pipet()

  

test_vortex(VIAL_FILE,2, 3)