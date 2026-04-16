#Test the capping and decapping of vials

import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E


lash_e = Lash_E(None, show_gui=False)    
lash_e.c9.open_gripper()
