import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E

lash_e = Lash_E("../utoronto_demo/status/peroxide_assay.txt")

lash_e.nr_robot.get_pipet(0)
lash_e.nr_robot.remove_pipet()