#Test opening and closing the Cytation

import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E

lash_e = Lash_E("../utoronto_demo/status/sample_input_vials.csv")

def move_carrier_out_and_in():
    lash_e.cytation.CarrierOut()
    lash_e.cytation.CarrierIn()


move_carrier_out_and_in()