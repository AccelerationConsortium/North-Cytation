import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E

def test_measure_wellplate(): #tests track movement of wellplate to and from cytation + obtaining biotek measurements (default: wells 0,1,2)
 
    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(None, simulate=True)

    #protocol = r"C:\Protocols\300_900_sweep.prt"
    #protocol = r"C:\Protocols\SQ_degradation_sweep.prt" 
    protocol = None

    data = lash_e.measure_wellplate(protocol, [0,1,2], plate_type='96 WELL PLATE')

    #data = lash_e.cytation.run_protocol(protocols, wells=range(0,3), plate_type="48 WELL PLATE")

    print (data)
 

#asd
test_measure_wellplate()