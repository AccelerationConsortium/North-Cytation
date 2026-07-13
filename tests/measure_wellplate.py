import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E

def test_measure_wellplate(plate_type): #tests track movement of wellplate to and from cytation + obtaining biotek measurements (default: wells 0,1,2)
 
    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(None)

    protocol = None #Change if you want to get actual data or test an actual protocol
    wells = [0,1,2] #Change if you want to get actual data or test an actual protocol

    data = lash_e.measure_wellplate(protocol, wells, plate_type=plate_type)

    print (data)

#asd
for i in range (0, 10):
    test_measure_wellplate(plate_type='96 WELL PLATE')


'''
What to look for:
- Is the wellplate aligned in x/z for both the cytation and north positions
- Is the wellplate settling in both positions correctly?
- Is this working for quartz, 96 and 48 well plates?
'''