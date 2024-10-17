import sys
sys.path.append("../utoronto_demo")
from biotek import Biotek

# Cytation 5
readerType = 21
ComPort = 4
appName = 'Gen5.Application'

gen5 = Biotek(readerType, ComPort, appName)

gen5.CarrierOut()
gen5.CarrierIn()

