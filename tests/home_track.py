#Home the track of a NorthC9

import sys
sys.path.append("../utoronto_demo") 
from north import NorthC9


c9 = NorthC9('A', network_serial='AU06CNCF')

for i in range (6,8):
    c9.home_axis(i) 
