import sys
sys.path.append("../utoronto_demo") 
from north import NorthC9
from Locator import *
import os

c9 = NorthC9('A', network_serial='AU06CNCF')
#print(c9.pumps)

for i in range (6,8):
    c9.home_axis(i)

#c9.move_axis(7,80000)

#c9.home_axis(2)


#for i in range (0,7):
#    c9.home_axis(i)
