from north import NorthC9
from Locator import *
import os
from north import NorthC9


c9 = NorthC9('A', network_serial='AU06CNCF')

c9.home_axis(7)
c9.home_axis(6)

#for i in range (0,7):
#    c9.home_axis(i)

c9 = None
os._exit(0)