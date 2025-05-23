import sys
sys.path.append("C:\\Users\\Imaging Controller\\Desktop\\utoronto_demo")
sys.path.append("C:\\Users\\Imaging Controller\\Desktop\\utoronto_demo\\status")
import North_Safe
from Locator import *
from north import NorthC9

c9 = NorthC9('A', network_serial='AU06CNCF')
nr_track = North_Safe.North_Track(c9)

try:
    # for i in range(1):
    #     nr_track.get_new_wellplate()
    #     nr_track.discard_wellplate()
    nr_track.check_input_file()
    nr_track.get_new_wellplate()
    nr_track.discard_wellplate()
    nr_track.origin()

    #TODO: general path looks ok? testing edge cases & test w/ actual WP

    # nr_track.get_next_WP_from_source()
    
    # #looping to check multiple
    # for i in range(2):
    #     nr_track.get_next_WP_from_source()

    #     input("Empty the WP -> Enter")
    # print("Complete")

except KeyboardInterrupt:
    nr_track.c9.move_axis(6, 0, vel=5)