import sys
sys.path.append("../utoronto_demo")
from north import NorthC9
from Locator import *
import os
from prefect import flow


@flow
def move_track(location:int):
    c9 = NorthC9('A', network_serial='AU06CNCF')

    c9.home_axis(7)
    c9.move_axis(7,50000,vel=25)

    c9 = None
    os._exit(0)

if __name__=="__main__":
    move_track.serve(name="move_track")
