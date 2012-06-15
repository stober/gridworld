#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: TEST_GUI.PY
Date: Wednesday, June  6 2012
Description: Code to test gui changes.
"""

from gridworld.gridworldgui import GridworldGui
from lspi import LSPI
import numpy as np

endstates = [32, 2016, 1024, 1040, 1056, 1072]
gw = GridworldGui(nrows=32,ncols=64,endstates=endstates,walls=[])
t = gw.trace(10000)
z = np.zeros(gw.nfeatures())
#import pdb
#pdb.set_trace()
w = LSPI(t,0.0001,gw,z)
print gw.phi(0,0)
print gw.phi(0,1)
print w

