#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: TEST_SOLUTION_METHODS.PY
Date: Wednesday, July  4 2012
Description: Try different solution methods for gridworld problems.
"""

from gridworld.gridworldgui import GridworldGui
from lspi import LSPI
import numpy as np

endstates = [28]
gw = GridworldGui(nrows=9,ncols=9,endstates=endstates, walls=[])
gw.draw_state_labels()


