#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: TEST_GUI.PY
Date: Wednesday, June  6 2012
Description: Code to test gui changes.
"""

from gridworld.gridworldgui import GridworldGui

endstates = [32, 2016, 1024, 1040, 1056, 1072]
gw = GridworldGui(nrows=32,ncols=64,goal=endstates,walls=[])
t = gw.trace(1000)



