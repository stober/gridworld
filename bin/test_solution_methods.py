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
from td import TDQ,TD,Sarsa

endstates = [40]
gw = GridworldGui(nrows=9,ncols=9,endstates=endstates, walls=[])
#gw.draw_state_labels()


#learner = TDQ(8,81,0.1,0.9,0.9)
#learner = TD(81,0.1,0.9,0.9)
learner = Sarsa(8,81, 0.3, 0.9,0.9, 0.1)
#learner.learn(100,gw,verbose=True)

v,pi = gw.value_iteration()

#vals = { s : learner.value(s) for s in range(gw.nstates) }
#print vals
#gw.draw_values(vals)

for s in range(gw.nstates):
    #a = learner.best(s)
    a = pi[s]
    gw.draw_arrow(s,a)

gw.redraw()


def policy(state):
    return pi[state]

for i in gw.sindices:
    gw.follow(i,policy)

