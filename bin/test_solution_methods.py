#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: TEST_SOLUTION_METHODS.PY
Date: Wednesday, July  4 2012
Description: Try different solution methods for gridworld problems.
"""

from gridworld.gridworldgui import GridworldGui
import pdb
from lspi import LSPI
import numpy as np
from td import TDQ,TD,Sarsa,SampleModelValueIteration
import time

endstates = [40]
gw = GridworldGui(nrows=9,ncols=9,endstates=endstates, walls=[])
#gw.updategui=False
#gw.draw_state_labels()


#learner = TDQ(8,81,0.1,0.9,0.9)
#learner = TD(81,0.1,0.9,0.9)
#learner = Sarsa(8,81, 0.3, 0.9,0.9, 0.4)
learner = SampleModelValueIteration(8,81)

# rw_model, transition_model
# pdb.set_trace()
v,pi = learner.learn(100,gw,verbose=True)

#v,pi = gw.value_iteration()
#pi = np.ones(gw.nstates,dtype='int')

#vals = { s : learner.value(s) for s in range(gw.nstates) }
#print vals
#gw.draw_values(vals)

# for s in range(gw.nstates):
#     a = learner.best(s)
#     pi[s] = learner.best(s)

# gw.redraw()

# for i in gw.sindices:
#     pi[i] = i % 8
gw.set_arrows(pi)


# for i in gw.sindices[11:]:
#     gw.state2circle(i)
#     gw.current = i
#     for a in gw.actions:
#         gw.move(a)
#         print a
#         time.sleep(.1)
#         gw.current = i

# s = gw.endstates[0]
# gw.follow(s-1,pi__getitem__)

for i in gw.sindices:
    gw.follow(i,pi.__getitem__)

