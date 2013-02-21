#!/usr/bin/env python
'''
@author jstober

Alignment task encoded as an MDP.
'''

import os, sys, getopt, pdb, string
import random as pr
import cPickle as pickle
import numpy as np
import numpy.linalg as la
from markovdp import MDP,FastMDP,SparseMDP, Features, AliasFeatures
from utils import sp_create, sp_create_data

# 1. state representation needs to indicate where the goal is
# think about what features the agent gets (e.g. duplicate peripheral coordinates.)
# 2. goal shifts every trail

class Vergence( SparseMDP ):

    def __init__(self, size = 10, rsize = 10):
       
        self.size = 10
        self.goal_index = 0
        self.states = [(x,y) for x in range(self.size) for y in range(self.size)] # x is the agent location, y is the goal location
        self.actions = [0,1] # left or right
        self.nstates = self.size ** 2
        self.nactions = 2
        self.endstates = []
        for (i,s) in enumerate(self.states):
            if s[0] == s[1]:
                self.endstates.append(i)
        SparseMDP.__init__(self, nstates = self.nstates, nactions = self.nactions)

    def initialize_rewards(self):
        r = np.zeros(self.size ** 2)
        for es in self.endstates:
            r[es] = 1.0
        return r

    def initialize_model(self, a, i):
        cs = self.states[i][0]

        if a:
            cs = cs - 1
        else:
            cs = cs + 1
        
        if cs < 0:
            cs = 0
        if cs >= self.size:
            cs = self.size - 1

        ns = (cs,self.states[i][1])
        ni = self.states.index(ns)
        
        return [(ni,1.0)]

    def is_state(self, s):
        return s in self.states

if __name__ == '__main__':

    v = Vergence()

    # double check the dynamics
    print v.rewards
    for i in range(10):
        print v.current, v.states[v.current], v.rewards[v.current]
        print v.move(0)
    for i in range(10):
        print v.current, v.states[v.current], v.rewards[v.current]
        print v.move(1)

    # todo - add resets on success
    t = v.trace(1000, reset_on_endstate = True)
    print t

    # show reward transitions in the trace
    for i in t:
        if i[2] == 1.0:
            print i